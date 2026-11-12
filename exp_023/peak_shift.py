"""Replay one fixed, causal timing correction; see plan.md for the decision rule."""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "exp_021"))
import improve_network as net
from backtest_network import load_folds, metric

WEIGHT = .20
DAYS = (1, 2, 3, 7)
SHIFTS = np.array([0, -1, 1])  # positive means a later peak
SATURDAYS = pd.date_range("2025-08-16 10:30", periods=6, freq="14D")
RAW = ROOT.parent / "rwb_case_data"


def analog(history, relative_slots):
    t = len(history)-1
    return np.mean([history[t-48*d+relative_slots] for d in DAYS], axis=0).T


def normalized(profile):
    return profile/profile.mean(axis=1, keepdims=True).clip(min=1)


def directions(history):
    if len(history) < 349:
        raise ValueError("Insufficient route history")
    offsets = np.arange(-11, 1)
    observed = history[-12:].T
    ref = normalized(observed)
    errors = np.stack([abs(ref-normalized(analog(history, offsets-k))).mean(axis=1)
                       for k in SHIFTS])
    best = errors.argmin(axis=0)
    use = ((errors.min(axis=0) < .95*errors[0]) & (errors[1] != errors[2])
           & (observed.sum(axis=1) > 0))
    return np.where(use, SHIFTS[best], 0)


def mixtures(b):
    """Row-matched alternatives in SHIFTS order, before column balancing."""
    result = [b]
    for k in SHIFTS[1:]:
        shifted = b[:, (np.arange(8)-k).clip(0, 7)]
        mass = shifted.sum(axis=1, keepdims=True)
        scaled = np.divide(shifted*b.sum(axis=1, keepdims=True), mass,
                           out=b.copy(), where=mass > 0)
        result.append((1-WEIGHT)*b + WEIGHT*scaled)
    return np.stack(result)


def balance(p, b):
    p = p.copy()
    rows = b.sum(axis=1, keepdims=True)
    cols = b.sum(axis=0)
    for _ in range(200):
        sums = p.sum(axis=0)
        p *= np.divide(cols, sums, out=np.ones_like(sums), where=sums > 0)
        if np.allclose(p.sum(axis=1, keepdims=True), rows, rtol=1e-12, atol=1e-9):
            np.testing.assert_allclose(p.sum(axis=0), cols, rtol=1e-12, atol=1e-9)
            return p
        sums = p.sum(axis=1, keepdims=True)
        p *= np.divide(rows, sums, out=np.ones_like(sums), where=sums > 0)
    raise ValueError("Route/horizon balancing did not converge")


def apply(b, shifts, smooth=False):
    variants = mixtures(b)
    index = np.select([shifts == -1, shifts == 1], [1, 2], default=0)
    p = variants[index, np.arange(len(b))].copy()
    if smooth:
        active = shifts != 0
        p[active] = (variants[1, active]+variants[2, active])/2
    return balance(p, b)


def test_submission(test, submission, shift_by_route):
    routes = shift_by_route.index
    r = routes.get_indexer(test.route_id)
    if (r < 0).any() or len(routes) != 1000:
        raise ValueError("Missing route")
    h = test.h.to_numpy()
    b = np.empty((len(routes), 8))
    b[r, h] = net.aligned(test, submission)
    p = apply(b, shift_by_route.to_numpy())
    return pd.DataFrame(dict(id=test.id, y_pred=p[r, h]))


def self_test():
    rng = np.random.default_rng(23)
    prefix = np.ones((400, 3))
    offsets = np.arange(-12, 2)
    pattern = np.array([1, 2, 3, 5, 2, 2, 9, 3, 2, 1, 1, 6, 3, 4])
    for d in DAYS:
        prefix[399-48*d+offsets, :2] = pattern[:, None]
    prefix[-12:, 0] = pattern[:12]
    prefix[-12:, 1] = pattern[2:]
    np.testing.assert_array_equal(directions(prefix), [1, -1, 0])
    ramp = np.arange(1., 9.)[None, :]
    later = np.array([[1., 1, 2, 3, 4, 5, 6, 7]])
    np.testing.assert_allclose(mixtures(ramp)[2], .8*ramp+.2*later*ramp.sum()/later.sum())
    origin = pd.Timestamp("2025-10-25 10:30")
    routes = np.arange(1000)*7
    grid = pd.MultiIndex.from_product([routes, pd.date_range(origin+net.STEP, periods=8,
        freq=net.STEP)], names=["route_id", "timestamp"]).to_frame(index=False)
    grid["id"] = rng.permutation(8000).astype(str)
    grid = net.validate_grid(grid.sample(frac=1, random_state=7), routes, origin)
    # Opposite shifts for two complementary peak shapes give testable directions.
    b = np.ones((1000, 8))
    b[::2, 3] = 10
    b[1::2, 4] = 10
    r = grid.route_id.to_numpy()//7
    h = grid.h.to_numpy()
    sub = pd.DataFrame(dict(id=grid.id, y_pred=b[r, h])).sample(frac=1, random_state=2)
    shifts = pd.Series(np.where(np.arange(1000)%2, -1, 1), index=routes)
    p = net.aligned(grid, test_submission(grid, sub, shifts.sample(frac=1, random_state=9)))
    expected = b.copy()
    expected[::2, 3:5] = [8.2, 2.8]
    expected[1::2, 3:5] = [2.8, 8.2]
    np.testing.assert_allclose(p, expected[r, h])
    np.testing.assert_allclose(np.bincount(h, weights=p), b.sum(axis=0))
    np.testing.assert_allclose(np.bincount(r, weights=p), b.sum(axis=1))


@dataclass
class Case:
    label: str
    origin: pd.Timestamp
    source: str
    y: np.ndarray
    b: np.ndarray
    q: np.ndarray


def record(rows, label, source, name, y, b, p, q, shifts, origin=""):
    def add(group, yt, base, pred):
        old = abs(yt-base).sum()
        ae = abs(yt-pred).sum()
        rows.append(dict(sample=label, origin=origin, source=source, method=name, group=group,
                         n=yt.size, **metric(yt, pred), abs_error=ae, delta_ae=ae-old,
                         delta_ae_relative=(ae-old)/old,
                         shift_early=(shifts==-1).sum() if group=="all" else None,
                         shift_late=(shifts==1).sum() if group=="all" else None))
    add("all", y, b, p)
    for k in range(4):
        mask = q == k
        add(f"Q{k+1}", y[mask], b[mask], p[mask])
    for h in range(8):
        add(f"h{h+1}", y[:, h], b[:, h], p[:, h])


def evaluate(rows, cases, label, methods, panel):
    pooled = {m: [] for m in methods}
    for case in cases:
        prefix = panel.loc[:case.origin].to_numpy()
        shift = directions(prefix) if "causal" in methods else None
        for name in methods:
            if name == "base":
                s, p = np.zeros(len(case.b), dtype=int), case.b
            elif name == "oracle":
                loss = abs(mixtures(case.b)-case.y).sum(axis=2)
                s = SHIFTS[loss.argmin(axis=0)]
                p = apply(case.b, s)
            else:
                s = shift
                p = apply(case.b, s, smooth=name=="smooth")
            record(rows, case.label, case.source, name, case.y, case.b, p, case.q, s, case.origin)
            pooled[name].append((case.y, case.b, p, case.q, s))
    for name, parts in pooled.items():
        record(rows, label, cases[0].source, name, *map(np.concatenate, zip(*parts)))
    return pooled


def submit(panel):
    origin = panel.index[-1]
    test = net.validate_grid(pd.read_parquet(RAW/"test_solo_track.parquet"), panel.columns, origin)
    sub = pd.read_csv(net.EXP/"submission_model_021a.csv", dtype={"id": str})
    b = net.aligned(test, sub)
    shifts = pd.Series(directions(panel.to_numpy()), index=panel.columns)
    candidate = test_submission(test, sub, shifts)
    with NamedTemporaryFile(dir=OUT, suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    try:
        candidate.to_csv(tmp, index=False)
        p = net.aligned(test, pd.read_csv(tmp, dtype={"id": str}))
        r = panel.columns.get_indexer(test.route_id)
        for key in [r, test.h]:
            np.testing.assert_allclose(np.bincount(key, weights=p), np.bincount(key, weights=b),
                                       rtol=2e-12, atol=1e-8)
        budgets = net.budgets(np.bincount(test.h, weights=p), net.q0_at(panel, origin))
        print("Test min budget, mean/max absolute change, max horizon drift:",
              budgets.min(), abs(p-b).mean(), abs(p-b).max(),
              abs(np.bincount(test.h, weights=p)-np.bincount(test.h, weights=b)).max())
        tmp.replace(OUT/"submission_model_023a.csv")
    finally:
        tmp.unlink(missing_ok=True)


def main():
    self_test()
    series, panel = net.load_train(RAW/"train_solo_track.parquet")
    origins, ys, _, old = load_folds(panel, ROOT/"reports")
    rule = json.loads((net.EXP/"rule.json").read_text())
    cases, rows = [], []

    def quartiles(origin):
        volumes = panel.loc[:origin].iloc[-336:].mean()
        return pd.qcut(volumes.rank(method="first"), 4, labels=False).to_numpy()

    for fold, (o, b, y) in enumerate(zip(origins, old, ys), 3):
        sums = b.sum(axis=0)
        f = net.fit_predict(series, o, [rule["model"]])[rule["model"]]
        proposed = net.proposed_totals(sums, f, rule["mode"], rule["alpha"], rule["cap"])
        totals, _ = net.safe_totals(sums, proposed, net.q0_at(panel, o))
        cases.append(Case(f"fold{fold}", o, "021a proxy on 008b", y.astype(float),
                          b*(totals/sums), quartiles(o)))
    base_wape = metric(ys, [c.b for c in cases])["wape"]
    np.testing.assert_allclose(base_wape, .3241313, rtol=0, atol=1e-7)
    evaluate(rows, cases[:2], "dev", ["base", "oracle"], panel)
    def row(label, method, group="all"):
        return next(r for r in rows if (r["sample"],r["method"],r["group"])==(label,method,group))

    potential = row("dev","base")["wape"]-row("dev","oracle")["wape"]
    print("008b score, proxy WAPE, oracle dev gain:", metric(ys, old)["score"], base_wape,
          potential, flush=True)
    if potential < .001 or any(row(c.label,"oracle")["delta_ae"] >= 0 for c in cases[:2]):
        pd.DataFrame(rows).to_csv(OUT/"experiments.csv", index=False)
        return
    dev = evaluate(rows, cases[:2], "dev", ["causal", "smooth"], panel)
    # Base development rows already exist, append only the fold 5 measurements.
    last = evaluate(rows, cases[2:], "single", ["base", "causal", "smooth"], panel)
    rows[:] = [r for r in rows if r["sample"] != "single"]
    for name in ("base", "causal", "smooth"):
        if name == "base":
            parts = [(c.y, c.b, c.b, c.q, np.zeros(len(c.b))) for c in cases]
        else:
            parts = dev[name]+last[name]
        record(rows, "main", cases[0].source, name, *map(np.concatenate, zip(*parts)))
    sats = []
    for o in SATURDAYS:
        b = analog(panel.loc[:o].to_numpy(), np.arange(1, 9))
        y = panel.loc[o+net.STEP:o+8*net.STEP].to_numpy(dtype=float).T
        sats.append(Case(o.date().isoformat(), o, "route analog 1/2/3/7 days", y, b, quartiles(o)))
    evaluate(rows, sats, "saturdays", ["base", "causal", "smooth"], panel)
    pd.DataFrame(rows).to_csv(OUT/"experiments.csv", index=False)
    checks = {
        "pooled_gains": all(row(k,"causal")["delta_ae"] < 0 for k in ("dev","main","saturdays")),
        "fold5": row("fold5","causal")["delta_ae"] < 0,
        "main_wins": sum(row(c.label,"causal")["delta_ae"] < 0 for c in cases) >= 2,
        "sat_wins": sum(row(c.label,"causal")["delta_ae"] < 0 for c in sats) >= 4,
        "beats_smoothing": all(row(k,"causal")["wape"] < row(k,"smooth")["wape"]
                               for k in ("dev","saturdays")),
        "top_quartile": all(row(c.label,"causal","Q4")["delta_ae_relative"] <= .005
                            for c in [cases[-1], *sats])}
    print("Decision checks", {k:bool(v) for k,v in checks.items()})
    if all(checks.values()):
        submit(panel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    if parser.parse_args().test_only:
        self_test()
    else:
        main()
