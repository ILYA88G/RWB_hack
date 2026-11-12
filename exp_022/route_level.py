"""One route-level target, chronological selection, and budget-neutral redistribution."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "exp_021"))
import improve_network as net
from backtest_network import load_folds, metric

SEED = 22
CAP = .05
WEIGHTS = (.10, .25)
RAW = ROOT.parent / "rwb_case_data"


class History:
    def __init__(self):
        cols = [net.TARGET, *net.STATUS]
        df = pd.read_parquet(RAW / "train_solo_track.parquet",
                             columns=["timestamp", "route_id", *cols])
        df["timestamp"] = pd.to_datetime(df.timestamp)
        self.series = df.groupby("timestamp")[cols].sum().astype(float)
        df = df.set_index(["timestamp", "route_id"]).sort_index()
        self.panel = df[net.TARGET].unstack()
        expected = pd.MultiIndex.from_product([self.panel.index, self.panel.columns])
        assert df.index.equals(expected) and self.panel.shape == (4630, 1000)
        assert self.panel.index.as_unit("ns").equals(pd.date_range(
            "2025-07-28", periods=4630, freq=net.STEP).as_unit("ns"))
        self.y = self.panel.to_numpy()
        self.status = [df[c].to_numpy(dtype=np.float32).reshape(self.y.shape) for c in net.STATUS]
        for a in [self.y, *self.status]:
            assert np.isfinite(a).all() and a.min() >= 0
        self.cs = np.pad(np.cumsum(self.y, axis=0, dtype=float), ((1, 0), (0, 0)))
        self.r = np.arange(self.y.shape[1])
        self.cache = {}
        self.rng = np.random.default_rng(SEED)
        self.times = {day: day*48 + self.rng.integers(0, 48, size=len(self.r))
                      for day in range(8, len(self.y)//48+1)}

    def mean(self, t, w):
        return (self.cs[t+1, self.r] - self.cs[t+1-w, self.r]) / w

    def features(self, t):
        r = self.r
        recent = self.mean(t, 8)
        scale = self.mean(t, 336).clip(1)
        x = {f"lag{lag}": self.y[t-lag, r]/scale for lag in (0, 1, 2, 4, 8)}
        x.update(week=scale, recent=recent/scale, mean4=self.mean(t, 4)/scale,
                 mean48=self.mean(t, 48)/scale)
        anchor = np.zeros(len(r))
        for lag in (48, 336):
            ratio = (recent+1)/(self.mean(t-lag, 8)+1)
            x[f"ratio{lag}"] = ratio
            profile = np.stack([self.y[t-lag+h, r] for h in range(1, 9)])
            anchor += .5 * ratio.clip(.8, 1.2) * profile.sum(axis=0, dtype=float)
            for h, value in enumerate(profile):
                x[f"profile{lag}_{h+1}"] = value/scale
        for j, s in enumerate(self.status):
            x[f"status{j+1}"] = s[t, r]/scale
            x[f"change{j+1}"] = (s[t, r]-s[t-2, r])/scale
        x.update(slot=t%48, weekday=(t//48)%7, anchor=anchor/scale)
        return pd.DataFrame(x, dtype=np.float32), anchor, scale

    def predict(self, origin, learned=True):
        end = self.panel.index.get_loc(origin)
        x, anchor, scale = self.features(end)
        estimates = {"seasonal": anchor}
        if not learned:
            return estimates, 0
        batches, labels, weights = [], [], []
        for day, t in self.times.items():
            valid = (t >= end-84*48) & (t+8 <= end)
            if not valid.any():
                continue
            # Cache complete, already observed days only. The partial last day
            # uses a causal placeholder for masked rows, discarded before fitting.
            if day in self.cache:
                data, residual, size = self.cache[day]
            else:
                at = np.minimum(t, end-8)
                data, prior, size = self.features(at)
                target = self.cs[at+9, self.r]-self.cs[at+1, self.r]
                residual = (target-prior)/size
                if t.max()+8 <= end:
                    self.cache[day] = data, residual, size
            batches.append(data.loc[valid])
            labels.append(residual[valid])
            weights.append(size[valid])
        model = LGBMRegressor(objective="regression_l1", n_estimators=90,
                              num_leaves=15, learning_rate=.07, min_child_samples=100,
                              random_state=SEED, deterministic=True, force_col_wise=True,
                              n_jobs=3, verbosity=-1)
        model.fit(pd.concat(batches), np.concatenate(labels), sample_weight=np.concatenate(weights))
        estimates["lgbm"] = (anchor + scale*model.predict(x)).clip(0)
        return estimates, sum(map(len, labels))


def adjust(b, a, weight):
    assert np.isfinite(a).all() and a.min() >= 0 and a.sum() > 0
    total = b.sum(axis=1)
    a = a * (total.sum()/a.sum())
    ratio = np.divide(a, total, out=np.ones_like(a), where=total > 1)
    change = weight*(ratio-1)
    adjusted = b*(1+change.clip(-CAP, CAP))[:, None]
    adjusted *= b.sum(axis=0)/adjusted.sum(axis=0)
    np.testing.assert_allclose(adjusted.sum(axis=0), b.sum(axis=0), rtol=1e-12)
    return adjusted, np.mean(abs(change) >= CAP)


def apply_test(test, submission, estimates, weight):
    routes = estimates.index.sort_values()
    idx = routes.get_indexer(test.route_id)
    assert routes.is_unique and idx.min() >= 0
    h = test.h.to_numpy()
    b = np.empty((len(routes), 8))
    b[idx, h] = net.aligned(test, submission)
    p, _ = adjust(b, estimates.reindex(routes).to_numpy(), weight)
    return pd.DataFrame(dict(id=test.id, y_pred=p[idx, h]))


def self_test():
    origin = pd.Timestamp("2025-09-01")
    routes = np.arange(1000)*3+4
    grid = pd.MultiIndex.from_product([routes, pd.date_range(origin+net.STEP, periods=8,
        freq=net.STEP)], names=["route_id", "timestamp"]).to_frame(index=False)
    grid["id"] = (np.arange(8000)*11+37).astype(str)
    grid = net.validate_grid(grid.sample(frac=1, random_state=4), routes, origin)
    b = pd.DataFrame(dict(id=grid.id, y_pred=10.+grid.h)).sample(frac=1, random_state=6)
    a = pd.Series(1., index=routes)
    a.iloc[:500] = 3.
    out = apply_test(grid, b, a.sample(frac=1, random_state=3), .25)
    p = net.aligned(grid, out)
    old = net.aligned(grid, b)
    np.testing.assert_allclose(p, old*np.where(grid.route_id < routes[500], 1.05, .95))
    np.testing.assert_allclose(np.bincount(grid.h, weights=p), np.bincount(grid.h, weights=old))


def compare(rows, fold, base_name, key, truth, base, pred, quartiles, **meta):
    groups = {"all": (truth, base, pred)}
    for q in range(4):
        groups[f"Q{q+1}"] = tuple(a[quartiles == q] for a in (truth, base, pred))
    for h in range(8):
        groups[f"h{h+1}"] = tuple(a[:, h:h+1] for a in (truth, base, pred))
    total_ae = abs(truth-base).sum()
    for group, (y, b, p) in groups.items():
        ae, old = abs(y-p).sum(), abs(y-b).sum()
        relative = (p[b>0]/b[b>0])-1
        rows.append(dict(fold=fold, base=base_name, model=key[0], weight=key[1],
                         cap=CAP if key[1] else 0, group=group, n=y.size, **meta,
                         **metric(y, p), abs_error=ae, delta_ae=ae-old,
                         baseline_error_share=old/total_ae, delta_over_total_ae=(ae-old)/total_ae,
                         level_ae=abs(y.sum(axis=1)-p.sum(axis=1)).sum(),
                         min_relative=relative.min(), max_relative=relative.max()))


def main():
    self_test()
    data = History()
    reports = ROOT/"reports"
    origins, ys, _, old = load_folds(data.panel, reports)
    rule = json.loads((net.EXP/"rule.json").read_text())
    bases = []
    for o, b in zip(origins, old):
        forecast = net.fit_predict(data.series, o, [rule["model"]])[rule["model"]]
        sums = b.sum(axis=0)
        proposed = net.proposed_totals(sums, forecast, rule["mode"], rule["alpha"], rule["cap"])
        totals, _ = net.safe_totals(sums, proposed, net.q0_at(data.panel, o))
        bases.append(b*(totals/sums))
    assert abs(metric(ys, bases)["wape"]-.3241313) < 1e-7
    rows, history = [], {}
    baseline = ("base", 0)
    grid = [(m, w) for m in ("seasonal", "lgbm") for w in WEIGHTS]
    proxy = "021a proxy on 008b"

    def evaluate(fold, origin, y, b, keys, name):
        y = y.astype(float)
        levels, n = data.predict(origin, any(m == "lgbm" for m, _ in keys))
        volume = data.panel.loc[:origin].iloc[-336:].mean()
        q = pd.qcut(volume.rank(method="first"), 4, labels=False).to_numpy()
        history[fold] = y, b, q, {}
        for key in [baseline, *keys]:
            m, w = key
            p, frac = adjust(b, levels[m], w) if w else (b, 0.)
            compare(rows, fold, name, key, y, b, p, q, origin=str(origin),
                    train_n=n if m=="lgbm" else 0, seed=SEED, cap_fraction=frac)
            history[fold][3][key] = p

    def pool(label, folds, keys, name):
        batch = [history[f] for f in folds]
        y, b, q = [np.concatenate([item[col] for item in batch]) for col in range(3)]
        for key in keys:
            compare(rows, label, name, key, y, b,
                    np.concatenate([item[3][key] for item in batch]), q)

    for i in (0, 1):
        evaluate(i+3, origins[i], ys[i], bases[i], grid, proxy)
    pool("dev", (3, 4), [baseline, *grid], proxy)
    dev = [r for r in rows if r["group"]=="all" and r["fold"]=="dev"]
    best = [min((r for r in dev if r["model"]==m), key=lambda r:r["wape"])
            for m in ("seasonal", "lgbm")]
    selected = min(best, key=lambda r:r["wape"])
    key = selected["model"], selected["weight"]
    print("Development selection", key, CAP, "delta AE", selected["delta_ae"], flush=True)
    selected["frozen"] = True
    pd.DataFrame(rows).to_csv(OUT/"experiments.csv", index=False)
    if selected["delta_ae"] >= 0:
        return
    finalists = [(r["model"], r["weight"]) for r in best]
    evaluate(5, origins[2], ys[2], bases[2], finalists, proxy)
    pool("pooled", (3, 4, 5), [baseline, *finalists], proxy)

    def result(f, group="all"):
        return next(r for r in rows if (r["fold"], r["group"], r["model"], r["weight"])
                    == (f, group, *key))

    def top_ok(f):
        r = result(f, "Q4")
        return r["delta_ae"] <= .005*(r["abs_error"]-r["delta_ae"])

    passed = (result("pooled")["delta_ae"] < 0 and result(5)["delta_ae"] < 0
              and sum(result(f)["delta_ae"] < 0 for f in (3, 4, 5)) >= 2)
    if passed:
        for f in (1, 2):
            with np.load(reports/f"exp_005/ensemble_005b_fold_{f}.npz") as a:
                evaluate(f, pd.Timestamp(a["origin"].item()), a["y_true"],
                         a["lgbm_pred"].astype(float).clip(0), [key], "005a")
        pool("external", (1, 2), [baseline, key], "005a")
        passed = (result("external")["delta_ae"] < 0 and
                  all(top_ok(f) for f in (1, 2, 5)))
    pd.DataFrame(rows).to_csv(OUT/"experiments.csv", index=False)
    if passed:
        o = data.panel.index[-1]
        test = net.validate_grid(pd.read_parquet(RAW/"test_solo_track.parquet"), data.panel.columns, o)
        base = pd.read_csv(net.EXP/"submission_model_021a.csv", dtype={"id":str})
        levels, n = data.predict(o, key[0]=="lgbm")
        pred = apply_test(test, base, pd.Series(levels[key[0]], index=data.panel.columns), key[1])
        tmp = OUT/"candidate.tmp"
        try:
            pred.to_csv(tmp, index=False)
            reread = net.aligned(test, pd.read_csv(tmp, dtype={"id":str}))
            old = net.aligned(test, base)
            sums = np.bincount(test.h, weights=reread)
            offsets = sums-np.bincount(test.h, weights=old)
            np.testing.assert_allclose(offsets, 0, atol=1e-5)
            print("budgets", net.budgets(sums, net.q0_at(data.panel, o)), "sum differences", offsets)
            rel = reread/old-1
            print("test n, mean/max absolute change, relative min/max/mean absolute",
                  n, abs(reread-old).mean(), abs(reread-old).max(), rel.min(), rel.max(), abs(rel).mean())
            tmp.replace(OUT/"submission_model_022a.csv")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    if parser.parse_args().test_only:
        self_test()
    else:
        main()
