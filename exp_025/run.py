"""Run exp_025 (first horizon) and exp_026 (network level) on shared inputs."""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
NET_OUT = ROOT / "exp_026"
sys.path[:0] = [str(ROOT / "exp_024"), str(ROOT)]
import long_cat as replay
import train_network_calibration_008b as legacy
from backtest_network import load_folds, metric

net = replay.net
STEP = net.STEP


def features(v, half, t, r):
    y = v[:, :, 0]
    # Same clock slot on seven preceding days, available even for training rows.
    level = np.maximum(np.mean([y[t-48*d, r] for d in range(1, 8)], axis=0), 100)
    x = {"log_level": np.log1p(level), "slot": t % 48, "weekday": (t//48) % 7}
    for lag in (0, 1, 2, 3, 4, 6, 8, 48, 96):
        x[f"y_{lag}"] = y[t-lag, r]/level
        x[f"half_{lag}"] = half[t-lag, r]/level
    for d in (1, 2, 3, 7):
        x[f"next_half_day_{d}"] = half[t+1-48*d, r]/level
    for j in range(1, 7):
        x[f"status_{j}"] = v[t, r, j]/level
        x[f"change_{j}"] = (v[t, r, j]-v[t-2, r, j])/level
    return pd.DataFrame(x, dtype=np.float32), level


def first_horizon(v, half):
    end = len(v)-1
    n = v.shape[1]
    rng = np.random.default_rng(25)
    # One draw per route in each of three 8-hour strata, for each of 42 days.
    starts = end-42*48 + 16*np.arange(42*3)
    t = (starts[:, None] + rng.integers(0, 16, (len(starts), n))).ravel()
    r = np.tile(np.arange(n), len(starts))
    assert t.min() >= 336 and (t+1 <= end).all()
    x, scale = features(v, half, t, r)
    # delta = incoming - outgoing; outgoing = y[t] - reconstructed half[t].
    incoming = (v[t+1, r, 0] - half[t, r])/scale
    model = LGBMRegressor(objective="regression_l1", n_estimators=120, num_leaves=15,
                          learning_rate=.06, min_child_samples=80, n_jobs=3, random_state=25,
                          deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(x, incoming, sample_weight=scale/scale.mean())
    query, level = features(v, half, np.full(n, end), np.arange(n))
    return np.maximum(half[-1] + model.predict(query)*level, 0)


def network_factors(prefix, sums, q0):
    recent = []
    for distance in (16, 8):
        end = len(prefix)-distance
        previous = replay.fixed_network_forecast(prefix.iloc[:end])
        actual = prefix[net.TARGET].iloc[end:end+8]
        assert len(actual) == 8 and actual.index[-1] <= prefix.index[-1]
        recent.append(actual.sum()/previous.sum())
    forecast = replay.fixed_network_forecast(prefix).sum()
    level = np.exp(.5*np.mean(np.log(recent)))
    factors = []
    for calibration in (1., level):
        factor = np.clip(1 + .25*(forecast*calibration/sums.sum()-1), .99, 1.01)
        safe, fallback = net.safe_totals(sums, sums*factor, q0)
        factor = 1. if fallback else factor
        np.testing.assert_allclose(safe, sums*factor, rtol=1e-12)
        factors.append(factor)
    return *factors, *recent


def saturday_base(prefix, observations, q0):
    end = len(observations)-1
    y = observations[:, :, 0]
    analog = np.mean([y[end-48*d+1:end-48*d+9].T for d in (1, 2, 3, 7)], axis=0, dtype=float)
    data, cols = legacy.build_network(prefix.rename_axis("timestamp").reset_index())
    forecast = legacy.predict_network(data, cols, prefix.index[-1], "lgbm28")
    base8, _ = legacy.corrected(analog, forecast, .8, "per_step")
    old = base8.sum(axis=0)
    shape = net.proposed_totals(old, replay.fixed_network_forecast(prefix), "shape", .15, .02)
    safe, _ = net.safe_totals(old, shape, q0)
    return base8*(safe/old)


def adjust_h1(base, specialist, weight=.2):
    b = base.copy()
    new = specialist*(base[:, 0].sum()/specialist.sum())
    b[:, 0] = (1-weight)*base[:, 0] + weight*new
    np.testing.assert_allclose(b.sum(axis=0), base.sum(axis=0), rtol=1e-12)
    np.testing.assert_array_equal(b[:, 1:], base[:, 1:])
    return b


@dataclass
class Case:
    origin: pd.Timestamp
    truth: np.ndarray
    q4: np.ndarray
    predictions: dict


def make_case(o, truth, q4, b, specialist, factor0, factor, pure_cat=False):
    h1 = adjust_h1(b, specialist)
    preds = {"base": b, "h1": h1}
    if pure_cat:
        preds["specialist"] = adjust_h1(b, specialist, 1.)
    else:
        preds.update(control=b*factor0, level=b*factor, both=h1*factor)
    for p in preds.values():
        assert p.shape == truth.shape == (1000, 8) and np.isfinite(p).all() and p.min() >= 0
    return Case(o, truth, q4, preds)


def summarize(cohorts):
    rows = []
    for name, cases in cohorts.items():
        for batch in [[c] for c in cases] + [cases]:
            y = np.concatenate([c.truth for c in batch])
            q4 = np.concatenate([c.q4 for c in batch])
            date = str(batch[0].origin) if len(batch) == 1 else "pooled"
            for method in batch[0].predictions:
                p = np.concatenate([c.predictions[method] for c in batch])
                b = np.concatenate([c.predictions["base"] for c in batch])
                for group, idx in (("all", np.s_[:, :]), ("h1", np.s_[:, 0]), ("Q4", np.s_[q4, :])):
                    stats = metric(y[idx], p[idx])
                    original = metric(y[idx], b[idx])
                    rows.append(dict(base=name, date=date, method=method, group=group,
                                     n=y[idx].size, **stats,
                                     delta_score=stats["score"]-original["score"],
                                     delta_wape=stats["wape"]-original["wape"],
                                     relative_ae=stats["wape"]/original["wape"]-1))
    return pd.DataFrame(rows)


def choose(table):
    full = table[table.group == "all"].set_index(["base", "method", "date"]).sort_index()
    def pooled(base, method):
        return full.loc[(base, method, "pooled")]
    def passes(method):
        for name, wins in (("M", 2), ("Saturday", 4)):
            group = full.loc[(name, method)]
            row = group.loc["pooled"]
            if row.delta_score >= 0 or row.delta_wape >= 0 or (group.drop("pooled").delta_score < 0).sum() < wins:
                return False
        q = table[(table.group == "Q4") & (table.base == "M") & (table.date != "pooled") & (table.method == method)]
        return q.relative_ae.max() <= .005 and pooled("B", method).delta_score < 0
    pure = table[(table.base == "C") & (table.group == "h1") & (table.date == "pooled") & (table.method == "h1")]
    h1 = passes("h1") and pure.delta_score.item() <= 0
    level = passes("level") and all(pooled(b, "level").score < pooled(b, "control").score for b in ("M", "Saturday"))
    if h1 and level and passes("both") and all(
        pooled(b, "both").score < min(pooled(b, "h1").score, pooled(b, "level").score) for b in ("M", "Saturday")
    ):
        return "both"
    if h1:
        return "h1"
    return "level" if level else None


def write_candidate(method, v, half, series, routes):
    test = net.validate_grid(pd.read_parquet(net.RAW/"test_solo_track.parquet"), routes, series.index[-1])
    sub = pd.read_csv(net.EXP/"submission_model_021a.csv", dtype=str)
    baseline = net.aligned(test, sub)
    r = pd.Index(routes).get_indexer(test.route_id)
    h = test.h.to_numpy()
    matrix = np.empty((len(routes), 8))
    matrix[r, h] = baseline
    pred = matrix.copy()
    if method in ("h1", "both"):
        pred = adjust_h1(pred, first_horizon(v, half))
    factor = 1.
    if method in ("level", "both"):
        _, factor, *_ = network_factors(series, matrix.sum(axis=0), half[-1].sum(dtype=float))
        pred *= factor
    sums = matrix.sum(axis=0)*factor
    net.budgets(sums, half[-1].sum(dtype=float))
    path = (OUT if method == "h1" else NET_OUT)/f"candidate_{method}.csv"
    data = sub.set_index("id").loc[test.id].reset_index()
    changed = (h == 0) if method == "h1" else np.ones(len(h), dtype=bool)
    data.loc[changed, "y_pred"] = [str(float(x)) for x in pred[r[changed], h[changed]]]
    data.to_csv(path, index=False)
    restored = net.aligned(test, pd.read_csv(path, dtype={"id": str}))
    np.testing.assert_allclose(restored, pred[r, h], rtol=1e-12)
    np.testing.assert_allclose(np.bincount(h, weights=restored), sums, rtol=1e-12)
    print(f"Verified {path.name}; mean |change| {np.abs(restored-baseline).mean():.3f}")


def main():
    started = time.perf_counter()
    NET_OUT.mkdir(exist_ok=True)
    reader, builder = replay.load_recipe()
    v, times, routes = reader(net.RAW/"train_solo_track.parquet")
    np.testing.assert_array_equal(routes, np.arange(1000))
    series = pd.DataFrame(v.sum(axis=1, dtype=float), index=times, columns=[net.TARGET, *net.STATUS])
    panel = pd.DataFrame(v[:, :, 0], index=times, columns=routes)
    origins, truths, _, old = load_folds(panel, ROOT/"reports")
    half_fn = builder.features.__globals__["halfhour_causal"]
    # The original recursion only writes half[t] from observations through t.
    half = half_fn(v[:, :, 0])
    with np.load(replay.ARCHIVE) as a:
        np.testing.assert_array_equal(a["origin"], pd.DatetimeIndex(origins).to_numpy(dtype="datetime64[ns]"))
        np.testing.assert_array_equal(a["route_id"], routes)
        np.testing.assert_array_equal(a["horizon"], replay.H)
        base = replay.normalize(old, a["B_totals"])
        cat = replay.normalize(a["cat_raw"], a["B_totals"])
    mix = .75*base + .25*cat
    np.testing.assert_allclose(metric(truths, mix)["score"], .326441755, atol=1e-9, rtol=0)
    sats = list(pd.date_range("2025-09-20 10:30", periods=6, freq="7D"))
    cohorts = {name: [] for name in ("B", "M", "C", "Saturday")}
    diagnostics = []
    for o in [*origins, *sats]:
        end = times.get_loc(o)+1
        prefix = series.iloc[:end]
        v_p = v[:end]
        half_p = half[:end]
        # Audit causality of the full-pass helper at each origin.
        expected, _ = net.last_halfhour(panel, o)
        np.testing.assert_array_equal(half_p[-1], expected)
        q0 = expected.sum(dtype=float)
        specialist = first_horizon(v_p, half_p)
        means = v_p[-336:, :, 0].mean(axis=0, dtype=float)
        q4 = pd.qcut(pd.Series(means).rank(method="first"), 4, labels=False).to_numpy() == 3
        if o in origins:
            i = origins.index(o)
            bases = {"B": base[i], "M": mix[i], "C": cat[i]}
            y = truths[i].astype(float)
        else:
            bases = {"Saturday": saturday_base(prefix, v_p, q0)}
            y = panel.iloc[end:end+8].to_numpy(dtype=float).T
        sums = next(iter(bases.values())).sum(axis=0)
        f0, f, recent16, recent8 = network_factors(prefix, sums, q0)
        diagnostics.append(dict(origin=o, control_factor=f0, level_factor=f,
                                ratio_older=recent16, ratio_latest=recent8))
        for name, b in bases.items():
            np.testing.assert_allclose(b.sum(axis=0), sums, rtol=1e-12)
            cohorts[name].append(make_case(o, y, q4, b, specialist, f0, f, name == "C"))
        print(f"Finished {o}", flush=True)
    results = summarize(cohorts)
    sat = results.query("base == 'Saturday' and method == 'base' and date == 'pooled' and group == 'all'").iloc[0]
    np.testing.assert_allclose(sat.score, .39640928323955504, rtol=0, atol=1e-9)
    results[results.method.isin(["base", "h1", "specialist"])].to_csv(OUT/"experiments.csv", index=False)
    results[(results.base != "C") & results.method.isin(["base", "control", "level", "both"])].to_csv(
        NET_OUT/"experiments.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(NET_OUT/"factors.csv", index=False)
    decision = choose(results)
    if decision:
        write_candidate(decision, v, half, series, routes)
    print(f"Decision: {decision}. Total elapsed {time.perf_counter()-started:.1f} s.")


if __name__ == "__main__":
    main()
