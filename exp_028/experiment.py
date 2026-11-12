"""Freeze a decision on development dates and score it on later full-recipe folds."""
import argparse
import itertools
import json
import sys

from components import (DATES, DEV, DIAGNOSTIC, HOLDOUT, OUT, ROOT, ROUTES, TEST,
                        folder, origin, read_stage, replay, write_stage)
import predict_model_019b as old
import upgrade_model_020a as patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "exp_025"))
from run import network_factors

FAMILIES = (("lag_half", "lag"), ("fresh",), ("early", "forecast"), ("network",))
SINGLES = [item for family in FAMILIES for item in family]
BASE = "020a"


def local_components(times, values, totals):
    half = old.halfhour_causal(values[:, :, 0])
    # 336 slots of feature history + 14 days of fit + 8 labels + origin.
    window = slice(-1017, None)
    args = (values[window], times[window], half[window], totals)
    base = patch.local_adjustment(*args)
    lag = patch.local_adjustment(*args, upstream=True) - base
    lag[:, :2] = 0
    series = pd.DataFrame(values.sum(axis=1, dtype=float), index=times, columns=replay.CHANNELS)
    factor, _, _, _ = network_factors(series, totals, half[-1].sum(dtype=float))
    q = pd.qcut(pd.Series(values[-336:, :, 0].mean(axis=0, dtype=float)).rank(method="first"),
                4, labels=False).to_numpy()+1
    return base, lag, factor, q


def build(day):
    if (folder(day) / "variants.npz").exists():
        return
    sums = read_stage(day, "network")["totals"]
    short = read_stage(day, "short")
    norm = replay.archive.normalize
    base15 = (.25*norm(short["linear"], sums) + .25*norm(short["boosted"], sums)
              + .5*norm(short["cat"], sums))
    long = norm(read_stage(day, "long")["pred"], sums)
    fresh = norm(read_stage(day, "fresh")["pred"], sums)
    tcn = read_stage(day, "temporal")
    if day == DIAGNOSTIC:
        archived = read_stage(day, "tcn")
        for k in ("adapted", "scores", "known"):
            np.testing.assert_allclose(tcn[k], archived[k], rtol=1e-6, atol=.05)
    def allocate(key):
        return old.allocate(tcn[key], tcn["known"], sums)
    adapted = allocate("adapted")
    times, values = replay.history(end=origin(day))
    local, lag, factor, q = local_components(times, values, sums)
    base = .25 * base15 + .25 * long + .5 * adapted + local
    assert base.shape == (1000, 8) and base.min() >= 0
    np.testing.assert_allclose(base.sum(axis=0), sums, rtol=1e-12)
    write_stage(day, "variants", base=base, lag=lag, fresh=.25*(fresh-long),
                early=.5*(allocate("early")-adapted), forecast=.5*(allocate("forecast")-adapted),
                network_factor=factor, q=q, totals=sums, adapted=adapted, short15=base15,
                long_cat=long, local=local)


def predict(data, name):
    p = data["base"].copy()
    keys = [] if name == BASE else name.split("+")
    for key in keys:
        if key == "lag_half":
            p += .5 * data["lag"]
        elif key != "network":
            p += data[key]
    totals = data["totals"]
    affected = (p < 0).any(axis=0)
    if affected.any():
        p[:, affected] = replay.archive.normalize(np.maximum(p[:, affected], 0), totals[affected])
    if keys in (["lag"], ["lag_half"]):
        np.testing.assert_array_equal(p[:, :2], data["base"][:, :2])
    if "network" in keys:
        p *= data["network_factor"]
        totals = totals * data["network_factor"]
    if p.shape != (1000, 8) or not np.isfinite(p).all() or p.min() < 0:
        raise ValueError("Invalid forecast matrix")
    np.testing.assert_allclose(p.sum(axis=0), totals, rtol=1e-12)
    return p, totals


def round_trip(day, p, totals, dest):
    t = origin(day)
    if day == TEST:
        test = pd.read_parquet(replay.net.RAW / "test_solo_track.parquet")
        reference = pd.read_csv(ROOT / "submissions/submission_model_020a.csv", dtype={"id": str})
    else:
        test = pd.MultiIndex.from_product(
            [ROUTES, pd.date_range(t+replay.net.STEP, periods=8, freq=replay.net.STEP)],
            names=["route_id", "timestamp"]).to_frame(index=False)
        test["id"] = np.arange(len(test)).astype(str)
        test = test.sample(frac=1, random_state=28)
        reference = pd.DataFrame({"id": test.id, "y_pred": 1.})
    test = replay.net.validate_grid(test, ROUTES, t)
    csv = old.make_submission(test, reference, p, pd.DatetimeIndex([t]), ROUTES, target_totals=totals)
    temp = dest.with_suffix(".tmp.csv")
    csv.to_csv(temp, index=False)
    values = replay.net.aligned(test, pd.read_csv(temp, dtype={"id": str}))
    np.testing.assert_allclose(values, p[test.route_id, test.h], rtol=1e-12, atol=1e-7)
    np.testing.assert_allclose(np.bincount(test.h, weights=values), totals, rtol=1e-12)
    temp.replace(dest)


def labels(day):
    t = origin(day)
    future = pd.read_parquet(replay.TRAIN, columns=["route_id", "timestamp", replay.net.TARGET],
                             filters=[("timestamp", ">", t), ("timestamp", "<=", t+8*replay.net.STEP)])
    matrix = future.pivot(index="route_id", columns="timestamp", values=replay.net.TARGET)
    np.testing.assert_array_equal(matrix.index, ROUTES)
    assert matrix.columns.as_unit("ns").equals(pd.date_range(t+replay.net.STEP, periods=8,
                                                            freq=replay.net.STEP).as_unit("ns"))
    return matrix.to_numpy(dtype=float)


def evaluate_dates(days, names):
    rows = []
    for day in days:
        data = read_stage(day, "variants")
        forecasts = {name: predict(data, name)[0] for name in names}
        # Verify the real writer on an intentionally shuffled route/time grid.
        round_trip(day, *predict(data, "network"), folder(day)/"network_export.csv")
        actual = labels(day)
        groups = {"all": np.s_[:, :], **{f"h{i+1}": np.s_[:, i] for i in range(8)},
                  **{f"Q{i}": np.s_[data["q"] == i, :] for i in range(1, 5)}}
        for name, p in forecasts.items():
            for group, idx in groups.items():
                y = actual[idx]
                metrics = replay.metric(y, p[idx])
                ref = replay.metric(y, data["base"][idx])
                if group == "all" and "network" not in name.split("+"):
                    np.testing.assert_allclose(metrics["signed_bias"], ref["signed_bias"], atol=1e-12, rtol=0)
                rows.append(dict(day=day, group=group, model=name, **metrics,
                                 delta_score=metrics["score"]-ref["score"],
                                 delta_wape=metrics["wape"]-ref["wape"],
                                 contribution=(np.abs(p[idx]-y).sum()-np.abs(data["base"][idx]-y).sum())/actual.sum()))
    return pd.DataFrame(rows)


def rank(table):
    return (table.query("group == 'all'").groupby("model").delta_score.agg(["mean", "max"])
            .sort_values(["mean", "max"]))


def select():
    singles = evaluate_dates(DEV, [BASE, *SINGLES])
    ranks = rank(singles)
    best = []
    for family in FAMILIES:
        eligible = ranks.loc[ranks.index.isin(family) & (ranks["max"] < 0)]
        if not eligible.empty:
            best.append(eligible.index[0])
    combos = ["+".join(group) for n in range(2, len(best)+1)
              for group in itertools.combinations(best, n)]
    dev = evaluate_dates(DEV, [BASE, *SINGLES, *combos])
    ranks = rank(dev)
    allowed = ranks.loc[ranks.index.isin([BASE, *best, *combos]) & (ranks["max"] <= 0)]
    chosen = allowed.index[0]
    shares = next(name for name in allowed.index if "network" not in name.split("+"))
    dev.to_csv(OUT/"development.csv", index=False)
    decision = dict(selected=chosen, shares=shares, development=DEV, holdout=HOLDOUT, eligible=best,
                    combinations=combos)
    path = OUT/"selection.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(decision)):
        raise ValueError("Frozen selection differs. Investigate changed predictions.")
    path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(decision, flush=True)


def assess():
    decision = json.loads((OUT/"selection.json").read_text())
    names = [BASE, *SINGLES, *decision["combinations"]]
    table = evaluate_dates((*HOLDOUT, DIAGNOSTIC), names)
    table.to_csv(OUT/"validation.csv", index=False)
    selected = decision["selected"]
    def passes(name):
        hold = table.query("group == 'all' and model == @name and day in @HOLDOUT")
        return name != BASE and len(hold) == len(HOLDOUT) and bool((hold.delta_score < 0).all())
    passed = passes(selected)
    choice = selected if passed else BASE
    all_dates = pd.concat([pd.read_csv(OUT/"development.csv"), table], ignore_index=True)
    by_date = all_dates.query("group == 'all'").copy()
    by_date["split"] = by_date.day.map(
        {**dict.fromkeys(DEV, "development"), **dict.fromkeys(HOLDOUT, "holdout"), DIAGNOSTIC: "diagnostic"})
    summary = by_date.groupby(["split", "model"]).agg(
        days=("day", "size"), wape=("wape", "mean"), mean_daily_score=("score", "mean"),
        delta_wape=("delta_wape", "mean"), delta_score=("delta_score", "mean"),
        worst_delta=("delta_score", "max"), wins=("delta_score", lambda s: int((s < 0).sum())))
    summary.to_csv(OUT/"summary.csv")
    diag = []
    for day in DATES:
        actual = labels(day)
        variant = read_stage(day, "variants")
        tcn = read_stage(day, "temporal")
        for key in ("early", "adapted", "forecast"):
            for budget, totals in (("008b", variant["totals"]), ("oracle", actual.sum(0))):
                p = old.allocate(tcn[key], tcn["known"], totals)
                diag.append(dict(day=day, adapter=key, budget=budget, **replay.metric(actual, p)))
    pd.DataFrame(diag).to_csv(OUT/"tcn_budgets.csv", index=False)
    print(all_dates.query("group == 'all' and model in [@BASE, @selected]").to_string(index=False))
    shares = decision["shares"]
    (OUT/"result.json").write_text(json.dumps(dict(selected=selected, final=choice, holdout_passed=passed,
                                                   shares=shares, shares_passed=passes(shares)), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("build", "select", "assess"))
    p.add_argument("days", nargs="*", default=DATES)
    args = p.parse_args()
    if args.stage == "build":
        for d in args.days:
            build(d)
    elif args.stage == "select":
        select()
    else:
        assess()
