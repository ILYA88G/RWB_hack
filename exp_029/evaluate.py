"""Select one rich-model blend on development and measure subsequent dates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from train import OUT, ROOT, DEV, HOLDOUT, DIAGNOSTIC, TEST, origin, replay
sys.path.insert(0, str(ROOT/"exp_028"))
from components import read_stage
from experiment import labels, round_trip


def load(day, mode):
    with np.load(OUT/"folds"/day/(mode+".npz")) as a:
        assert pd.Timestamp(a["origin"][()]) == origin(day)
        np.testing.assert_array_equal(a["route_id"], replay.ROUTES)
        return a["pred"]


def candidates():
    return ["020a"] + [f"{mode}_{portion}_{weight}" for mode in ("broad", "clock", "mix")
                       for portion in ("all", "long") for weight in (.125, .25, .5)]


def prediction(day, name):
    data = read_stage(day, "variants")
    base = data["base"]
    if name == "020a":
        return base
    mode, portion, weight = name.split("_")
    norm = replay.archive.normalize
    matrices = {m: norm(load(day, m), data["totals"]) for m in ("broad", "clock")}
    matrices["mix"] = .5*(matrices["broad"]+matrices["clock"])
    p = base.copy()
    idx = np.s_[:, :] if portion == "all" else np.s_[:, 2:]
    p[idx] = (1-float(weight))*base[idx]+float(weight)*matrices[mode][idx]
    np.testing.assert_allclose(p.sum(0), data["totals"], rtol=1e-12)
    return p


def score(days):
    rows = []
    for day in days:
        forecasts = {name: prediction(day, name) for name in candidates()}
        actual = labels(day)
        data = read_stage(day, "variants")
        groups = {"all": np.s_[:, :], **{f"h{i+1}": np.s_[:, i] for i in range(8)},
                  **{f"Q{i}": np.s_[data["q"] == i, :] for i in range(1, 5)}}
        for name, p in forecasts.items():
            for group, idx in groups.items():
                stats = replay.metric(actual[idx], p[idx])
                base = replay.metric(actual[idx], forecasts["020a"][idx])
                rows.append(dict(day=day, model=name, group=group, **stats,
                                 delta_wape=stats["wape"]-base["wape"],
                                 delta_score=stats["score"]-base["score"]))
    return pd.DataFrame(rows)


def select():
    table = score(DEV)
    table.to_csv(OUT/"development.csv", index=False)
    ranked = table.query("group=='all'").groupby("model").delta_wape.agg(["mean", "max"]).sort_values(["mean", "max"])
    selected = ranked.index[0]
    path = OUT/"selection.json"
    choice = dict(selected=selected, development=DEV, holdout=HOLDOUT)
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(choice)):
        raise ValueError("The frozen selection has changed")
    path.write_text(json.dumps(choice, indent=2))
    print(ranked.head(10).to_string(), flush=True)


def assess():
    choice = json.loads((OUT/"selection.json").read_text())["selected"]
    table = score((*HOLDOUT, DIAGNOSTIC))
    table.to_csv(OUT/"validation.csv", index=False)
    checked = table.query("group=='all' and model==@choice and day in @HOLDOUT")
    mean_delta = checked.delta_score.mean()
    result = dict(selected=choice, holdout_mean_delta=float(mean_delta),
                  holdout_wins=int((checked.delta_score<0).sum()), n_holdout=len(checked),
                  candidate=choice != "020a" and mean_delta < 0)
    (OUT/"result.json").write_text(json.dumps(result, indent=2))
    print(table.query("group=='all' and model in ['020a', @choice]").to_string(index=False), flush=True)


def export():
    result = json.loads((OUT/"result.json").read_text())
    choice = result["selected"]
    t = origin(TEST)
    test = replay.net.validate_grid(pd.read_parquet(replay.net.RAW/"test_solo_track.parquet"), replay.ROUTES, t)
    source = pd.read_csv(ROOT/"submissions/submission_model_020a.csv", dtype={"id": str})
    base = np.empty((1000, 8))
    base[test.route_id, test.h] = replay.net.aligned(test, source)
    totals = base.sum(0)
    mode, portion, weight = choice.split("_")
    models = ("broad", "clock") if mode == "mix" else (mode,)
    new = np.mean([replay.archive.normalize(load(TEST, kind), totals) for kind in models], axis=0)
    p = base.copy()
    idx = np.s_[:, :] if portion == "all" else np.s_[:, 2:]
    p[idx] = (1-float(weight))*base[idx]+float(weight)*new[idx]
    round_trip(TEST, p, totals, OUT/"submission_model_029a.csv")
    print("Verified submission_model_029a.csv", choice, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "assess", "export"])
    args = parser.parse_args()
    globals()[args.stage]()
