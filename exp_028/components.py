"""Build prefix-only 020a components. Each stage is resumable per origin."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "exp_027"))
import replay

DEV = ("2025-10-11", "2025-10-13")
HOLDOUT = ("2025-10-18", "2025-10-20")
DIAGNOSTIC = "2025-10-25"
DATES = (*DEV, *HOLDOUT, DIAGNOSTIC)
TEST = "2025-11-01"
ROUTES = replay.ROUTES


def origin(day):
    return pd.Timestamp(day) + pd.Timedelta(hours=10, minutes=30)


def folder(day):
    p = OUT / "folds" / day
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_stage(day, name, **arrays):
    path = folder(day) / (name + ".npz")
    assert all(np.isfinite(v).all() for v in arrays.values())
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, origin=origin(day).to_datetime64(), route_id=ROUTES, **arrays)
    temp.replace(path)


def read_stage(day, name):
    path = folder(day) / (name + ".npz")
    if day == DIAGNOSTIC and name in ("long", "network", "short", "tcn") and not path.exists():
        path = replay.OUT / (name + ".npz")
    with np.load(path, allow_pickle=False) as a:
        assert pd.Timestamp(a["origin"][()]) == origin(day)
        np.testing.assert_array_equal(a["route_id"], ROUTES)
        return {k: a[k] for k in a.files if k not in ("origin", "route_id")}


def long_cat(day, fresh):
    from catboost import CatBoostRegressor
    import predict_model_019b as old
    from train_components_019b import fit_long_cat
    name = "fresh" if fresh else "long"
    if (folder(day) / (name + ".npz")).exists() or (day == DIAGNOSTIC and not fresh):
        return
    end = origin(day)
    cutoff = end if fresh else end - pd.Timedelta(days=2)
    models = OUT / "models"
    models.mkdir(exist_ok=True)
    path = models / f"cat_{cutoff.date()}.cbm"
    times, v = replay.history(end=end)
    if not path.exists():
        model = fit_long_cat(v, times, ROUTES, times.get_loc(cutoff), threads=5)
        temp = path.with_suffix(".tmp.cbm")
        model.save_model(str(temp))
        temp.replace(path)
        del model
        gc.collect()
    model = CatBoostRegressor().load_model(str(path))
    metadata = model.get_metadata()
    assert pd.Timestamp(metadata["training_cutoff"]) == cutoff
    np.testing.assert_array_equal(json.loads(metadata["route_order"]), ROUTES)
    engine = old.CatFeatures(v, times, ROUTES)
    x, scale, offset = engine.features(np.full(8000, len(times)-1), np.repeat(ROUTES, 8),
                                       np.tile(np.arange(1, 9), 1000))
    assert list(x) == model.feature_names_ and model.tree_count_ == 1400
    pred = (np.maximum(model.predict(x, thread_count=1), 0) * scale + offset).reshape(1000, 8)
    write_stage(day, name, pred=pred, cutoff_index=times.get_loc(cutoff))


def upstream(day):
    for name in ("network", "short"):
        if (folder(day)/(name+".npz")).exists() or day == DIAGNOSTIC:
            continue
        if name == "network":
            data = replay.network_components(origin(day))
        else:
            linear, boosted, cat = replay.short_components(origin(day))
            data = dict(linear=linear, boosted=boosted, cat=cat)
        write_stage(day, name, **data)
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["cat", "upstream"])
    parser.add_argument("days", nargs="*", default=DATES)
    args = parser.parse_args()
    for day in args.days:
        print(day, args.stage, flush=True)
        if args.stage == "cat":
            for fresh in (False, True):
                long_cat(day, fresh)
        else:
            upstream(day)
