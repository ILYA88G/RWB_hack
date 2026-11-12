"""Fit rich route-level LightGBM models on strictly available history."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, log_evaluation

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT/"exp_028"))
from components import DEV, HOLDOUT, DIAGNOSTIC, TEST, origin, replay
import predict_model_019b as old


class RichFeatures(old.CatFeatures):
    def lagged(self, array, index, route):
        valid = index >= 0
        result = np.full(len(index), np.nan, dtype=np.float32)
        result[valid] = array[index[valid], route[valid]]
        return result

    def features(self, t, r, h):
        frame, scale, offset = super().features(t, r, h)
        extra = {}
        y = self.v[:, :, 0]
        for name, array in (("y", y), ("q", self.half)):
            for weeks in (1, 2, 3, 4):
                extra[f"{name}_week{weeks}"] = self.lagged(array, t+h-weeks*336, r)/scale
            recent = np.stack([self.lagged(array, t+h-day*48, r) for day in range(1, 29)])
            extra[f"{name}_clock_median28"] = np.nanmedian(recent, axis=0)/scale
            extra[f"{name}_clock_mean28"] = np.nanmean(recent, axis=0)/scale
            extra[f"{name}_clock_std28"] = np.nanstd(recent, axis=0)/scale
            extra[f"{name}_clock_mean7"] = np.nanmean(recent[:7], axis=0)/scale
            extra[f"{name}_clock_median7"] = np.nanmedian(recent[:7], axis=0)/scale
            now = np.stack([self.lagged(array, t-day*48, r) for day in range(1, 8)])
            extra[f"{name}_clock_level"] = array[t, r]/np.maximum(np.nanmean(now, axis=0), 100.)
        for channel in range(1, 7):
            den = np.maximum(self.means[336][t, r, channel], 1.)
            array = self.v[:, :, channel]
            for lag in (12, 16, 24, 48, 96, 336):
                past = self.lagged(array, t-lag, r)
                extra[f"s{channel}_lag{lag}"] = past/den
                extra[f"s{channel}_change{lag}"] = (array[t, r]-past)/den
            same_clock = np.stack([self.lagged(array, t-day*48, r) for day in range(1, 8)])
            extra[f"s{channel}_clock_level"] = array[t, r]/np.maximum(np.nanmean(same_clock, axis=0), 1.)
            extra[f"s{channel}_to_target"] = array[t, r]/scale
        future = self.times[t] + pd.to_timedelta(h*30, unit="min")
        extra["future_slot"] = future.hour.to_numpy()*2+future.minute.to_numpy()//30
        extra["weekend"] = (future.dayofweek.to_numpy() >= 5).astype(np.float32)
        return pd.concat([frame, pd.DataFrame(extra, dtype=np.float32)], axis=1), scale, offset


def fit(day, kind, rows=600000, iterations=1000):
    dest = OUT/"folds"/day
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/(kind+".npz")
    if path.exists():
        return
    times, values = replay.history(end=origin(day))
    engine = RichFeatures(values, times, replay.ROUTES)
    end = len(times)-9
    rng = np.random.default_rng(29)
    t = np.r_[rng.integers(max(337, end-21*48), end+1, size=rows//2),
              rng.integers(max(337, end-84*48), end+1, size=rows-rows//2)]
    if kind == "clock":
        # Move each sampled source to a nearby occurrence of the contest clock.
        t = t//48*48 + 21 + rng.integers(-4, 5, size=rows)
        t = np.clip(t, 337, end)
    r = rng.integers(0, 1000, size=rows)
    h = rng.integers(1, 9, size=rows)
    assert (t+h <= len(times)-1).all()
    x, scale, offset = engine.features(t, r, h)
    label = np.maximum((values[t+h, r, 0]-offset)/scale, 0)
    weight = scale/scale.mean()*np.exp(-np.log(2.)*(len(times)-1-t)/48/28)
    query, query_scale, query_offset = engine.features(
        np.full(8000, len(times)-1), np.repeat(replay.ROUTES, 8), np.tile(np.arange(1, 9), 1000))
    params = dict(objective="regression_l1", n_estimators=iterations, learning_rate=.035,
                  num_leaves=63, min_child_samples=180, reg_lambda=20., reg_alpha=.1,
                  colsample_bytree=.85, subsample=.85, subsample_freq=1, max_bin=127,
                  n_jobs=6, random_state=29, deterministic=True, force_col_wise=True, verbosity=-1)
    last_label = str(times[int((t+h).max())])
    del engine, values, t, r, h, scale, offset
    gc.collect()
    print(day, kind, x.shape, last_label, flush=True)
    model = LGBMRegressor(**params)
    began = time.monotonic()
    model.fit(x, label, sample_weight=weight, categorical_feature=["route_id"],
              eval_set=[(x, label)], eval_sample_weight=[weight], callbacks=[log_evaluation(200)])
    raw = (np.maximum(model.predict(query), 0)*query_scale+query_offset).reshape(1000, 8)
    assert np.isfinite(raw).all() and raw.min() >= 0
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, pred=raw, origin=origin(day).to_datetime64(), route_id=replay.ROUTES)
    temp.replace(path)
    model.booster_.save_model(str(dest/(kind+".txt")))
    pd.DataFrame(dict(feature=x.columns, gain=model.booster_.feature_importance("gain"))).sort_values(
        "gain", ascending=False).to_csv(dest/(kind+"_importance.csv"), index=False)
    (dest/(kind+".json")).write_text(json.dumps(dict(params=params, rows=len(x), columns=list(x),
                                                    last_training_label=last_label, seconds=time.monotonic()-began),
                                               indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("broad", "clock"))
    parser.add_argument("days", nargs="*", default=DEV)
    args = parser.parse_args()
    for day in args.days:
        fit(day, args.kind)
