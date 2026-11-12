"""Causal network forecasts and ID-aligned recalibration for RWB 021a."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "exp_021"
RAW = ROOT / "data/raw"
if not (RAW / "train_solo_track.parquet").exists():
    RAW = ROOT.parent / "rwb_case_data"
STEP = pd.Timedelta(minutes=30)
TARGET = "target_1h"
STATUS = [f"status_{j}" for j in range(1, 7)]
MODELS = ("seasonal", "ridge", "lgbm")


def load_train(path):
    df = pd.read_parquet(path, columns=["timestamp", "route_id", TARGET, *STATUS])
    df["timestamp"] = pd.to_datetime(df.timestamp)
    if df.isna().any().any() or df.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Missing values or duplicate route/time")
    values = df[[TARGET, *STATUS]]
    if (values.min() < 0).any() or not np.isfinite(values.max()).all():
        raise ValueError("Invalid train values")
    net = df.groupby("timestamp", sort=True)[[TARGET, *STATUS]].sum().astype(float)
    targets = df.pivot(index="timestamp", columns="route_id", values=TARGET).sort_index().sort_index(axis=1)
    if targets.shape != (4630, 1000) or targets.isna().any().any():
        raise ValueError("Incomplete train panel")
    if not net.index.as_unit("ns").equals(pd.date_range("2025-07-28", periods=4630, freq=STEP).as_unit("ns")):
        raise ValueError("Unexpected time grid")
    return net, targets


def last_halfhour(panel, origin):
    # Same causal interval midpoint and float32 rounding as halfhour_causal.
    y = panel.loc[:origin].to_numpy(dtype=float)
    lower = np.zeros(y.shape[1])
    upper = y[0].copy()
    intercept = np.zeros_like(lower)
    for t, row in enumerate(y):
        if t:
            intercept = row - intercept
        if t % 2:
            upper = np.minimum(upper, intercept)
        else:
            lower = np.maximum(lower, -intercept)
    half = np.maximum((-1)**t * (lower + upper) / 2 + intercept, 0)
    # Keep the signed interval for audit. Legacy width is max(signed width, 0).
    return half.astype(np.float32), (upper-lower).astype(np.float32)


def q0_at(panel, origin):
    half, _ = last_halfhour(panel, origin)
    return half.sum(dtype=np.float64)


def budgets(totals, q0):
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (8,) or not np.isfinite(q0) or q0 < 0:
        raise ValueError("Invalid budget inputs")
    result = []
    for total in totals:
        q0 = total - q0
        if not np.isfinite(q0) or q0 <= 0:
            raise ValueError("Nonpositive or nonfinite half-hour budget")
        result.append(q0)
    return np.asarray(result)


def features(net):
    y = net[TARGET]
    f = {}
    for col in [TARGET, *STATUS]:
        for lag in ((0,1,2,4,8,16,48,96,336) if col == TARGET else (0,1,4,8,48)):
            f[col + f"_lag{lag}"] = net[col].shift(lag)
    for w in (4,8,16,48,96,336):
        mean = y.rolling(w).mean()
        f[f"mean{w}"] = mean
        f[f"median{w}"] = y.rolling(w).median()
        f[f"relative{w}"] = y / mean.clip(lower=1) - 1
        f[f"slope{w}"] = (y - y.shift(w-1)) / (w-1)
    slot = net.index.hour*2 + net.index.minute//30
    for name, value, period in [("slot", slot, 48), ("dow", net.index.day_of_week, 7)]:
        f[name] = value
        f[name + "_sin"] = np.sin(2*np.pi*value/period)
        f[name + "_cos"] = np.cos(2*np.pi*value/period)
    # Adapt previous profiles to the current level using only observed hours.
    profiles = []
    for lag, weight in [(48, .5), (336, .5)]:
        ratio = (y.rolling(8).mean()/y.shift(lag).rolling(8).mean()).clip(.8, 1.2)
        profile = pd.concat([y.shift(lag-h) for h in range(1,9)], axis=1)
        for h in range(8):
            f[f"analog{lag}_{h+1}"] = profile.iloc[:,h]
        profiles.append(profile.mul(ratio, axis=0)*weight)
    return pd.DataFrame(f, index=net.index), profiles[0]+profiles[1]


def fit_predict(net, origin, models=MODELS):
    prefix = net.loc[:origin]
    if prefix.index[-1] != origin:
        raise ValueError("No observation at origin")
    x, anchor = features(prefix)
    y = np.lib.stride_tricks.sliding_window_view(prefix[TARGET].to_numpy()[1:], 8)
    end = len(x)-8                       # exclusive; last label equals origin
    start = max(344, end-42*48)
    if start >= end:
        raise ValueError("Insufficient prefix")
    rows = slice(start, end)
    residual = (y[rows] - anchor.iloc[rows].to_numpy()) / 1e6
    base = anchor.iloc[-1].to_numpy()
    result = {}
    if "seasonal" in models:
        result["seasonal"] = base
    if "ridge" in models:
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=180))
        ridge.fit(x.iloc[rows], residual)
        result["ridge"] = base + 1e6*ridge.predict(x.iloc[[-1]])[0]
    if "lgbm" in models:
        adjustments = []
        for j in range(8):
            model = LGBMRegressor(n_estimators=70, learning_rate=.055, num_leaves=7,
                                  min_child_samples=50, reg_lambda=5, random_state=42,
                                  n_jobs=2, verbosity=-1)
            model.fit(x.iloc[rows], residual[:,j])
            adjustments.append(model.predict(x.iloc[[-1]])[0])
        result["lgbm"] = base + 1e6*np.asarray(adjustments)
    return result


def proposed_totals(old, forecast, mode, alpha, cap):
    if not np.isfinite(forecast).all() or np.any(forecast<=0):
        return np.full(8, np.nan)
    ratio = forecast / old
    global_ratio = forecast.sum() / old.sum()
    if mode == "shape":
        centered = alpha * (ratio / global_ratio - 1)
        # A common shrink factor preserves the zero weighted mean after capping.
        return old * (1 + centered * min(1, cap/max(1e-12, np.abs(centered).max())))
    if mode == "global":
        ratio = global_ratio
    elif mode == "smoothed":
        ratio = (ratio + global_ratio)/2
    elif mode != "per_step":
        raise ValueError(mode)
    return old * np.clip(1 + alpha*(ratio-1), 1-cap, 1+cap)


def safe_totals(old, proposed, q0):
    try:
        budgets(old, q0)
        budgets(proposed, q0)
    except ValueError:
        return old.copy(), True
    return proposed, False


def validate_grid(test, routes, origin):
    test = test.copy()
    if test[["route_id","timestamp","id"]].isna().any().any():
        raise ValueError("Missing test keys")
    test["timestamp"] = pd.to_datetime(test.timestamp)
    test["id"] = test.id.astype(str)
    expected = pd.MultiIndex.from_product([routes, pd.date_range(origin+STEP, periods=8, freq=STEP)])
    actual = pd.MultiIndex.from_frame(test[["route_id","timestamp"]])
    if len(test) != 8000 or not test.id.is_unique or not actual.is_unique or len(expected.difference(actual)):
        raise ValueError("Invalid test IDs or route/time grid")
    test["h"] = ((test.timestamp-origin)//STEP-1).astype(int)
    return test


def aligned(test, submission):
    if list(submission) != ["id","y_pred"] or len(submission) != 8000 or submission.id.isna().any():
        raise ValueError("Invalid submission schema")
    ids = submission.id.astype(str)
    if not ids.is_unique or set(ids) != set(test.id):
        raise ValueError("Invalid submission IDs")
    p = submission.set_index(ids).y_pred.reindex(test.id).to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid predictions")
    return p


def rescale(test, baseline, totals):
    p = aligned(test, baseline)
    h = test.h.to_numpy()
    old = np.bincount(h, weights=p, minlength=8)
    if (old <= 0).any():
        raise ValueError("Empty baseline horizon")
    result = pd.DataFrame(dict(id=test.id.to_numpy(), y_pred=p*(totals/old)[h]))
    np.testing.assert_allclose(np.bincount(h, weights=aligned(test,result)), totals, rtol=1e-12)
    return result


def parser(output):
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=RAW/"train_solo_track.parquet")
    p.add_argument("--test", type=Path, default=RAW/"test_solo_track.parquet")
    p.add_argument("--baseline", type=Path, default=ROOT/"submissions/submission_model_020a.csv")
    p.add_argument("--output", type=Path, default=EXP/output)
    return p


def main():
    p = parser("submission_model_021a.csv")
    p.add_argument("--rule", type=Path, default=EXP/"rule.json")
    args = p.parse_args()
    if args.output.resolve() == args.baseline.resolve() or (args.output.exists() and args.output.resolve() != EXP/"submission_model_021a.csv"):
        raise ValueError("Refusing to overwrite an input or unrelated file")
    net, panel = load_train(args.train)
    origin = net.index[-1]
    test = validate_grid(pd.read_parquet(args.test), panel.columns, origin)
    base = pd.read_csv(args.baseline, dtype={"id":str})
    old = np.bincount(test.h, weights=aligned(test, base))
    rule = json.loads(args.rule.read_text())
    forecast = fit_predict(net, origin, [rule["model"]])[rule["model"]]
    proposal = proposed_totals(old, forecast, rule["mode"], rule["alpha"], rule["cap"])
    q0 = q0_at(panel, origin)
    totals, fallback = safe_totals(old, proposal, q0)
    result = rescale(test, base, totals)
    with NamedTemporaryFile(dir=args.output.parent,delete=False,suffix=".csv") as tmp:
        temp = Path(tmp.name)
    try:
        result.to_csv(temp,index=False)
        reread = pd.read_csv(temp,dtype={"id":str})
        np.testing.assert_allclose(np.bincount(test.h, weights=aligned(test,reread)), totals, rtol=1e-12)
        temp.replace(args.output)
    finally:
        temp.unlink(missing_ok=True)
    print(dict(q0=q0, old=old.tolist(), totals=totals.tolist(),
               q=None if fallback else budgets(totals,q0).tolist(), fallback=fallback))


if __name__ == "__main__":
    main()

