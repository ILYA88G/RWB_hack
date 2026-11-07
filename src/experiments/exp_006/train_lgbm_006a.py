"""EXP-006A: fast, leakage-safe LightGBM with known-current-target features.

Run from project root:
    python -m src.experiments.exp_006.train_lgbm_006a

Trains on 3 latest rolling folds, compares with cached EXP-005A OOF,
then generates a submission (id,y_pred) if the new model/ensemble wins.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    STATUS_COLS,
    FUTURE_TARGET_COLS,
    FORECAST_POINTS,
    HORIZON,
    STEP,
    TRAIN_DAYS,
    load_data,
    wape_plus_relative_bias,
)
from src.experiments.exp_004.train_ridge_target_lags_004a import (
    get_origins,
    select_training_rows,
)

EXP_DIR = REPORTS_DIR / "exp_006"
BASE_CACHE_DIR = REPORTS_DIR / "exp_005"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

# Validate on the three most recent origins (folds 3, 4, 5 of EXP-005A).
# Switching to all 5 folds requires changing only the following tuple.
VALIDATION_FOLDS = (3, 4, 5)
WEIGHTS_NEW = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_GAIN_TO_SUBMIT = 0.0010
FEATURE_VERSION = "006a_current_target_weekly_v1"

PARAMS = dict(
    objective="regression",
    n_estimators=240,
    learning_rate=0.055,
    num_leaves=39,
    min_child_samples=100,
    max_bin=127,
    colsample_bytree=0.9,
    reg_lambda=5.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


def build_features(train: pd.DataFrame, origins: list[pd.Timestamp]):
    earliest_fit = min(
        origin - HORIZON - pd.Timedelta(days=TRAIN_DAYS)
        for origin in origins
    )
    # 336 half-hour intervals = one full week of available history.
    history = STEP * 336
    work = train.loc[
        train["timestamp"] >= earliest_fit - history
    ].copy()
    work.sort_values(["route_id", "timestamp"], inplace=True)
    work.reset_index(drop=True, inplace=True)

    grouped = work.groupby("route_id", sort=False)
    target_group = grouped[TARGET_COL]

    # CRUCIAL: target_1h at forecast origin is known from train data.
    # We predict t+1,...,t+8, never the current t.
    cols = ["route_id", TARGET_COL, *STATUS_COLS]

    for col in STATUS_COLS:
        for lag in (1, 2, 4, 8):
            name = f"{col}_lag_{lag}"
            work[name] = grouped[col].shift(lag).astype("float32")
            cols.append(name)
        for lag in (1, 4, 8):
            name = f"{col}_delta_{lag}"
            work[name] = (work[col] - work[f"{col}_lag_{lag}"]).astype("float32")
            cols.append(name)

    for lag in (1, 2, 4, 8, 16, 48, 96, 336):
        name = f"target_lag_{lag}"
        work[name] = target_group.shift(lag).astype("float32")
        cols.append(name)

    for lag in (1, 4, 8, 48):
        name = f"target_delta_{lag}"
        work[name] = (work[TARGET_COL] - work[f"target_lag_{lag}"]).astype("float32")
        cols.append(name)

    # Rolling statistics INCLUDE the current target, which is already observed.
    for window in (4, 8, 16, 48):
        name = f"target_mean_{window}"
        work[name] = target_group.transform(
            lambda series: series.rolling(window, min_periods=window).mean()
        ).astype("float32")
        cols.append(name)

    for window in (8, 48):
        name = f"target_std_{window}"
        work[name] = target_group.transform(
            lambda series: series.rolling(window, min_periods=window).std()
        ).astype("float32")
        cols.append(name)

    local = work[STATUS_COLS[:3]].sum(axis=1)
    upstream = work[STATUS_COLS[3:]].sum(axis=1)
    work["status_local_sum"] = local.astype("float32")
    work["status_upstream_sum"] = upstream.astype("float32")
    work["status_total_sum"] = (local + upstream).astype("float32")
    work["status_upstream_ratio"] = (upstream / (local + 1.0)).astype("float32")
    cols += [
        "status_local_sum", "status_upstream_sum",
        "status_total_sum", "status_upstream_ratio",
    ]

    # Contemporaneous totals are available at forecast origin for all routes.
    total_target = work.groupby("timestamp", sort=False)[TARGET_COL].sum()
    work["network_target_now"] = work["timestamp"].map(total_target).astype("float32")
    cols.append("network_target_now")

    tm = work["timestamp"]
    slot = tm.dt.hour * 2 + tm.dt.minute // 30
    dow = tm.dt.dayofweek
    work["origin_slot"] = slot.astype("int8")
    work["origin_dow"] = dow.astype("int8")
    work["is_weekend"] = (dow >= 5).astype("int8")
    work["slot_sin"] = np.sin(2 * np.pi * slot / 48).astype("float32")
    work["slot_cos"] = np.cos(2 * np.pi * slot / 48).astype("float32")
    cols += ["origin_slot", "origin_dow", "is_weekend", "slot_sin", "slot_cos"]

    # Labels are STRICTLY in the future of each historical forecast origin.
    for step, name in enumerate(FUTURE_TARGET_COLS, start=1):
        work[name] = target_group.shift(-step)

    # Trim only after all historical features have been computed.
    work = work.loc[work["timestamp"] >= earliest_fit].copy()
    work["route_id"] = work["route_id"].astype("category")
    print(f"Features: {len(cols)} | Rows: {len(work):,} | Start: {earliest_fit}", flush=True)
    return work, cols


def add_future_time(X: pd.DataFrame, timestamps: pd.Series, step: int):
    future_time = timestamps + STEP * step
    X["future_slot"] = (
        future_time.dt.hour * 2 + future_time.dt.minute // 30
    ).to_numpy(dtype=np.int8)
    X["future_dow"] = future_time.dt.dayofweek.to_numpy(dtype=np.int8)
    return X


def train_predict(fit_rows: pd.DataFrame, predict_rows: pd.DataFrame, feature_cols):
    # Reuse numeric/categorical design matrix across all forecast horizons.
    X_fit = fit_rows[feature_cols].copy()
    X_pred = predict_rows[feature_cols].copy()
    predictions = np.zeros((len(predict_rows), FORECAST_POINTS), dtype=np.float64)

    for index, target_name in enumerate(FUTURE_TARGET_COLS):
        start = time.perf_counter()
        step = index + 1
        add_future_time(X_fit, fit_rows["timestamp"], step)
        add_future_time(X_pred, predict_rows["timestamp"], step)
        model = LGBMRegressor(**PARAMS)
        model.fit(
            X_fit,
            fit_rows[target_name].to_numpy(dtype=np.float32),
            categorical_feature=["route_id"],
        )
        predictions[:, index] = model.predict(X_pred)
        print(f"  Step {step}/8: {time.perf_counter()-start:.1f}s", flush=True)

    return np.clip(predictions, 0, None)


def baseline_submission_path() -> Path:
    candidates = (
        SUBMISSION_DIR / "submission_lightgbm_005a_fixed.csv",
        SUBMISSION_DIR / "submission_ensemble_005b.csv",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No baseline submission found in submissions/")


def validate(data, cols, origins, route_count):
    all_true, all_old, all_new = [], [], []
    report = []

    for fold_num in VALIDATION_FOLDS:
        origin = origins[fold_num - 1]
        cache = EXP_DIR / f"oof_fold_{fold_num}.npz"
        old_cache = BASE_CACHE_DIR / f"ensemble_005b_fold_{fold_num}.npz"
        if not old_cache.exists():
            raise FileNotFoundError(f"Missing 005B OOF cache: {old_cache}")

        valid = data.loc[data["timestamp"].eq(origin)].copy()
        valid = valid.loc[valid[FUTURE_TARGET_COLS].notna().all(axis=1)]
        if len(valid) != route_count:
            raise ValueError(f"Fold {fold_num}: {len(valid)} routes instead of {route_count}")
        truth = valid[FUTURE_TARGET_COLS].to_numpy(dtype=np.float64)

        with np.load(old_cache) as stored:
            old_truth = stored["y_true"]
            old_pred = stored["lgbm_pred"].copy()
        if not np.allclose(old_truth, truth, rtol=0, atol=0):
            raise ValueError(f"Fold {fold_num}: 005A cache validation order/targets differ")

        print(f"\nFOLD {fold_num}/5: {origin}", flush=True)
        if cache.exists():
            with np.load(cache) as stored:
                cached_origin = str(stored["origin"].item())
                cached_version = str(stored["version"].item())
                if cached_origin != str(origin) or cached_version != FEATURE_VERSION:
                    raise ValueError(f"Stale feature cache: {cache}. Delete it to retrain.")
                new_pred = stored["new_pred"].copy()
            print("  Reused cached EXP-006A predictions", flush=True)
        else:
            fit = select_training_rows(data, origin)
            print(f"  Train rows: {len(fit):,}", flush=True)
            new_pred = train_predict(fit, valid, cols)
            np.savez_compressed(
                cache, origin=str(origin), version=FEATURE_VERSION, new_pred=new_pred
            )

        base_score = wape_plus_relative_bias(truth, old_pred)["score"]
        new_score = wape_plus_relative_bias(truth, new_pred)["score"]
        print(f"  Baseline 005A: {base_score:.6f} | New 006A: {new_score:.6f}", flush=True)
        report.append({"fold": fold_num, "origin": origin, "baseline_score": base_score,
                       "new_score": new_score})
        all_true.append(truth)
        all_old.append(old_pred)
        all_new.append(new_pred)

    pd.DataFrame(report).to_csv(EXP_DIR / "fold_scores_006a.csv", index=False)
    return np.concatenate(all_true), np.concatenate(all_old), np.concatenate(all_new)


def pick_weight(truth, old, new):
    baseline_score = wape_plus_relative_bias(truth, old)["score"]
    rows = []
    for weight in WEIGHTS_NEW:
        blended = np.clip((1 - weight) * old + weight * new, 0, None)
        metrics = wape_plus_relative_bias(truth, blended)
        rows.append({"new_weight": weight, **metrics})
    table = pd.DataFrame(rows).sort_values(["score", "wape"])
    table.to_csv(EXP_DIR / "weight_search_006a.csv", index=False)
    print("\nWEIGHT SEARCH (same 3 folds for all models)")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    best = table.iloc[0]
    gained = baseline_score - float(best["score"])
    selected = float(best["new_weight"]) if gained >= MIN_GAIN_TO_SUBMIT else 0.0
    print(f"\nBaseline: {baseline_score:.6f} | Best: {best['score']:.6f}")
    print(f"Gain: {gained:.6f} | Selected NEW weight: {selected:.2f}")
    return selected


def make_submission(data, cols, train, test, origin, new_weight):
    baseline = pd.read_csv(baseline_submission_path(), dtype={"id": str})
    if baseline.columns.tolist() != ["id", "y_pred"]:
        raise ValueError("Baseline submission must have columns id,y_pred")
    if baseline["id"].duplicated().any():
        raise ValueError("Duplicate IDs in baseline submission")
    old_lookup = baseline.set_index("id")["y_pred"]
    old_predictions = old_lookup.reindex(test["id"].astype(str)).to_numpy(dtype=float)
    if not np.isfinite(old_predictions).all():
        raise ValueError("Baseline submission does not match test IDs")

    if new_weight > 0:
        print("\nTraining final EXP-006A models...", flush=True)
        fit = select_training_rows(data, origin)
        observed = data.loc[data["timestamp"].eq(origin)].copy()
        if len(observed) != train["route_id"].nunique():
            raise ValueError("Missing routes at final forecast origin")
        new_matrix = train_predict(fit, observed, cols)
        route_ids = observed["route_id"].astype(train["route_id"].dtype).to_numpy()
        forecast_parts = [
            pd.DataFrame({
                "route_id": route_ids,
                "timestamp": origin + STEP * (step + 1),
                "new_pred": new_matrix[:, step],
            })
            for step in range(FORECAST_POINTS)
        ]
        forecast = pd.concat(forecast_parts, ignore_index=True)
        aligned = test[["id", "route_id", "timestamp"]].merge(
            forecast, on=["route_id", "timestamp"], how="left",
            validate="one_to_one", sort=False,
        )
        new_predictions = aligned["new_pred"].to_numpy(dtype=float)
        if not np.isfinite(new_predictions).all():
            raise ValueError("Unmatched test rows or invalid forecasts")
        predictions = np.clip(
            (1 - new_weight) * old_predictions + new_weight * new_predictions,
            0, None,
        )
    else:
        predictions = old_predictions

    result = pd.DataFrame({"id": test["id"], "y_pred": predictions})
    if len(result) != 8000 or not result["id"].is_unique:
        raise ValueError("Submission must have 8000 unique ids")
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite forecasts")
    destination = SUBMISSION_DIR / "submission_lgbm_006a.csv"
    result.to_csv(destination, index=False)
    print(f"\nSUBMISSION SAVED: {destination}")
    print(f"Shape: {result.shape} | Columns: {result.columns.tolist()}")
    if new_weight == 0:
        print("NO CV IMPROVEMENT: predictions equal EXP-005A; do not resubmit.")
    else:
        print("New predictions generated. Compare the public score after upload.")


def main():
    start = time.perf_counter()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    last_timestamp = train["timestamp"].max()
    origins = get_origins(last_timestamp)
    selected_origins = [origins[n - 1] for n in VALIDATION_FOLDS]

    print("\nEXP-006A: CURRENT TARGET + TEMPORAL FEATURES")
    print(f"Validation folds: {VALIDATION_FOLDS}")
    print("Target at time t is used only to predict times t+1,...,t+8.")
    data, cols = build_features(train, selected_origins)
    truth, old, new = validate(data, cols, origins, train["route_id"].nunique())
    weight = pick_weight(truth, old, new)
    make_submission(data, cols, train, test, last_timestamp, weight)
    print(f"\nTotal runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
