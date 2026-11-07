
"""EXP-007A: robust LightGBM challengers + calibrated ensemble.

Place at src/experiments/exp_007/train_lgbm_007a.py, run from project root:
    python -m src.experiments.exp_007.train_lgbm_007a

Uses the exact same three latest rolling origins as EXP-006A and its cached
OOF predictions. No lookahead in features; each direct target is t+1 ... t+8.
Only emits a NEW submission when robust rolling CV beats the saved EXP-006A.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Wide feature frames produce harmless pandas fragmentation warnings.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from train_ridge_baseline import (
    PROJECT_ROOT, REPORTS_DIR, TARGET_COL, STATUS_COLS, FUTURE_TARGET_COLS,
    FORECAST_POINTS, STEP, HORIZON, TRAIN_DAYS, load_data,
    wape_plus_relative_bias,
)
from src.experiments.exp_004.train_ridge_target_lags_004a import (
    get_origins, select_training_rows,
)

REPORT_DIR = REPORTS_DIR / "exp_007"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"
BASE005_DIR = REPORTS_DIR / "exp_005"
BASE006_DIR = REPORTS_DIR / "exp_006"
VALIDATION_FOLDS = (3, 4, 5)
CACHE_VERSION = "007a_rich_causal_v1"
BASE006_NEW_WEIGHT = 0.50
PUBLIC_006A = 0.340  # Shown rounded to three decimals on public leaderboard.
EXPECTED_BASE_OOF = 0.335336
MIN_GAIN = 0.0015
MIN_LOFO_GAIN = 0.0005
MIN_FOLDS_IMPROVED = 2
WEIGHTS = (0.25, 0.50, 0.75, 1.0)
CALIBRATION = ("none", "global_50", "global_100", "step_50", "step_100")

# Two deliberately different losses: squared-error models conditional mean,
# absolute-error models conditional median (then calibrated on held-out folds).
MODEL_PARAMS = {
    "l2": dict(
        objective="regression", n_estimators=320, learning_rate=0.045,
        num_leaves=47, min_child_samples=125, max_bin=191,
        colsample_bytree=0.85, reg_lambda=10.0,
        random_state=27, n_jobs=-1, verbosity=-1,
    ),
    "l1": dict(
        objective="regression_l1", n_estimators=330, learning_rate=0.045,
        num_leaves=31, min_child_samples=180, max_bin=191,
        colsample_bytree=0.85, reg_lambda=12.0,
        random_state=53, n_jobs=-1, verbosity=-1,
    ),
}


def add_rolling(work: pd.DataFrame, group_name: str, source: str,
                windows: tuple[int, ...], feature_cols: list[str],
                statistic: str = "mean") -> None:
    group = work.groupby(group_name, sort=False)[source]
    for window in windows:
        name = f"{source}_roll_{statistic}_{window}"
        if statistic == "mean":
            series = group.transform(
                lambda x, w=window: x.rolling(w, min_periods=w).mean()
            )
        elif statistic == "std":
            series = group.transform(
                lambda x, w=window: x.rolling(w, min_periods=w).std()
            )
        else:
            raise ValueError(statistic)
        work[name] = series.astype("float32")
        feature_cols.append(name)


def build_features(train: pd.DataFrame,
                   validation_origins: list[pd.Timestamp]) -> tuple[pd.DataFrame, list[str]]:
    earliest_fit = min(
        origin - HORIZON - pd.Timedelta(days=TRAIN_DAYS)
        for origin in validation_origins
    )
    # Keep enough history before the first fit row for a 336-step weekly lag.
    work = train.loc[
        train["timestamp"] >= earliest_fit - STEP * 336
    ].copy()
    work = work.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    routes = work.groupby("route_id", sort=False)
    targets = routes[TARGET_COL]

    cols: list[str] = ["route_id", TARGET_COL, *STATUS_COLS]

    for lag in (1, 2, 3, 4, 6, 8, 12, 16, 24, 47, 48, 49, 96, 144, 336):
        name = f"target_lag_{lag}"
        work[name] = targets.shift(lag).astype("float32")
        cols.append(name)

    for lag in (1, 2, 4, 8, 48):
        name = f"target_delta_{lag}"
        work[name] = (work[TARGET_COL] - work[f"target_lag_{lag}"]).astype("float32")
        cols.append(name)

    add_rolling(work, "route_id", TARGET_COL, (4, 8, 16, 48, 96), cols)
    add_rolling(work, "route_id", TARGET_COL, (8, 48), cols, "std")

    for col in STATUS_COLS:
        group = routes[col]
        for lag in (1, 2, 4, 8, 16, 48):
            name = f"{col}_lag_{lag}"
            work[name] = group.shift(lag).astype("float32")
            cols.append(name)
        for lag in (1, 4, 8):
            name = f"{col}_delta_{lag}"
            work[name] = (work[col] - work[f"{col}_lag_{lag}"]).astype("float32")
            cols.append(name)
        add_rolling(work, "route_id", col, (4, 8), cols)

    local = work[STATUS_COLS[:3]].sum(axis=1)
    inbound = work[STATUS_COLS[3:]].sum(axis=1)
    work["local_status_sum"] = local.astype("float32")
    work["inbound_status_sum"] = inbound.astype("float32")
    work["total_status_sum"] = (local + inbound).astype("float32")
    work["inbound_local_ratio"] = (inbound / (local + 1)).astype("float32")
    cols.extend(("local_status_sum", "inbound_status_sum", "total_status_sum",
                 "inbound_local_ratio"))

    # The current target is known at the forecast origin. Relative to that
    # observation, seasonal levels and concurrent logistics signals are causal.
    for denom in ("target_1h_roll_mean_8", "target_1h_roll_mean_48",
                  "target_1h_roll_mean_96", "target_lag_48"):
        name = "current_over_" + denom
        work[name] = (work[TARGET_COL] / (work[denom] + 1.0)).astype("float32")
        cols.append(name)

    work["current_over_local_status"] = (
        work[TARGET_COL] / (work["local_status_sum"] + 1.0)
    ).astype("float32")
    cols.append("current_over_local_status")

    # Across-route aggregate at the CURRENT timestamp, available at inference.
    # Grouping over timestamp never mixes in future data.
    time_aggs = work.groupby("timestamp", sort=True).agg(
        net_target=(TARGET_COL, "sum"),
        net_local=("local_status_sum", "sum"),
        net_inbound=("inbound_status_sum", "sum"),
    ).sort_index()
    for original in ("net_target", "net_local", "net_inbound"):
        work[original] = work["timestamp"].map(time_aggs[original]).astype("float32")
        cols.append(original)
    for lag in (1, 4, 48):
        name = f"net_target_lag_{lag}"
        work[name] = work["timestamp"].map(
            time_aggs["net_target"].shift(lag)
        ).astype("float32")
        cols.append(name)
    work["route_share_of_network"] = (
        work[TARGET_COL] / (work["net_target"] + 1.0)
    ).astype("float32")
    cols.append("route_share_of_network")

    ts = work["timestamp"]
    slot = ts.dt.hour * 2 + ts.dt.minute // 30
    dow = ts.dt.day_of_week
    work["origin_slot"] = slot.astype("int8")
    work["origin_dow"] = dow.astype("int8")
    work["origin_is_weekend"] = (dow >= 5).astype("int8")
    work["origin_slot_sin"] = np.sin(2 * np.pi * slot / 48).astype("float32")
    work["origin_slot_cos"] = np.cos(2 * np.pi * slot / 48).astype("float32")
    cols.extend(("origin_slot", "origin_dow", "origin_is_weekend",
                 "origin_slot_sin", "origin_slot_cos"))

    for step, target_name in enumerate(FUTURE_TARGET_COLS, start=1):
        work[target_name] = targets.shift(-step)

    # Retain only rows whose entire historical context is available.
    work = work.loc[work["timestamp"] >= earliest_fit].copy()
    work["route_id"] = work["route_id"].astype("category")
    if len(set(cols)) != len(cols):
        raise ValueError("Duplicate feature names")
    print(f"Feature rows: {len(work):,} | Features: {len(cols)} | "
          f"First: {work['timestamp'].min()}", flush=True)
    return work, cols


def add_future_features(X: pd.DataFrame, timestamps: pd.Series,
                        step: int) -> None:
    future = timestamps + STEP * step
    future_slot = future.dt.hour * 2 + future.dt.minute // 30
    future_dow = future.dt.day_of_week
    X["future_slot"] = future_slot.to_numpy(dtype=np.int8)
    X["future_dow"] = future_dow.to_numpy(dtype=np.int8)
    X["future_is_weekend"] = (future_dow >= 5).to_numpy(dtype=np.int8)
    X["future_slot_sin"] = np.sin(2 * np.pi * future_slot / 48).to_numpy(dtype=np.float32)
    X["future_slot_cos"] = np.cos(2 * np.pi * future_slot / 48).to_numpy(dtype=np.float32)


def train_predict(fit_rows: pd.DataFrame, predict_rows: pd.DataFrame,
                  cols: list[str], model_name: str) -> np.ndarray:
    X_fit = fit_rows[cols].copy()
    X_predict = predict_rows[cols].copy()
    predictions = np.empty((len(predict_rows), FORECAST_POINTS), dtype=np.float64)
    for step, target_col in enumerate(FUTURE_TARGET_COLS, start=1):
        started = time.perf_counter()
        add_future_features(X_fit, fit_rows["timestamp"], step)
        add_future_features(X_predict, predict_rows["timestamp"], step)
        model = LGBMRegressor(**MODEL_PARAMS[model_name])
        model.fit(
            X_fit,
            fit_rows[target_col].to_numpy(dtype=np.float32),
            categorical_feature=["route_id"],
        )
        predictions[:, step - 1] = model.predict(X_predict)
        print(f"  {model_name} step {step}/8: "
              f"{time.perf_counter() - started:.1f}s", flush=True)
    return np.clip(predictions, 0.0, None)


def load_existing_fold(fold: int, origin: pd.Timestamp,
                       y_true: np.ndarray) -> np.ndarray:
    path005 = BASE005_DIR / f"ensemble_005b_fold_{fold}.npz"
    path006 = BASE006_DIR / f"oof_fold_{fold}.npz"
    if not path005.exists() or not path006.exists():
        raise FileNotFoundError(
            f"Missing historical OOF caches: {path005} / {path006}. "
            "Run EXP-005B and EXP-006A first."
        )
    with np.load(path005) as a:
        if str(a["origin"].item()) != str(origin):
            raise ValueError(f"005B cache origin mismatch: fold {fold}")
        if not np.array_equal(a["y_true"], y_true):
            raise ValueError(f"005B OOF target or order mismatch: fold {fold}")
        pred005 = a["lgbm_pred"].copy()
    with np.load(path006) as b:
        if str(b["origin"].item()) != str(origin):
            raise ValueError(f"006A cache origin mismatch: fold {fold}")
        pred006 = b["new_pred"].copy()
    if pred005.shape != y_true.shape or pred006.shape != y_true.shape:
        raise ValueError(f"Unexpected cached prediction shapes for fold {fold}")
    return (1.0 - BASE006_NEW_WEIGHT) * pred005 + BASE006_NEW_WEIGHT * pred006


def collect_oof(data: pd.DataFrame, cols: list[str], origins: list[pd.Timestamp],
                route_count: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    all_y, all_base = [], []
    all_models = {name: [] for name in MODEL_PARAMS}
    for fold in VALIDATION_FOLDS:
        origin = origins[fold - 1]
        valid = data.loc[
            data["timestamp"].eq(origin)
            & data[FUTURE_TARGET_COLS].notna().all(axis=1)
        ].copy()
        if len(valid) != route_count:
            raise ValueError(f"Fold {fold}: {len(valid)} rows instead of {route_count}")
        y_true = valid[FUTURE_TARGET_COLS].to_numpy(dtype=np.float64)
        base = load_existing_fold(fold, origin, y_true)
        print(f"\n{'=' * 60}\nFOLD {fold}/5: {origin}\n"
              f"Old ensemble score: {wape_plus_relative_bias(y_true, base)['score']:.6f}",
              flush=True)
        fit = None
        for name in MODEL_PARAMS:
            cache = REPORT_DIR / f"oof_{CACHE_VERSION}_{name}_fold{fold}.npz"
            if cache.exists():
                with np.load(cache) as stored:
                    if (str(stored["origin"].item()) != str(origin)
                            or str(stored["version"].item()) != CACHE_VERSION):
                        raise ValueError(f"Stale cache: {cache}; delete it before retrying")
                    candidate = stored["pred"].copy()
                print(f"  Loaded {name} cache", flush=True)
            else:
                if fit is None:
                    fit = select_training_rows(data, origin)
                    print(f"  Training rows: {len(fit):,}", flush=True)
                candidate = train_predict(fit, valid, cols, name)
                np.savez_compressed(
                    cache, origin=str(origin), version=CACHE_VERSION, pred=candidate
                )
            if candidate.shape != y_true.shape or not np.isfinite(candidate).all():
                raise ValueError(f"Invalid prediction from model {name} fold {fold}")
            all_models[name].append(candidate)
            print(f"  {name} raw score: "
                  f"{wape_plus_relative_bias(y_true, candidate)['score']:.6f}", flush=True)
        all_y.append(y_true)
        all_base.append(base)
    return (
        np.stack(all_y), np.stack(all_base),
        {name: np.stack(parts) for name, parts in all_models.items()},
    )


def calibrate_train_test(y_train: np.ndarray, p_train: np.ndarray,
                         p_test: np.ndarray, setting: str) -> np.ndarray:
    if setting == "none":
        return np.clip(p_test, 0.0, None)
    per_step = setting.startswith("step")
    strength = 0.5 if setting.endswith("50") else 1.0
    if per_step:
        sums_true = y_train.sum(axis=(0, 1))
        sums_pred = p_train.sum(axis=(0, 1))
        ratio = np.divide(
            sums_true, sums_pred, out=np.ones_like(sums_true),
            where=sums_pred > 0,
        )
    else:
        denom = p_train.sum()
        ratio = float(y_train.sum() / denom) if denom > 0 else 1.0
    factor = np.clip(1.0 + strength * (ratio - 1.0), 0.80, 1.25)
    return np.clip(p_test * factor, 0.0, None)


def cross_calibrate(y_true: np.ndarray, predictions: np.ndarray,
                    setting: str) -> np.ndarray:
    if setting == "none":
        return predictions.copy()
    adjusted = np.empty_like(predictions)
    for held in range(y_true.shape[0]):
        train_mask = np.arange(y_true.shape[0]) != held
        adjusted[held] = calibrate_train_test(
            y_true[train_mask], predictions[train_mask],
            predictions[held], setting,
        )
    return adjusted


def score(y: np.ndarray, p: np.ndarray) -> float:
    return float(wape_plus_relative_bias(y, p)["score"])


def make_blend(y_cal: np.ndarray, model_cal: np.ndarray,
               model_test: np.ndarray, base_test: np.ndarray,
               setting: str, weight: float) -> np.ndarray:
    calibrated = calibrate_train_test(y_cal, model_cal, model_test, setting)
    return np.clip((1.0 - weight) * base_test + weight * calibrated, 0.0, None)


def select_ensemble(y_true: np.ndarray, base: np.ndarray,
                    candidates: dict[str, np.ndarray]) -> tuple[str | None, str, float, float]:
    """Two-stage validation: pooled cross-fitted calibration + nested 3-fold LOFO.

    On each held-out fold, the model/calibration/weight combination is selected
    from the OTHER TWO folds. This avoids tuning ensemble weights on the fold
    that is being evaluated. Still only 3 folds: results remain uncertain.
    """
    baseline = score(y_true, base)
    print(f"\nEXP-006A three-fold baseline: {baseline:.6f}", flush=True)
    if abs(baseline - EXPECTED_BASE_OOF) > 0.0001:
        raise ValueError(
            "EXP-006A baseline differs from 0.335336. "
            "Check OOF cache alignment and baseline weight."
        )
    baseline_folds = [score(y_true[i], base[i]) for i in range(len(y_true))]
    combos = [(name, setting, weight)
              for name in candidates for setting in CALIBRATION for weight in WEIGHTS]

    results = []
    for model_name, setting, weight in combos:
        preds = candidates[model_name]
        calibrated = cross_calibrate(y_true, preds, setting)
        blended = np.clip((1.0 - weight) * base + weight * calibrated, 0.0, None)
        folded = [score(y_true[i], blended[i]) for i in range(len(y_true))]
        metrics = wape_plus_relative_bias(y_true, blended)
        results.append({
            "model": model_name, "calibration": setting,
            "new_weight": weight, **metrics,
            "folds_improved": sum(n < o for n, o in zip(folded, baseline_folds)),
            "last_fold_gain": baseline_folds[-1] - folded[-1],
            "gain": baseline - float(metrics["score"]),
        })
    table = pd.DataFrame(results).sort_values(["score", "wape"])
    table.to_csv(REPORT_DIR / "weight_search_007a.csv", index=False)
    print("\nWEIGHT SEARCH — cross-fitted calibration")
    print(table.head(15).to_string(index=False, float_format=lambda a: f"{a:.6f}"))

    # Nested leave-one-fold-out parameter selection.
    # Train-side scoring uses calibration fitted on the OTHER train-side fold;
    # the held-out truth NEVER enters fitting or candidate selection.
    lofo_predictions = np.empty_like(base)
    lofo_rows = []
    for held in range(len(y_true)):
        training = [idx for idx in range(len(y_true)) if idx != held]
        selected_combo = None
        selected_score = float("inf")
        for model_name, setting, weight in combos:
            model_pred = candidates[model_name]
            train_predictions = np.empty_like(base[training])
            for j, fold in enumerate(training):
                other = training[1 - j]
                train_predictions[j] = make_blend(
                    y_true[other:other + 1],
                    model_pred[other:other + 1],
                    model_pred[fold], base[fold], setting, weight,
                )
            candidate_score = score(y_true[training], train_predictions)
            if candidate_score < selected_score:
                selected_combo = (model_name, setting, weight)
                selected_score = candidate_score
        model_name, setting, weight = selected_combo
        lofo_predictions[held] = make_blend(
            y_true[training], candidates[model_name][training],
            candidates[model_name][held], base[held], setting, weight,
        )
        lofo_rows.append({
            "fold": VALIDATION_FOLDS[held], "chosen_model": model_name,
            "calibration": setting, "new_weight": weight,
            "baseline_score": baseline_folds[held],
            "lofo_score": score(y_true[held], lofo_predictions[held]),
        })
    lofo = pd.DataFrame(lofo_rows)
    lofo.to_csv(REPORT_DIR / "nested_lofo_007a.csv", index=False)
    lofo_score = score(y_true, lofo_predictions)
    lofo_gain = baseline - lofo_score
    print("\nNESTED LOFO — parameters selected without using the held-out fold")
    print(lofo.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Nested LOFO pooled score: {lofo_score:.6f} | "
          f"gain versus EXP-006A: {lofo_gain:+.6f}", flush=True)

    eligible = table.loc[
        (table["gain"] >= MIN_GAIN)
        & (table["folds_improved"] >= MIN_FOLDS_IMPROVED)
        & (table["last_fold_gain"] >= -0.0020)
    ]
    if eligible.empty or lofo_gain < MIN_LOFO_GAIN:
        print("\nNO ROBUST CV IMPROVEMENT — keep the accepted EXP-006A submission")
        return None, "none", 0.0, float(baseline)
    winner = eligible.iloc[0]
    print(f"\nWINNER: {winner['model']}, calibration={winner['calibration']}, "
          f"new_weight={winner['new_weight']:.2f}")
    print(f"CV score: {winner['score']:.6f}; "
          f"gain: {winner['gain']:.6f}; "
          f"folds improved: {int(winner['folds_improved'])}/3")
    print("Indicative public-score extrapolation (very uncertain): "
          f"{PUBLIC_006A - float(winner['gain']):.3f}")
    return (str(winner["model"]), str(winner["calibration"]),
            float(winner["new_weight"]), float(winner["score"]))


def read_submission(path: Path, test: pd.DataFrame) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing accepted submission: {path}")
    df = pd.read_csv(path, dtype={"id": str})
    if df.columns.tolist() != ["id", "y_pred"]:
        raise ValueError(f"Unexpected submission columns in {path}")
    if len(df) != len(test) or not df["id"].is_unique:
        raise ValueError(f"Invalid row count or duplicate IDs: {path}")
    lookup = df.set_index("id")["y_pred"]
    result = lookup.reindex(test["id"].astype(str)).to_numpy(dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Missing/nonfinite predictions in {path}")
    return result


def generate_submission(data: pd.DataFrame, cols: list[str],
                        train: pd.DataFrame, test: pd.DataFrame,
                        origin: pd.Timestamp,
                        y_true: np.ndarray, candidate_oof: np.ndarray,
                        model_name: str, setting: str, weight: float) -> None:
    path006 = SUBMISSION_DIR / "submission_lgbm_006a.csv"
    base = read_submission(path006, test)

    fit = select_training_rows(data, origin)
    rows = data.loc[data["timestamp"].eq(origin)].copy()
    if len(rows) != train["route_id"].nunique():
        raise ValueError("Missing route observations at the final forecast origin")
    print("\nTraining final winner with the latest 14-day window...", flush=True)
    pred_new = train_predict(fit, rows, cols, model_name)
    calibrated = calibrate_train_test(y_true, candidate_oof, pred_new, setting)

    routes = rows["route_id"].astype(train["route_id"].dtype).to_numpy()
    forecasts = pd.concat([
        pd.DataFrame({
            "route_id": routes,
            "timestamp": origin + STEP * step,
            "new_y_pred": calibrated[:, step - 1],
        }) for step in range(1, FORECAST_POINTS + 1)
    ], ignore_index=True)
    matched = test[["id", "route_id", "timestamp"]].merge(
        forecasts, on=["route_id", "timestamp"],
        how="left", validate="one_to_one", sort=False,
    )
    new = matched["new_y_pred"].to_numpy(dtype=np.float64)
    if not np.isfinite(new).all():
        raise ValueError("Missing new predictions after matching test rows")
    final_pred = np.clip((1.0 - weight) * base + weight * new, 0.0, None)

    submission = pd.DataFrame({"id": test["id"], "y_pred": final_pred})
    if len(submission) != len(test) or not submission["id"].is_unique:
        raise ValueError("Submission must contain one row per unique test ID")
    if not np.isfinite(submission["y_pred"]).all():
        raise ValueError("Invalid predictions")
    output = SUBMISSION_DIR / "submission_lgbm_007a.csv"
    submission.to_csv(output, index=False)
    print(f"\nSUBMISSION SAVED: {output}")
    print(f"Shape: {submission.shape} | Columns: {submission.columns.tolist()}")
    print(f"Prediction mean: {submission['y_pred'].mean():.3f}")


def main() -> None:
    started = time.perf_counter()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_data()
    last_timestamp = train["timestamp"].max()
    origins = get_origins(last_timestamp)
    select_origins = [origins[fold - 1] for fold in VALIDATION_FOLDS]
    print("\nEXP-007A — TWO-LOSS LIGHTGBM + ROBUST ENSEMBLING", flush=True)
    print(f"Folds: {VALIDATION_FOLDS}; models: {list(MODEL_PARAMS)}", flush=True)
    data, features = build_features(train, select_origins)
    y_true, base_oof, candidate_oof = collect_oof(
        data, features, origins, train["route_id"].nunique(),
    )
    winner, setting, weight, score = select_ensemble(
        y_true, base_oof, candidate_oof,
    )
    if winner is None:
        print("No submission created; EXP-006A remains best.")
        return
    generate_submission(
        data, features, train, test, last_timestamp,
        y_true, candidate_oof[winner], winner, setting, weight,
    )
    print(f"Validation score: {score:.6f}; public comparison only after upload.")
    print(f"Total runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
