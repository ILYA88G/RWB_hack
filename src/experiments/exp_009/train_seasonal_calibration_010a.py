"""EXP-010A: fast, causal seasonal shrinkage of accepted EXP-008B.

Run from project root:
    python -m src.experiments.exp_010.train_seasonal_calibration_010a

Requires 005B/006A OOF caches and submission_network_008b.csv.
No route-level model training. All 8 test targets are predicted at the final
known origin; earlier historical values only are used as analogues.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from train_ridge_baseline import (
    PROJECT_ROOT, REPORTS_DIR, TARGET_COL, STATUS_COLS,
    STEP, FORECAST_POINTS, wape_plus_relative_bias,
)
from src.experiments.exp_004.train_ridge_target_lags_004a import get_origins
from src.experiments.exp_008.train_network_calibration_008b import (
    build_network, predict_network, load_baseline_oof, corrected,
)

ROOT = REPORTS_DIR / "exp_010"
SUBMISSIONS = PROJECT_ROOT / "submissions"
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train_solo_track.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "test_solo_track.parquet"
FOLDS = (3, 4, 5)
BASELINE_CV = 0.33025948686895884
MAX_WEIGHT = 0.20
WEIGHTS = (0.02, 0.04, 0.07, 0.10, 0.15, 0.20)
MIN_TRAIN_GAIN = 0.0005
MIN_HOLDOUT_GAIN = 0.0005
MIN_TOTAL_GAIN = 0.0007


def metric(y: np.ndarray, p: np.ndarray) -> float:
    return float(wape_plus_relative_bias(y, p)["score"])


def history_matrix(train: pd.DataFrame) -> pd.DataFrame:
    """Rows = sorted routes, columns = regular 30-minute timestamps."""
    pivot = train.pivot(
        index="route_id", columns="timestamp", values=TARGET_COL
    ).sort_index().sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Missing route-time observations in history")
    time_grid = pd.DatetimeIndex(pivot.columns)
    if not (time_grid[1:] - time_grid[:-1] == STEP).all():
        raise ValueError("History does not have a continuous 30-minute grid")
    return pivot


def historical_analogs(
    values: np.ndarray,
    times: pd.DatetimeIndex,
    origin: pd.Timestamp,
) -> dict[str, np.ndarray]:
    """No values after origin are read, including for future test horizons."""
    t = int(times.get_loc(origin))
    if t < 680:
        raise ValueError(f"Insufficient two-week history for {origin}")

    # x: [routes, time]. Last observed value is x[:, t].
    mean16 = values[:, t - 15:t + 1].mean(axis=1, keepdims=True)
    mean48 = values[:, t - 47:t + 1].mean(axis=1, keepdims=True)
    mean96 = values[:, t - 95:t + 1].mean(axis=1, keepdims=True)
    yesterday = values[:, t - 47:t - 39]  # t+s-48, s=1..8
    week1 = values[:, t - 335:t - 327]  # t+s-336
    week2 = values[:, t - 671:t - 663]  # t+s-672

    # Clamp adjustment of the previous-week profile to recent route activity.
    last_week_level = values[:, t - 383:t - 335].mean(axis=1, keepdims=True)
    recent_to_week = np.clip(
        (mean48 + 20_000.0) / (last_week_level + 20_000.0),
        0.80, 1.20,
    )
    week_adjusted = week1 * recent_to_week
    week_mean = 0.6 * week1 + 0.4 * week2

    candidates = {
        "day_level": np.repeat(mean48, FORECAST_POINTS, axis=1),
        "recent_level": np.repeat(mean16, FORECAST_POINTS, axis=1),
        "two_day_level": np.repeat(0.7 * mean48 + 0.3 * mean96,
                                   FORECAST_POINTS, axis=1),
        "yesterday": yesterday,
        "weekly": week1,
        "two_weeks": week_mean,
        "weekly_adjusted": week_adjusted,
        "daily_pattern": 0.75 * mean48 + 0.25 * yesterday,
        "weekly_pattern": 0.75 * mean48 + 0.25 * week_adjusted,
        "weekly_mean_pattern": 0.75 * mean48 + 0.25 * week_mean,
    }
    for name, arr in candidates.items():
        if arr.shape != (values.shape[0], FORECAST_POINTS):
            raise ValueError(f"Bad shape for {name}: {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite historical analogue: {name}")
    return candidates


def blend_preserve_totals(
    base: np.ndarray, seasonal: np.ndarray, weight: float,
) -> np.ndarray:
    """Correct route proportions without undoing EXP-008B network calibration."""
    pred = np.clip((1 - weight) * base + weight * seasonal, 0.0, None)
    base_totals = base.sum(axis=0)
    pred_totals = pred.sum(axis=0)
    if (base_totals <= 0).any() or (pred_totals <= 0).any():
        raise ValueError("Nonpositive network totals")
    return pred * (base_totals / pred_totals)[None, :]


def reconstruct_oof(
    train: pd.DataFrame,
    origins: list[pd.Timestamp],
    pivot: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    y, old_006a = load_baseline_oof(origins)
    if pivot.shape[0] != y.shape[1]:
        raise ValueError("Unexpected route count in OOF")
    matrix = pivot.to_numpy(dtype=np.float64, copy=False)
    times = pd.DatetimeIndex(pivot.columns)
    network, features = build_network(train)
    pred_008b = []
    for j, fold in enumerate(FOLDS):
        origin = origins[fold - 1]
        t = int(times.get_loc(origin))
        expected_y = matrix[:, t + 1:t + 1 + FORECAST_POINTS]
        if expected_y.shape != y[j].shape or not np.array_equal(expected_y, y[j]):
            raise ValueError(f"OOF target/route order mismatch in fold {fold}")

        cache = ROOT / f"network_fold{fold}.npy"
        if cache.exists():
            net_forecast = np.load(cache)
        else:
            net_forecast = predict_network(network, features, origin, "lgbm28")
            np.save(cache, net_forecast)
        pred, _ = corrected(old_006a[j], net_forecast, 0.80, "per_step")
        pred_008b.append(pred)
        print(f"Fold {fold}: EXP-008B score {metric(y[j], pred):.6f}", flush=True)
    base = np.stack(pred_008b)
    baseline = metric(y, base)
    print(f"Reconstructed EXP-008B CV: {baseline:.6f}", flush=True)
    if abs(baseline - BASELINE_CV) > 0.00015:
        raise ValueError("Baseline mismatch: verify 005B/006A caches and EXP-008B")
    return y, base


def select(
    y: np.ndarray, base: np.ndarray,
    matrix: np.ndarray, times: pd.DatetimeIndex,
    origins: list[pd.Timestamp],
) -> tuple[str, float] | None:
    base_folds = [metric(y[j], base[j]) for j in range(3)]
    base_training = metric(y[:2], base[:2])
    analogs = [historical_analogs(matrix, times, origins[f - 1]) for f in FOLDS]
    results = []
    for name in analogs[0]:
        for weight in WEIGHTS:
            p = np.stack([
                blend_preserve_totals(base[j], analogs[j][name], weight)
                for j in range(3)
            ])
            s3, s4, s5 = [metric(y[j], p[j]) for j in range(3)]
            train_score = metric(y[:2], p[:2])
            total_score = metric(y, p)
            results.append({
                "analogue": name,
                "weight": weight,
                "train_score": train_score,
                "train_gain": base_training - train_score,
                "fold3_gain": base_folds[0] - s3,
                "fold4_gain": base_folds[1] - s4,
                "holdout_score": s5,
                "holdout_gain": base_folds[2] - s5,
                "pooled_score": total_score,
                "pooled_gain": metric(y, base) - total_score,
            })
    report = pd.DataFrame(results).sort_values(
        ["train_score", "weight"], ascending=[True, True]
    )
    report.to_csv(ROOT / "seasonal_010a_search.csv", index=False)
    print("\nTOP CANDIDATES (ranked using folds 3 and 4 only):")
    print(report.head(12).to_string(index=False, float_format=lambda z: f"{z:.6f}"))

    # Choose on folds 3+4 ONLY, before inspecting the held-out fold 5.
    best = report.iloc[0]
    print("\nFORWARD CHECK (fold 5 NOT used to tune analogue/weight):")
    print(f"Chosen analogue: {best['analogue']} | weight: {best['weight']:.2f}")
    print(f"Training gain (folds 3+4): {best['train_gain']:+.6f}")
    print(f"Holdout gain (fold 5):      {best['holdout_gain']:+.6f}")
    print(f"Pooled gain (all 3):       {best['pooled_gain']:+.6f}")
    passed = (
        float(best["train_gain"]) >= MIN_TRAIN_GAIN
        and float(best["holdout_gain"]) >= MIN_HOLDOUT_GAIN
        and float(best["pooled_gain"]) >= MIN_TOTAL_GAIN
        and min(float(best["fold3_gain"]), float(best["fold4_gain"])) >= -0.001
    )
    pd.DataFrame([{"passed": passed, **best.to_dict()}]).to_csv(
        ROOT / "seasonal_010a_forward_check.csv", index=False
    )
    if not passed:
        print("\nNO RELIABLE IMPROVEMENT. Keep EXP-008B (Public Score 0.337).")
        return None
    print("\nPASSED: preparing new submission.")
    return str(best["analogue"]), float(best["weight"])


def create_submission(
    test: pd.DataFrame, pivot: pd.DataFrame,
    final_origin: pd.Timestamp, choice: tuple[str, float],
) -> None:
    source_path = SUBMISSIONS / "submission_network_008b.csv"
    previous = pd.read_csv(source_path, dtype={"id": "string"})
    if previous.columns.tolist() != ["id", "y_pred"]:
        raise ValueError("Wrong EXP-008B submission columns")
    if len(previous) != len(test) or not previous["id"].is_unique:
        raise ValueError("Invalid EXP-008B submission size or IDs")
    base_by_id = previous.set_index("id")["y_pred"]
    values = base_by_id.reindex(test["id"].astype("string")).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Test IDs do not match EXP-008B submission")
    route_index = pd.Index(pivot.index).get_indexer(test["route_id"])
    offsets = ((test["timestamp"] - final_origin) / STEP).to_numpy(dtype=float)
    if not np.allclose(offsets, np.rint(offsets)):
        raise ValueError("Irregular test timestamps")
    steps = np.rint(offsets).astype(int) - 1
    if (route_index < 0).any() or (steps < 0).any() or (steps >= 8).any():
        raise ValueError("Unrecognized test route/time")
    base = np.full((len(pivot), FORECAST_POINTS), np.nan, dtype=float)
    base[route_index, steps] = values
    if not np.isfinite(base).all():
        raise ValueError("Missing route/time combinations")
    mode, weight = choice
    analogs = historical_analogs(
        pivot.to_numpy(dtype=np.float64, copy=False),
        pd.DatetimeIndex(pivot.columns), final_origin,
    )
    blended = blend_preserve_totals(base, analogs[mode], weight)
    submission = pd.DataFrame({
        "id": test["id"],
        "y_pred": blended[route_index, steps],
    })
    if len(submission) != 8000 or not submission["id"].is_unique:
        raise ValueError("Wrong submission shape / duplicate IDs")
    if not np.isfinite(submission["y_pred"]).all():
        raise ValueError("Invalid submission predictions")
    path = SUBMISSIONS / "submission_seasonal_010a.csv"
    submission.to_csv(path, index=False)
    print(f"\nSUBMISSION SAVED: {path}")
    print(f"Rows: {len(submission)} | Columns: {submission.columns.tolist()}")


def main() -> None:
    started = time.perf_counter()
    ROOT.mkdir(parents=True, exist_ok=True)
    print("EXP-010A: seasonal shrinkage, no route-level retraining", flush=True)
    # Load only necessary columns and only the latest 39 days (enough for
    # 28-day network fits, weekly feature history and two-week analogues).
    test = pd.read_parquet(TEST_PATH)
    # Test begins one half-hour after the latest observed origin.
    # We verify this against the filtered train data before using it.
    last_timestamp = pd.Timestamp(test["timestamp"].min()) - STEP
    origins = get_origins(last_timestamp)
    earliest = origins[FOLDS[0] - 1] - pd.Timedelta(days=39)
    cols = ["route_id", "timestamp", TARGET_COL, *STATUS_COLS]
    train = pd.read_parquet(
        TRAIN_PATH, columns=cols, filters=[("timestamp", ">=", earliest)]
    )
    if train["timestamp"].max() != last_timestamp:
        raise ValueError("Test does not start immediately after training")
    print(f"History rows: {len(train):,} | Test rows: {len(test):,}", flush=True)
    pivot = history_matrix(train)
    y, base = reconstruct_oof(train, origins, pivot)
    selected = select(y, base, pivot.to_numpy(dtype=np.float64, copy=False),
                      pd.DatetimeIndex(pivot.columns), origins)
    if selected is not None:
        create_submission(test, pivot, last_timestamp, selected)
    print(f"Elapsed: {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
