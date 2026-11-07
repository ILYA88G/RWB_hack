"""Leakage-aware Ridge baseline for the RWB 'Отгрузки без простоев' task.

Run from any working directory:
    python train_ridge_baseline.py

Expected project layout:
    RWB_hack/
      train_ridge_baseline.py
      data/raw/train_solo_track.parquet
      data/raw/test_solo_track.parquet

The script:
1) loads and validates Train/Test;
2) creates 8 direct future targets (30-minute steps);
3) evaluates a Ridge multi-output model on a holdout origin that mimics Test;
4) refits the model using all eligible recent training history;
5) predicts the 8 Test points per route and writes a valid submission CSV.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train_solo_track.parquet"
TEST_PATH = DATA_DIR / "test_solo_track.parquet"

SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "target_1h"
STATUS_COLS = [f"status_{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["route_id"]
FEATURE_COLS = CATEGORICAL_COLS + STATUS_COLS

FORECAST_POINTS = 8
STEP = pd.Timedelta(minutes=30)
HORIZON = STEP * FORECAST_POINTS  # 4 hours
TRAIN_DAYS = 14
MAX_TRAIN_ROWS = 1_500_000
RIDGE_ALPHA = 4.0
RANDOM_STATE = 42

FUTURE_TARGET_COLS = [
    f"target_step_{i}" for i in range(1, FORECAST_POINTS + 1)
]


# ---------------------------------------------------------------------------
# 2. Competition metric
# ---------------------------------------------------------------------------
def wape_plus_relative_bias(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Exact competition metric: WAPE + absolute Relative Bias."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Metric received NaN or infinite values")

    denominator = float(y_true.sum())
    if denominator == 0.0:
        raise ZeroDivisionError("Sum of y_true is zero; WAPE/Relative Bias are undefined")

    wape = float(np.abs(y_pred - y_true).sum() / denominator)
    relative_bias = float(abs(y_pred.sum() / denominator - 1.0))
    return {
        "wape": wape,
        "relative_bias": relative_bias,
        "score": wape + relative_bias,
    }


def print_metric_report(title: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    overall = wape_plus_relative_bias(y_true, y_pred)
    print(f"\n{title}")
    print(f"WAPE:            {overall['wape']:.6f}")
    print(f"Relative Bias:   {overall['relative_bias']:.6f}")
    print(f"Competition score (lower is better): {overall['score']:.6f}")

    rows = []
    for step_idx in range(y_true.shape[1]):
        m = wape_plus_relative_bias(y_true[:, step_idx], y_pred[:, step_idx])
        rows.append({"step": step_idx + 1, **m})

    per_step = pd.DataFrame(rows)
    print("\nMetric by forecast step:")
    print(per_step.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return per_step


# ---------------------------------------------------------------------------
# 3. Data loading and structural validation
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}")
    print("Loading Parquet files...")

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)

    required_train = {"route_id", "timestamp", TARGET_COL, *STATUS_COLS}
    required_test = {"id", "route_id", "timestamp"}
    missing_train = required_train - set(train.columns)
    missing_test = required_test - set(test.columns)
    if missing_train:
        raise ValueError(f"Train is missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test is missing columns: {sorted(missing_test)}")

    train["timestamp"] = pd.to_datetime(train["timestamp"], errors="raise")
    test["timestamp"] = pd.to_datetime(test["timestamp"], errors="raise")

    train = train.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    test = test.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    if train.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Train has duplicate (route_id, timestamp) rows")
    if test.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Test has duplicate (route_id, timestamp) rows")
    if test["id"].duplicated().any():
        raise ValueError("Test contains duplicate id values")
    if train[list(required_train)].isna().any().any():
        raise ValueError("Train contains missing values in required columns")
    if test[list(required_test)].isna().any().any():
        raise ValueError("Test contains missing values in required columns")

    train_routes = set(train["route_id"].unique())
    test_routes = set(test["route_id"].unique())
    if train_routes != test_routes:
        raise ValueError(
            "Train/Test route sets differ: "
            f"missing in Test={len(train_routes - test_routes)}, "
            f"new in Test={len(test_routes - train_routes)}"
        )

    test_counts = test.groupby("route_id").size()
    if not (test_counts == FORECAST_POINTS).all():
        raise ValueError(
            "Expected exactly 8 Test rows per route. Counts: "
            f"{test_counts.value_counts().sort_index().to_dict()}"
        )

    # Check that Test contains exactly the next 8 half-hour timestamps per route.
    last_train_ts = train.groupby("route_id")["timestamp"].max()
    test_check = test.sort_values(["route_id", "timestamp"]).copy()
    test_check["step"] = test_check.groupby("route_id").cumcount() + 1
    expected = test_check["route_id"].map(last_train_ts) + STEP * test_check["step"]
    if not (test_check["timestamp"].to_numpy() == expected.to_numpy()).all():
        raise ValueError("Test timestamps are not the next 8 half-hour steps after Train")

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"Routes:      {train['route_id'].nunique()}")
    print(f"Train range: {train['timestamp'].min()} -> {train['timestamp'].max()}")
    print(f"Test range:  {test['timestamp'].min()} -> {test['timestamp'].max()}")
    return train, test


# ---------------------------------------------------------------------------
# 4. Create a compact supervised window with 8 future targets
# ---------------------------------------------------------------------------
def make_supervised_window(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    last_timestamp = train["timestamp"].max()

    # One holdout origin: last Train timestamp minus the 4-hour forecast horizon.
    # Its next 8 target values are still observable in Train, so validation mimics Test.
    validation_origin = last_timestamp - HORIZON

    # The model trained for an origin can only use rows whose 8th future target
    # is known no later than that origin. This creates a 4-hour purge gap.
    validation_fit_end = validation_origin - HORIZON
    validation_fit_start = validation_fit_end - pd.Timedelta(days=TRAIN_DAYS)

    # Keep only the history needed for validation and final training, not all 4.63M rows.
    final_fit_start = (last_timestamp - HORIZON) - pd.Timedelta(days=TRAIN_DAYS)
    window_start = min(validation_fit_start, final_fit_start)
    work = train.loc[train["timestamp"] >= window_start].copy()
    work = work.sort_values(["route_id", "timestamp"]).reset_index(drop=True)

    grouped_target = work.groupby("route_id", sort=False)[TARGET_COL]
    for step_idx, col in enumerate(FUTURE_TARGET_COLS, start=1):
        work[col] = grouped_target.shift(-step_idx)

    print(f"\nSupervised window: {work.shape}")
    print(f"Validation forecast origin: {validation_origin}")
    print(f"Validation fit source range: {validation_fit_start} -> {validation_fit_end}")
    print(f"Final fit source range:      {final_fit_start} -> {last_timestamp - HORIZON}")
    return work, validation_origin


def select_fit_rows(
    supervised: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    fit_end = origin - HORIZON
    fit_start = fit_end - pd.Timedelta(days=TRAIN_DAYS)
    mask = (
        supervised["timestamp"].between(fit_start, fit_end)
        & supervised[FUTURE_TARGET_COLS].notna().all(axis=1)
    )
    fit_rows = supervised.loc[mask].copy()

    if fit_rows.empty:
        raise ValueError(f"No training rows available for origin {origin}")

    if len(fit_rows) > MAX_TRAIN_ROWS:
        fit_rows = fit_rows.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        ).sort_values(["route_id", "timestamp"])
        print(f"Training rows capped deterministically at {MAX_TRAIN_ROWS:,}")

    return fit_rows


# ---------------------------------------------------------------------------
# 5. Build and fit the Ridge multi-output pipeline
# ---------------------------------------------------------------------------
def make_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, STATUS_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def fit_model(fit_rows: pd.DataFrame, label: str) -> Pipeline:
    X_fit = fit_rows[FEATURE_COLS]
    y_fit = fit_rows[FUTURE_TARGET_COLS].to_numpy(dtype=np.float64)

    model = make_model()
    print(f"\nFitting {label} model...")
    print(f"Fit rows: {len(fit_rows):,}")
    print(f"X shape:  {X_fit.shape}")
    print(f"y shape:  {y_fit.shape}")

    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    elapsed = time.perf_counter() - start
    print(f"Training completed in {elapsed:.1f} seconds")
    return model


# ---------------------------------------------------------------------------
# 6. Validation that mirrors the actual 8-step Test forecast
# ---------------------------------------------------------------------------
def run_validation(
    supervised: pd.DataFrame,
    validation_origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = select_fit_rows(supervised, validation_origin)
    valid_rows = supervised.loc[
        supervised["timestamp"].eq(validation_origin)
        & supervised[FUTURE_TARGET_COLS].notna().all(axis=1)
    ].copy()

    expected_routes = supervised["route_id"].nunique()
    if len(valid_rows) != expected_routes:
        raise ValueError(
            f"Expected one validation origin row per route ({expected_routes}), "
            f"got {len(valid_rows)}"
        )

    model = fit_model(fit_rows, "validation")
    X_valid = valid_rows[FEATURE_COLS]
    y_valid = valid_rows[FUTURE_TARGET_COLS].to_numpy(dtype=np.float64)
    y_valid_pred = np.clip(model.predict(X_valid), 0.0, None)

    per_step = print_metric_report(
        "LEAKAGE-AWARE VALIDATION (one origin per route, 8 steps)",
        y_valid,
        y_valid_pred,
    )

    summary = wape_plus_relative_bias(y_valid, y_valid_pred)
    summary_df = pd.DataFrame(
        [{"validation_origin": validation_origin, **summary}]
    )
    return summary_df, per_step


# ---------------------------------------------------------------------------
# 7. Refit on all eligible recent history and predict Test
# ---------------------------------------------------------------------------
def train_final_and_predict(
    train: pd.DataFrame,
    supervised: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    final_origin = train["timestamp"].max()
    final_fit_rows = select_fit_rows(supervised, final_origin)
    final_model = fit_model(final_fit_rows, "final")

    # One latest known feature row per route, matching the organizers' inference logic.
    latest_rows = (
        train.sort_values(["route_id", "timestamp"])
        .groupby("route_id", sort=False)
        .tail(1)
        .sort_values("route_id")
        .reset_index(drop=True)
    )
    X_test_origin = latest_rows[FEATURE_COLS]
    predictions = np.clip(final_model.predict(X_test_origin), 0.0, None)

    # Expand (route, 8 predictions) into (route, future timestamp, prediction).
    forecast_parts = []
    for step_idx in range(1, FORECAST_POINTS + 1):
        forecast_parts.append(
            pd.DataFrame(
                {
                    "route_id": latest_rows["route_id"].to_numpy(),
                    "timestamp": latest_rows["timestamp"].to_numpy() + STEP * step_idx,
                    "y_pred": predictions[:, step_idx - 1],
                }
            )
        )
    forecast_long = pd.concat(forecast_parts, ignore_index=True)

    # Join by route and timestamp; then restore Test's original id order.
    submission = test[["id", "route_id", "timestamp"]].merge(
        forecast_long,
        on=["route_id", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    submission = submission.sort_values("id", kind="stable").reset_index(drop=True)
    submission = submission[["id", "y_pred"]]

    # Strict platform-format checks.
    if len(submission) != len(test):
        raise ValueError(f"Submission has {len(submission)} rows; expected {len(test)}")
    if submission.columns.tolist() != ["id", "y_pred"]:
        raise ValueError(f"Wrong submission columns: {submission.columns.tolist()}")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids")
    if submission["y_pred"].isna().any():
        missing = int(submission["y_pred"].isna().sum())
        raise ValueError(f"Submission has {missing} missing predictions")
    if not np.isfinite(submission["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError("Submission contains non-finite predictions")
    if (submission["y_pred"] < 0).any():
        raise ValueError("Submission contains negative predictions")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "ridge_baseline.joblib"
    submission_path = SUBMISSIONS_DIR / "ridge_baseline_submission.csv"
    joblib.dump(final_model, model_path)
    submission.to_csv(submission_path, index=False)

    print("\nSubmission checks passed")
    print(f"Rows:        {len(submission):,}")
    print(f"Columns:     {submission.columns.tolist()}")
    print(f"ID unique:   {submission['id'].is_unique}")
    print(f"Pred min:    {submission['y_pred'].min():.3f}")
    print(f"Pred max:    {submission['y_pred'].max():.3f}")
    print(f"Saved CSV:   {submission_path}")
    print(f"Saved model: {model_path}")

    return submission


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    start_total = time.perf_counter()
    train, test = load_data()
    supervised, validation_origin = make_supervised_window(train)

    validation_summary, per_step_metrics = run_validation(
        supervised,
        validation_origin,
    )

    validation_summary.to_csv(
        REPORTS_DIR / "ridge_validation_summary.csv",
        index=False,
    )
    per_step_metrics.to_csv(
        REPORTS_DIR / "ridge_validation_by_step.csv",
        index=False,
    )

    submission = train_final_and_predict(train, supervised, test)

    print(f"\nTotal execution time: {time.perf_counter() - start_total:.1f} seconds")
    print("\nDone. This is the Ridge baseline; compare future models against its validation score.")


if __name__ == "__main__":
    main()
