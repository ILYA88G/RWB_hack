
"""EXP-004A: Ridge with status features and historical target lags."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    STATUS_COLS,
    FUTURE_TARGET_COLS,
    FORECAST_POINTS,
    HORIZON,
    TRAIN_DAYS,
    MAX_TRAIN_ROWS,
    RIDGE_ALPHA,
    RANDOM_STATE,
    load_data,
    wape_plus_relative_bias,
)


EXPERIMENT_REPORTS_DIR = REPORTS_DIR / "exp_004"
TARGET_LAGS = (1, 2, 4, 8)
STATUS_LAGS = (1, 2, 4)


def get_origins(last_timestamp: pd.Timestamp) -> list[pd.Timestamp]:
    return [
        last_timestamp - pd.Timedelta(days=days)
        for days in range(4, 0, -1)
    ] + [last_timestamp - HORIZON]


def build_features(
    train: pd.DataFrame,
    origins: list[pd.Timestamp],
) -> tuple[pd.DataFrame, list[str]]:

    earliest_start = min(
        origin - HORIZON - pd.Timedelta(days=TRAIN_DAYS)
        for origin in origins
    )

    # Keep enough history to calculate the longest lag (8 steps = 4 hours).
    work = train.loc[
        train["timestamp"] >= earliest_start - HORIZON
    ].copy()

    work = work.sort_values(
        ["route_id", "timestamp"]
    ).reset_index(drop=True)

    grouped = work.groupby("route_id", sort=False)

    feature_cols = ["route_id"] + STATUS_COLS

    # Historical status lags and deltas.
    for col in STATUS_COLS:
        for lag in STATUS_LAGS:
            lag_col = f"{col}_lag_{lag}"
            work[lag_col] = grouped[col].shift(lag)
            feature_cols.append(lag_col)

        delta_col = f"{col}_delta_1"
        work[delta_col] = work[col] - work[f"{col}_lag_1"]
        feature_cols.append(delta_col)

    # Historical target lags. Only past values are used.
    target_group = work.groupby(
        "route_id", sort=False
    )[TARGET_COL]

    for lag in TARGET_LAGS:
        lag_col = f"{TARGET_COL}_lag_{lag}"
        work[lag_col] = target_group.shift(lag)
        feature_cols.append(lag_col)

    # Eight direct future targets.
    for step, col in enumerate(FUTURE_TARGET_COLS, start=1):
        work[col] = target_group.shift(-step)

    # Remove rows outside the required feature window.
    work = work.loc[
        work["timestamp"] >= earliest_start
    ].copy()

    print(f"Feature window shape: {work.shape}")
    print(f"Numeric feature count: {len(feature_cols) - 1}")
    print(f"Total feature count: {len(feature_cols)}")
    print(f"Feature window start: {work['timestamp'].min()}")

    return work, feature_cols


def select_training_rows(
    data: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:

    fit_end = origin - HORIZON
    fit_start = fit_end - pd.Timedelta(days=TRAIN_DAYS)

    mask = (
        data["timestamp"].between(fit_start, fit_end)
        & data[FUTURE_TARGET_COLS].notna().all(axis=1)
    )

    rows = data.loc[mask].copy()

    if rows.empty:
        raise ValueError(f"No training rows for origin {origin}")

    if len(rows) > MAX_TRAIN_ROWS:
        rows = rows.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
        ).sort_values(["route_id", "timestamp"])

    return rows


def make_model(feature_cols: list[str]) -> Pipeline:

    numeric_cols = [
        col for col in feature_cols if col != "route_id"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["route_id"]),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=RIDGE_ALPHA)),
    ])


def main() -> None:

    start = time.perf_counter()

    EXPERIMENT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = load_data()

    last_timestamp = train["timestamp"].max()
    expected_routes = train["route_id"].nunique()

    origins = get_origins(last_timestamp)

    print("\nEXP-004A — ROLLING VALIDATION")
    print(f"Ridge alpha: {RIDGE_ALPHA}")
    print(f"Target lags: {TARGET_LAGS}")
    print(f"Status lags: {STATUS_LAGS}")

    print("\nOrigins:")
    for origin in origins:
        print(f"  {origin}")

    data, feature_cols = build_features(train, origins)

    numeric_cols = [
        col for col in feature_cols if col != "route_id"
    ]

    fold_results = []
    step_results = []

    all_true = []
    all_pred = []

    for fold_idx, origin in enumerate(origins, start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx}/{len(origins)} — {origin}")
        print("=" * 60)

        fit_rows = select_training_rows(data, origin)

        valid_rows = data.loc[
            data["timestamp"].eq(origin)
            & data[FUTURE_TARGET_COLS].notna().all(axis=1)
        ].copy()

        if len(valid_rows) != expected_routes:
            raise ValueError(
                f"Expected {expected_routes} validation rows, "
                f"got {len(valid_rows)}"
            )

        X_train = fit_rows[feature_cols]
        y_train = fit_rows[FUTURE_TARGET_COLS].to_numpy(
            dtype=np.float64
        )

        X_valid = valid_rows[feature_cols]
        y_true = valid_rows[FUTURE_TARGET_COLS].to_numpy(
            dtype=np.float64
        )

        model = make_model(feature_cols)

        print(f"Training rows: {len(fit_rows):,}")
        print(f"Numeric features: {len(numeric_cols)}")

        fit_start = time.perf_counter()
        model.fit(X_train, y_train)

        print(
            f"Fit time: {time.perf_counter() - fit_start:.1f}s"
        )

        y_pred = np.clip(
            model.predict(X_valid),
            0.0,
            None,
        )

        metrics = wape_plus_relative_bias(y_true, y_pred)

        fold_results.append({
            "origin": origin,
            "fit_rows": len(fit_rows),
            **metrics,
        })

        all_true.append(y_true)
        all_pred.append(y_pred)

        for step_idx in range(FORECAST_POINTS):

            step_metric = wape_plus_relative_bias(
                y_true[:, step_idx],
                y_pred[:, step_idx],
            )

            step_results.append({
                "origin": origin,
                "step": step_idx + 1,
                **step_metric,
            })

        print(
            f"Score: {metrics['score']:.6f} | "
            f"WAPE: {metrics['wape']:.6f} | "
            f"Bias: {metrics['relative_bias']:.6f}"
        )

    # Aggregate results.
    summary = pd.DataFrame(fold_results)
    by_step = pd.DataFrame(step_results)

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)

    aggregate = wape_plus_relative_bias(
        y_true_all,
        y_pred_all,
    )

    aggregate_steps = []

    for step_idx in range(FORECAST_POINTS):

        metrics = wape_plus_relative_bias(
            y_true_all[:, step_idx],
            y_pred_all[:, step_idx],
        )

        aggregate_steps.append({
            "step": step_idx + 1,
            **metrics,
        })

    aggregate_steps = pd.DataFrame(aggregate_steps)

    # Print summary.
    print("\n" + "=" * 60)
    print("EXP-004A — ROLLING VALIDATION SUMMARY")
    print("=" * 60)

    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    print("\nAggregate metrics:")
    print(f"WAPE:  {aggregate['wape']:.6f}")
    print(f"Bias:  {aggregate['relative_bias']:.6f}")
    print(f"Score: {aggregate['score']:.6f}")

    print("\nAggregate metrics by step:")
    print(aggregate_steps.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    # Save reports.
    summary.to_csv(
        EXPERIMENT_REPORTS_DIR / "ridge_target_lags_004a_summary.csv",
        index=False,
    )

    by_step.to_csv(
        EXPERIMENT_REPORTS_DIR / "ridge_target_lags_004a_by_step.csv",
        index=False,
    )

    aggregate_steps.to_csv(
        EXPERIMENT_REPORTS_DIR / "ridge_target_lags_004a_aggregate_by_step.csv",
        index=False,
    )

    print("\nReports saved in:", EXPERIMENT_REPORTS_DIR)
    print(f"Total time: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
