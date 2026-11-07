
"""Rolling-origin validation for the Ridge baseline.

Run:
    python validate_ridge_rolling.py

The original train_ridge_baseline.py is not modified.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    STATUS_COLS,
    FUTURE_TARGET_COLS,
    FEATURE_COLS,
    FORECAST_POINTS,
    STEP,
    HORIZON,
    TRAIN_DAYS,
    load_data,
    select_fit_rows,
    fit_model,
    wape_plus_relative_bias,
)


def make_origins(last_timestamp: pd.Timestamp) -> list[pd.Timestamp]:
    """Four daily origins plus the original Test-mimicking origin."""
    origins = [
        last_timestamp - pd.Timedelta(days=days)
        for days in range(4, 0, -1)
    ]
    origins.append(last_timestamp - HORIZON)
    return origins


def make_rolling_window(
    train: pd.DataFrame,
    origins: list[pd.Timestamp],
) -> pd.DataFrame:
    """Keep enough history for the earliest rolling-origin fold."""

    earliest_fit_start = min(
        origin - HORIZON - pd.Timedelta(days=TRAIN_DAYS)
        for origin in origins
    )

    work = train.loc[
        train["timestamp"] >= earliest_fit_start
    ].copy()

    work = work.sort_values(
        ["route_id", "timestamp"]
    ).reset_index(drop=True)

    grouped_target = work.groupby(
        "route_id", sort=False
    )[TARGET_COL]

    for step_idx, col in enumerate(
        FUTURE_TARGET_COLS, start=1
    ):
        work[col] = grouped_target.shift(-step_idx)

    print(f"Rolling supervised window: {work.shape}")
    print(f"Window start: {earliest_fit_start}")
    print(f"Window end:   {work['timestamp'].max()}")

    return work


def validate_one_origin(
    supervised: pd.DataFrame,
    origin: pd.Timestamp,
    expected_routes: int,
) -> tuple[dict, list[dict], np.ndarray, np.ndarray, pd.DataFrame]:

    fit_rows = select_fit_rows(supervised, origin)

    valid_rows = supervised.loc[
        supervised["timestamp"].eq(origin)
        & supervised[FUTURE_TARGET_COLS].notna().all(axis=1)
    ].copy()

    if len(valid_rows) != expected_routes:
        raise ValueError(
            f"Origin {origin}: expected {expected_routes} validation rows, "
            f"got {len(valid_rows)}"
        )

    model = fit_model(fit_rows, f"origin {origin}")

    X_valid = valid_rows[FEATURE_COLS]
    y_true = valid_rows[FUTURE_TARGET_COLS].to_numpy(
        dtype=np.float64
    )

    y_pred = np.clip(model.predict(X_valid), 0.0, None)

    overall = wape_plus_relative_bias(y_true, y_pred)

    fold_summary = {
        "origin": origin,
        "fit_rows": len(fit_rows),
        **overall,
    }

    step_metrics = []

    for step_idx in range(FORECAST_POINTS):
        metrics = wape_plus_relative_bias(
            y_true[:, step_idx],
            y_pred[:, step_idx],
        )

        step_metrics.append({
            "origin": origin,
            "step": step_idx + 1,
            **metrics,
        })

    predictions = []

    for row_idx, route_id in enumerate(valid_rows["route_id"]):
        for step_idx in range(FORECAST_POINTS):
            predictions.append({
                "origin": origin,
                "route_id": route_id,
                "step": step_idx + 1,
                "y_true": y_true[row_idx, step_idx],
                "y_pred": y_pred[row_idx, step_idx],
            })

    return (
        fold_summary,
        step_metrics,
        y_true,
        y_pred,
        pd.DataFrame(predictions),
    )


def main() -> None:
    start = time.perf_counter()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = load_data()

    last_timestamp = train["timestamp"].max()
    expected_routes = train["route_id"].nunique()

    origins = make_origins(last_timestamp)

    print("\nRolling origins:")
    for origin in origins:
        print(f"  {origin}")

    supervised = make_rolling_window(train, origins)

    fold_summaries = []
    all_step_metrics = []
    all_predictions = []

    all_y_true = []
    all_y_pred = []

    for fold_idx, origin in enumerate(origins, start=1):
        print("\n" + "=" * 65)
        print(f"FOLD {fold_idx}/{len(origins)}")
        print(f"Forecast origin: {origin}")
        print(f"Forecast period: {origin + STEP} -> {origin + HORIZON}")
        print("=" * 65)

        (
            fold_summary,
            step_metrics,
            y_true,
            y_pred,
            predictions,
        ) = validate_one_origin(
            supervised,
            origin,
            expected_routes,
        )

        fold_summaries.append(fold_summary)
        all_step_metrics.extend(step_metrics)
        all_predictions.append(predictions)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        print(
            f"\nFold score: {fold_summary['score']:.6f} "
            f"(WAPE={fold_summary['wape']:.6f}, "
            f"Bias={fold_summary['relative_bias']:.6f})"
        )

    summary_df = pd.DataFrame(fold_summaries)
    step_df = pd.DataFrame(all_step_metrics)
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    y_true_all = np.concatenate(all_y_true, axis=0)
    y_pred_all = np.concatenate(all_y_pred, axis=0)

    aggregate = wape_plus_relative_bias(
        y_true_all,
        y_pred_all,
    )

    print("\n" + "=" * 65)
    print("ROLLING-ORIGIN VALIDATION SUMMARY")
    print("=" * 65)

    print(summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    print("\nAggregate metrics across all folds:")
    print(f"WAPE:            {aggregate['wape']:.6f}")
    print(f"Relative Bias:   {aggregate['relative_bias']:.6f}")
    print(f"Competition score: {aggregate['score']:.6f}")

    aggregate_step_rows = []

    for step_idx in range(FORECAST_POINTS):
        metrics = wape_plus_relative_bias(
            y_true_all[:, step_idx],
            y_pred_all[:, step_idx],
        )

        aggregate_step_rows.append({
            "step": step_idx + 1,
            **metrics,
        })

    aggregate_step_df = pd.DataFrame(aggregate_step_rows)

    print("\nAggregate metrics by forecast step:")
    print(aggregate_step_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    ))

    summary_df.to_csv(
        REPORTS_DIR / "ridge_rolling_summary.csv",
        index=False,
    )

    step_df.to_csv(
        REPORTS_DIR / "ridge_rolling_by_fold_and_step.csv",
        index=False,
    )

    aggregate_step_df.to_csv(
        REPORTS_DIR / "ridge_rolling_aggregate_by_step.csv",
        index=False,
    )

    predictions_df.to_csv(
        REPORTS_DIR / "ridge_rolling_predictions.csv",
        index=False,
    )

    print("\nSaved reports:")
    print(REPORTS_DIR / "ridge_rolling_summary.csv")
    print(REPORTS_DIR / "ridge_rolling_by_fold_and_step.csv")
    print(REPORTS_DIR / "ridge_rolling_aggregate_by_step.csv")
    print(REPORTS_DIR / "ridge_rolling_predictions.csv")

    print(f"\nTotal execution time: {time.perf_counter() - start:.1f} seconds")


if __name__ == "__main__":
    main()
