
"""EXP-005A: LightGBM rolling validation and automatic submission."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    FUTURE_TARGET_COLS,
    FORECAST_POINTS,
    STEP,
    load_data,
    wape_plus_relative_bias,
)

from src.experiments.exp_004.train_ridge_target_lags_004a import (
    get_origins,
    build_features,
    select_training_rows,
    make_model as make_ridge,
)


EXP_DIR = REPORTS_DIR / "exp_005"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

RIDGE_BEST_SCORE = 0.340798

LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 220,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "max_bin": 127,
    "colsample_bytree": 0.9,
    "subsample": 0.9,
    "subsample_freq": 1,
    "reg_lambda": 5.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def train_lightgbm(
    fit_rows: pd.DataFrame,
    predict_rows: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:

    X_train = fit_rows[feature_cols]
    X_predict = predict_rows[feature_cols]

    predictions = np.zeros(
        (len(predict_rows), FORECAST_POINTS),
        dtype=np.float64,
    )

    for step_idx, target_col in enumerate(FUTURE_TARGET_COLS):

        started = time.perf_counter()

        model = LGBMRegressor(**LGBM_PARAMS)

        model.fit(
            X_train,
            fit_rows[target_col].to_numpy(dtype=np.float32),
            categorical_feature=["route_id"],
        )

        predictions[:, step_idx] = model.predict(X_predict)

        print(
            f"  Step {step_idx + 1}/{FORECAST_POINTS}: "
            f"{time.perf_counter() - started:.1f}s",
            flush=True,
        )

    return np.clip(predictions, 0.0, None)


def train_ridge(
    fit_rows: pd.DataFrame,
    predict_rows: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:

    model = make_ridge(feature_cols)

    model.fit(
        fit_rows[feature_cols],
        fit_rows[FUTURE_TARGET_COLS].to_numpy(dtype=np.float64),
    )

    return np.clip(
        model.predict(predict_rows[feature_cols]),
        0.0,
        None,
    )


def validate(
    data: pd.DataFrame,
    origins: list[pd.Timestamp],
    feature_cols: list[str],
    expected_routes: int,
) -> float:

    fold_results = []
    step_results = []

    all_true = []
    all_pred = []

    for fold_idx, origin in enumerate(origins, start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx}/{len(origins)} — {origin}")
        print("=" * 60, flush=True)

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

        print(f"Training rows: {len(fit_rows):,}", flush=True)

        y_true = valid_rows[
            FUTURE_TARGET_COLS
        ].to_numpy(dtype=np.float64)

        y_pred = train_lightgbm(
            fit_rows,
            valid_rows,
            feature_cols,
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
            step_metrics = wape_plus_relative_bias(
                y_true[:, step_idx],
                y_pred[:, step_idx],
            )

            step_results.append({
                "origin": origin,
                "step": step_idx + 1,
                **step_metrics,
            })

        print(
            f"Score: {metrics['score']:.6f} | "
            f"WAPE: {metrics['wape']:.6f} | "
            f"Bias: {metrics['relative_bias']:.6f}",
            flush=True,
        )

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

    summary.to_csv(
        EXP_DIR / "lgbm_005a_summary.csv",
        index=False,
    )

    by_step.to_csv(
        EXP_DIR / "lgbm_005a_by_step.csv",
        index=False,
    )

    aggregate_steps.to_csv(
        EXP_DIR / "lgbm_005a_aggregate_by_step.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("EXP-005A — ROLLING VALIDATION SUMMARY")
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

    return float(aggregate["score"])


def make_submission(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    origin: pd.Timestamp,
    feature_cols: list[str],
    use_lightgbm: bool,
) -> None:

    print("\n" + "=" * 60)
    print("FINAL MODEL — SUBMISSION")
    print("=" * 60)

    fit_rows = select_training_rows(data, origin)

    origin_rows = data.loc[
        data["timestamp"].eq(origin)
    ].copy()

    expected_routes = train["route_id"].nunique()

    if len(origin_rows) != expected_routes:
        raise ValueError(
            f"Expected {expected_routes} origin rows, "
            f"got {len(origin_rows)}"
        )

    print(f"Training rows: {len(fit_rows):,}")

    if use_lightgbm:
        print("Selected model: LightGBM")

        predictions = train_lightgbm(
            fit_rows,
            origin_rows,
            feature_cols,
        )

        model_name = "lightgbm_005a"

    else:
        print("Selected model: Ridge EXP-004A")

        predictions = train_ridge(
            fit_rows,
            origin_rows,
            feature_cols,
        )

        model_name = "ridge_004a"

    forecast_parts = []

    route_values = origin_rows["route_id"].astype(
        train["route_id"].dtype
    ).to_numpy()

    for step_idx in range(FORECAST_POINTS):

        forecast_parts.append(pd.DataFrame({
            "route_id": route_values,
            "timestamp": origin + STEP * (step_idx + 1),
            TARGET_COL: predictions[:, step_idx],
        }))

    forecasts = pd.concat(
        forecast_parts,
        ignore_index=True,
    )

    submission = test[
        ["id", "route_id", "timestamp"]
    ].merge(
        forecasts,
        on=["route_id", "timestamp"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(submission) != len(test):
        raise ValueError("Incorrect submission row count")

    if submission[TARGET_COL].isna().any():
        raise ValueError("Missing predictions in submission")

    if not np.isfinite(
        submission[TARGET_COL].to_numpy()
    ).all():
        raise ValueError("Non-finite predictions in submission")

    submission = submission[["id", TARGET_COL]]

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        SUBMISSION_DIR
        / f"submission_{model_name}.csv"
    )

    submission.to_csv(output_path, index=False)

    print(f"Submission shape: {submission.shape}")
    print(f"Prediction min: {submission[TARGET_COL].min():.3f}")
    print(f"Prediction max: {submission[TARGET_COL].max():.3f}")
    print(f"Prediction mean: {submission[TARGET_COL].mean():.3f}")
    print(f"\nSUBMISSION SAVED: {output_path}")


def main() -> None:

    started = time.perf_counter()

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_data()

    last_timestamp = train["timestamp"].max()
    expected_routes = train["route_id"].nunique()

    origins = get_origins(last_timestamp)

    print("\nEXP-005A — LIGHTGBM")
    print(f"Number of folds: {len(origins)}")
    print(f"Forecast steps: {FORECAST_POINTS}")
    print(f"Estimators per model: {LGBM_PARAMS['n_estimators']}")

    # Reuse EXACTLY the EXP-004A features.
    data, feature_cols = build_features(train, origins)

    # Native categorical handling in LightGBM.
    data["route_id"] = data["route_id"].astype("category")

    print(f"Total features: {len(feature_cols)}")

    # 1. Five-fold rolling validation.
    score = validate(
        data,
        origins,
        feature_cols,
        expected_routes,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"Ridge EXP-004A: {RIDGE_BEST_SCORE:.6f}")
    print(f"LightGBM 005A:  {score:.6f}")

    use_lightgbm = score < RIDGE_BEST_SCORE

    if use_lightgbm:
        print("WINNER: LightGBM")
    else:
        print("WINNER: Ridge EXP-004A")

    # 2. Train selected model on the latest window.
    # 3. Predict all 8 future points for each route.
    # 4. Save submission with columns id, target_1h.
    make_submission(
        data,
        train,
        test,
        last_timestamp,
        feature_cols,
        use_lightgbm,
    )

    print(
        f"\nTotal time: "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
