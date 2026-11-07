
"""EXP-005B: Ridge + LightGBM ensemble with rolling validation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    FUTURE_TARGET_COLS,
    FORECAST_POINTS,
    STEP,
    HORIZON,
    load_data,
    wape_plus_relative_bias,
)

from src.experiments.exp_004.train_ridge_target_lags_004a import (
    get_origins,
    build_features,
    select_training_rows,
)

from src.experiments.exp_005.train_lgbm_005a import (
    train_lightgbm,
    train_ridge,
)


EXP_DIR = REPORTS_DIR / "exp_005"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

WEIGHTS = np.round(np.linspace(0.0, 1.0, 21), 2)

# Weight represents the LightGBM contribution.
# 0.0 = Ridge only; 1.0 = LightGBM only.


def blend_predictions(
    ridge_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    weight: float,
) -> np.ndarray:

    return np.clip(
        (1.0 - weight) * ridge_pred
        + weight * lgbm_pred,
        0.0,
        None,
    )


def collect_oof_predictions(
    data: pd.DataFrame,
    origins: list[pd.Timestamp],
    feature_cols: list[str],
    expected_routes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_true = []
    all_ridge = []
    all_lgbm = []

    for fold_idx, origin in enumerate(origins, start=1):

        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx}/{len(origins)} — {origin}")
        print("=" * 60, flush=True)

        # Cache prevents unnecessary retraining
        # if the script is launched again.
        cache_path = (
            EXP_DIR / f"ensemble_005b_fold_{fold_idx}.npz"
        )

        if cache_path.exists():
            with np.load(cache_path) as cached:

                saved_origin = str(cached["origin"].item())

                if saved_origin != str(origin):
                    raise ValueError(
                        f"Cache origin mismatch: {cache_path}"
                    )

                y_true = cached["y_true"].copy()
                ridge_pred = cached["ridge_pred"].copy()
                lgbm_pred = cached["lgbm_pred"].copy()

            if y_true.shape != (expected_routes, FORECAST_POINTS):
                raise ValueError(
                    f"Unexpected cached shape: {cache_path}"
                )

            print("Loaded cached predictions.", flush=True)

        else:
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

            y_true = valid_rows[
                FUTURE_TARGET_COLS
            ].to_numpy(dtype=np.float64)

            print(
                f"Training rows: {len(fit_rows):,}",
                flush=True,
            )

            print("\nTraining Ridge...", flush=True)
            start = time.perf_counter()

            ridge_pred = train_ridge(
                fit_rows,
                valid_rows,
                feature_cols,
            )

            print(
                f"Ridge time: {time.perf_counter() - start:.1f}s",
                flush=True,
            )

            print("\nTraining LightGBM...", flush=True)

            lgbm_pred = train_lightgbm(
                fit_rows,
                valid_rows,
                feature_cols,
            )

            np.savez_compressed(
                cache_path,
                origin=str(origin),
                y_true=y_true,
                ridge_pred=ridge_pred,
                lgbm_pred=lgbm_pred,
            )

            print("Fold predictions cached.", flush=True)

        all_true.append(y_true)
        all_ridge.append(ridge_pred)
        all_lgbm.append(lgbm_pred)

        ridge_metrics = wape_plus_relative_bias(
            y_true,
            ridge_pred,
        )

        lgbm_metrics = wape_plus_relative_bias(
            y_true,
            lgbm_pred,
        )

        print(
            f"Ridge: {ridge_metrics['score']:.6f} | "
            f"LightGBM: {lgbm_metrics['score']:.6f}",
            flush=True,
        )

    return (
        np.stack(all_true),
        np.stack(all_ridge),
        np.stack(all_lgbm),
    )


def optimize_ensemble(
    y_true: np.ndarray,
    ridge_pred: np.ndarray,
    lgbm_pred: np.ndarray,
) -> float:

    # Shape: [folds, routes, forecast_steps].
    # Flatten only the fold/route dimensions for
    # pooled WAPE and Relative Bias calculations.

    results = []

    for weight in WEIGHTS:

        predictions = blend_predictions(
            ridge_pred,
            lgbm_pred,
            float(weight),
        )

        metrics = wape_plus_relative_bias(
            y_true,
            predictions,
        )

        results.append({
            "lgbm_weight": float(weight),
            "ridge_weight": float(1.0 - weight),
            **metrics,
        })

    search = pd.DataFrame(results).sort_values(
        ["score", "wape"]
    )

    search.to_csv(
        EXP_DIR / "ensemble_005b_weight_search.csv",
        index=False,
    )

    best_weight = float(search.iloc[0]["lgbm_weight"])
    best_score = float(search.iloc[0]["score"])

    baseline_metrics = wape_plus_relative_bias(
        y_true,
        lgbm_pred,
    )
    baseline_score = float(baseline_metrics["score"])

    print("\n" + "=" * 60)
    print("ENSEMBLE WEIGHT SEARCH")
    print("=" * 60)

    print(
        search.head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nLightGBM score: {baseline_score:.6f}")
    print(f"Best ensemble score: {best_score:.6f}")
    print(f"Best LightGBM weight: {best_weight:.2f}")

    # --------------------------------------------------
    # Leave-one-fold-out weight validation.
    #
    # Each held-out fold gets a weight selected using
    # only the other four folds.
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("LEAVE-ONE-FOLD-OUT WEIGHT VALIDATION")
    print("=" * 60)

    fold_count = y_true.shape[0]

    lofo_predictions = np.empty_like(lgbm_pred)
    lofo_results = []

    for holdout in range(fold_count):

        train_mask = np.arange(fold_count) != holdout

        candidate_scores = []

        for weight in WEIGHTS:

            train_predictions = blend_predictions(
                ridge_pred[train_mask],
                lgbm_pred[train_mask],
                float(weight),
            )

            metrics = wape_plus_relative_bias(
                y_true[train_mask],
                train_predictions,
            )

            candidate_scores.append(metrics["score"])

        selected_weight = float(
            WEIGHTS[int(np.argmin(candidate_scores))]
        )

        holdout_pred = blend_predictions(
            ridge_pred[holdout],
            lgbm_pred[holdout],
            selected_weight,
        )

        lofo_predictions[holdout] = holdout_pred

        holdout_metrics = wape_plus_relative_bias(
            y_true[holdout],
            holdout_pred,
        )

        lofo_results.append({
            "fold": holdout + 1,
            "lgbm_weight": selected_weight,
            **holdout_metrics,
        })

    lofo_summary = pd.DataFrame(lofo_results)

    lofo_summary.to_csv(
        EXP_DIR / "ensemble_005b_lofo_summary.csv",
        index=False,
    )

    lofo_metrics = wape_plus_relative_bias(
        y_true,
        lofo_predictions,
    )

    lofo_score = float(lofo_metrics["score"])

    print(
        lofo_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nLOFO aggregate score: {lofo_score:.6f}")
    print(f"Pure LightGBM score:  {baseline_score:.6f}")

    # Conservative selection:
    # use blending only if both the pooled search
    # and leave-one-fold-out validation improve
    # on pure LightGBM.
    if (
        best_score < baseline_score
        and lofo_score < baseline_score
    ):
        final_weight = best_weight
        print("\nSELECTED: Ridge + LightGBM ensemble")
    else:
        final_weight = 1.0
        print("\nSELECTED: Pure LightGBM (safer baseline)")

    print(f"Final LightGBM weight: {final_weight:.2f}")
    print(f"Final Ridge weight: {1.0 - final_weight:.2f}")

    final_oof_pred = blend_predictions(
        ridge_pred,
        lgbm_pred,
        final_weight,
    )

    final_metrics = wape_plus_relative_bias(
        y_true,
        final_oof_pred,
    )

    print("\nFinal selected model OOF metrics:")
    print(f"WAPE:  {final_metrics['wape']:.6f}")
    print(f"Bias:  {final_metrics['relative_bias']:.6f}")
    print(f"Score: {final_metrics['score']:.6f}")

    step_results = []

    for step_idx in range(FORECAST_POINTS):

        metrics = wape_plus_relative_bias(
            y_true[:, :, step_idx],
            final_oof_pred[:, :, step_idx],
        )

        step_results.append({
            "step": step_idx + 1,
            **metrics,
        })

    step_report = pd.DataFrame(step_results)

    step_report.to_csv(
        EXP_DIR / "ensemble_005b_by_step.csv",
        index=False,
    )

    print("\nMetrics by forecast step:")
    print(
        step_report.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    return final_weight


def make_submission(
    data: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    origin: pd.Timestamp,
    feature_cols: list[str],
    lgbm_weight: float,
) -> None:

    print("\n" + "=" * 60)
    print("FINAL TRAINING AND SUBMISSION")
    print("=" * 60, flush=True)

    fit_rows = select_training_rows(
        data,
        origin,
    )

    predict_rows = data.loc[
        data["timestamp"].eq(origin)
    ].copy()

    expected_routes = train["route_id"].nunique()

    if len(predict_rows) != expected_routes:
        raise ValueError(
            f"Expected {expected_routes} routes, "
            f"got {len(predict_rows)}"
        )

    predictions = np.zeros(
        (len(predict_rows), FORECAST_POINTS),
        dtype=np.float64,
    )

    if lgbm_weight > 0:
        print("\nFinal LightGBM training...", flush=True)

        predictions += lgbm_weight * train_lightgbm(
            fit_rows,
            predict_rows,
            feature_cols,
        )

    if lgbm_weight < 1:
        print("\nFinal Ridge training...", flush=True)

        predictions += (1.0 - lgbm_weight) * train_ridge(
            fit_rows,
            predict_rows,
            feature_cols,
        )

    predictions = np.clip(predictions, 0.0, None)

    route_values = (
        predict_rows["route_id"]
        .astype(train["route_id"].dtype)
        .to_numpy()
    )

    forecast_parts = []

    for step_idx in range(FORECAST_POINTS):

        forecast_parts.append(pd.DataFrame({
            "route_id": route_values,
            "timestamp": origin + STEP * (step_idx + 1),
            "y_pred": predictions[:, step_idx],
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

    # Strict competition format: id, y_pred.
    submission = submission[["id", "y_pred"]]

    if len(submission) != len(test):
        raise ValueError("Incorrect submission row count")

    if submission["id"].isna().any():
        raise ValueError("Missing IDs")

    if not submission["id"].is_unique:
        raise ValueError("Duplicate IDs")

    if not np.isfinite(
        submission["y_pred"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Invalid or missing predictions")

    if (submission["y_pred"] < 0).any():
        raise ValueError("Negative predictions")

    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        SUBMISSION_DIR
        / "submission_ensemble_005b.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSubmission shape: {submission.shape}")
    print(f"Columns: {submission.columns.tolist()}")
    print(f"Prediction mean: {submission['y_pred'].mean():.3f}")
    print(f"Prediction min: {submission['y_pred'].min():.3f}")
    print(f"Prediction max: {submission['y_pred'].max():.3f}")

    print(f"\nSUBMISSION SAVED: {output_path}")


def main() -> None:

    started = time.perf_counter()

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, test = load_data()

    last_timestamp = train["timestamp"].max()
    expected_routes = train["route_id"].nunique()

    origins = get_origins(last_timestamp)

    print("\nEXP-005B — RIDGE + LIGHTGBM ENSEMBLE")
    print(f"Folds: {len(origins)}")
    print(f"Weights: {WEIGHTS.tolist()}")

    # Exactly the same feature set as EXP-004A/005A.
    data, feature_cols = build_features(
        train,
        origins,
    )

    data["route_id"] = data["route_id"].astype("category")

    y_true, ridge_pred, lgbm_pred = collect_oof_predictions(
        data,
        origins,
        feature_cols,
        expected_routes,
    )

    best_weight = optimize_ensemble(
        y_true,
        ridge_pred,
        lgbm_pred,
    )

    make_submission(
        data,
        train,
        test,
        last_timestamp,
        feature_cols,
        best_weight,
    )

    print(
        f"\nTotal time: "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
