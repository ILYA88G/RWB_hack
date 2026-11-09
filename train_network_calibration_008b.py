
"""EXP-008B: fast network-total calibration of EXP-006A."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_ridge_baseline import (
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET_COL,
    STATUS_COLS,
    STEP,
    HORIZON,
    FORECAST_POINTS,
    load_data,
    wape_plus_relative_bias,
)

from src.experiments.exp_004.train_ridge_target_lags_004a import (
    get_origins,
)


REPORT_DIR = REPORTS_DIR / "exp_008"
BASE_005 = REPORTS_DIR / "exp_005"
BASE_006 = REPORTS_DIR / "exp_006"
SUB_DIR = PROJECT_ROOT / "submissions"

FOLDS = (3, 4, 5)

ALPHAS = (0.15, 0.25, 0.40, 0.60, 0.80, 1.0)
MODES = ("global", "smoothed", "per_step")

MAX_RELATIVE_CHANGE = 0.05

MIN_CV_GAIN = 0.0004
MIN_LOFO_GAIN = 0.0001


def build_network(train: pd.DataFrame):

    source = [TARGET_COL, *STATUS_COLS]

    net = (
        train.groupby("timestamp", sort=True)[source]
        .sum()
        .sort_index()
        / 1_000_000.0
    )

    features = {}

    for col in source:

        features[f"{col}_now"] = (
            net[col].to_numpy(dtype=np.float32)
        )

        if col == TARGET_COL:
            lags = (1, 2, 4, 8, 16, 48, 96, 336)
        else:
            lags = (1, 4, 8, 48)

        for lag in lags:
            features[f"{col}_lag_{lag}"] = (
                net[col]
                .shift(lag)
                .to_numpy(dtype=np.float32)
            )

    for window in (4, 8, 16, 48, 96):

        features[f"target_mean_{window}"] = (
            net[TARGET_COL]
            .rolling(window, min_periods=window)
            .mean()
            .to_numpy(dtype=np.float32)
        )

    for window in (8, 48):

        features[f"target_std_{window}"] = (
            net[TARGET_COL]
            .rolling(window, min_periods=window)
            .std()
            .to_numpy(dtype=np.float32)
        )

    slots = (
        net.index.hour * 2
        + net.index.minute // 30
    )

    dow = net.index.dayofweek

    features["slot"] = np.asarray(
        slots, dtype=np.float32
    )

    features["dow"] = np.asarray(
        dow, dtype=np.float32
    )

    features["slot_sin"] = np.asarray(
        np.sin(2 * np.pi * slots / 48),
        dtype=np.float32,
    )

    features["slot_cos"] = np.asarray(
        np.cos(2 * np.pi * slots / 48),
        dtype=np.float32,
    )

    features["weekend"] = np.asarray(
        dow >= 5,
        dtype=np.float32,
    )

    X = pd.DataFrame(features, index=net.index)

    feature_cols = list(X.columns)

    for step in range(1, FORECAST_POINTS + 1):

        X[f"future_{step}"] = (
            net[TARGET_COL]
            .shift(-step)
            .to_numpy(dtype=np.float64)
        )

    return X, feature_cols


def predict_network(
    data: pd.DataFrame,
    features: list[str],
    origin: pd.Timestamp,
    model_name: str,
) -> np.ndarray:

    train_days = (
        14 if model_name == "ridge14" else 28
    )

    fit_end = origin - HORIZON
    fit_start = (
        fit_end - pd.Timedelta(days=train_days)
    )

    label_cols = [
        f"future_{i}"
        for i in range(1, FORECAST_POINTS + 1)
    ]

    fit = data.loc[
        (data.index >= fit_start)
        & (data.index <= fit_end)
    ].dropna(subset=label_cols)

    if len(fit) < 300 or origin not in data.index:
        raise ValueError(
            f"Not enough network history for {origin}"
        )

    x_fit = fit[features]
    x_pred = data.loc[[origin], features]

    y_fit = fit[label_cols].to_numpy(
        dtype=np.float64
    )

    if model_name.startswith("ridge"):

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=100.0),
        )

        model.fit(x_fit, y_fit)

        result = model.predict(
            x_pred
        ).reshape(-1)

    elif model_name == "lgbm28":

        result = np.empty(
            FORECAST_POINTS,
            dtype=np.float64,
        )

        for j in range(FORECAST_POINTS):

            model = LGBMRegressor(
                objective="regression",
                n_estimators=100,
                learning_rate=0.07,
                num_leaves=7,
                min_child_samples=40,
                reg_lambda=3.0,
                n_jobs=4,
                random_state=42,
                verbosity=-1,
            )

            model.fit(
                x_fit,
                y_fit[:, j],
            )

            result[j] = model.predict(
                x_pred
            )[0]

    else:
        raise ValueError(model_name)

    if (
        not np.isfinite(result).all()
        or (result <= 0).any()
    ):
        raise ValueError(
            "Invalid network forecasts"
        )

    return result


def load_baseline_oof(origins):

    ys = []
    preds = []

    for fold in FOLDS:

        a_path = (
            BASE_005
            / f"ensemble_005b_fold_{fold}.npz"
        )

        b_path = (
            BASE_006
            / f"oof_fold_{fold}.npz"
        )

        if not a_path.exists() or not b_path.exists():
            raise FileNotFoundError(
                f"Need cached predictions: "
                f"{a_path} and {b_path}"
            )

        with np.load(a_path) as a, np.load(b_path) as b:

            if str(a["origin"].item()) != str(
                origins[fold - 1]
            ):
                raise ValueError(
                    f"005B fold {fold} timestamp mismatch"
                )

            if str(b["origin"].item()) != str(
                origins[fold - 1]
            ):
                raise ValueError(
                    f"006A fold {fold} timestamp mismatch"
                )

            y = a["y_true"].astype(np.float64)
            old = a["lgbm_pred"].astype(np.float64)
            new = b["new_pred"].astype(np.float64)

        if (
            y.shape != old.shape
            or y.shape != new.shape
            or y.shape[1] != FORECAST_POINTS
        ):
            raise ValueError(
                f"OOF shape mismatch, fold {fold}"
            )

        ys.append(y)

        preds.append(
            np.clip(
                0.5 * old + 0.5 * new,
                0.0,
                None,
            )
        )

    return np.stack(ys), np.stack(preds)


def corrected(
    base: np.ndarray,
    network: np.ndarray,
    alpha: float,
    mode: str,
):

    baseline_total = (
        base.sum(axis=0) / 1_000_000.0
    )

    if (baseline_total <= 0).any():
        raise ValueError(
            "Invalid base forecast totals"
        )

    ratio = network / baseline_total

    total_ratio = (
        network.sum()
        / baseline_total.sum()
    )

    if mode == "global":

        ratio = np.full(
            FORECAST_POINTS,
            total_ratio,
        )

    elif mode == "smoothed":

        ratio = (
            0.5 * ratio
            + 0.5 * total_ratio
        )

    elif mode != "per_step":
        raise ValueError(mode)

    factors = np.clip(
        1.0 + alpha * (ratio - 1.0),
        1.0 - MAX_RELATIVE_CHANGE,
        1.0 + MAX_RELATIVE_CHANGE,
    )

    return (
        np.clip(
            base * factors[None, :],
            0.0,
            None,
        ),
        factors,
    )


def score(y, p) -> float:

    return float(
        wape_plus_relative_bias(
            y, p
        )["score"]
    )


def select_candidate(
    y,
    base,
    net_predictions,
):

    base_score = score(y, base)

    options = []

    for name in net_predictions:

        for mode in MODES:

            for alpha in ALPHAS:

                pred = np.stack([
                    corrected(
                        base[j],
                        net_predictions[name][j],
                        alpha,
                        mode,
                    )[0]
                    for j in range(len(FOLDS))
                ])

                fold_scores = [
                    score(y[j], pred[j])
                    for j in range(len(FOLDS))
                ]

                base_folds = [
                    score(y[j], base[j])
                    for j in range(len(FOLDS))
                ]

                options.append({
                    "model": name,
                    "mode": mode,
                    "alpha": alpha,
                    "score": score(y, pred),
                    "gain": (
                        base_score - score(y, pred)
                    ),
                    "improved_folds": sum(
                        a < b
                        for a, b in zip(
                            fold_scores,
                            base_folds,
                        )
                    ),
                    "last_fold_gain": (
                        base_folds[-1]
                        - fold_scores[-1]
                    ),
                })

    table = (
        pd.DataFrame(options)
        .sort_values("score")
        .reset_index(drop=True)
    )

    table.to_csv(
        REPORT_DIR / "network_008b_search.csv",
        index=False,
    )

    print(
        "\nNETWORK CORRECTION SEARCH "
        "(lower Score is better):"
    )

    print(
        table.head(12).to_string(
            index=False,
            float_format=lambda n: f"{n:.6f}",
        )
    )

    print(
        f"\nEXP-006A baseline score: "
        f"{base_score:.6f}"
    )

    # Nested leave-one-fold-out:
    # Select parameters on two folds,
    # evaluate on the third.

    held_predictions = np.empty_like(base)

    rows = []

    for held in range(len(FOLDS)):

        training = [
            i for i in range(len(FOLDS))
            if i != held
        ]

        ranked = []

        for entry in options:

            name = entry["model"]
            mode = entry["mode"]
            alpha = entry["alpha"]

            preds = np.stack([
                corrected(
                    base[j],
                    net_predictions[name][j],
                    alpha,
                    mode,
                )[0]
                for j in training
            ])

            ranked.append(
                score(y[training], preds)
            )

        jbest = int(np.argmin(ranked))

        winner = options[jbest]

        pred, _ = corrected(
            base[held],
            net_predictions[winner["model"]][held],
            winner["alpha"],
            winner["mode"],
        )

        held_predictions[held] = pred

        rows.append({
            "fold": FOLDS[held],
            "model": winner["model"],
            "mode": winner["mode"],
            "alpha": winner["alpha"],
            "baseline": score(
                y[held], base[held]
            ),
            "lofo": score(
                y[held], pred
            ),
        })

    lofo = pd.DataFrame(rows)

    lofo.to_csv(
        REPORT_DIR / "network_008b_lofo.csv",
        index=False,
    )

    lofo_gain = (
        base_score
        - score(y, held_predictions)
    )

    print("\nNESTED LOFO:")

    print(
        lofo.to_string(
            index=False,
            float_format=lambda n: f"{n:.6f}",
        )
    )

    print(
        f"LOFO gain: {lofo_gain:+.6f}"
    )

    eligible = table.loc[
        (table.gain >= MIN_CV_GAIN)
        & (table.improved_folds >= 2)
        & (table.last_fold_gain >= -0.0005)
    ]

    if (
        eligible.empty
        or lofo_gain < MIN_LOFO_GAIN
    ):

        print(
            "\nNO ROBUST IMPROVEMENT: "
            "keep EXP-006A, "
            "do NOT upload network correction."
        )

        return None

    winner = eligible.iloc[0]

    print(
        "\nWINNER:",
        winner[
            ["model", "mode", "alpha", "score", "gain"]
        ].to_dict(),
    )

    return winner


def save_submission(
    test: pd.DataFrame,
    network: np.ndarray,
    winner,
):

    baseline_path = (
        SUB_DIR / "submission_lgbm_006a.csv"
    )

    df = pd.read_csv(
        baseline_path,
        dtype={"id": str},
    )

    if (
        df.columns.tolist()
        != ["id", "y_pred"]
        or not df["id"].is_unique
    ):
        raise ValueError(
            "Incorrect EXP-006A submission format"
        )

    lookup = df.set_index("id")["y_pred"]

    forecast = lookup.reindex(
        test["id"].astype(str)
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(forecast).all():
        raise ValueError(
            "Test IDs do not match "
            "EXP-006A submission"
        )

    first = test["timestamp"].min()

    steps = (
        (test["timestamp"] - first) / STEP
    ).to_numpy(dtype=float)

    if not np.allclose(
        steps,
        np.rint(steps),
    ):
        raise ValueError(
            "Timestamps do not match 30 minute step"
        )

    step_idx = np.rint(
        steps
    ).astype(int)

    if (
        step_idx.min() != 0
        or step_idx.max() != FORECAST_POINTS - 1
    ):
        raise ValueError(
            "Unexpected test forecast steps"
        )

    baseline_totals = np.array([
        forecast[
            step_idx == i
        ].sum() / 1_000_000.0

        for i in range(FORECAST_POINTS)
    ])

    ratio = network / baseline_totals

    global_ratio = (
        network.sum()
        / baseline_totals.sum()
    )

    mode = str(winner["mode"])

    if mode == "global":

        ratio = np.full(
            FORECAST_POINTS,
            global_ratio,
        )

    elif mode == "smoothed":

        ratio = (
            0.5 * ratio
            + 0.5 * global_ratio
        )

    alpha = float(winner["alpha"])

    factors = np.clip(
        1 + alpha * (ratio - 1),
        1 - MAX_RELATIVE_CHANGE,
        1 + MAX_RELATIVE_CHANGE,
    )

    result = pd.DataFrame({
        "id": test["id"],
        "y_pred": np.clip(
            forecast * factors[step_idx],
            0,
            None,
        ),
    })

    if (
        len(result) != len(test)
        or not result["id"].is_unique
        or not np.isfinite(result.y_pred).all()
    ):
        raise ValueError(
            "Invalid submission"
        )

    path = (
        SUB_DIR / "submission_network_008b.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print(
        "\nFINAL NETWORK ESTIMATES (millions):",
        np.round(network, 3),
    )

    print(
        "FINAL CORRECTION FACTORS:",
        np.round(factors, 5),
    )

    print(
        "SUBMISSION SAVED:",
        path,
    )

    print(
        "Shape:",
        result.shape,
        "Columns:",
        result.columns.tolist(),
    )


def main():

    started = time.perf_counter()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "EXP-008B: network-total "
        "calibration of EXP-006A",
        flush=True,
    )

    train, test = load_data()

    origins = get_origins(
        train["timestamp"].max()
    )

    data, features = build_network(train)

    print(
        f"Network training points: {len(data):,}; "
        f"features: {len(features)}",
        flush=True,
    )

    y, base = load_baseline_oof(origins)

    if (
        y.shape[1]
        != train["route_id"].nunique()
    ):
        raise ValueError(
            "OOF route count differs from source data"
        )

    net_predictions = {
        name: []
        for name in (
            "ridge14",
            "ridge28",
            "lgbm28",
        )
    }

    for i, fold in enumerate(FOLDS):

        origin = origins[fold - 1]

        true_total = (
            y[i].sum(axis=0) / 1_000_000.0
        )

        frame_total = data.loc[
            origin,
            [
                f"future_{k}"
                for k in range(1, 9)
            ],
        ].to_numpy(dtype=float)

        if not np.allclose(
            true_total,
            frame_total,
            atol=0.0001,
        ):
            raise ValueError(
                f"Validation aggregate target "
                f"mismatch on fold {fold}"
            )

        print(
            f"\nFold {fold}: {origin}",
            flush=True,
        )

        for name in net_predictions:

            p = predict_network(
                data,
                features,
                origin,
                name,
            )

            net_predictions[name].append(p)

            print(
                f"  {name}: network WAPE="
                f"{np.abs(p - true_total).sum() / true_total.sum():.4f}",
                flush=True,
            )

    net_predictions = {
        k: np.stack(v)
        for k, v in net_predictions.items()
    }

    winner = select_candidate(
        y,
        base,
        net_predictions,
    )

    if winner is None:

        print(
            f"Total time: "
            f"{time.perf_counter() - started:.1f}s"
        )

        return

    latest = train["timestamp"].max()

    forecast = predict_network(
        data,
        features,
        latest,
        str(winner["model"]),
    )

    save_submission(
        test,
        forecast,
        winner,
    )

    print(
        f"Total time: "
        f"{time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
