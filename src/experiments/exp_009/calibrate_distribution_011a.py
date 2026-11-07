"""
EXP-009: прямой прогноз отгрузок на восемь получасовых шагов.

Читает только явно указанные --train и --test.
Public Score определяется после загрузки submission на платформу.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .calibration import apply_config, evaluate_forward, metric


TARGET = "target_1h"
VALUES = [TARGET] + [f"status_{i}" for i in range(1, 7)]

STEP = pd.Timedelta(minutes=30)
H = 8
SCALE = 1000.0
WARMUP = 336


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    return pd.read_csv(
        path,
        dtype={"route_id": str, "id": str},
    )


def validate_data(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()

    requirements = [
        (train, ["route_id", "timestamp", *VALUES]),
        (test, ["id", "route_id", "timestamp"]),
    ]

    for frame, required in requirements:
        missing = set(required) - set(frame)

        if missing:
            raise ValueError(f"Отсутствуют колонки: {sorted(missing)}")

        if frame[required].isna().any().any():
            raise ValueError("Есть пропуски в обязательных колонках")

        frame["route_id"] = frame["route_id"].astype(str)
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            errors="raise",
        )

        if frame.duplicated(["route_id", "timestamp"]).any():
            raise ValueError("Повторяющиеся пары route_id/timestamp")

    if test["id"].astype(str).duplicated().any():
        raise ValueError("Повторяющиеся ID в test")

    if set(train.route_id) != set(test.route_id):
        raise ValueError("Наборы маршрутов в train и test различаются")

    numbers = train[VALUES].to_numpy(dtype=np.float64)

    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError(
            "Статусы и отгрузки должны быть конечными и неотрицательными"
        )

    latest = train.groupby("route_id").timestamp.max()

    if latest.nunique() != 1:
        raise ValueError(
            "Последняя известная дата должна совпадать у всех маршрутов"
        )

    ordered = test.sort_values(["route_id", "timestamp"])

    if not ordered.groupby("route_id").size().eq(H).all():
        raise ValueError("В test должно быть восемь строк на маршрут")

    expected = (
        ordered.route_id.map(latest)
        + STEP * (ordered.groupby("route_id").cumcount() + 1)
    )

    if not ordered.timestamp.eq(expected).all():
        raise ValueError(
            "Test должен содержать следующие восемь получасовых шагов"
        )

    return train, test


class History:
    """Массивы с порядком осей: время, маршрут, показатель."""

    def __init__(self, train: pd.DataFrame):
        self.routes = np.array(sorted(train.route_id.unique()))
        self.times = pd.DatetimeIndex(
            sorted(train.timestamp.unique())
        ).as_unit("ns")

        self.n_routes = len(self.routes)

        if len(self.times) <= WARMUP + H:
            raise ValueError("Нужно больше семи дней непрерывной истории")

        if not np.all(np.diff(self.times.asi8) == STEP.value):
            raise ValueError("В train отсутствуют получасовые отметки")

        expected_rows = len(self.times) * self.n_routes

        if len(train) != expected_rows:
            raise ValueError(
                "Неполная сетка маршрут/время. "
                "Пропуски необходимо обработать явно до обучения."
            )

        grid = train.sort_values(["timestamp", "route_id"])

        self.v = (
            grid[VALUES]
            .to_numpy(dtype=np.float32)
            .reshape(len(self.times), self.n_routes, len(VALUES))
            / SCALE
        )

        self.means = {}
        self.stds = {}

        cs = np.concatenate([
            np.zeros_like(self.v[:1], dtype=np.float64),
            np.cumsum(self.v, axis=0, dtype=np.float64),
        ])

        for w in (4, 8, 16, 48, 96, 336):
            out = np.full(
                self.v.shape,
                np.nan,
                dtype=np.float32,
            )
            out[w - 1:] = (cs[w:] - cs[:-w]) / w

            self.means[w] = (
                out
                if w in (4, 8)
                else out[:, :, :1].copy()
            )

        y = self.v[:, :, 0]

        square_cs = np.concatenate([
            np.zeros_like(y[:1], dtype=np.float64),
            np.cumsum(
                y.astype(np.float64) ** 2,
                axis=0,
            ),
        ])

        for w in (8, 48):
            out = np.full(
                y.shape,
                np.nan,
                dtype=np.float32,
            )

            variance = (
                (square_cs[w:] - square_cs[:-w]) / w
                - self.means[w][w - 1:, :, 0] ** 2
            )

            out[w - 1:] = np.sqrt(np.maximum(variance, 0))
            self.stds[w] = out

        self.net = self.v.sum(
            axis=1,
            dtype=np.float64,
        ).astype(np.float32)

        self.net_mean = {}

        nc = np.vstack([
            np.zeros((1, len(VALUES))),
            np.cumsum(self.net, axis=0, dtype=np.float64),
        ])

        for w in (8, 48):
            out = np.full(
                self.net.shape,
                np.nan,
                dtype=np.float32,
            )
            out[w - 1:] = (nc[w:] - nc[:-w]) / w
            self.net_mean[w] = out

    def features(
        self,
        t: np.ndarray,
        r: np.ndarray,
    ) -> pd.DataFrame:
        if np.min(t) < WARMUP:
            raise ValueError("Недостаточно истории для лагов")

        f = {"route_id": r.astype(np.int32)}

        for j, col in enumerate(VALUES):
            f[col] = self.v[t, r, j]

            lags = (
                (1, 2, 4, 8, 16, 48, 96, 336)
                if j == 0
                else (1, 4, 8)
            )

            for lag in lags:
                f[f"{col}_lag{lag}"] = self.v[t - lag, r, j]

            windows = (
                (4, 8, 16, 48, 96, 336)
                if j == 0
                else (4, 8)
            )

            for w in windows:
                column = j if w in (4, 8) else 0
                f[f"{col}_mean{w}"] = self.means[w][t, r, column]

        for w in (8, 48):
            f[f"target_std{w}"] = self.stds[w][t, r]

        f["target_trend4"] = (
            self.v[t, r, 0] - self.v[t - 4, r, 0]
        )

        f["target_trend_mean"] = (
            self.means[4][t, r, 0]
            - self.means[48][t, r, 0]
        )

        denominator = np.maximum(
            self.means[48][t, r, 0],
            0.001,
        )

        for j, col in enumerate(VALUES):
            f[f"{col}_relative"] = (
                self.v[t, r, j] / denominator
            )
            f[f"network_{col}"] = self.net[t, j]

        for lag in (1, 4, 8, 48):
            f[f"network_target_lag{lag}"] = self.net[t - lag, 0]

        for w in (8, 48):
            f[f"network_target_mean{w}"] = self.net_mean[w][t, 0]

        f["route_share"] = (
            self.v[t, r, 0]
            / np.maximum(self.net[t, 0], 0.001)
        )

        f["elapsed_days"] = t.astype(np.float32) / 48

        ts = self.times[t]
        slot = ts.hour.to_numpy() * 2 + ts.minute.to_numpy() / 30

        f["slot_sin"] = np.sin(
            slot * 2 * np.pi / 48
        ).astype(np.float32)

        f["slot_cos"] = np.cos(
            slot * 2 * np.pi / 48
        ).astype(np.float32)

        f["dow"] = ts.dayofweek.to_numpy(dtype=np.float32)

        return pd.DataFrame(f)

    def add_future_features(self, x, t, r, h):
        # Календарь будущей точки известен заранее.
        future = self.times[t] + STEP * h
        slot = (
            future.hour.to_numpy() * 2
            + future.minute.to_numpy() / 30
        )

        x["future_slot_sin"] = np.sin(
            slot * 2 * np.pi / 48
        ).astype(np.float32)

        x["future_slot_cos"] = np.cos(
            slot * 2 * np.pi / 48
        ).astype(np.float32)

        x["future_dow"] = future.dayofweek.to_numpy(
            dtype=np.float32
        )

        x["future_weekend"] = (
            future.dayofweek.to_numpy() >= 5
        ).astype(np.float32)

        # При h <= 8 обе ссылки находятся раньше момента t.
        x["future_yesterday"] = self.v[t + h - 48, r, 0]
        x["future_lastweek"] = self.v[t + h - 336, r, 0]


RIDGE_COLS = (
    [TARGET]
    + [f"{TARGET}_lag{k}" for k in (1, 2, 4, 8, 48, 96)]
    + [f"{TARGET}_mean{w}" for w in (4, 8, 48)]
    + [f"status_{i}" for i in range(1, 7)]
    + [f"status_{i}_mean4" for i in range(1, 7)]
    + ["slot_sin", "slot_cos"]
)


def network_prediction(history, origin, args, save_dir=None):
    """Независимая модель суммарного объёма всей сети."""
    start = max(
        WARMUP,
        origin - H - args.train_days * 48,
    )
    times = np.arange(start, origin - H + 1)

    def features(t):
        parts = [history.net[t]]

        for lag in (1, 4, 8, 48, 96, 336):
            parts.append(history.net[t - lag])

        for w in (8, 48):
            parts.append(history.net_mean[w][t])

        ts = history.times[t]
        slot = (
            ts.hour.to_numpy() * 2
            + ts.minute.to_numpy() / 30
        )

        parts.append(np.column_stack([
            np.sin(slot * 2 * np.pi / 48),
            np.cos(slot * 2 * np.pi / 48),
            ts.dayofweek.to_numpy(),
        ]))

        return np.column_stack(parts)

    x = features(times)
    xp = features(np.array([origin]))

    y = np.column_stack([
        history.net[times + h, 0]
        for h in range(1, H + 1)
    ])

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0),
    )

    model.fit(x, y)

    result = np.maximum(
        model.predict(xp).ravel() * SCALE,
        0,
    )

    if save_dir is not None:
        joblib.dump(
            model,
            save_dir / "network_ridge.joblib",
        )

    return result


def predict_origin(
    history: History,
    origin: int,
    args,
    save_dir: Path | None = None,
):
    n = history.n_routes

    start = max(
        WARMUP,
        origin - H - args.train_days * 48,
    )
    end = origin - H

    available = np.arange(start, end + 1)

    if len(available) < args.min_fit_days * 48:
        raise ValueError(
            f"Для {history.times[origin]} доступно "
            f"{len(available) / 48:.1f} дней обучения; "
            f"нужно {args.min_fit_days}. "
            "Нужна более длинная история или меньше проверочных дат."
        )

    if args.max_rows < n * 10:
        raise ValueError(
            "--max-rows должен позволять хотя бы десять дат на маршрут"
        )

    # Одинаковые обучающие даты для всех маршрутов.
    rng = np.random.default_rng(args.seed)
    keep = min(len(available), args.max_rows // n)

    chosen = np.sort(
        rng.choice(available, keep, replace=False)
    )

    t = np.repeat(chosen, n)
    r = np.tile(np.arange(n), keep)

    tp = np.full(n, origin)
    rp = np.arange(n)

    x = history.features(t, r)
    xp = history.features(tp, rp)

    y = np.column_stack([
        history.v[t + h, r, 0]
        for h in range(1, H + 1)
    ])

    preds = {
        "persistence": np.repeat(
            history.v[origin, :, 0, None],
            H,
            axis=1,
        ) * SCALE
    }

    # Отдельная линейная модель каждого маршрута.
    xr = (
        x[RIDGE_COLS]
        .to_numpy(dtype=np.float64)
        .reshape(keep, n, -1)
    )

    xpr = xp[RIDGE_COLS].to_numpy(dtype=np.float64)
    yr = y.reshape(keep, n, H)

    ridge_preds = np.empty((n, H))
    ridge_states = []

    for route in range(n):
        a = xr[:, route]
        mean = a.mean(axis=0)
        std = np.maximum(a.std(axis=0), 1e-4)

        model = Ridge(alpha=args.ridge_alpha)
        model.fit((a - mean) / std, yr[:, route])

        row = ((xpr[route] - mean) / std).reshape(1, -1)
        ridge_preds[route] = model.predict(row)[0]

        if save_dir is not None:
            ridge_states.append((mean, std, model))

    preds["route_ridge"] = np.maximum(
        ridge_preds * SCALE,
        0,
    )

    # Восемь отдельных моделей будущих шагов.
    boosted = np.empty((n, H))

    for h in range(1, H + 1):
        history.add_future_features(x, t, r, h)
        history.add_future_features(xp, tp, rp, h)

        model = LGBMRegressor(
            objective="regression",
            n_estimators=args.trees,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,
            reg_lambda=10.0,
            reg_alpha=0.1,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            n_jobs=args.threads,
            random_state=args.seed + h,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )

        # Все обучающие ответы известны не позднее origin.
        model.fit(
            x,
            np.ascontiguousarray(y[:, h - 1]),
            categorical_feature=["route_id"],
        )

        boosted[:, h - 1] = model.predict(xp) * SCALE

        if save_dir is not None:
            model.booster_.save_model(
                str(save_dir / f"lgbm_step_{h}.txt")
            )

            if h == 1:
                importance = pd.DataFrame({
                    "feature": x.columns,
                    "gain": model.booster_.feature_importance("gain"),
                })

                importance.sort_values(
                    "gain",
                    ascending=False,
                ).to_csv(
                    save_dir / "feature_importance.csv",
                    index=False,
                )

    preds["lgbm"] = np.maximum(boosted, 0)

    # Дополнительный кандидат с коррекцией общего объёма.
    mix = (
        0.5 * preds["lgbm"]
        + 0.5 * preds["route_ridge"]
    )

    totals = mix.sum(axis=0)
    net = network_prediction(history, origin, args, save_dir)

    if (totals > 0).all() and net.sum() > 0:
        ratio = (
            0.5 * (net / totals)
            + 0.5 * (net.sum() / totals.sum())
        )

        factors = np.clip(
            1 + 0.5 * (ratio - 1),
            0.90,
            1.10,
        )

        preds["reconciled"] = mix * factors[None, :]
    else:
        preds["reconciled"] = mix

    if save_dir is not None:
        joblib.dump(
            {
                "routes": history.routes,
                "columns": RIDGE_COLS,
                "states": ridge_states,
                "value_scale": SCALE,
            },
            save_dir / "route_ridge.joblib",
        )

    return preds


def make_submission(test, history, prediction):
    if prediction.shape != (history.n_routes, H):
        raise ValueError("Неправильная форма прогнозов")

    if (
        not np.isfinite(prediction).all()
        or (prediction < 0).any()
    ):
        raise ValueError("Некорректные значения прогнозов")

    route_index = pd.Index(
        history.routes
    ).get_indexer(test.route_id)

    h = (
        (test.timestamp - history.times[-1]) / STEP
    ).to_numpy(dtype=float)

    if (
        (route_index < 0).any()
        or not np.allclose(h, np.rint(h))
        or np.min(h) != 1
        or np.max(h) != H
    ):
        raise ValueError("Ошибка соответствия прогнозов строкам test")

    return pd.DataFrame({
        "id": test.id.to_numpy(),
        "y_pred": prediction[
            route_index,
            np.rint(h).astype(int) - 1,
        ],
    })


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    project_root = Path(__file__).resolve().parents[3]

    p.add_argument(
        "--train",
        type=Path,
        default=project_root / "data" / "raw" / "train_solo_track.parquet",
    )

    p.add_argument(
        "--test",
        type=Path,
        default=project_root / "data" / "raw" / "test_solo_track.parquet",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("results_exp009"),
    )

    p.add_argument("--train-days", type=int, default=21)
    p.add_argument("--min-fit-days", type=int, default=7)
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--fold-spacing-days", type=int, default=1)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--max-rows", type=int, default=250_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--ridge-alpha", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--reference-submission",
        type=Path,
        help="Старый CSV: только сравнение ID и изменений прогнозов",
    )

    args = p.parse_args()

    if (
        args.folds < 3
        or min(
            args.train_days,
            args.min_fit_days,
            args.fold_spacing_days,
            args.trees,
            args.threads,
        ) < 1
        or args.min_fit_days > args.train_days
    ):
        p.error(
            "Нужны folds>=3, положительные настройки "
            "и min-fit-days<=train-days"
        )

    if args.ridge_alpha <= 0:
        p.error("ridge-alpha должен быть положительным")

    return args


def main():
    args = parse_args()
    started = time.perf_counter()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Загрузка явно указанных train и test", flush=True)

    train, test = validate_data(
        read_table(args.train),
        read_table(args.test),
    )

    history_days = (
        args.train_days
        + args.folds * args.fold_spacing_days
        + 8
    )

    cutoff = (
        train.timestamp.max()
        - pd.Timedelta(days=history_days)
    )

    train = train.loc[train.timestamp >= cutoff].copy()
    history = History(train)

    del train
    gc.collect()

    latest = len(history.times) - 1

    origins = [
        latest - d * 48
        for d in range(
            args.folds * args.fold_spacing_days,
            0,
            -args.fold_spacing_days,
        )
    ]

    if (
        origins[0]
        < WARMUP + H + args.min_fit_days * 48 - 1
    ):
        raise ValueError(
            "Недостаточно истории для заданных фолдов, "
            "недельных лагов и минимального периода обучения"
        )

    ys = []
    cache = {
        name: []
        for name in (
            "persistence",
            "route_ridge",
            "lgbm",
            "reconciled",
        )
    }

    for i, origin in enumerate(origins):
        print(
            f"Fold {i + 1}/{len(origins)}: "
            f"{history.times[origin]}",
            flush=True,
        )

        pred = predict_origin(history, origin, args)

        truth = np.stack([
            history.v[origin + h, :, 0]
            for h in range(1, H + 1)
        ], axis=1) * SCALE

        ys.append(truth)

        for name in cache:
            cache[name].append(pred[name])

            print(
                f"  {name}: {metric(truth, pred[name])}",
                flush=True,
            )

        np.savez_compressed(
            args.output / f"oof_{i + 1:02d}.npz",
            y_true=truth,
            route_id=history.routes,
            origin=str(history.times[origin]),
            **pred,
        )

    y = np.stack(ys)

    predictions = {
        name: np.stack(parts)
        for name, parts in cache.items()
    }

    config, report = evaluate_forward(
        y,
        predictions,
        min_history=2,
    )

    report["origins"] = [
        str(history.times[o])
        for o in origins
    ]

    report["public_score"] = None
    report["public_score_note"] = (
        "Это историческая валидация. "
        "Public Score определяется на платформе соревнования."
    )

    report["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    print(
        "Конфигурация, выбранная без последнего holdout:",
        config,
        flush=True,
    )

    print(
        "Последний независимый holdout:",
        report["locked_holdout"],
        flush=True,
    )

    (args.output / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (args.output / "ensemble.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    final_dir = args.output / "models"
    final_dir.mkdir(exist_ok=True)

    final = predict_origin(
        history,
        latest,
        args,
        save_dir=final_dir,
    )

    final_pred = apply_config(final, config)
    submission = make_submission(test, history, final_pred)

    submission.to_csv(
        args.output / "submission_exp009.csv",
        index=False,
    )

    for name, pred in final.items():
        make_submission(
            test,
            history,
            pred,
        ).to_csv(
            args.output / f"submission_{name}.csv",
            index=False,
        )

    if args.reference_submission:
        old = pd.read_csv(
            args.reference_submission,
            dtype={"id": str},
        ).set_index("id")

        if (
            not old.index.is_unique
            or set(old.index) != set(submission.id.astype(str))
        ):
            raise ValueError("ID старого submission не совпадают")

        a = old.loc[
            submission.id.astype(str),
            "y_pred",
        ].to_numpy(dtype=float)

        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Некорректные старые прогнозы")

        diagnostics = {
            "old_total": float(a.sum()),
            "new_total": float(final_pred.sum()),
            "mean_absolute_change": float(
                np.abs(a - submission.y_pred.to_numpy()).mean()
            ),
            "note": (
                "Изменение прогнозов само по себе "
                "не измеряет улучшение качества."
            ),
        }

        (args.output / "reference_comparison.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        f"Сохранено {len(submission)} прогнозов: "
        f"{args.output / 'submission_exp009.csv'}",
        flush=True,
    )

    print(
        f"Время выполнения: "
        f"{time.perf_counter() - started:.1f} секунд",
        flush=True,
    )


if __name__ == "__main__":
    main()