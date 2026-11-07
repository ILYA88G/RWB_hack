from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

try:
    import train_model_014a as base
except ModuleNotFoundError as exc:
    if exc.name == "train_model_014a":
        raise SystemExit(
            "Положи train_model_015a.py рядом с train_model_014a.py."
        ) from exc
    raise

try:
    import catboost
    from catboost import CatBoostRegressor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Установи CatBoost: python -m pip install catboost"
    ) from exc


STEP = pd.Timedelta(minutes=30)

CAT_PARAMS = dict(
    loss_function="MAE",
    iterations=1000,
    depth=6,
    learning_rate=0.04,
    l2_leaf_reg=10.0,
    random_strength=0.2,
    bootstrap_type="Bernoulli",
    subsample=0.8,
    border_count=64,
    thread_count=2,
    random_seed=42,
    allow_writing_files=False,
    verbose=200,
)


class Engine(base.Engine):
    def __init__(self, v, times, routes):
        super().__init__(v, times, routes)
        self.extra = {}

        for name, a in [
            ("y", self.v[:, :, 0]),
            ("half", self.half),
        ]:
            cs = np.vstack([
                np.zeros_like(a[:1], dtype=np.float64),
                np.cumsum(a, axis=0, dtype=np.float64),
            ])
            ss = np.vstack([
                np.zeros_like(a[:1], dtype=np.float64),
                np.cumsum(a.astype(np.float64)**2, axis=0),
            ])
            zz = np.vstack([
                np.zeros_like(a[:1], dtype=np.float64),
                np.cumsum(a <= 50, axis=0, dtype=np.float64),
            ])

            for w in (8, 48, 336):
                mean = (cs[w:] - cs[:-w]) / w

                std = np.full_like(a, np.nan)
                std[w-1:] = np.sqrt(np.maximum(
                    (ss[w:] - ss[:-w]) / w - mean**2,
                    0,
                ))

                zeros = np.full_like(a, np.nan)
                zeros[w-1:] = (zz[w:] - zz[:-w]) / w

                self.extra[f"{name}_std{w}"] = std
                self.extra[f"{name}_zeros{w}"] = zeros

    def features(self, t, r, h):
        x, scale, offset = super().features(t, r, h)

        for name, a in self.extra.items():
            x[name] = (
                a[t, r]
                if "_zeros" in name
                else a[t, r] / scale
            )

        return x, scale, offset


def predict(model, engine, origin=None):
    origin = len(engine.times)-1 if origin is None else int(origin)

    r = np.repeat(np.arange(len(engine.routes)), 8)
    h = np.tile(np.arange(1, 9), len(engine.routes))
    t = np.full(len(r), origin)

    x, scale, offset = engine.features(t, r, h)
    result = (
        np.maximum(model.predict(x), 0) * scale + offset
    ).reshape(-1, 8)

    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Некорректные прогнозы CatBoost")

    return result


def train_predict_cat(engine, origin=None):
    origin = len(engine.times)-1 if origin is None else int(origin)
    start = max(337, origin - 8 - 21*48)

    if origin - 8 - start < 7*48:
        raise ValueError("Недостаточно истории для обучения")

    rng = np.random.default_rng(42)

    t = rng.integers(start, origin-8+1, size=400000)
    r = rng.integers(0, len(engine.routes), size=400000)
    h = rng.integers(1, 9, size=400000)

    x, scale, offset = engine.features(t, r, h)
    label = np.maximum(
        (engine.v[t+h, r, 0] - offset) / scale,
        0,
    )

    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(
        x,
        label,
        sample_weight=scale / scale.mean(),
        cat_features=["route_id"],
    )

    return predict(model, engine, origin), model


def predict_network(v, times, half, origin=None):
    origin = len(v)-1 if origin is None else int(origin)

    n = v.sum(axis=1, dtype=np.float64) / 1e6
    half = half.sum(axis=1, dtype=np.float64) / 1e6

    c = np.vstack([
        np.zeros((1, 7)),
        np.cumsum(n, axis=0),
    ])
    hc = np.r_[0.0, np.cumsum(half)]

    def features(t, h):
        t = np.asarray(t)
        f = []

        for lag in (0, 1, 2, 3, 4, 8, 16):
            f.append(half[t-lag])

        for j in (1, 2, 3):
            for lag in (0, 1, 2, 4, 8):
                f.append(n[t-lag, j])

            for w in (4, 8):
                f.append(
                    (c[t+1, j] - c[t+1-w, j]) / w
                )

        for w in (4, 8, 16, 48, 96, 336):
            f.append((hc[t+1] - hc[t+1-w]) / w)

        for lag in (48, 96, 144, 336):
            f.append(half[t+h-lag])

        future = times[t] + pd.to_timedelta(h*30, unit="min")
        slot = (
            future.hour.to_numpy()*2
            + future.minute.to_numpy()/30
        )

        for k in (1, 2, 3):
            f.extend([
                np.sin(slot*2*np.pi/48*k),
                np.cos(slot*2*np.pi/48*k),
            ])

        f.append(
            (future.day_of_week.to_numpy() >= 5).astype(float)
        )

        return np.column_stack(f)

    forecasts = []

    for h in range(1, 9):
        t = np.arange(
            max(336, origin - 21*48),
            origin - h + 1,
        )

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=30.0),
        )

        model.fit(features(t, h), half[t+h])

        value = float(
            model.predict(features(np.array([origin]), h))[0]
        )
        forecasts.append(max(value, 0))

    forecasts = np.asarray(forecasts)

    return (
        forecasts + np.r_[half[origin], forecasts[:-1]]
    ) * 1e6


def baseline_matrix(test, old, times, routes):
    if old.columns.tolist() != ["id", "y_pred"]:
        raise ValueError(
            "В исходном submission нужны колонки id,y_pred"
        )

    # Проверяем ID, маршруты, даты и значения исходного CSV.
    base.make_submission(
        test, old, np.ones((len(routes), 8)), times, routes
    )

    t = test.copy()
    t["id"] = t.id.astype(str)
    t["route_id"] = pd.to_numeric(
        t.route_id, errors="raise"
    ).astype(np.int64)
    t["timestamp"] = pd.to_datetime(
        t.timestamp, errors="raise"
    )

    step = np.rint(
        ((t.timestamp - times[-1]) / STEP).to_numpy(dtype=float)
    ).astype(int) - 1

    route = pd.Index(routes).get_indexer(t.route_id)

    aligned = old.copy()
    aligned["id"] = aligned.id.astype(str)

    values = pd.to_numeric(
        aligned.set_index("id").loc[t.id, "y_pred"],
        errors="raise",
    ).to_numpy(dtype=float)

    matrix = np.empty((len(routes), 8), dtype=np.float64)
    matrix[route, step] = values

    return matrix, t, step, values


def make_candidates(test, old, cat_prediction, network, times, routes):
    original, aligned, step, values = baseline_matrix(
        test, old, times, routes
    )

    old_totals = original.sum(axis=0, dtype=np.float64)
    cat_totals = cat_prediction.sum(axis=0, dtype=np.float64)

    if (
        cat_prediction.shape != original.shape
        or not np.isfinite(cat_prediction).all()
        or (cat_prediction < 0).any()
        or (cat_totals <= 0).any()
        or np.shape(network) != (8,)
        or not np.isfinite(network).all()
        or (network <= 0).any()
    ):
        raise ValueError("Некорректные прогнозы отдельных моделей")

    shares = (
        0.5 * original / old_totals[None, :]
        + 0.5 * cat_prediction / cat_totals[None, :]
    )

    a = base.make_submission(
        test, old, shares, times, routes
    )

    new_totals = 0.5 * old_totals + 0.5 * network

    revised = pd.DataFrame({
        "id": aligned.id.to_numpy(),
        "y_pred": values * (new_totals / old_totals)[step],
    })

    b = base.make_submission(
        test, revised, shares, times, routes
    )

    report = {
        "old_totals": old_totals.tolist(),
        "cat_totals": cat_totals.tolist(),
        "network_totals": network.tolist(),
        "015a_totals": old_totals.tolist(),
        "015b_totals": new_totals.tolist(),
        "share_weights": [0.5, 0.5],
        "015b_network_weight": 0.5,
        "public_score": None,
    }

    return a, b, report


def main():
    root = base.find_root()

    candidates = [
        root / "submissions/submission_model_014a.csv",
        root / "submission_model_014a.csv",
    ]

    p = argparse.ArgumentParser()
    p.add_argument(
        "--train", type=Path,
        default=root / "data/raw/train_solo_track.parquet",
    )
    p.add_argument(
        "--test", type=Path,
        default=root / "data/raw/test_solo_track.parquet",
    )
    p.add_argument(
        "--base", type=Path,
        default=next(
            (f for f in candidates if f.is_file()),
            candidates[0],
        ),
    )
    p.add_argument(
        "--outputdir", type=Path,
        default=root / "submissions",
    )
    args = p.parse_args()

    for name in ("train", "test", "base"):
        path = getattr(args, name)
        if not path.is_file():
            p.error(
                f"Файл не найден: {path}. "
                f"Укажи путь через --{name}."
            )

    outputs = [
        args.outputdir / name
        for name in (
            "submission_model_015a.csv",
            "submission_model_015b.csv",
            "model_015a.cbm",
            "model_015a.json",
        )
    ]

    if (
        {f.resolve() for f in outputs}
        & {
            args.train.resolve(),
            args.test.resolve(),
            args.base.resolve(),
        }
    ):
        p.error("Выходные файлы должны отличаться от исходных")

    started = time.perf_counter()

    print("Загрузка последних 37 дней истории...", flush=True)
    v, times, routes = base.load_history(args.train)
    times = pd.DatetimeIndex(times).as_unit("ns")

    test = base.read_table(
        args.test, ["id", "route_id", "timestamp"]
    )
    old = pd.read_csv(args.base, dtype={"id": str})
    baseline_matrix(test, old, times, routes)

    engine = Engine(v, times, routes)

    print(
        "Обучение CatBoost: 400000 примеров, 1000 деревьев...",
        flush=True,
    )
    prediction, model = train_predict_cat(engine)

    print("Сборка двух submission...", flush=True)
    network = predict_network(v, times, engine.half)

    a, b, report = make_candidates(
        test, old, prediction, network, times, routes
    )

    report.update({
        "origin": str(times[-1]),
        "base_submission": str(args.base.resolve()),
        "catboost_version": catboost.__version__,
        "cat_params": CAT_PARAMS,
        "training_days": 21,
        "training_rows": 400000,
        "sample_seed": 42,
        "routes": routes.tolist(),
        "feature_names": model.feature_names_,
        "network_params": {
            "train_days": 21,
            "ridge_alpha": 30.0,
        },
        "seconds": time.perf_counter() - started,
    })

    args.outputdir.mkdir(parents=True, exist_ok=True)

    a.to_csv(outputs[0], index=False)
    b.to_csv(outputs[1], index=False)
    model.save_model(str(outputs[2]))

    outputs[3].write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for path in outputs:
        print(f"Сохранено: {path.resolve()}", flush=True)

    print("Public Score новых файлов пока неизвестен.", flush=True)


if __name__ == "__main__":
    main()