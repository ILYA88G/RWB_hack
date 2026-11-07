from pathlib import Path
from types import SimpleNamespace
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


VALUES = ["target_1h"] + [f"status_{i}" for i in range(1, 7)]
STEP = pd.Timedelta(minutes=30)


def find_root():
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        if (root / "data/raw/train_solo_track.parquet").is_file():
            return root
    return Path.cwd()


def read_table(path, columns=None):
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def halfhour_causal(y, return_uncertainty=False):
    """Восстановление получасовых отгрузок только по прошлым значениям."""
    y = np.asarray(y, dtype=np.float64)
    half = np.empty(y.shape, dtype=np.float32)
    width = np.empty_like(half)

    lower = np.zeros(y.shape[1])
    upper = y[0].copy()
    intercept = np.zeros(y.shape[1])

    for t in range(len(y)):
        if t:
            intercept = y[t] - intercept

        sign = 1.0 if t % 2 == 0 else -1.0

        if sign > 0:
            lower = np.maximum(lower, -intercept)
        else:
            upper = np.minimum(upper, intercept)

        half[t] = np.maximum(
            sign * 0.5 * (lower + upper) + intercept, 0
        )
        width[t] = np.maximum(upper - lower, 0)

    return (half, width) if return_uncertainty else half


def linear_features(v, times, cs, flow, t, scale):
    f = []

    for lag in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48, 96, 336):
        f.append(v[t-lag, :, 0] / scale[None, :, 0])

    for j in (1, 2, 3):
        for lag in (0, 1, 2, 3, 4, 5, 6, 8):
            f.append(v[t-lag, :, j] / scale[None, :, j])

    for j in (4, 5, 6):
        f.append(v[t, :, j] / scale[None, :, j])
        for lag in (1, 4, 8, 16):
            f.append(
                (v[t, :, j] - v[t-lag, :, j]) / scale[None, :, j]
            )

    for w in (4, 8, 16, 48, 96):
        f.append(
            (cs[t+1, :, 0] - cs[t+1-w, :, 0])
            / w / scale[None, :, 0]
        )

    for j in (1, 2, 3):
        for w in (3, 6, 12):
            f.append(
                (cs[t+1, :, j] - cs[t+1-w, :, j])
                / w / scale[None, :, j]
            )

    slot = times[t].hour.to_numpy()*2 + times[t].minute.to_numpy()/30
    shape = (len(t), v.shape[1])

    for k in (1, 2, 3):
        f.append(np.broadcast_to(
            np.sin(slot*2*np.pi*k/48)[:, None], shape
        ))
        f.append(np.broadcast_to(
            np.cos(slot*2*np.pi*k/48)[:, None], shape
        ))

    dow = times[t].day_of_week.to_numpy()
    f.append(np.broadcast_to(
        np.sin(dow*2*np.pi/7)[:, None], shape
    ))
    f.append(np.broadcast_to(
        np.cos(dow*2*np.pi/7)[:, None], shape
    ))

    for lag in (0, 1, 2, 3, 4, 5, 6, 8, 16, 48):
        f.append(flow[t-lag] / scale[None, :, 0])

    return np.stack(f, axis=-1).astype(np.float32)


def train_linear(v, times):
    origin = len(v) - 1
    end = origin - 8
    start = max(336, end - 14*48)

    if end - start < 7*48:
        raise ValueError("Недостаточно истории для обучения")

    t = np.arange(start, end+1, 2)
    scale = np.maximum(
        np.mean(v[start:end+1], axis=0), 1
    ).astype(np.float32)

    flow = halfhour_causal(v[:, :, 0])
    cs = np.concatenate([
        np.zeros((1, v.shape[1], 7)),
        np.cumsum(v, axis=0, dtype=np.float64),
    ])

    x = linear_features(v, times, cs, flow, t, scale)
    xp = linear_features(
        v, times, cs, flow, np.array([origin]), scale
    )[0]

    y = np.stack([
        v[t+h, :, 0] / scale[None, :, 0]
        for h in range(1, 9)
    ], axis=-1).astype(np.float32)

    yf = np.stack([
        flow[t+h] / scale[None, :, 0]
        for h in range(1, 9)
    ], axis=-1).astype(np.float32)

    y = np.concatenate([y, yf], axis=-1)
    xmean = x.mean(axis=0)
    ymean = y.mean(axis=0)
    xc = x - xmean[None]
    yc = y - ymean[None]
    std = np.maximum(xc.std(axis=(0, 1)), 0.001)
    xc /= std[None, None]
    xpc = (xp - xmean) / std[None]

    weight = scale[:, 0] / scale[:, 0].mean()
    model = Ridge(alpha=100.0, fit_intercept=False)
    model.fit(
        xc.reshape(-1, xc.shape[-1]),
        yc.reshape(-1, yc.shape[-1]),
        sample_weight=np.tile(weight, len(t)),
    )

    prediction = model.predict(xpc) + ymean
    prediction = np.maximum(
        prediction[:, :8] * scale[:, 0, None], 0
    ).astype(np.float64)

    state = {
        "model": model,
        "scale": scale,
        "xmean": xmean,
        "ymean": ymean,
        "feature_std": std,
    }
    return prediction, state


def load_history(path):
    train = read_table(path, ["route_id", "timestamp", *VALUES])

    if train.isna().any().any():
        raise ValueError("В train есть пропуски")

    train["timestamp"] = pd.to_datetime(
        train.timestamp, errors="raise"
    )
    latest = train.timestamp.max()
    train = train.loc[
        train.timestamp >= latest - pd.Timedelta(days=37)
    ].copy()

    train["route_id"] = pd.to_numeric(
        train.route_id, errors="raise"
    ).astype(np.int64)

    if train.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Повторяющиеся пары route_id/timestamp")

    routes = np.sort(train.route_id.unique())
    times = pd.DatetimeIndex(
        np.sort(train.timestamp.unique())
    ).as_unit("ns")

    if len(train) != len(routes)*len(times):
        raise ValueError("Неполная сетка маршрут/время")

    if not np.all(np.diff(times.asi8) == STEP.value):
        raise ValueError("В train отсутствуют получасовые отметки")

    ordered = train.sort_values(["timestamp", "route_id"])
    v = ordered[VALUES].to_numpy(dtype=np.float32).reshape(
        len(times), len(routes), 7
    )

    if not np.isfinite(v).all() or (v < 0).any():
        raise ValueError("Некорректные значения в train")

    return v, times, routes


def make_submission(test, old, prediction, times, routes):
    if test[["id", "route_id", "timestamp"]].isna().any().any():
        raise ValueError("Пропуски в test")

    test = test.copy()
    test["id"] = test.id.astype(str)
    test["route_id"] = pd.to_numeric(
        test.route_id, errors="raise"
    ).astype(np.int64)
    test["timestamp"] = pd.to_datetime(
        test.timestamp, errors="raise"
    )

    old = old.copy()
    old["id"] = old.id.astype(str)

    if test.id.duplicated().any() or old.id.duplicated().any():
        raise ValueError("Повторяющиеся ID")

    if test.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Повторяющиеся маршрут/время в test")

    if set(test.id) != set(old.id) or len(test) != len(routes)*8:
        raise ValueError("ID старого submission и test различаются")

    if (
        set(test.route_id) != set(routes)
        or not test.groupby("route_id").size().eq(8).all()
    ):
        raise ValueError("Нужно восемь строк test на каждый маршрут")

    h = ((test.timestamp - times[-1]) / STEP).to_numpy(dtype=float)

    if (
        not np.isfinite(h).all()
        or not np.allclose(h, np.rint(h))
        or h.min() != 1
        or h.max() != 8
    ):
        raise ValueError("Неправильные временные точки test")

    step = np.rint(h).astype(int) - 1
    route = pd.Index(routes).get_indexer(test.route_id)

    base = pd.to_numeric(
        old.set_index("id").loc[test.id, "y_pred"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Некорректный старый submission")

    base_total = np.bincount(step, weights=base, minlength=8)
    new_total = prediction.sum(axis=0, dtype=np.float64)

    if (base_total <= 0).any() or (new_total <= 0).any():
        raise ValueError("Неположительные общие объёмы")

    reconciled = prediction * (base_total / new_total)[None, :]
    values = reconciled[route, step]

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Некорректный итоговый прогноз")

    np.testing.assert_allclose(
        np.bincount(step, weights=values, minlength=8),
        base_total,
        rtol=1e-10,
    )

    return pd.DataFrame({
        "id": test.id.to_numpy(),
        "y_pred": values,
    })


class Engine:
    def __init__(self, v, times, routes):
        self.v = np.asarray(v, dtype=np.float32)
        self.times = times
        self.routes = np.asarray(routes)

        self.half, self.unc = halfhour_causal(
            self.v[:, :, 0], return_uncertainty=True
        )

        cs = np.concatenate([
            np.zeros_like(self.v[:1], dtype=np.float64),
            np.cumsum(self.v, axis=0, dtype=np.float64),
        ])

        self.means = {}
        for w in (2, 4, 8, 16, 48, 96, 336):
            out = np.full_like(self.v, np.nan)
            out[w-1:] = (cs[w:] - cs[:-w]) / w
            self.means[w] = out

        hc = np.vstack([
            np.zeros_like(self.half[:1], dtype=np.float64),
            np.cumsum(self.half, axis=0, dtype=np.float64),
        ])

        self.hm = {}
        for w in (4, 8, 16, 48):
            out = np.full_like(self.half, np.nan)
            out[w-1:] = (hc[w:] - hc[:-w]) / w
            self.hm[w] = out

        self.net = self.v.sum(axis=1, dtype=np.float64)

    def features(self, t, r, h):
        scale = np.maximum(self.means[336][t, r, 0], 100.0)

        f = {
            "route_id": r.astype(np.int32),
            "horizon": h.astype(np.float32),
            "route_scale": np.log1p(scale).astype(np.float32),
        }

        for j in range(7):
            normal = np.maximum(self.means[336][t, r, j], 1.0)
            denominator = scale if j == 0 else normal

            lags = (
                (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48, 96)
                if j == 0 else (0, 1, 2, 3, 4, 6, 8)
            )
            for lag in lags:
                f[f"c{j}_lag{lag}"] = (
                    self.v[t-lag, r, j] / denominator
                )

            windows = (
                (2, 4, 8, 16, 48, 96, 336)
                if j == 0 else (4, 16, 48, 336)
            )
            for w in windows:
                f[f"c{j}_mean{w}"] = (
                    self.means[w][t, r, j] / denominator
                )

            f[f"c{j}_level"] = np.log1p(normal).astype(np.float32)

        for lag in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48):
            f[f"half_lag{lag}"] = self.half[t-lag, r] / scale

        for w in (4, 8, 16, 48):
            f[f"half_mean{w}"] = self.hm[w][t, r] / scale

        f["half_uncertainty"] = self.unc[t, r] / scale

        for lag in (48, 96, 336):
            f[f"future_y_lag{lag}"] = (
                self.v[t+h-lag, r, 0] / scale
            )
            f[f"future_half_lag{lag}"] = (
                self.half[t+h-lag, r] / scale
            )

        seasonal = np.zeros(len(t), dtype=np.float32)
        for day in range(1, 8):
            for delta in (-1, 0, 1):
                seasonal += (
                    self.v[t+h-day*48+delta, r, 0] / 21
                )

        f["future_profile7"] = seasonal / scale

        future = self.times[t] + pd.to_timedelta(h*30, unit="min")
        slot = future.hour.to_numpy()*2 + future.minute.to_numpy()/30

        f["slot_sin"] = np.sin(
            2*np.pi*slot/48
        ).astype(np.float32)
        f["slot_cos"] = np.cos(
            2*np.pi*slot/48
        ).astype(np.float32)
        f["dow"] = future.day_of_week.to_numpy(dtype=np.float32)

        for j in range(7):
            f[f"network_c{j}"] = (
                self.net[t, j] / 1_000_000
            ).astype(np.float32)

        offset = np.where(h == 1, self.half[t, r], 0)
        return pd.DataFrame(f), scale, offset


def train_boosted(engine, threads):
    origin = len(engine.times) - 1
    rng = np.random.default_rng(42)

    start = max(337, origin - 8 - 21*48)
    t = rng.integers(start, origin-8+1, size=400000)
    r = rng.integers(0, len(engine.routes), size=400000)
    h = rng.integers(1, 9, size=400000)

    x, scale, offset = engine.features(t, r, h)
    label = (engine.v[t+h, r, 0] - offset) / scale
    weight = scale / scale.mean()

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=150,
        reg_lambda=15.0,
        reg_alpha=0.1,
        colsample_bytree=0.9,
        subsample=0.9,
        subsample_freq=1,
        max_bin=127,
        n_jobs=threads,
        random_state=42,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )

    model.fit(
        x,
        np.maximum(label, 0),
        sample_weight=weight,
        categorical_feature=["route_id"],
    )

    rp = np.repeat(np.arange(len(engine.routes)), 8)
    hp = np.tile(np.arange(1, 9), len(engine.routes))
    tp = np.full(len(rp), origin)

    xp, sp, op = engine.features(tp, rp, hp)
    prediction = (
        np.maximum(model.predict(xp), 0) * sp + op
    ).reshape(-1, 8)

    return prediction, model


def main():
    root = find_root()

    old_candidates = [
        root / "submissions/submission_network_008b.csv",
        root / "submission_network_008b.csv",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=Path,
        default=root / "data/raw/train_solo_track.parquet",
    )
    parser.add_argument(
        "--test", type=Path,
        default=root / "data/raw/test_solo_track.parquet",
    )
    parser.add_argument(
        "--old", type=Path,
        default=next(
            (p for p in old_candidates if p.is_file()),
            old_candidates[0],
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "submissions/submission_model_014a.csv",
    )
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("threads должен быть положительным")

    for name, path in (
        ("train", args.train),
        ("test", args.test),
        ("old", args.old),
    ):
        if not path.is_file():
            parser.error(
                f"Файл не найден: {path}. Укажи путь через --{name}."
            )

    if args.output.resolve() in (
        args.old.resolve(),
        args.train.resolve(),
        args.test.resolve(),
    ):
        parser.error("Итоговый файл должен отличаться от исходных")

    started = time.perf_counter()
    print("Загрузка последних 37 дней истории...", flush=True)
    v, times, routes = load_history(args.train)

    print("Обучение Ridge...", flush=True)
    linear, state = train_linear(v, times)

    print("Обучение LightGBM...", flush=True)
    engine = Engine(v, times, routes)
    boosted, boosted_model = train_boosted(engine, args.threads)

    linear_total = linear.sum(axis=0, dtype=np.float64)
    boosted_total = boosted.sum(axis=0, dtype=np.float64)

    if (linear_total <= 0).any() or (boosted_total <= 0).any():
        raise ValueError("Неположительные объёмы отдельных моделей")

    shares = (
        0.5 * linear / linear_total[None, :]
        + 0.5 * boosted / boosted_total[None, :]
    )

    test = read_table(args.test, ["id", "route_id", "timestamp"])
    old = pd.read_csv(args.old, dtype={"id": str})

    result = make_submission(test, old, shares, times, routes)

    state.update({
        "routes": routes,
        "origin": str(times[-1]),
        "network_reference": str(args.old),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    model_path = root / "models/model_014a.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "linear": state,
        "lightgbm": boosted_model,
        "share_weights": [0.5, 0.5],
    }, model_path)

    print(
        f"Сохранено {len(result)} прогнозов: "
        f"{args.output.resolve()}",
        flush=True,
    )
    print(
        f"Время: {time.perf_counter() - started:.1f} секунд",
        flush=True,
    )


if __name__ == "__main__":
    main()