"""019B: 25% BASE015A + 25% long CatBoost + 50% long TCN."""

from pathlib import Path
import argparse
import gc
import json
import math
import time

try:
    import resource
except ModuleNotFoundError:
    resource = None

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Установи PyTorch командой: "
        "python -m pip install torch "
        "--index-url https://download.pytorch.org/whl/cpu"
    ) from exc


VALUES = ["target_1h"] + [f"status_{i}" for i in range(1, 7)]
STEP = pd.Timedelta(minutes=30)

EPOCHS = 34
WEIGHT = 0.5

CONFIG = dict(
    history=48,
    channels=16,
    dilations=[1, 2, 4, 8, 16],
    kernel=3,
    groups=150,
    days=84,
    recent_days=21,
    recent_fraction=0.5,
    max_epochs=40,
    batch_groups=1,
    seed=42,
    learning_rate=0.001,
    weight_decay=0.001,
    gradient_clip=5.0,
    raw_clip=12.0,
    normalized_clip=8.0,
    route_embedding=8,
    threads=1,
    decoder="q",
)


def find_root():
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        if (root / "data/raw/train_solo_track.parquet").is_file():
            return root
    return Path.cwd()


def read_table(path, columns=None):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path, columns=columns)
    dtype = {"id": str} if columns and "id" in columns else None
    return pd.read_csv(path, usecols=columns, dtype=dtype)


def halfhour_causal(y, return_uncertainty=False):
    """Приближённое причинное восстановление получасовых объёмов."""
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


def load_history(path):
    print(f"Чтение train: {Path(path).resolve()}", flush=True)

    train = read_table(path, ["route_id", "timestamp", *VALUES])

    if train.empty:
        raise ValueError("Train пустой")
    if train.isna().any().any():
        raise ValueError("В обучающих данных есть пропуски")

    train["timestamp"] = pd.to_datetime(
        train["timestamp"], errors="raise"
    )
    train["route_id"] = pd.to_numeric(
        train["route_id"], errors="raise"
    ).astype(np.int64)

    if train.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Повторяющиеся пары route_id/timestamp")

    routes = np.sort(train["route_id"].unique())
    times = pd.DatetimeIndex(
        np.sort(train["timestamp"].unique())
    ).as_unit("ns")

    if len(times) < 682:
        raise ValueError("Недостаточно истории для причинных признаков")
    if len(train) != len(routes) * len(times):
        raise ValueError("Неполная сетка маршрут/время в train")
    if not np.all(np.diff(times.asi8) == STEP.value):
        raise ValueError("В train отсутствуют получасовые отметки")

    ordered = train.sort_values(["timestamp", "route_id"])
    values = ordered[VALUES].to_numpy(dtype=np.float32)
    values = values.reshape(len(times), len(routes), 7)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Некорректные значения статусов или отгрузок")

    print(
        f"Прочитано {len(train):,} строк; "
        f"маршрутов: {len(routes)}; "
        f"период: {times[0]} — {times[-1]}",
        flush=True,
    )
    return values, times, routes


def make_submission(test, old, prediction, times, routes, *, target_totals=None):
    if test[["id", "route_id", "timestamp"]].isna().any().any():
        raise ValueError("Пропуски в test")

    test = test.copy()
    test["id"] = test["id"].astype(str)
    test["route_id"] = pd.to_numeric(
        test["route_id"], errors="raise"
    ).astype(np.int64)
    test["timestamp"] = pd.to_datetime(
        test["timestamp"], errors="raise"
    )

    old = old.copy()
    old["id"] = old["id"].astype(str)

    if test["id"].duplicated().any() or old["id"].duplicated().any():
        raise ValueError("Повторяющиеся ID")
    if test.duplicated(["route_id", "timestamp"]).any():
        raise ValueError("Повторяющиеся пары route_id/timestamp в test")
    if set(test["id"]) != set(old["id"]):
        raise ValueError("ID исходного submission и test не совпадают")
    if len(test) != len(routes) * 8:
        raise ValueError("В test должно быть 8 строк на каждый маршрут")
    if set(test["route_id"]) != set(routes):
        raise ValueError("Маршруты train и test не совпадают")
    if not test.groupby("route_id").size().eq(8).all():
        raise ValueError("В test должно быть 8 строк на каждый маршрут")

    h = ((test["timestamp"] - times[-1]) / STEP).to_numpy(dtype=float)

    if (
        not np.isfinite(h).all()
        or not np.allclose(h, np.rint(h))
        or h.min() != 1
        or h.max() != 8
    ):
        raise ValueError(
            "Test должен содержать следующие 8 получасовых шагов"
        )

    step = np.rint(h).astype(int) - 1
    route = pd.Index(routes).get_indexer(test["route_id"])

    base = pd.to_numeric(
        old.set_index("id").loc[test["id"], "y_pred"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("Некорректные прогнозы в исходном submission")

    base_total = np.bincount(step, weights=base, minlength=8)
    if target_totals is not None:
        base_total = np.asarray(target_totals, dtype=np.float64)
        if base_total.shape != (8,) or not np.isfinite(base_total).all():
            raise ValueError("Expected eight finite horizon totals")
    new_total = prediction.sum(axis=0, dtype=np.float64)

    if (base_total <= 0).any() or (new_total <= 0).any():
        raise ValueError("Неположительные общие объёмы прогнозов")

    reconciled = prediction * (base_total / new_total)[None, :]
    values = reconciled[route, step]

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Некорректный итоговый прогноз")

    np.testing.assert_allclose(
        np.bincount(step, weights=values, minlength=8),
        base_total,
        rtol=1e-10,
    )

    return pd.DataFrame(
        {"id": test["id"].to_numpy(), "y_pred": values}
    )


def baseline_matrix(test, old, times, routes):
    if old.columns.tolist() != ["id", "y_pred"]:
        raise ValueError(
            "В исходном submission нужны колонки id,y_pred"
        )

    make_submission(
        test, old, np.ones((len(routes), 8)), times, routes
    )

    table = test.copy()
    table["id"] = table["id"].astype(str)
    table["route_id"] = pd.to_numeric(
        table["route_id"], errors="raise"
    ).astype(np.int64)
    table["timestamp"] = pd.to_datetime(
        table["timestamp"], errors="raise"
    )

    step = np.rint(
        ((table["timestamp"] - times[-1]) / STEP).to_numpy(dtype=float)
    ).astype(int) - 1
    route = pd.Index(routes).get_indexer(table["route_id"])

    aligned = old.copy()
    aligned["id"] = aligned["id"].astype(str)
    values = pd.to_numeric(
        aligned.set_index("id").loc[table["id"], "y_pred"],
        errors="raise",
    ).to_numpy(dtype=float)

    matrix = np.empty((len(routes), 8), dtype=np.float64)
    matrix[route, step] = values

    return matrix, table, step, values


class RawHistory:
    def __init__(self, values, times, routes):
        self.v = np.asarray(values, dtype=np.float32)
        self.times = pd.DatetimeIndex(times)
        self.routes = np.asarray(routes)

        self.half, self.uncertainty = halfhour_causal(
            self.v[:, :, 0], True
        )

        cs = np.concatenate(
            [
                np.zeros_like(self.v[:1], dtype=np.float64),
                np.cumsum(self.v, axis=0, dtype=np.float64),
            ]
        )
        self.mean336 = np.full_like(self.v, np.nan)
        self.mean336[335:] = (cs[336:] - cs[:-336]) / 336


def sample_sources(cutoff, epoch, groups=None):
    groups = CONFIG["groups"] if groups is None else groups
    end = cutoff - 8
    start = max(337, end - CONFIG["days"] * 48)
    recent_start = max(337, end - CONFIG["recent_days"] * 48)

    if end - start < 336:
        raise ValueError("Недостаточно причинной истории")

    rng = np.random.default_rng(CONFIG["seed"] + 1009 * epoch)
    recent_groups = int(groups * CONFIG["recent_fraction"])

    sources = np.concatenate(
        (
            rng.integers(recent_start, end + 1, recent_groups),
            rng.integers(start, end + 1, groups - recent_groups),
        )
    )
    rng.shuffle(sources)

    assert (sources + 8 <= cutoff).all()
    return sources


def raw_group(history, source, labels=True):
    source = int(source)

    if source < 337 or source >= len(history.v):
        raise ValueError("Некорректный момент построения признаков")

    mean = np.maximum(history.mean336[source], 1.0)
    scale = np.maximum(mean[:, 0], 100.0)
    mean[:, 0] = scale

    raw = history.v[source - 47:source + 1]

    sequence = np.concatenate(
        [
            raw[:, :, :1] / scale[None, :, None],
            history.half[
                source - 47:source + 1, :, None
            ] / scale[None, :, None],
            raw[:, :, 1:] / mean[None, :, 1:],
        ],
        axis=2,
    )
    sequence = np.ascontiguousarray(sequence.transpose(1, 2, 0))
    np.clip(sequence, 0, CONFIG["raw_clip"], out=sequence)

    h = np.arange(1, 9)
    profile = np.zeros((len(history.routes), 8), dtype=np.float32)
    qprofile = np.zeros_like(profile)

    for day in range(1, 8):
        for delta in (-1, 0, 1):
            index = source + h - day * 48 + delta
            profile += history.v[index, :, 0].T / 21
            qprofile += history.half[index].T / 21

    known = np.zeros_like(profile)
    known[:, 0] = history.half[source]

    normalized_profile = profile / scale[:, None]
    remaining = np.maximum(qprofile / scale[:, None], 0.025)
    base = np.where(
        remaining > 15,
        remaining,
        np.log(np.expm1(np.minimum(remaining, 15))),
    )

    future = history.times[source] + pd.to_timedelta(
        h * 30, unit="min"
    )
    slot = future.hour.to_numpy() * 2 + future.minute.to_numpy() / 30
    dow = future.day_of_week.to_numpy()

    calendar = np.stack(
        [
            np.sin(slot * 2 * np.pi / 48),
            np.cos(slot * 2 * np.pi / 48),
            np.sin(dow * 2 * np.pi / 7),
            np.cos(dow * 2 * np.pi / 7),
        ],
        axis=1,
    ).ravel()

    current = np.concatenate(
        [
            history.v[source, :, :1],
            history.half[source, :, None],
            history.v[source, :, 1:],
        ],
        axis=1,
    )

    static = np.concatenate(
        [
            np.log1p(mean),
            np.log1p(current),
            normalized_profile,
            qprofile / scale[:, None],
            (history.uncertainty[source] / scale)[:, None],
            np.broadcast_to(calendar, (len(scale), len(calendar))),
        ],
        axis=1,
    ).astype(np.float32)

    result = dict(
        x=sequence,
        static=static,
        base=base.astype(np.float32),
        scale=scale.astype(np.float32),
        known=known,
    )

    if labels:
        if source + 8 >= len(history.v):
            raise ValueError("Цель выходит за доступную историю")

        y = history.v[source + h, :, 0].T.copy()
        network_q(y.sum(axis=0), known[:, 0].sum())
        result["y"] = y

    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("В признаках обнаружены NaN или inf")

    return result


def build_groups(history, sources, normalization=None, labels=True):
    first = raw_group(history, sources[0], labels)

    result = {
        key: np.empty((len(sources), *row.shape), dtype=row.dtype)
        for key, row in first.items()
    }

    for key, row in first.items():
        result[key][0] = row

    for i, source in enumerate(sources[1:], 1):
        row = raw_group(history, source, labels)
        for key in result:
            result[key][i] = row[key]

    if normalization is None:
        x = result["x"]
        xmean = x.mean(
            axis=(0, 1, 3), dtype=np.float64
        ).astype(np.float32)
        xstd = np.maximum(
            x.std(axis=(0, 1, 3), dtype=np.float64), 0.001
        ).astype(np.float32)

        static = result["static"]
        smean = static.mean(
            axis=(0, 1), dtype=np.float64
        ).astype(np.float32)
        sstd = np.maximum(
            static.std(axis=(0, 1), dtype=np.float64), 0.001
        ).astype(np.float32)

        normalization = dict(
            xmean=xmean,
            xstd=xstd,
            smean=smean,
            sstd=sstd,
        )

    result["x"] -= normalization["xmean"][None, None, :, None]
    result["x"] /= normalization["xstd"][None, None, :, None]
    result["static"] -= normalization["smean"]
    result["static"] /= normalization["sstd"]

    for key in ("x", "static"):
        np.clip(
            result[key],
            -CONFIG["normalized_clip"],
            CONFIG["normalized_clip"],
            out=result[key],
        )

    tensors = {
        key: torch.from_numpy(array)
        for key, array in result.items()
    }
    return tensors, normalization


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(
            channels, channels, 3, dilation=dilation
        )

    def forward(self, x):
        return F.gelu(
            x + self.conv(F.pad(x, (self.left, 0)))
        )


class TemporalAllocator(nn.Module):
    def __init__(self, static_features, routes):
        super().__init__()

        channels = CONFIG["channels"]
        self.route = nn.Embedding(
            routes, CONFIG["route_embedding"]
        )
        self.input = nn.Conv1d(8, channels, 1)
        self.temporal = nn.Sequential(
            *[
                CausalBlock(channels, dilation)
                for dilation in CONFIG["dilations"]
            ]
        )

        width = (
            static_features
            + 2 * channels
            + CONFIG["route_embedding"]
        )

        self.linear = nn.Linear(static_features, 8)
        self.head = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )

        nn.init.normal_(self.route.weight, std=0.02)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        x = batch["x"]
        static = batch["static"]
        groups, routes = x.shape[:2]

        z = self.temporal(
            F.gelu(
                self.input(
                    x.reshape(-1, 8, CONFIG["history"])
                )
            )
        )

        z = torch.cat(
            (z[:, :, -1], z.mean(dim=2)), dim=1
        ).reshape(groups, routes, -1)

        route = self.route.weight[None].expand(groups, -1, -1)

        residual = self.linear(static) + self.head(
            torch.cat((z, static, route), dim=-1)
        )
        logits = torch.clamp(
            batch["base"] + residual, -15, 30
        )

        return F.softplus(logits) * batch["scale"][:, :, None]


def network_q(totals, q0):
    previous = q0
    result = []

    for h in range(8):
        current = totals[..., h] - previous
        if bool((current <= 0).any()):
            raise ValueError(
                "Общие объёмы дают неположительный "
                "получасовой бюджет"
            )
        result.append(current)
        previous = current

    if torch.is_tensor(totals):
        return torch.stack(result, dim=-1)
    return np.stack(result, axis=-1)


def group_prediction(model, batch, totals):
    scores = model(batch)
    q0 = batch["known"][:, :, 0]
    budget = network_q(totals, q0.sum(dim=1))

    q = scores * (budget / scores.sum(dim=1))[:, None, :]
    previous = torch.cat(
        (q0[:, :, None], q[:, :, :-1]), dim=2
    )

    return q + previous


def epoch_pass(model, data, optimizer=None, generator=None, forecast_totals=None):
    training = optimizer is not None
    model.train(training)

    groups = len(data["x"])
    if forecast_totals is not None:
        forecast_totals = torch.as_tensor(forecast_totals, dtype=data["y"].dtype)
        if forecast_totals.shape != (groups, 8) or not torch.isfinite(forecast_totals).all():
            raise ValueError("Expected eight finite forecast totals for every group")
    order = (
        torch.randperm(groups, generator=generator)
        if training
        else torch.arange(groups)
    )

    absolute = 0.0
    mass = 0.0

    for start in range(0, groups, CONFIG["batch_groups"]):
        selected = order[start:start + CONFIG["batch_groups"]]
        batch = {
            key: value[selected]
            for key, value in data.items()
        }
        y = batch["y"]

        with torch.set_grad_enabled(training):
            totals = y.sum(dim=1) if forecast_totals is None else forecast_totals[selected]
            prediction = group_prediction(model, batch, totals)
            error = (prediction - y).abs().sum()
            loss = error / y.sum()

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), CONFIG["gradient_clip"]
                )
                optimizer.step()

        absolute += float(error.detach())
        mass += float(y.sum())

    return absolute / mass


def train_model(
    history, cutoff, epochs, validation_sources=None, tag="train", on_epoch=None,
    validation_totals=None,
):
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    generator = torch.Generator().manual_seed(CONFIG["seed"])

    norm = None
    model = None
    validation = None
    records = []
    began = time.monotonic()

    for epoch in range(epochs):
        before = time.monotonic()
        sources = sample_sources(cutoff, epoch)
        data, norm = build_groups(history, sources, norm)

        if model is None:
            model = TemporalAllocator(
                data["static"].shape[-1], len(history.routes)
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG["learning_rate"],
                weight_decay=CONFIG["weight_decay"],
            )

            if validation_sources is not None:
                validation, _ = build_groups(
                    history, validation_sources, norm
                )

            print(
                tag,
                "parameters",
                sum(parameter.numel() for parameter in model.parameters()),
                "sequence_shape",
                list(data["x"].shape),
                "static_features",
                data["static"].shape[-1],
                "preparation_seconds",
                time.monotonic() - before,
                flush=True,
            )

        lr = CONFIG["learning_rate"] * (
            0.1
            + 0.9
            * 0.5
            * (
                1
                + math.cos(
                    math.pi * epoch / CONFIG["max_epochs"]
                )
            )
        )

        for group in optimizer.param_groups:
            group["lr"] = lr

        loss = epoch_pass(model, data, optimizer, generator)
        val = (
            None
            if validation is None
            else epoch_pass(model, validation, forecast_totals=validation_totals)
        )

        record = dict(
            epoch=epoch + 1,
            train_group_wape=loss,
            validation_group_wape=val,
            validation_budget=(None if validation is None else
                               "observed" if validation_totals is None else "forecast"),
            seconds=time.monotonic() - before,
            lr=lr,
            source_min=int(sources.min()),
            source_max=int(sources.max()),
            source_min_time=str(history.times[sources.min()]),
            source_max_time=str(history.times[sources.max()]),
            label_max=int(sources.max() + 8),
            source_count=len(sources),
            max_rss_mib=(
                None
                if resource is None
                else resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss / 1024
            ),
        )

        records.append(record)
        if on_epoch is not None:
            on_epoch(model, norm, records)
        print(tag, json.dumps(record), flush=True)

        del data
        gc.collect()

    return model, norm, records, time.monotonic() - began


def save_checkpoint(path, model, norm, history, records):
    torch.save(
        dict(
            state_dict=model.state_dict(),
            normalization={
                key: torch.from_numpy(value)
                for key, value in norm.items()
            },
            config=CONFIG,
            origin_index=len(history.times) - 1,
            routes=history.routes.tolist(),
            static_features=len(norm["smean"]),
            history=records,
            origin=str(history.times[-1]),
            history_start=str(history.times[0]),
        ),
        path,
    )


def load_checkpoint(path):
    state = torch.load(
        path, weights_only=True, map_location="cpu"
    )

    if state["config"].get("decoder", "q") != "q":
        raise ValueError("Нужен checkpoint с decoder='q'")

    for key in (
        "history",
        "channels",
        "dilations",
        "kernel",
        "route_embedding",
    ):
        if state["config"][key] != CONFIG[key]:
            raise ValueError("Архитектура checkpoint не совпадает")

    model = TemporalAllocator(
        state["static_features"], len(state["routes"])
    )
    model.load_state_dict(state["state_dict"])
    model.eval()

    model.train_origin = state["origin_index"]
    model.train_time = (
        pd.Timestamp(state["origin"])
        if "origin" in state
        else None
    )
    model.history_start = (
        pd.Timestamp(state["history_start"])
        if "history_start" in state
        else None
    )
    model.trained_routes = np.asarray(state["routes"])

    norm = {
        key: value.numpy()
        for key, value in state["normalization"].items()
    }

    return model, norm, state


def predict_scores(model, history, norm, origin=None):
    origin = (
        len(history.times) - 1
        if origin is None
        else int(origin)
    )

    if origin < getattr(model, "train_origin", -1):
        raise ValueError("Прогноз раньше границы обучения")

    if (
        getattr(model, "train_time", None) is not None
        and history.times[origin] < model.train_time
    ):
        raise ValueError("Дата прогноза раньше границы обучения")

    if (
        getattr(model, "history_start", None) is not None
        and history.times[0] != model.history_start
    ):
        raise ValueError(
            "Checkpoint требует исходную историю "
            "с той же начальной датой"
        )

    if hasattr(model, "trained_routes"):
        np.testing.assert_array_equal(
            history.routes, model.trained_routes
        )

    data, _ = build_groups(
        history, [origin], norm, labels=False
    )
    model.eval()

    with torch.no_grad():
        scores = model(data)[0].numpy().astype(float)

    known = data["known"][0].numpy().astype(float)
    return scores, known


def allocate(scores, known, totals):
    scores, known, totals = [
        np.asarray(array, dtype=np.float64)
        for array in (scores, known, totals)
    ]

    if (
        scores.ndim != 2
        or scores.shape[1] != 8
        or scores.shape != known.shape
        or totals.shape != (8,)
        or (scores <= 0).any()
        or (known < 0).any()
        or not all(
            np.isfinite(array).all()
            for array in (scores, known, totals)
        )
    ):
        raise ValueError("Некорректные массивы распределения")

    budget = network_q(totals, known[:, 0].sum())
    q = scores * (budget / scores.sum(axis=0))[None, :]

    prediction = q + np.concatenate(
        (known[:, :1], q[:, :-1]), axis=1
    )

    np.testing.assert_allclose(
        prediction.sum(axis=0), totals, rtol=1e-12
    )
    return prediction


class BaseCatFeatures:
    def __init__(self, values, times, routes):
        self.v = np.asarray(values, dtype=np.float32)
        self.times = times
        self.routes = np.asarray(routes)

        self.half, self.unc = halfhour_causal(
            self.v[:, :, 0], return_uncertainty=True
        )
        self.means = {}

        cs = np.concatenate(
            [
                np.zeros_like(self.v[:1], dtype=np.float64),
                np.cumsum(self.v, axis=0, dtype=np.float64),
            ]
        )

        for window in (2, 4, 8, 16, 48, 96, 336):
            out = np.full_like(self.v, np.nan)
            out[window - 1:] = (
                cs[window:] - cs[:-window]
            ) / window
            self.means[window] = out

        hc = np.vstack(
            [
                np.zeros_like(self.half[:1], dtype=np.float64),
                np.cumsum(self.half, axis=0, dtype=np.float64),
            ]
        )
        self.hm = {}

        for window in (4, 8, 16, 48):
            out = np.full_like(self.half, np.nan)
            out[window - 1:] = (
                hc[window:] - hc[:-window]
            ) / window
            self.hm[window] = out

        self.net = self.v.sum(axis=1, dtype=np.float64)

    def features(self, t, r, h, mode="deconv"):
        scale = np.maximum(
            self.means[336][t, r, 0], 100.0
        )

        features = {
            "route_id": r.astype(np.int32),
            "horizon": h.astype(np.float32),
            "route_scale": np.log1p(scale).astype(np.float32),
        }

        for channel in range(7):
            normal = np.maximum(
                self.means[336][t, r, channel], 1.0
            )

            lags = (
                (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48, 96)
                if channel == 0
                else (0, 1, 2, 3, 4, 6, 8)
            )
            windows = (
                (2, 4, 8, 16, 48, 96, 336)
                if channel == 0
                else (4, 16, 48, 336)
            )
            denominator = scale if channel == 0 else normal

            for lag in lags:
                features[f"c{channel}_lag{lag}"] = (
                    self.v[t - lag, r, channel] / denominator
                )

            for window in windows:
                features[f"c{channel}_mean{window}"] = (
                    self.means[window][t, r, channel] / denominator
                )

            features[f"c{channel}_level"] = np.log1p(
                normal
            ).astype(np.float32)

        if mode != "plain":
            for lag in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48):
                features[f"half_lag{lag}"] = (
                    self.half[t - lag, r] / scale
                )

            for window in (4, 8, 16, 48):
                features[f"half_mean{window}"] = (
                    self.hm[window][t, r] / scale
                )

            features["half_uncertainty"] = self.unc[t, r] / scale

        for lag in (48, 96, 336):
            features[f"future_y_lag{lag}"] = (
                self.v[t + h - lag, r, 0] / scale
            )

            if mode != "plain":
                features[f"future_half_lag{lag}"] = (
                    self.half[t + h - lag, r] / scale
                )

        seasonal = np.zeros(len(t), dtype=np.float32)

        for day in range(1, 8):
            for delta in (-1, 0, 1):
                seasonal += (
                    self.v[
                        t + h - day * 48 + delta, r, 0
                    ] / 21
                )

        features["future_profile7"] = seasonal / scale

        future = self.times[t] + pd.to_timedelta(
            h * 30, unit="min"
        )
        slot = (
            future.hour.to_numpy() * 2
            + future.minute.to_numpy() / 30
        )

        features["slot_sin"] = np.sin(
            2 * np.pi * slot / 48
        ).astype(np.float32)
        features["slot_cos"] = np.cos(
            2 * np.pi * slot / 48
        ).astype(np.float32)
        features["dow"] = future.day_of_week.to_numpy(
            dtype=np.float32
        )

        for channel in range(7):
            features[f"network_c{channel}"] = (
                self.net[t, channel] / 1_000_000.0
            ).astype(np.float32)

        offset = (
            np.where(h == 1, self.half[t, r], 0)
            if mode == "deconv"
            else np.zeros(len(t))
        )

        return pd.DataFrame(features), scale, offset


class CatFeatures(BaseCatFeatures):
    def __init__(self, values, times, routes):
        super().__init__(values, times, routes)
        self.extra = {}

        for name, array in [
            ("y", self.v[:, :, 0]),
            ("half", self.half),
        ]:
            cs = np.vstack(
                [
                    np.zeros_like(array[:1], dtype=np.float64),
                    np.cumsum(array, axis=0, dtype=np.float64),
                ]
            )
            ss = np.vstack(
                [
                    np.zeros_like(array[:1], dtype=np.float64),
                    np.cumsum(
                        array.astype(np.float64) ** 2,
                        axis=0,
                    ),
                ]
            )
            zz = np.vstack(
                [
                    np.zeros_like(array[:1], dtype=np.float64),
                    np.cumsum(
                        array <= 50,
                        axis=0,
                        dtype=np.float64,
                    ),
                ]
            )

            for window in (8, 48, 336):
                mean = (
                    cs[window:] - cs[:-window]
                ) / window

                std = np.full_like(array, np.nan)
                std[window - 1:] = np.sqrt(
                    np.maximum(
                        (ss[window:] - ss[:-window]) / window
                        - mean ** 2,
                        0,
                    )
                )

                zeros = np.full_like(array, np.nan)
                zeros[window - 1:] = (
                    zz[window:] - zz[:-window]
                ) / window

                self.extra[f"{name}_std{window}"] = std
                self.extra[f"{name}_zeros{window}"] = zeros

    def features(self, t, r, h):
        x, scale, offset = super().features(t, r, h)

        for name, array in self.extra.items():
            x[name] = (
                array[t, r]
                if "_zeros" in name
                else array[t, r] / scale
            )

        return x, scale, offset


def main():
    from catboost import CatBoostRegressor

    here = Path(__file__).resolve().parent
    root = find_root()

    parser = argparse.ArgumentParser(
        description="019B: создание submission из обученных моделей"
    )
    parser.add_argument(
        "--train", type=Path,
        default=root / "data/raw/train_solo_track.parquet",
    )
    parser.add_argument(
        "--test", type=Path,
        default=here / "test_007.csv.gz",
    )
    parser.add_argument(
        "--base", type=Path,
        default=here / "submission_model_015a.csv",
    )
    parser.add_argument(
        "--tcn", type=Path,
        default=here / "model019a.pt",
    )
    parser.add_argument(
        "--cat", type=Path,
        default=here / "long_cat_1030_1030.cbm",
    )
    parser.add_argument(
        "--output", type=Path,
        default=here / "submission_model_019b.csv",
    )
    parser.add_argument(
        "--details", type=Path,
        help="Необязательный NPZ с компонентами прогноза",
    )
    args = parser.parse_args()

    for name in ("train", "test", "base", "tcn", "cat"):
        if not getattr(args, name).is_file():
            parser.error(
                f"Файл {name} не найден: {getattr(args, name)}"
            )

    inputs = {
        getattr(args, name).resolve()
        for name in ("train", "test", "base", "tcn", "cat")
    }
    outputs = [args.output]
    if args.details:
        outputs.append(args.details)

    if (
        inputs & {path.resolve() for path in outputs}
        or len({path.resolve() for path in outputs}) != len(outputs)
    ):
        parser.error(
            "Выходные файлы должны отличаться от входных "
            "и друг от друга"
        )

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    values, times, routes = load_history(args.train)
    test = read_table(args.test, ["id", "route_id", "timestamp"])
    base = pd.read_csv(args.base, dtype={"id": str})

    original = baseline_matrix(
        test, base, times, routes
    )[0]
    totals = original.sum(axis=0, dtype=np.float64)

    print("Прогноз TCN...", flush=True)
    history = RawHistory(values, times, routes)
    model, norm, state = load_checkpoint(args.tcn)

    scores, known = predict_scores(model, history, norm)
    temporal = allocate(scores, known, totals)

    del history, model, norm, scores, known
    gc.collect()

    print("Прогноз CatBoost...", flush=True)

    if times[0] != pd.Timestamp("2025-07-28 00:00:00"):
        raise ValueError(
            "CatBoost требует полный train начиная с 28 июля"
        )
    cat = CatBoostRegressor()
    cat.load_model(str(args.cat))
    metadata = cat.get_metadata()
    # The archived 019b checkpoint predates the embedded training metadata.
    cat_cutoff = (metadata["training_cutoff"] if "training_cutoff" in metadata
                  else "2025-10-30 10:30:00")
    if times[-1] < pd.Timestamp(cat_cutoff):
        raise ValueError(
            "Дата прогноза раньше границы обучения CatBoost"
        )

    np.testing.assert_array_equal(routes, np.arange(1000))
    if "route_order" in metadata:
        np.testing.assert_array_equal(routes, json.loads(metadata["route_order"]))

    engine = CatFeatures(values, times, routes)
    rp = np.repeat(np.arange(len(routes)), 8)
    hp = np.tile(np.arange(1, 9), len(routes))

    features, scale, offset = engine.features(
        np.full(len(rp), len(times) - 1), rp, hp
    )

    if cat.feature_names_ != list(features.columns):
        raise ValueError(
            "Названия или порядок признаков CatBoost не совпадают"
        )

    raw = (
        np.maximum(
            cat.predict(features, thread_count=1), 0
        ) * scale + offset
    ).reshape(-1, 8)

    cat_pred = raw * (
        totals / raw.sum(axis=0, dtype=np.float64)
    )[None, :]

    prediction = (
        0.25 * original
        + 0.25 * cat_pred
        + 0.5 * temporal
    )

    if (
        not np.isfinite(prediction).all()
        or (prediction < 0).any()
    ):
        raise ValueError("Некорректный итоговый прогноз")

    np.testing.assert_allclose(
        prediction.sum(axis=0), totals, rtol=1e-12
    )

    submission = make_submission(
        test, base, prediction, times, routes
    )
    checked = baseline_matrix(
        test, submission, times, routes
    )[0]

    np.testing.assert_allclose(
        checked.sum(axis=0), totals, rtol=1e-10
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.details,
            pred=prediction,
            baseline=original,
            cat=cat_pred,
            temporal=temporal,
            teacher_totals=totals,
            routes=routes,
            origin=str(times[-1]),
            cat_training_origin=cat_cutoff,
        )

    print(
        f"Сохранено {len(submission)} прогнозов: "
        f"{args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
