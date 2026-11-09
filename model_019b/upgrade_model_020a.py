"""Обновление 019b: недельный адаптер и локальная поправка."""

from pathlib import Path
import argparse
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import Ridge

import predict_model_019b as old


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def long_features(history, source, norm):
    mean = np.maximum(history.mean336[source], 1.)
    scale = np.maximum(mean[:, 0], 100.)
    raw = history.v[source - 335:source + 1]

    sequence = np.concatenate([
        raw[:, :, :1] / scale[None, :, None],
        history.half[
            source - 335:source + 1, :, None
        ] / scale[None, :, None],
        raw[:, :, 1:] / mean[None, :, 1:]
    ], axis=2)

    np.clip(sequence, 0, 12, out=sequence)

    pooled = sequence.reshape(
        84, 4, len(history.routes), 8
    ).mean(axis=1)
    pooled = pooled.transpose(1, 2, 0).copy()

    pooled -= norm["xmean"][None, :, None]
    pooled /= norm["xstd"][None, :, None]
    np.clip(pooled, -8, 8, out=pooled)

    return np.ascontiguousarray(
        pooled.reshape(len(history.routes), 672)
    )


def groups(history, sources, norm, labels=True):
    data, _ = old.build_groups(
        history, sources, norm, labels=labels
    )

    patches = np.empty(
        (len(sources), len(history.routes), 672),
        dtype=np.float32
    )
    for i, source in enumerate(sources):
        patches[i] = long_features(history, int(source), norm)

    data["long"] = torch.from_numpy(patches)
    return data


class LongPatchAllocator(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.eval()
        self.base.requires_grad_(False)

        self.adapter = nn.Sequential(
            nn.Linear(672, 64),
            nn.GELU(),
            nn.Linear(64, 8)
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            x, static = batch["x"], batch["static"]
            count, routes = x.shape[:2]

            z = self.base.temporal(
                F.gelu(
                    self.base.input(x.reshape(-1, 8, 48))
                )
            )
            z = torch.cat(
                (z[:, :, -1], z.mean(dim=2)), dim=1
            ).reshape(count, routes, -1)

            route = self.base.route.weight[None].expand(
                count, -1, -1
            )
            residual = self.base.linear(static) + self.base.head(
                torch.cat((z, static, route), dim=-1)
            )
            logits = torch.clamp(
                batch["base"] + residual, -15, 30
            )

        adjustment = 0.5 * torch.tanh(
            self.adapter(batch["long"])
        )
        return (
            F.softplus(logits + adjustment)
            * batch["scale"][:, :, None]
        )


def fit_adapter(model, history, norm):
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=0.0005,
        weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(42)
    records = []

    for epoch in range(8):
        began = time.monotonic()
        sources = old.sample_sources(
            len(history.times) - 1, epoch, groups=100
        )
        data = groups(history, sources, norm)
        order = torch.randperm(
            len(sources), generator=generator
        )
        model.train()
        absolute = mass = 0.

        for selected in order:
            i = int(selected)
            batch = {
                key: value[i:i + 1]
                for key, value in data.items()
            }
            y = batch["y"]

            prediction = old.group_prediction(
                model, batch, y.sum(dim=1)
            )
            error = (prediction - y).abs().sum()

            optimizer.zero_grad(set_to_none=True)
            (error / y.sum()).backward()
            nn.utils.clip_grad_norm_(
                model.adapter.parameters(), 5.
            )
            optimizer.step()

            absolute += float(error.detach())
            mass += float(y.sum())

        row = dict(
            epoch=epoch + 1,
            wape=absolute / mass,
            seconds=time.monotonic() - began,
            latest_training_label=int(sources.max() + 8)
        )
        records.append(row)
        print(json.dumps(row), flush=True)

        del data
        gc.collect()

    model.eval()
    return records


def _features(v, times, cs, t, scale, flow):
    f = []

    for lag in (0, 1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 48, 96, 336):
        f.append(v[t - lag, :, 0] / scale[None, :, 0])

    for j in (1, 2, 3):
        for lag in (0, 1, 2, 3, 4, 5, 6, 8):
            f.append(v[t - lag, :, j] / scale[None, :, j])

    for j in (4, 5, 6):
        f.append(v[t, :, j] / scale[None, :, j])
        for lag in (1, 4, 8, 16):
            f.append(
                (v[t, :, j] - v[t - lag, :, j])
                / scale[None, :, j]
            )

    for w in (4, 8, 16, 48, 96):
        f.append(
            (cs[t + 1, :, 0] - cs[t + 1 - w, :, 0])
            / w / scale[None, :, 0]
        )

    for j in (1, 2, 3):
        for w in (3, 6, 12):
            f.append(
                (cs[t + 1, :, j] - cs[t + 1 - w, :, j])
                / w / scale[None, :, j]
            )

    slot = (
        times[t].hour.to_numpy() * 2
        + times[t].minute.to_numpy() / 30
    )
    for k in (1, 2, 3):
        for z in (
            np.sin(slot * 2 * np.pi * k / 48),
            np.cos(slot * 2 * np.pi * k / 48)
        ):
            f.append(np.broadcast_to(z[:, None], (len(t), v.shape[1])))

    dow = times[t].day_of_week.to_numpy()
    for z in (
        np.sin(dow * 2 * np.pi / 7),
        np.cos(dow * 2 * np.pi / 7)
    ):
        f.append(np.broadcast_to(z[:, None], (len(t), v.shape[1])))

    for lag in (0, 1, 2, 3, 4, 5, 6, 8, 16, 48):
        f.append(flow[t - lag] / scale[None, :, 0])

    return np.stack(f, axis=-1).astype(np.float32)


def _local(v, cs, t, scale, flow):
    f = [
        v[t, :, 0] / scale[None, :, 0],
        (v[t, :, 0] - v[t - 3, :, 0]) / scale[None, :, 0],
        flow[t] / scale[None, :, 0]
    ]

    for j in (1, 2, 3):
        recent = (cs[t + 1, :, j] - cs[t - 2, :, j]) / 3
        older = (cs[t - 2, :, j] - cs[t - 6, :, j]) / 4
        f.extend([
            recent / scale[None, :, j],
            (recent - older) / scale[None, :, j]
        ])

    f.append(
        (cs[t + 1, :, 0] - cs[t - 47, :, 0])
        / 48 / scale[None, :, 0]
    )
    return np.stack(f, axis=-1).astype(np.float64)


def fit_prefix(v, times, flow, origin, days=14):
    origin = int(origin)
    if origin < 700 or len(v) <= origin or len(flow) <= origin:
        raise ValueError("Недостаточно истории для локальной модели")

    v = np.asarray(v[:origin + 1])
    flow = np.asarray(flow[:origin + 1])
    cs = np.concatenate([
        np.zeros((1, v.shape[1], v.shape[2])),
        np.cumsum(v, axis=0, dtype=np.float64)
    ])

    end = origin - 8
    start = max(336, end - int(days) * 48)
    t = np.arange(start, end + 1, 2)

    scale = np.maximum(
        v[start:end + 1].mean(axis=0), 1
    ).astype(np.float32)

    x = _features(v, times, cs, t, scale, flow)
    xp = _features(
        v, times, cs, np.array([origin]), scale, flow
    )[0]

    y = np.stack([
        v[t + h, :, 0] / scale[None, :, 0]
        for h in range(1, 9)
    ], axis=-1).astype(np.float32)

    yf = np.stack([
        flow[t + h] / scale[None, :, 0]
        for h in range(1, 9)
    ], axis=-1).astype(np.float32)

    y = np.concatenate([y, yf], axis=-1)
    xm = x.mean(axis=0)
    ym = y.mean(axis=0)
    xc = x - xm[None]
    yc = y - ym[None]

    std = np.maximum(xc.std(axis=(0, 1)), 0.001)
    xc /= std[None, None]
    xpc = (xp - xm) / std[None]

    weight = scale[:, 0] / scale[:, 0].mean()
    model = Ridge(alpha=100., fit_intercept=False)
    model.fit(
        xc.reshape(-1, xc.shape[-1]),
        yc.reshape(-1, 16),
        sample_weight=np.tile(weight, len(t))
    )

    global_pred = (model.predict(xpc) + ym)[:, :8]
    residual = yc[:, :, :8] - np.einsum(
        "trf,hf->trh", xc, model.coef_[:8], optimize=True
    )

    h = np.linspace(-1, 1, 8)
    basis = np.linalg.qr(
        np.column_stack([np.ones(8), h, h * h])
    )[0]
    target = residual @ basis

    z = _local(v, cs, t, scale, flow)
    zp = _local(
        v, cs, np.array([origin]), scale, flow
    )[0]

    mean = z.mean(axis=0)
    z -= mean[None]
    zp -= mean

    zstd = np.maximum(np.sqrt(np.mean(z * z, axis=0)), 0.05)
    z /= zstd[None]
    zp /= zstd

    return dict(
        global_pred=global_pred.astype(np.float64),
        scale=scale[:, 0].astype(np.float64),
        z=z,
        zp=zp,
        basis=basis,
        target=target,
        residual=residual.astype(np.float64)
    )


def predict_robust(stats, alpha=400., iterations=6):
    z = np.concatenate([
        np.ones((*stats["z"].shape[:2], 1)),
        stats["z"]
    ], axis=2)
    zp = np.concatenate([
        np.ones((stats["zp"].shape[0], 1)),
        stats["zp"]
    ], axis=1)

    target = stats["target"]
    basis = stats["basis"]
    _, _, features = z.shape
    pred = np.zeros_like(stats["residual"])

    route_scale = np.maximum(
        np.median(np.abs(stats["residual"]), axis=(0, 2)),
        0.05
    )
    penalty = np.full(features, float(alpha))
    penalty[0] = alpha / 4

    for _ in range(iterations):
        deviation = np.mean(
            np.abs(stats["residual"] - pred), axis=2
        )
        w = np.minimum(
            1,
            0.35 * route_scale[None, :]
            / np.maximum(deviation, 0.001)
        )
        w /= w.mean(axis=0)[None, :]

        gram = np.einsum(
            "trf,trg,tr->rfg", z, z, w, optimize=True
        )
        rhs = np.einsum(
            "trf,trh,tr->rfh", z, target, w, optimize=True
        )
        gram[:, np.arange(features), np.arange(features)] += (
            penalty[None, :]
        )
        coef = np.linalg.solve(gram, rhs)
        pred = np.einsum(
            "trf,rfk->trk", z, coef, optimize=True
        ) @ basis.T

    local = np.einsum(
        "rf,rfk->rk", zp, coef
    ) @ basis.T
    raw = (
        stats["global_pred"] + local
    ) * stats["scale"][:, None]

    return np.maximum(raw, 0)


def local_adjustment(values, times, half, totals):
    values = np.asarray(values, dtype=np.float32)
    half = np.asarray(half, dtype=np.float32)
    totals = np.asarray(totals, dtype=np.float64)

    stats = fit_prefix(
        values, times, half, len(values) - 1, days=14
    )
    raw_local = predict_robust(stats, alpha=400., iterations=6)
    raw_ridge = np.maximum(
        stats["global_pred"] * stats["scale"][:, None], 0
    )

    local_sum = raw_local.sum(axis=0, dtype=np.float64)
    ridge_sum = raw_ridge.sum(axis=0, dtype=np.float64)

    if (
        not np.isfinite(raw_local).all()
        or not np.isfinite(raw_ridge).all()
        or (local_sum <= 0).any()
        or (ridge_sum <= 0).any()
    ):
        raise ValueError("Некорректные локальные прогнозы")

    local = raw_local * (totals / local_sum)[None, :]
    ridge = raw_ridge * (totals / ridge_sum)[None, :]
    adjustment = 0.0625 * (local - ridge)

    np.testing.assert_allclose(
        adjustment.sum(axis=0), 0., atol=1e-6
    )
    return adjustment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument(
        "--base", type=Path, required=True,
        help="Лучший submission_model_019b.csv"
    )
    parser.add_argument(
        "--tcn", type=Path, required=True,
        help="Сохранённый model019a.pt от 019b"
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path(__file__).parent / "refitted/model020a_adapter.pt"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("submissions/submission_model_020a.csv")
    )
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    for key in ("train", "test", "base", "tcn"):
        if not getattr(args, key).is_file():
            parser.error(
                f"Файл {key} не найден: {getattr(args, key)}"
            )

    if args.predict_only and not args.checkpoint.is_file():
        parser.error(
            f"Не найден адаптер: {args.checkpoint}"
        )

    inputs = {
        getattr(args, key).resolve()
        for key in ("train", "test", "base", "tcn")
    }
    outputs = {args.output.resolve()}
    if not args.predict_only:
        outputs.add(args.checkpoint.resolve())

    if (
        inputs & outputs
        or args.output.resolve() == args.checkpoint.resolve()
    ):
        parser.error("Выходные файлы должны отличаться от входных")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    torch.manual_seed(42)
    np.random.seed(42)

    values, times, routes = old.load_history(args.train)
    test = old.read_table(
        args.test, ["id", "route_id", "timestamp"]
    )
    best = pd.read_csv(args.base, dtype={"id": str})
    incumbent = old.baseline_matrix(
        test, best, times, routes
    )[0]
    totals = incumbent.sum(0, dtype=np.float64)

    history = old.RawHistory(values, times, routes)
    frozen, norm, state = old.load_checkpoint(args.tcn)

    if times[0] != pd.Timestamp("2025-07-28 00:00:00"):
        raise ValueError("Нужен полный train начиная с 28 июля")
    if state["origin_index"] > len(times) - 1:
        raise ValueError("Checkpoint обучен позже доступной истории")

    np.testing.assert_array_equal(
        routes, np.asarray(state["routes"])
    )
    model = LongPatchAllocator(frozen)

    if args.predict_only:
        saved = torch.load(
            args.checkpoint,
            weights_only=True,
            map_location="cpu"
        )
        if saved["base_checkpoint_sha256"] != file_hash(args.tcn):
            raise ValueError(
                "Адаптер относится к другому исходному checkpoint"
            )
        if saved["train_origin"] > len(times) - 1:
            raise ValueError("Адаптер обучен позже доступной истории")
        model.adapter.load_state_dict(saved["adapter"])
    else:
        print(
            "Обучение недельного адаптера: 8 эпох. "
            "Исходная TCN заморожена.",
            flush=True
        )
        records = fit_adapter(model, history, norm)
        args.checkpoint.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            dict(
                adapter=model.adapter.state_dict(),
                base_checkpoint_sha256=file_hash(args.tcn),
                train_origin=len(times) - 1,
                history_start=str(times[0]),
                routes=routes.tolist(),
                records=records
            ),
            args.checkpoint
        )

    data = groups(
        history, [len(times) - 1], norm, labels=False
    )
    model.eval()
    with torch.no_grad():
        old_scores = frozen(data)[0].numpy().astype(float)
        new_scores = model(data)[0].numpy().astype(float)

    known = data["known"][0].numpy().astype(float)
    old_temporal = old.allocate(old_scores, known, totals)
    new_temporal = old.allocate(new_scores, known, totals)

    prediction = (
        incumbent + 0.5 * (new_temporal - old_temporal)
    )

    print("Локальная поправка...", flush=True)
    prediction += local_adjustment(
        values, times, history.half, totals
    )

    if (
        not np.isfinite(prediction).all()
        or (prediction < 0).any()
    ):
        raise ValueError(
            "Некорректный прогноз: проверь соответствие "
            "исходного CSV и checkpoint модели 019b"
        )

    np.testing.assert_allclose(
        prediction.sum(0), totals, rtol=1e-12
    )
    submission = old.make_submission(
        test, best, prediction, times, routes
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    print(
        f"Сохранено {len(submission)} прогнозов: "
        f"{args.output.resolve()}",
        flush=True
    )


if __name__ == "__main__":
    main()