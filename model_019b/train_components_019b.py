"""Обучение компонентов 019B на полном train."""

from pathlib import Path
import argparse
import gc
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

import predict_model_019b as recipe


def main():
    here = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=(
            recipe.find_root()
            / "data/raw/train_solo_track.parquet"
        ),
    )
    parser.add_argument(
        "--outputdir",
        type=Path,
        default=here / "refitted",
    )
    parser.add_argument(
        "--component",
        choices=["both", "cat", "tcn"],
        default="both",
    )
    args = parser.parse_args()

    if not args.train.is_file():
        parser.error(
            f"Обучающий файл не найден: {args.train}"
        )

    args.outputdir.mkdir(parents=True, exist_ok=True)

    recipe.torch.set_num_threads(1)
    try:
        recipe.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    values, times, routes = recipe.load_history(args.train)

    if times[0] != pd.Timestamp("2025-07-28 00:00:00"):
        raise ValueError(
            "Нужен полный исходный train начиная с 28 июля"
        )

    np.testing.assert_array_equal(routes, np.arange(1000))

    if args.component in ("both", "cat"):
        cutoff = int(
            times.get_indexer(
                [pd.Timestamp("2025-10-30 10:30:00")]
            )[0]
        )
        if cutoff < 337:
            raise ValueError(
                "В train отсутствует граница "
                "2025-10-30 10:30:00"
            )

        print(
            "Подготовка 800000 обучающих строк CatBoost...",
            flush=True,
        )

        engine = recipe.CatFeatures(
            values[:cutoff + 1],
            times[:cutoff + 1],
            routes,
        )

        end = cutoff - 8
        rng = np.random.default_rng(42)

        t = np.r_[
            rng.integers(
                max(337, end - 21 * 48),
                end + 1,
                size=400000,
            ),
            rng.integers(
                max(337, end - 84 * 48),
                end + 1,
                size=400000,
            ),
        ]
        r = rng.integers(0, len(routes), size=len(t))
        h = rng.integers(1, 9, size=len(t))

        assert (t + h <= cutoff).all()

        x, scale, offset = engine.features(t, r, h)

        y = np.maximum(
            (engine.v[t + h, r, 0] - offset) / scale,
            0,
        )
        weights = (
            scale / scale.mean()
            * np.exp(
                -np.log(2.0) * (cutoff - t) / 48.0 / 28.0
            )
        )

        del engine, t, r, h, scale, offset
        gc.collect()

        params = dict(
            loss_function="MAE",
            iterations=1400,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=20.0,
            random_strength=0.2,
            bootstrap_type="Bernoulli",
            subsample=0.8,
            border_count=64,
            thread_count=1,
            random_seed=42,
            allow_writing_files=False,
            verbose=100,
        )

        print("Обучение CatBoost...", flush=True)
        model = CatBoostRegressor(**params)
        model.fit(
            x,
            y,
            sample_weight=weights,
            cat_features=["route_id"],
        )

        path = args.outputdir / "long_cat_1030_1030.cbm"
        model.save_model(str(path))
        print("Сохранено:", path.resolve(), flush=True)

        del x, y, weights, model
        gc.collect()

    if args.component in ("both", "tcn"):
        print(
            "Обучение TCN: 34 эпохи...",
            flush=True,
        )

        history = recipe.RawHistory(values, times, routes)

        model, normalization, records, seconds = (
            recipe.train_model(
                history,
                len(times) - 1,
                34,
            )
        )

        path = args.outputdir / "model019a.pt"
        recipe.save_checkpoint(
            path,
            model,
            normalization,
            history,
            records,
        )

        report = dict(
            config=recipe.CONFIG,
            epochs=34,
            fit_seconds=seconds,
            history=records,
            origin=str(times[-1]),
            public_score=None,
        )

        (args.outputdir / "tcn_training.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print("Сохранено:", path.resolve(), flush=True)


if __name__ == "__main__":
    main()