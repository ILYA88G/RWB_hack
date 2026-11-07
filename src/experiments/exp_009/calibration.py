from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np


SCALES = (1.0, 0.98, 1.02, 0.96, 1.04, 0.94, 1.06)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Точная метрика соревнования: WAPE + абсолютный Relative Bias."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    if y.shape != p.shape or not y.size:
        raise ValueError("Нужны непустые массивы одинаковой формы")

    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("В метрику переданы NaN или бесконечности")

    if (y < 0).any() or y.sum() <= 0:
        raise ValueError("Цели должны быть неотрицательными с положительной суммой")

    denominator = float(y.sum())
    wape = float(np.abs(y - p).sum() / denominator)
    bias = float(abs(p.sum() / denominator - 1.0))

    return {
        "wape": wape,
        "bias": bias,
        "score": wape + bias,
    }


def _validate_predictions(
    predictions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not predictions:
        raise ValueError("Нужна хотя бы одна модель")

    if any(
        not isinstance(name, str) or not name
        for name in predictions
    ):
        raise ValueError("Названия моделей должны быть непустыми строками")

    result = {
        name: np.asarray(p, dtype=np.float64)
        for name, p in predictions.items()
    }

    first = next(iter(result.values()))

    if not first.size:
        raise ValueError("Прогнозы не должны быть пустыми")

    for p in result.values():
        if p.shape != first.shape or not np.isfinite(p).all():
            raise ValueError(
                "Формы прогнозов должны совпадать; все значения — конечные"
            )

    return result


def _validate_oof(y, predictions):
    y = np.asarray(y, dtype=np.float64)
    predictions = _validate_predictions(predictions)

    if (
        y.ndim != 3
        or y.shape[0] == 0
        or y.shape[1] == 0
        or y.shape[2] != 8
    ):
        raise ValueError("Форма OOF должна быть (даты, маршруты, 8)")

    if next(iter(predictions.values())).shape != y.shape:
        raise ValueError("Формы ответов и прогнозов должны совпадать")

    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Ответы должны быть конечными и неотрицательными")

    if (y.sum(axis=(1, 2)) <= 0).any():
        raise ValueError("На каждой дате сумма ответов должна быть положительной")

    return y, predictions


def apply_config(
    predictions: Mapping[str, np.ndarray],
    config: dict,
) -> np.ndarray:
    """Применить зафиксированные веса и общий коэффициент."""
    predictions = _validate_predictions(predictions)
    weights = config.get("weights", {})

    if not weights or any(name not in predictions for name in weights):
        raise ValueError("В конфигурации указаны отсутствующие модели")

    values = np.asarray(list(weights.values()), dtype=np.float64)

    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError(
            "Веса должны быть конечными, неотрицательными и давать сумму 1"
        )

    scale = float(config.get("scale", 1.0))

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Коэффициент должен быть конечным и положительным")

    result = np.zeros_like(next(iter(predictions.values())))

    for name, weight in weights.items():
        result += float(weight) * np.maximum(predictions[name], 0.0)

    return np.maximum(result * scale, 0.0)


def _candidates(model_names):
    names = sorted(model_names)

    # Отдельные модели без ансамблирования.
    unscaled = [{name: 1.0} for name in names]

    # Простые попарные ансамбли.
    for left, right in combinations(names, 2):
        for weight in (0.25, 0.5, 0.75):
            unscaled.append({
                left: weight,
                right: 1.0 - weight,
            })

    for weights in unscaled:
        for scale in SCALES:
            yield {
                "weights": weights,
                "scale": scale,
            }


def _summary(y, p):
    per_origin = [
        metric(y[i], p[i])
        for i in range(len(y))
    ]

    return {
        "mean_score": float(np.mean([
            row["score"] for row in per_origin
        ])),
        "worst_score": float(max(
            row["score"] for row in per_origin
        )),
        "pooled": metric(y, p),
        "per_origin": per_origin,
    }


def select_config(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict:
    """Подбор по средней метрике отдельных дат."""
    y, predictions = _validate_oof(y, predictions)
    best_key = None
    best_config = None

    for config in _candidates(predictions):
        p = apply_config(predictions, config)
        summary = _summary(y, p)

        key = (
            summary["mean_score"],
            len(config["weights"]),
            abs(config["scale"] - 1.0),
            tuple(sorted(config["weights"].items())),
            config["scale"],
        )

        if best_key is None or key < best_key:
            best_key = key
            best_config = config

    return best_config


def evaluate_forward(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    min_history: int = 2,
):
    """
    Последняя дата полностью исключается из подбора.

    Для промежуточных проверок параметры выбираются только
    по предыдущим датам.
    """
    y, predictions = _validate_oof(y, predictions)

    if (
        isinstance(min_history, bool)
        or not isinstance(min_history, int)
        or min_history < 2
    ):
        raise ValueError("min_history должен быть целым числом >= 2")

    if len(y) < min_history + 1:
        raise ValueError(
            "Недостаточно дат для подбора и отдельного последнего holdout"
        )

    development = {
        name: p[:-1]
        for name, p in predictions.items()
    }

    config = select_config(y[:-1], development)
    dev_p = apply_config(development, config)

    hold_p = apply_config(
        {name: p[-1] for name, p in predictions.items()},
        config,
    )

    forward_rows = []
    forward_targets = []
    forward_preds = []

    for held in range(min_history, len(y) - 1):
        past = {
            name: p[:held]
            for name, p in predictions.items()
        }

        chosen = select_config(y[:held], past)

        pred = apply_config(
            {name: p[held] for name, p in predictions.items()},
            chosen,
        )

        forward_rows.append({
            "origin_index": held,
            "selection_indices": list(range(held)),
            "config": chosen,
            **metric(y[held], pred),
        })

        forward_targets.append(y[held])
        forward_preds.append(pred)

    forward_summary = None

    if forward_rows:
        forward_summary = _summary(
            np.stack(forward_targets),
            np.stack(forward_preds),
        )

    baselines = {}

    for name, p in predictions.items():
        clipped = np.maximum(p, 0.0)

        baselines[name] = {
            "development": _summary(y[:-1], clipped[:-1]),
            "locked_holdout": metric(y[-1], clipped[-1]),
        }

    report = {
        "frozen_config": config,
        "selection_indices": list(range(len(y) - 1)),
        "development": _summary(y[:-1], dev_p),
        "expanding_forward": forward_rows,
        "expanding_forward_summary": forward_summary,
        "locked_holdout": {
            "origin_index": len(y) - 1,
            **metric(y[-1], hold_p),
        },
        "baseline_metrics": baselines,
        "note": (
            "Последняя дата исключена из подбора. "
            "Development — результат поиска, а не независимая оценка."
        ),
    }

    return config, report