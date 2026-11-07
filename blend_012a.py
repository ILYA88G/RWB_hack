from pathlib import Path
import argparse

import numpy as np
import pandas as pd


WEIGHT_NEW = 0.30


def blend(old, new):
    for name, frame in (("old", old), ("new", new)):
        if frame.columns.tolist() != ["id", "y_pred"] or frame.empty:
            raise ValueError(
                f"{name}: нужны непустые колонки id,y_pred"
            )

        if frame.id.isna().any() or frame.id.astype(str).duplicated().any():
            raise ValueError(
                f"{name}: пропущенные или повторяющиеся ID"
            )

        values = pd.to_numeric(
            frame.y_pred, errors="raise"
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"{name}: некорректные значения прогнозов"
            )

    old = old.copy()
    new = new.copy()

    old["id"] = old.id.astype(str)
    new["id"] = new.id.astype(str)

    if set(old.id) != set(new.id):
        raise ValueError("Наборы ID в двух submission различаются")

    # Совмещаем прогнозы по ID, а не по порядку строк.
    old_values = old.y_pred.to_numpy(dtype=float)

    new_values = (
        new.set_index("id")
        .loc[old.id, "y_pred"]
        .to_numpy(dtype=float)
    )

    return pd.DataFrame({
        "id": old.id.to_numpy(),
        "y_pred": (
            (1 - WEIGHT_NEW) * old_values
            + WEIGHT_NEW * new_values
        ),
    })


def find_root():
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        if (
            root / "results_exp009" / "submission_exp009.csv"
        ).is_file():
            return root

    return Path.cwd()


def main():
    root = find_root()

    old_paths = [
        root / "submissions" / "submission_network_008b.csv",
        root / "submission_network_008b.csv",
    ]

    default_old = next(
        (path for path in old_paths if path.is_file()),
        old_paths[0],
    )

    parser = argparse.ArgumentParser(
        description="Смешивание EXP-008B и EXP-009 без обучения"
    )

    parser.add_argument(
        "--old",
        type=Path,
        default=default_old,
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=root / "results_exp009" / "submission_exp009.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "submissions" / "submission_blend_012a.csv",
    )

    args = parser.parse_args()

    for name, path in (("old", args.old), ("new", args.new)):
        if not path.is_file():
            parser.error(
                f"Файл не найден: {path}. "
                f"Укажи его расположение через --{name}."
            )

    if args.output.resolve() in (
        args.old.resolve(),
        args.new.resolve(),
    ):
        parser.error("Выходной файл должен отличаться от исходных")

    old = pd.read_csv(args.old, dtype={"id": str})
    new = pd.read_csv(args.new, dtype={"id": str})

    try:
        result = blend(old, new)
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"Готово: {args.output.resolve()}")
    print(f"Строк: {len(result)}")
    print("Веса: лучший прогноз — 70%, EXP-009 — 30%")
    print("Public Score этого файла пока неизвестен.")


if __name__ == "__main__":
    main()