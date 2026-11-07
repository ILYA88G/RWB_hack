from pathlib import Path
import hashlib
import json

import pandas as pd


def export_parts(source, output, days=37, limit=24 * 1024**2):
    columns = [
        "route_id", "timestamp", "target_1h",
        *[f"status_{i}" for i in range(1, 7)],
    ]

    train = pd.read_parquet(source, columns=columns)
    train["timestamp"] = pd.to_datetime(
        train["timestamp"], errors="raise"
    )

    latest = train.timestamp.max()
    cutoff = latest - pd.Timedelta(days=days)
    train = train.loc[train.timestamp >= cutoff].copy()

    if train.empty:
        raise ValueError("Выбранный период обучения пуст")

    output.mkdir(parents=True, exist_ok=True)
    parts = []

    def save(frame):
        path = output / f"train_part_{len(parts) + 1:03d}.parquet"
        frame.to_parquet(
            path, index=False, compression="zstd"
        )

        size = path.stat().st_size

        if size > limit:
            path.unlink()
            middle = len(frame) // 2

            if middle == 0:
                raise ValueError("Одна строка превышает лимит размера")

            save(frame.iloc[:middle])
            save(frame.iloc[middle:])
        else:
            parts.append({
                "file": path.name,
                "rows": len(frame),
                "bytes": size,
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            })

    routes = train.route_id.unique()

    for start in range(0, len(routes), 100):
        selected = routes[start:start + 100]
        save(train.loc[train.route_id.isin(selected)])

    manifest = {
        "rows": len(train),
        "routes": len(routes),
        "start": str(train.timestamp.min()),
        "end": str(latest),
        "parts": parts,
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Готово: {output.resolve()}")
    print(
        f"Строк: {len(train)}, маршрутов: {len(routes)}, "
        f"частей: {len(parts)}"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    export_parts(
        root / "data" / "raw" / "train_solo_track.parquet",
        root / "train_for_model",
    )