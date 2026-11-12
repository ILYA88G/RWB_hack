"""Generate and log a one-dimensional Public Score calibration."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "public_scale"
BASE = Path(__file__).resolve().parents[1] / "submissions/submission_model_020a.csv"


def make(factor, baseline=BASE, name="020a"):
    OUT.mkdir(exist_ok=True)
    source = pd.read_csv(baseline, dtype={"id": str})
    assert list(source) == ["id", "y_pred"] and len(source) == 8000 and source.id.is_unique
    values = source.y_pred.to_numpy(dtype=float)
    assert np.isfinite(values).all() and values.min() >= 0 and factor > 0
    source["y_pred"] = values * factor
    path = OUT / f"{name}_scale_{factor:.7f}.csv"
    source.to_csv(path, index=False)
    reread = pd.read_csv(path, dtype={"id": str})
    np.testing.assert_array_equal(reread.id, source.id)
    np.testing.assert_allclose(reread.y_pred, values*factor, rtol=1e-13)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(path.resolve(), digest)
    return path


def record(factor, score, name="020a"):
    OUT.mkdir(exist_ok=True)
    path = OUT/"scores.csv"
    rows = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "factor", "score"])
    prior = rows[(rows.model == name) & np.isclose(rows.factor.astype(float), factor, rtol=0, atol=1e-10)]
    if not prior.empty:
        np.testing.assert_allclose(prior.score.astype(float), score, atol=1e-8, rtol=0)
        return
    rows.loc[len(rows)] = [name, factor, score]
    rows.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factor", type=float)
    parser.add_argument("--score", type=float)
    parser.add_argument("--baseline", type=Path, default=BASE)
    parser.add_argument("--name", default="020a")
    args = parser.parse_args()
    if args.score is None:
        make(args.factor, args.baseline, args.name)
    else:
        record(args.factor, args.score, args.name)
