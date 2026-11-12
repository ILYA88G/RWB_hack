"""Replay the fixed 25% long CatBoost proxy on folds 3–5 (no model fitting)."""
import argparse
import ast
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "exp_021"))
import improve_network as net
from backtest_network import load_folds, metric

ORIGINS = pd.to_datetime(["2025-10-30 10:30", "2025-10-31 10:30", "2025-11-01 06:30"])
H = np.arange(1, 9)
ARCHIVE = OUT / "long_cat_folds.npz"
WEIGHTS = ROOT / "model_019b/refitted/long_cat_1030_1030.cbm"


def load_recipe():
    """Load just the original reader and feature definitions, without importing torch."""
    source = ROOT / "model_019b/predict_model_019b.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"read_table", "load_history", "halfhour_causal", "BaseCatFeatures", "CatFeatures"}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(np=np, pd=pd, Path=Path, VALUES=[net.TARGET, *net.STATUS], STEP=net.STEP)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), env)
    return env["load_history"], env["CatFeatures"]


def load_model():
    from catboost import CatBoostRegressor
    manifest = pd.read_csv(net.EXP / "input_manifest.csv")
    record, = manifest.loc[manifest.path.str.replace(chr(92), "/", regex=False)
                           == WEIGHTS.relative_to(ROOT).as_posix()].itertuples()
    assert WEIGHTS.stat().st_size == record.bytes
    with WEIGHTS.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == record.sha256
    model = CatBoostRegressor().load_model(str(WEIGHTS))
    params = model.get_params()
    assert (model.tree_count_, params["depth"], params["loss_function"], params["random_seed"]) == (1400, 6, "MAE", 42)
    assert model.get_cat_feature_indices() == [model.feature_names_.index("route_id")]
    return model


def fixed_network_forecast(prefix):
    """Evaluate the one prediction of exp_021 Ridge(alpha=180) as a linear smoother.

    Compute the weights of observed residuals on this query directly. The
    linear system depends only on historical covariates and the query.
    """
    x, anchor = net.features(prefix)
    end = len(x) - 8
    rows = slice(max(344, end - 42 * 48), end)
    past = x.iloc[rows].to_numpy(dtype=float)
    center = past.mean(axis=0)
    scale = past.std(axis=0)
    scale[scale == 0] = 1
    z = (past - center) / scale
    query = (x.iloc[-1].to_numpy(dtype=float) - center) / scale
    gram = z.T @ z + 180 * np.eye(z.shape[1])
    weights = z @ np.linalg.solve(gram, query)
    observed = np.lib.stride_tricks.sliding_window_view(prefix[net.TARGET].to_numpy()[1:], 8)
    residual = observed[rows] - anchor.iloc[rows].to_numpy()
    average = residual.mean(axis=0)
    return anchor.iloc[-1].to_numpy() + average + weights @ (residual - average)


def baseline(origins, old, series, panel):
    rule = json.loads((net.EXP / "rule.json").read_text())
    assert rule == dict(model="ridge", mode="shape", alpha=0.15, cap=0.02)
    base = []
    for origin, matrix in zip(origins, old, strict=True):
        sums = matrix.sum(axis=0)
        forecast = fixed_network_forecast(series.loc[:origin])
        proposal = net.proposed_totals(sums, forecast, rule["mode"], rule["alpha"], rule["cap"])
        totals, _ = net.safe_totals(sums, proposal, net.q0_at(panel, origin))
        base.append(matrix * (totals / sums))
    return np.stack(base)


def predict(model, builder, values, times, routes):
    assert times[0] == pd.Timestamp("2025-07-28") and times[-1] in ORIGINS
    engine = builder(values, times, routes)
    r = np.repeat(np.arange(len(routes)), 8)
    h = np.tile(H, len(routes))
    features, scale, offset = engine.features(np.full(len(r), len(times) - 1), r, h)
    assert model.feature_names_ == list(features.columns)
    np.testing.assert_array_equal(features.route_id, routes[r])
    np.testing.assert_array_equal(features.horizon, h)
    raw = np.maximum(model.predict(features, thread_count=1), 0) * scale + offset
    return raw.reshape(len(routes), 8)


def normalize(raw, totals):
    sums = raw.sum(axis=-2, dtype=np.float64)
    assert np.isfinite(raw).all() and raw.min() >= 0 and sums.min() > 0
    return raw * (totals / sums)[..., None, :]


def summarize(origins, truth, base, cat, mix, quartiles):
    rows = []
    for label, selection in [(str(i+3), slice(i, i+1)) for i in range(3)] + [("pooled", slice(None))]:
        y, b, c, m = (a[selection] for a in (truth, base, cat, mix))
        groups = {"all": np.ones(y.shape, dtype=bool)}
        for h in H:
            groups[f"h{h}"] = np.broadcast_to(H == h, y.shape)
        for q in range(1, 5):
            groups[f"Q{q}"] = np.broadcast_to(quartiles[selection, :, None] == q, y.shape)
        total_error = np.abs(b-y).sum()
        for group, mask in groups.items():
            base_error = np.abs(b[mask]-y[mask]).sum()
            for name, matrix in (("B", b), ("C", c), ("M", m)):
                error = np.abs(matrix[mask]-y[mask]).sum()
                delta = error - base_error
                rows.append(dict(fold=label, origin="|".join(map(str, origins[selection])),
                                 group=group, method=name, n=mask.sum(),
                                 **metric(y[mask], matrix[mask]), abs_error=error, delta_ae=delta,
                                 delta_over_group_B=delta/base_error, group_share_B=base_error/total_error,
                                 delta_over_total_B=delta/total_error, mean_abs_C_B=np.abs(c[mask]-b[mask]).mean()))
    return pd.DataFrame(rows)


def archive_wape(path):
    """Independent replay from archived raw C and budgets, plus existing OOF."""
    ys, mixtures = [], []
    with np.load(path, allow_pickle=False) as a:
        np.testing.assert_array_equal(a["origin"], ORIGINS.to_numpy(dtype="datetime64[ns]"))
        np.testing.assert_array_equal(a["fold"], [3, 4, 5])
        np.testing.assert_array_equal(a["route_id"], np.arange(1000))
        np.testing.assert_array_equal(a["horizon"], H)
        assert a["cat_raw"].shape == (3, 1000, 8) and a["B_totals"].shape == (3, 8)
        for i, fold in enumerate(a["fold"]):
            with np.load(ROOT / f"reports/exp_005/ensemble_005b_fold_{fold}.npz") as old, np.load(
                ROOT / f"reports/exp_006/oof_fold_{fold}.npz") as new:
                assert pd.Timestamp(old["origin"].item()) == pd.Timestamp(new["origin"].item()) == ORIGINS[i]
                share = np.maximum(0.5*old["lgbm_pred"].astype(float) + 0.5*new["new_pred"].astype(float), 0)
                ys.append(old["y_true"])
            b = normalize(share, a["B_totals"][i])
            c = normalize(a["cat_raw"][i], a["B_totals"][i])
            mixtures.append(0.75*b + 0.25*c)
    return metric(ys, mixtures)["wape"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Replay the archive against the saved OOF")
    args = parser.parse_args()
    if args.verify_only:
        value = archive_wape(ARCHIVE)
        table = pd.read_csv(OUT / "experiments.csv")
        expected = table.query("fold == 'pooled' and method == 'M' and group == 'all'").wape.item()
        np.testing.assert_allclose(value, expected, atol=1e-14, rtol=0)
        print(f"Archive M pooled WAPE {value:.12f}")
        return

    start = time.perf_counter()
    model = load_model()
    read_history, builder = load_recipe()
    values, times, routes = read_history(ROOT.parent / "rwb_case_data/train_solo_track.parquet")
    np.testing.assert_array_equal(routes, np.arange(1000))
    assert times.equals(pd.date_range("2025-07-28", periods=4630, freq=net.STEP).as_unit("ns"))
    panel = pd.DataFrame(values[:, :, 0], index=times, columns=routes)
    series = pd.DataFrame(values.sum(axis=1, dtype=float), index=times, columns=[net.TARGET, *net.STATUS])
    origins, truth, _, old = load_folds(panel, ROOT / "reports")
    assert pd.DatetimeIndex(origins).equals(ORIGINS.as_unit("ns"))
    truth = truth.astype(float)
    base = baseline(origins, old, series, panel)
    np.testing.assert_allclose(metric(truth, old)["score"], 0.3302594869, rtol=0, atol=5e-11)
    np.testing.assert_allclose(metric(truth, base)["wape"], 0.3241312979, rtol=0, atol=5e-11)
    # All 24 historical budgets are checked separately using archived evaluation metrics.
    prior = pd.read_csv(ROOT / "exp_023/experiments.csv")
    raw, quartiles = [], []
    for i, origin in enumerate(origins):
        for h in H:
            ref = prior[(prior["sample"] == f"fold{i+3}") & (prior.method == "base") & (prior.group == f"h{h}")]
            bias = metric(truth[i, :, h-1], base[i, :, h-1])["signed_bias"]
            np.testing.assert_allclose(bias, ref.signed_bias.item(), rtol=0, atol=1e-12)
        end = times.get_loc(origin) + 1
        volume = values[end-336:end, :, 0].mean(axis=0, dtype=float)
        quartiles.append(pd.qcut(pd.Series(volume).rank(method="first"), 4, labels=False).to_numpy()+1)
        raw.append(predict(model, builder, values[:end], times[:end], routes))
        gc.collect()
        print(f"Fold {i+3} complete", flush=True)

    raw = np.stack(raw)
    totals = base.sum(axis=1)
    cat = normalize(raw, totals)
    mix = 0.75*base + 0.25*cat
    for matrix in (base, cat, mix):
        assert np.isfinite(matrix).all() and matrix.min() >= 0
        np.testing.assert_allclose(matrix.sum(axis=1), totals, rtol=1e-12, atol=0)
        for y, b, p in zip(truth, base, matrix, strict=True):
            np.testing.assert_allclose(metric(y, b)["signed_bias"], metric(y, p)["signed_bias"], atol=1e-12, rtol=0)
    table = summarize(origins, truth, base, cat, mix, np.stack(quartiles))
    np.savez_compressed(ARCHIVE, fold=np.arange(3, 6), origin=ORIGINS.to_numpy(dtype="datetime64[ns]"),
                        route_id=routes, horizon=H, cat_raw=raw, B_totals=totals)
    value = archive_wape(ARCHIVE)
    np.testing.assert_allclose(value, metric(truth, mix)["wape"], atol=1e-14, rtol=0)
    table.to_csv(OUT / "experiments.csv", index=False)
    print(f"Archive M pooled WAPE {value:.12f}. Elapsed {time.perf_counter()-start:.2f} s.")


if __name__ == "__main__":
    main()
