"""One locked full-recipe fold. Run stages tcn, long, short, network, evaluate."""
import argparse
import gc
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path[:0] = [str(ROOT/"model_019b"), str(ROOT/"exp_024"), str(ROOT)]
import long_cat as archive
from backtest_network import metric

net = archive.net
ORIGIN = pd.Timestamp("2025-10-25 10:30")
LONG_END = ORIGIN - pd.Timedelta(days=2)
TRAIN = net.RAW/"train_solo_track.parquet"
CHANNELS = [net.TARGET, *net.STATUS]
ROUTES = np.arange(1000)


def frame(end=ORIGIN, start=None, columns=None):
    filters = [("timestamp", "<=", end)]
    if start is not None:
        filters.append(("timestamp", ">=", start))
    return pd.read_parquet(TRAIN, filters=filters, columns=columns or ["timestamp", "route_id", *CHANNELS])


def history(start=None, end=ORIGIN):
    df = frame(start=start, end=end).sort_values(["timestamp", "route_id"])
    times = pd.date_range(start or "2025-07-28", end, freq=net.STEP).as_unit("ns")
    keys = pd.MultiIndex.from_frame(df[["timestamp", "route_id"]])
    assert keys.equals(pd.MultiIndex.from_product([times, ROUTES]))
    values = df[CHANNELS].to_numpy(np.float32).reshape(len(times), 1000, 7)
    assert np.isfinite(values).all() and values.min() >= 0
    return times, values


def save(stage, **arrays):
    for a in arrays.values():
        assert np.isfinite(a).all()
    tmp = OUT/f"{stage}.tmp.npz"
    np.savez_compressed(tmp, origin=ORIGIN.to_datetime64(), route_id=ROUTES, **arrays)
    tmp.replace(OUT/f"{stage}.npz")


def short_components(origin, stored_cat=None):
    import train_model_014a as mod
    import train_model_015a as short_cat
    start = origin-pd.Timedelta(days=37)
    times, v = history(start, origin)
    linear, _ = mod.train_linear(v, times)
    engine = short_cat.Engine(v, times, ROUTES)
    # 014a LightGBM uses the base feature set, 015a CatBoost adds 12 columns.
    base_engine = types.SimpleNamespace(times=times, routes=ROUTES, v=v,
                                        features=mod.Engine.features.__get__(engine))
    boosted, _ = mod.train_boosted(base_engine, 2)
    if stored_cat is None:
        cat, _ = short_cat.train_predict_cat(engine)
    else:
        from catboost import CatBoostRegressor
        cat = short_cat.predict(CatBoostRegressor().load_model(str(stored_cat)), engine)
    return linear, boosted, cat


def short():
    linear, boosted, cat = short_components(ORIGIN)
    save("short", linear=linear, boosted=boosted, cat=cat)


def long():
    from train_components_019b import fit_long_cat
    from predict_model_019b import CatFeatures
    times, v = history()
    cutoff = times.get_loc(LONG_END)
    model = fit_long_cat(v, times, ROUTES, cutoff)
    engine = CatFeatures(v, times, ROUTES)
    r = np.repeat(ROUTES, 8)
    h = np.tile(np.arange(1, 9), len(ROUTES))
    features, scale, offset = engine.features(np.full(r.size, len(times)-1), r, h)
    assert model.feature_names_ == list(features) and model.tree_count_ == 1400
    pred = (np.maximum(model.predict(features, thread_count=1), 0)*scale+offset).reshape(1000, 8)
    save("long", pred=pred)


def tcn():
    import torch
    import predict_model_019b as old
    import upgrade_model_020a as patch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    times, v = history()
    hist = old.RawHistory(v, times, ROUTES)
    model, norm, records, _ = old.train_model(hist, len(times)-1, 34)
    assert max(row["label_max"] for row in records) < len(times)
    scores, known = old.predict_scores(model, hist, norm)
    torch.manual_seed(42)
    np.random.seed(42)
    # Original 020a reloads the frozen TCN, consuming its initialization RNG.
    old.TemporalAllocator(len(norm["smean"]), len(ROUTES))
    adapter = patch.LongPatchAllocator(model)
    records = patch.fit_adapter(adapter, hist, norm)
    assert max(row["latest_training_label"] for row in records) < len(times)
    data = patch.groups(hist, [len(times)-1], norm, labels=False)
    with torch.no_grad():
        adapted = adapter(data)[0].numpy().astype(float)
    save("tcn", scores=scores, known=known, adapted=adapted)


def network_components(origin):
    from src.experiments.exp_005 import train_lgbm_005a as five
    from src.experiments.exp_006 import train_lgbm_006a as six
    import train_network_calibration_008b as eight
    # Thread count is a resource limit; all feature and learner settings are
    # the original ones. Pass only completed labels to each fit.
    five.LGBM_PARAMS["n_jobs"] = six.PARAMS["n_jobs"] = 4
    df = frame(start=origin-pd.Timedelta(days=24), end=origin)
    pieces = []
    for build, fit_fn in [(five.build_features, five.train_lightgbm), (six.build_features, six.train_predict)]:
        work, cols = build(df, [origin])
        fit = five.select_training_rows(work, origin)
        query = work.loc[work.timestamp == origin].sort_values("route_id")
        np.testing.assert_array_equal(query.route_id.to_numpy(), ROUTES)
        assert fit.timestamp.max()+8*net.STEP <= origin
        pieces.append(fit_fn(fit, query, cols))
        del work, fit, query
        gc.collect()
    del df
    six_matrix = .5*pieces[0] + .5*pieces[1]
    times, v = history(end=origin)
    series = pd.DataFrame(v.sum(axis=1, dtype=float), index=times.rename("timestamp"), columns=CHANNELS)
    design, cols = eight.build_network(series.reset_index())
    forecast = eight.predict_network(design, cols, origin, "lgbm28")
    base8, _ = eight.corrected(six_matrix, forecast, .8, "per_step")
    totals = base8.sum(axis=0)
    import json
    rule = json.loads((net.EXP/"rule.json").read_text())
    assert rule == dict(model="ridge", mode="shape", alpha=.15, cap=.02)
    proposal = net.proposed_totals(totals, archive.fixed_network_forecast(series), "shape", .15, .02)
    q0 = net.q0_at(pd.DataFrame(v[:, :, 0], index=times), origin)
    final, _ = net.safe_totals(totals, proposal, q0)
    return dict(base8=base8, totals=totals, final=final)


def network():
    save("network", **network_components(ORIGIN))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["short", "long", "tcn", "network", "evaluate"])
    stage = parser.parse_args().stage
    started = time.monotonic()
    if stage == "evaluate":
        from evaluate import evaluate
        evaluate()
    else:
        {"tcn": tcn, "short": short, "long": long, "network": network}[stage]()
    print(f"{stage} finished in {time.monotonic()-started:.1f} seconds", flush=True)
