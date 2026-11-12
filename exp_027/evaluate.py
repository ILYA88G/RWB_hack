"""Reconcile components, freeze paired predictions, then reveal the eight labels."""
import sys

import numpy as np
import pandas as pd
import torch
from replay import OUT, ROOT, ORIGIN, TRAIN, ROUTES, net, archive, history, metric, save
import predict_model_019b as old
import upgrade_model_020a as patch

sys.path.insert(0, str(ROOT/"exp_025"))
import run as specialist


def assemble(base15, cat, scores, adapted, known, local, final_totals):
    totals = base15.sum(axis=0)
    temporal = old.allocate(scores, known, totals)
    new_temporal = old.allocate(adapted, known, totals)
    m19 = .25*base15 + .25*archive.normalize(cat, totals) + .5*temporal
    m20 = m19 + .5*(new_temporal-temporal) + local
    m21 = archive.normalize(m20, final_totals)
    for m, budget in ((m19, totals), (m20, totals), (m21, final_totals)):
        assert m.shape == (1000, 8) and np.isfinite(m).all() and m.min() >= 0
        np.testing.assert_allclose(m.sum(axis=0), budget, rtol=1e-12)
    return m19, m20, m21


def load(stage):
    with np.load(OUT/f"{stage}.npz", allow_pickle=False) as a:
        assert pd.Timestamp(a["origin"][()]) == ORIGIN
        np.testing.assert_array_equal(a["route_id"], ROUTES)
        return {k: a[k] for k in a.files if k not in ("origin", "route_id")}


def evaluate():
    torch.set_num_threads(1)
    times, v = history()
    net_data = load("network")
    short = load("short")
    tcn = load("tcn")
    sums = net_data["totals"]
    m14 = .5*archive.normalize(short["linear"], sums)+.5*archive.normalize(short["boosted"], sums)
    m15 = .5*m14+.5*archive.normalize(short["cat"], sums)
    half = old.halfhour_causal(v[:, :, 0])
    local = patch.local_adjustment(v, times, half, sums)
    m19, m20, m21 = assemble(m15, load("long")["pred"], tcn["scores"], tcn["adapted"],
                              tcn["known"], local, net_data["final"])
    h1 = specialist.adjust_h1(m21, specialist.first_horizon(v, half))
    # Also check the two previously fixed network alternatives on the full ensemble.
    series = pd.DataFrame(v.sum(axis=1, dtype=float), index=times, columns=[net.TARGET, *net.STATUS])
    _, factor, *_ = specialist.network_factors(series, m21.sum(axis=0), half[-1].sum(dtype=float))
    models = {"008b": net_data["base8"], "014a": m14, "015a": m15, "019b": m19, "020a": m20,
              "021a": m21, "025": h1, "026": m21*factor, "025+026": h1*factor}
    save("predictions", horizon=archive.H, **models)
    frozen = load("predictions")
    # Separate read, strictly after prediction files have been written.
    future = pd.read_parquet(TRAIN, columns=["timestamp", "route_id", net.TARGET],
                             filters=[("timestamp", ">", ORIGIN), ("timestamp", "<=", ORIGIN+8*net.STEP)])
    target_times = pd.date_range(ORIGIN+net.STEP, periods=8, freq=net.STEP).as_unit("ns")
    labels = future.pivot(index="route_id", columns="timestamp", values=net.TARGET)
    np.testing.assert_array_equal(labels.index, ROUTES)
    assert labels.columns.as_unit("ns").equals(target_times)
    y = labels.to_numpy(dtype=float)
    q = pd.qcut(pd.Series(v[-336:, :, 0].mean(axis=0, dtype=float)).rank(method="first"), 4, labels=False).to_numpy()+1
    rows = []
    groups = {"all": np.s_[:, :], **{f"h{i+1}": np.s_[:, i] for i in range(8)},
              **{f"Q{i}": np.s_[q == i, :] for i in range(1, 5)}}
    for name in models:
        for group, idx in groups.items():
            p = frozen[name][idx]
            ae = np.abs(p-y[idx]).sum()
            ref = metric(y[idx], m21[idx])
            stats = metric(y[idx], p)
            rows.append(dict(model=name, group=group, n=p.size, **stats, abs_error=ae,
                             delta_score=stats["score"]-ref["score"], delta_wape=stats["wape"]-ref["wape"],
                             truth_sum=y[idx].sum(), pred_sum=p.sum()))
    table = pd.DataFrame(rows)
    table.to_csv(OUT/"metrics.csv", index=False)
    print(table.query("group=='all'").to_string(index=False))


def verify_final():
    """Check the actual archived outputs downstream of 015a, without test labels."""
    from catboost import CatBoostRegressor
    from replay import CHANNELS
    torch.set_num_threads(1)
    v, times, routes = old.load_history(TRAIN)
    test = net.validate_grid(pd.read_parquet(net.RAW/"test_solo_track.parquet"), routes, times[-1])
    def csv_matrix(path):
        data = pd.read_csv(path, dtype={"id": str})
        matrix = np.empty((1000, 8))
        matrix[pd.Index(routes).get_indexer(test.route_id), test.h] = net.aligned(test, data)
        return matrix
    s15 = csv_matrix(ROOT/"submissions/submission_model_015a.csv")
    weights = ROOT/"model_019b/refitted"
    cat = CatBoostRegressor().load_model(str(weights/"long_cat_1030_1030.cbm"))
    engine = old.CatFeatures(v, times, routes)
    r = np.repeat(routes, 8)
    h = np.tile(archive.H, len(routes))
    x, scale, offset = engine.features(np.full(r.size, len(times)-1), r, h)
    assert list(x.columns) == cat.feature_names_
    raw = (np.maximum(cat.predict(x, thread_count=1), 0)*scale+offset).reshape(-1, 8)
    del engine
    hist = old.RawHistory(v, times, routes)
    model, norm, _ = old.load_checkpoint(weights/"model019a.pt")
    scores, known = old.predict_scores(model, hist, norm)
    adapter = patch.LongPatchAllocator(model)
    state = torch.load(weights/"model020a_adapter.pt", map_location="cpu", weights_only=True)
    assert state["base_checkpoint_sha256"] == patch.file_hash(weights/"model019a.pt")
    adapter.adapter.load_state_dict(state["adapter"])
    adapter.eval()
    with torch.no_grad():
        data = patch.groups(hist, [len(times)-1], norm, labels=False)
        adapted = adapter(data)[0].numpy().astype(float)
    sums = s15.sum(axis=0)
    series = pd.DataFrame(v.sum(axis=1, dtype=float), index=times, columns=CHANNELS)
    requested = net.proposed_totals(sums, archive.fixed_network_forecast(series), "shape", .15, .02)
    budgets, _ = net.safe_totals(sums, requested, hist.half[-1].sum(dtype=float))
    matrices = assemble(s15, raw, scores, adapted, known, patch.local_adjustment(v, times, hist.half, sums), budgets)
    rows = []
    for name, mat, folder in zip(("019b", "020a", "021a"), matrices, ("submissions", "submissions", "exp_021")):
        expected = csv_matrix(ROOT/folder/f"submission_model_{name}.csv")
        diff = np.abs(mat-expected)
        rows.append(dict(model=name, n=diff.size, mean_abs=diff.mean(), max_abs=diff.max(),
                         relative_l1=diff.sum()/expected.sum()))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(OUT/"final_recipe_check.csv", index=False)
    # CPU-library roundoff is permitted; the relative tolerances apply on all 8000 rows.
    assert table.relative_l1.max() < 1e-5 and table.max_abs.max() < 25


def verify_upstream():
    from replay import network_components, short_components
    origin = pd.Timestamp("2025-11-01 10:30")
    test = net.validate_grid(pd.read_parquet(net.RAW/"test_solo_track.parquet"), ROUTES, origin)
    network = network_components(origin)
    sums = network["totals"]
    linear, boosted, cat = short_components(origin, ROOT/"submissions/model_015a.cbm")
    m14 = .5*archive.normalize(linear, sums) + .5*archive.normalize(boosted, sums)
    m15 = .5*m14 + .5*archive.normalize(cat, sums)
    results = []
    for stage, p, file in [("008b", network["base8"], "submission_network_008b.csv"),
                           ("014a", m14, "submission_model_014a.csv"),
                           ("015a", m15, "submission_model_015a.csv")]:
        sub = pd.read_csv(ROOT/"submissions"/file, dtype={"id": str})
        reference = net.aligned(test, sub)
        diff = np.abs(p[test.route_id, test.h] - reference)
        results.append(dict(model=stage, n=diff.size, mean_abs=diff.mean(), max_abs=diff.max(),
                            relative_l1=diff.sum()/reference.sum()))
    table = pd.DataFrame(results)
    table.to_csv(OUT/"upstream_recipe_check.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="store_true")
    if parser.parse_args().upstream:
        verify_upstream()
    else:
        verify_final()
