"""Apply the frozen, holdout-tested deltas to the actual contest 020a forecast."""
import gc
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor

from components import OUT, ROOT, ROUTES, TEST, folder, long_cat, origin, read_stage, replay
import predict_model_019b as old
import upgrade_model_020a as patch
from experiment import BASE, local_components, network_factors, predict, round_trip
from temporal import budget_provider, temporal


def verify_contest_export():
    stamp = origin(TEST)
    test = replay.net.validate_grid(pd.read_parquet(replay.net.RAW/"test_solo_track.parquet"), ROUTES, stamp)
    ref = pd.read_csv(ROOT/"submissions/submission_model_020a.csv", dtype={"id": str})
    p = np.empty((len(ROUTES), 8))
    p[test.route_id, test.h] = replay.net.aligned(test, ref)
    totals = p.sum(axis=0)
    series, panel = replay.net.load_train(replay.TRAIN)
    assert series.index[-1] == stamp
    factor, _, _, _ = network_factors(series, totals, replay.net.q0_at(panel, stamp))
    csv = folder(TEST) / "network_export.csv"
    round_trip(TEST, p*factor, totals*factor, csv)
    reread = replay.net.aligned(test, pd.read_csv(csv, dtype={"id": str}))
    check = dict(origin=str(stamp), n=len(reread), network_factor=float(factor),
                 baseline_totals=totals.tolist(), target_totals=(totals*factor).tolist(),
                 exported_totals=np.bincount(test.h, weights=reread).tolist())
    (csv.parent/"export_check.json").write_text(json.dumps(check, indent=2))


def export_choice(choice, filename):
    stamp = origin(TEST)
    test = replay.net.validate_grid(pd.read_parquet(replay.net.RAW/"test_solo_track.parquet"), ROUTES, stamp)
    reference = ROOT/"submissions/submission_model_020a.csv"
    submission = pd.read_csv(reference, dtype={"id": str})
    base = np.empty((len(ROUTES), 8))
    base[test.route_id, test.h] = replay.net.aligned(test, submission)
    totals = base.sum(axis=0)
    data = dict(base=base, totals=totals)
    if choice == BASE:
        print("Retain", reference, flush=True)
        return reference
    keys = choice.split("+")
    if "fresh" in keys:
        long_cat(TEST, fresh=True)
    if "early" in keys:
        temporal(TEST)
    times, values = replay.history(end=stamp)
    weights = ROOT / "model_019b/refitted"
    if {"lag", "lag_half", "network"}.intersection(keys):
        _, data["lag"], data["network_factor"], _ = local_components(times, values, totals)
    if "fresh" in keys:
        model = CatBoostRegressor().load_model(str(weights/"long_cat_1030_1030.cbm"))
        engine = old.CatFeatures(values, times, ROUTES)
        x, scale, offset = engine.features(np.full(8000, len(times)-1), np.repeat(ROUTES, 8),
                                          np.tile(np.arange(1, 9), len(ROUTES)))
        assert model.feature_names_ == list(x)
        stale = (np.maximum(model.predict(x, thread_count=1), 0)*scale+offset).reshape(1000, 8)
        norm = replay.archive.normalize
        data["fresh"] = .25 * (norm(read_stage(TEST, "fresh")["pred"], totals) - norm(stale, totals))
        del engine, x, model
        gc.collect()
    if {"early", "forecast"}.intersection(keys):
        hist = old.RawHistory(values, times, ROUTES)
        torch.manual_seed(42)
        base_tcn, norm, state = old.load_checkpoint(weights/"model019a.pt")
        assert state["origin"] == str(stamp)
        adapter = patch.LongPatchAllocator(base_tcn)
        initial = {k: v.clone() for k, v in adapter.adapter.state_dict().items()}
        ckpt = torch.load(weights/"model020a_adapter.pt", map_location="cpu", weights_only=True)
        assert ckpt["base_checkpoint_sha256"] == patch.file_hash(weights/"model019a.pt")
        adapter.adapter.load_state_dict(ckpt["adapter"])
        adapter.eval()
        batch = patch.groups(hist, [len(times)-1], norm, labels=False)
        known = batch["known"][0].numpy().astype(float)
        def decoded():
            with torch.no_grad():
                return old.allocate(adapter(batch)[0].numpy().astype(float), known, totals)
        current = decoded()
        if "early" in keys:
            early = old.allocate(read_stage(TEST, "temporal")["early"], known, totals)
            data["early"] = .5*(early-current)
        if "forecast" in keys:
            adapter.adapter.load_state_dict(initial)
            patch.fit_adapter(adapter, hist, norm, budget_provider(hist.half))
            data["forecast"] = .5*(decoded()-current)
    result, target = predict(data, choice)
    dest = OUT/filename
    round_trip(TEST, result, target, dest)
    print("Verified", dest, "selected", choice, "totals", target.tolist(), flush=True)
    return dest


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    verify_contest_export()
    decision = json.loads((OUT/"result.json").read_text())
    export_choice(decision["final"], "submission_model_028a.csv")
    if decision["shares_passed"] and decision["shares"] != decision["final"]:
        export_choice(decision["shares"], "submission_model_028_routes.csv")
