"""Fit and checkpoint the two fixed TCN epochs and three adapter variants."""
import argparse
import gc
import numpy as np
import torch
from components import DATES, folder, origin, replay, write_stage
import predict_model_019b as old
import upgrade_model_020a as patch


def budget_provider(half):
    # Forecast network half-hours from completed historical analogues. The level
    # ratio uses only eight past slots at the source and at the analogue.
    q = half.sum(axis=1, dtype=np.float64)
    def forecast(t):
        weights = []
        for lag in (48, 96, 144, 336):
            ref = t - lag
            if ref < 7:
                continue
            scale = np.clip(q[t-7:t+1].mean() / max(q[ref-7:ref+1].mean(), 1), .8, 1.2)
            weights.append(q[ref+1:ref+9] * scale)
        future = np.maximum(np.mean(weights, axis=0), 1.)
        return future + np.r_[q[t], future[:-1]]
    return forecast


def temporal(day):
    dest = folder(day)
    if (dest / "temporal.npz").exists():
        return
    times, values = replay.history(end=origin(day))
    hist = old.RawHistory(values, times, replay.ROUTES)
    def checkpoint(model, norm, records):
        epoch = len(records)
        if epoch not in (24, 34):
            return
        assert max(r["label_max"] for r in records) <= len(times)-1
        temp = dest / f"epoch_{epoch}.tmp.pt"
        old.save_checkpoint(temp, model, norm, hist, records)
        temp.replace(dest / f"epoch_{epoch}.pt")
    if not all((dest / f"epoch_{n}.pt").exists() for n in (24, 34)):
        old.train_model(hist, len(times)-1, 34, on_epoch=checkpoint)
        gc.collect()
    result = {}
    for epoch, use_forecast, key in [(24, False, "early"), (34, False, "adapted"),
                                      (34, True, "forecast")]:
        torch.manual_seed(42)
        np.random.seed(42)
        base, norm, state = old.load_checkpoint(dest / f"epoch_{epoch}.pt")
        assert state["origin"] == str(times[-1]) and len(state["history"]) == epoch
        adapter = patch.LongPatchAllocator(base)
        data = patch.groups(hist, [len(times)-1], norm, labels=False)
        if epoch == 34 and not use_forecast:
            result["scores"], result["known"] = old.predict_scores(base, hist, norm)
        records = patch.fit_adapter(adapter, hist, norm, budget_provider(hist.half) if use_forecast else None)
        assert max(r["latest_training_label"] for r in records) < len(times)
        with torch.no_grad():
            result[key] = adapter(data)[0].numpy().astype(float)
        del adapter, data, base
        gc.collect()
    write_stage(day, "temporal", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("days", nargs="*", default=DATES)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for day in args.days:
        print(day, "TCN", flush=True)
        temporal(day)
