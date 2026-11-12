import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("The full 020a test suite requires the exp_027 PyTorch runtime") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp_028"))
from temporal import budget_provider, old, patch


class TestExperiment028(unittest.TestCase):
    def test_export_keeps_requested_totals_after_reordering(self):
        t = pd.Timestamp("2025-10-11 10:30")
        routes = np.array([3, 9])
        test = pd.MultiIndex.from_product(
            [routes, pd.date_range(t+old.STEP, periods=8, freq=old.STEP)],
            names=["route_id", "timestamp"]).to_frame(index=False)
        test["id"] = np.arange(16).astype(str)
        test = test.sample(frac=1, random_state=2)
        ref = pd.DataFrame({"id": test.id, "y_pred": 100.}).iloc[::-1]
        raw = np.arange(1., 17.).reshape(2, 8)
        for requested in (None, np.arange(201., 209.)):
            exported = old.make_submission(test, ref, raw, [t], routes, target_totals=requested)
            seen = test.merge(exported, on="id", validate="one_to_one").pivot(
                index="route_id", columns="timestamp", values="y_pred").to_numpy()
            target = np.full(8, 200.) if requested is None else requested
            np.testing.assert_allclose(seen, raw * target / raw.sum(0))

    def test_adapter_budgets_and_route_features_ignore_future(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(30, 100, (420, 3, 7))
        t = 375
        def features(v):
            flow = v[:, :, 0]
            cs = np.concatenate([np.zeros_like(v[:1]), np.cumsum(v, axis=0)])
            f = patch._local(v, cs, np.array([t]), np.ones((3, 7)), flow, upstream=True)
            return f, budget_provider(flow)(t)
        f, totals = features(x)
        x[t+1:] *= 1000
        f2, totals2 = features(x)
        np.testing.assert_array_equal(f, f2)
        np.testing.assert_array_equal(totals, totals2)
        self.assertTrue((old.network_q(totals, x[t, :, 0].sum()) > 0).all())

    def test_evaluation_uses_supplied_group_specific_budgets(self):
        class Uniform(torch.nn.Module):
            def forward(self, batch):
                return torch.ones_like(batch["y"])
        y = torch.full((2, 2, 8), 20.)
        data = dict(y=y, x=y, known=torch.full_like(y, 10.))
        model = Uniform()
        self.assertEqual(old.epoch_pass(model, data), 0.)
        budgets = np.array([[44.] * 8, [38.] * 8])
        expected = (4+2) / (40+40)
        self.assertAlmostEqual(old.epoch_pass(model, data, forecast_totals=budgets), expected)
