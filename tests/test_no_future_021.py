import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from exp_021 import improve_network as net


class TestCausal(unittest.TestCase):
    def test_every_model_ignores_future_and_ridge_labels_stop_at_origin(self):
        idx = pd.date_range("2025-01-01",periods=720,freq=net.STEP)
        rng = np.random.default_rng(91)
        series = pd.DataFrame(rng.uniform(1e6,2e6,(len(idx),7)), index=idx,columns=[net.TARGET,*net.STATUS])
        origin = idx[689]
        altered = series.copy()
        altered.loc[idx[690]:] *= 99
        original = net.fit_predict(series,origin)
        poisoned = net.fit_predict(altered,origin)
        for model in net.MODELS:
            np.testing.assert_array_equal(original[model],poisoned[model])
        _,anchor = net.features(series.loc[:origin])
        with patch.object(net,"make_pipeline") as factory:
            factory.return_value.predict.return_value = np.zeros((1,8))
            net.fit_predict(series,origin,["ridge"])
            x,residual = factory.return_value.fit.call_args.args
        self.assertEqual(x.index[-1],origin-8*net.STEP)
        expected = series.loc[x.index[-1]+net.STEP:origin,net.TARGET].to_numpy()
        np.testing.assert_allclose(residual[-1]*1e6+anchor.loc[x.index[-1]],expected)

