import unittest
import numpy as np
import pandas as pd
from exp_021.improve_network import STEP, aligned, rescale, validate_grid


class TestSubmission(unittest.TestCase):
    def setUp(self):
        routes = np.arange(1000)*3+2
        origin = pd.Timestamp("2025-11-01 10:30")
        grid = pd.MultiIndex.from_product([routes,pd.date_range(origin+STEP,periods=8,freq=STEP)])
        df = grid.to_frame(index=False,name=["route_id","timestamp"])
        df["id"] = ["key-"+str(i) for i in range(len(df))]
        self.test = validate_grid(df.sample(frac=1,random_state=8),routes,origin)
        self.base = pd.DataFrame(dict(id=df.id, y_pred=1+df.route_id.to_numpy()**.5))
        self.origin = origin
        self.routes = routes

    def test_export_keeps_new_horizon_sums_and_correct_ids(self):
        before = aligned(self.test,self.base)
        h = self.test.h.to_numpy()
        old = np.bincount(h,weights=before)
        factors = np.linspace(.98,1.02,8)
        result = rescale(self.test,self.base,old*factors)
        self.assertListEqual(result.columns.tolist(),["id","y_pred"])
        after = aligned(self.test,result)
        np.testing.assert_allclose(after,before*factors[h],rtol=1e-12)
        np.testing.assert_allclose(np.bincount(h,weights=after),old*factors,rtol=1e-12)
        self.assertFalse(np.allclose(old,np.bincount(h,weights=after)))

    def test_duplicate_route_slot_is_rejected(self):
        grid = self.test.sort_values(["route_id","timestamp"]).copy()
        grid.iloc[0,grid.columns.get_loc("timestamp")] = grid.timestamp.iloc[1]
        with self.assertRaises(ValueError):
            validate_grid(grid,self.routes,self.origin)

    def test_duplicate_or_unknown_id_is_rejected(self):
        for invalid in [self.base.id.iloc[1],"unknown"]:
            bad = self.base.copy()
            bad.loc[0,"id"] = invalid
            with self.assertRaises(ValueError):
                aligned(self.test,bad)

