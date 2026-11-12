import unittest
import numpy as np
import pandas as pd
from exp_021.improve_network import budgets, q0_at, safe_totals, proposed_totals


class TestBudget(unittest.TestCase):
    def test_recurrence_round_trip_and_rejection(self):
        halves = np.array([10.,9.,8.,20.,17.,2.,3.,12.,8.])
        totals = halves[:-1]+halves[1:]
        np.testing.assert_array_equal(budgets(totals,halves[0]),halves[1:])
        for val in (5,10,np.nan,np.inf):
            invalid = totals.copy()
            invalid[0] = val
            actual,fallback = safe_totals(totals,invalid,halves[0])
            self.assertTrue(fallback)
            np.testing.assert_array_equal(actual,totals)

    def test_causal_initial_budget_matches_identifiable_halves(self):
        q = np.array([[1,2],[0,0],[4,2],[7,8],[0,0],[2,8],[10,1]])
        y = pd.DataFrame(q[1:]+q[:-1])
        for t in (3,4,5):
            self.assertEqual(q0_at(y,t),q[t+1].sum())

    def test_shape_cap_preserves_bias_even_at_limit(self):
        old = np.arange(11,19,dtype=float)
        forecast = old*np.linspace(.6,1.4,8)
        revised = proposed_totals(old,forecast,"shape",.5,.02)
        self.assertAlmostEqual(revised.sum(),old.sum())
        self.assertAlmostEqual(np.abs(revised/old-1).max(),.02)

