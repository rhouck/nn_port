import unittest

import numpy as np
import pandas as pd

import opt_weights as ow
from gen_data import *


class TestCalcOptWeights(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=100)
        self.ret = pd.read_csv('tests/test_data/returns.csv', index_col=0, parse_dates=['Date',])

    def test_opt_weights_are_positive_and_sum_to_one_in_every_period(self):
        weights = ow.calc_opt_weights(self.ret, alpha=.1, norm_type=2)
        self.assertTrue(1. - sum(weights) < 1e-3)
        self.assertTrue(all(weights.map(lambda x: x > 0)))

    def test_high_l2_regularization_leads_to_equal_weight(self):
        weights = ow.calc_opt_weights(self.ret, alpha=100, norm_type=2)
        self.assertTrue(all(weights.sub(weights.mean()).abs().map(lambda x: x < 1e-3)))
    
    def test_results_are_same_for_both_norm_types_when_alpha_is_zero(self):
        weights_1 = ow.calc_opt_weights(self.ret, alpha=0, norm_type=1)
        weights_2 = ow.calc_opt_weights(self.ret, alpha=0, norm_type=2)
        self.assertTrue(weights_1.equals(weights_2))

    def test_raises_exception_if_contains_nan(self):
        ret_n = self.ret.copy(deep=True)
        ret_n.iloc[50,:] = np.nan
        with self.assertRaises(ValueError):
            weights = ow.calc_opt_weights(ret_n, alpha=.1, norm_type=2)


# class TestRollingFitOptWeights(unittest.TestCase):

#     def setUp(self):
#         np.random.seed(0)
#         self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
#         self.ys_labels = gen_random_onehot(self.dti, 10)
#         self.Xs = gen_random_normal(self.dti, 20)

#     """
#     tests:
#     for each date, never looks back in time
#     first date in first iteration is same as frist date in input data frame
#     look exits when fewer additional rows than look ahead input 
#     (i.e. final date in return df is look_ahead_per less than len of input df)
#     applied weights with minimum look ahead outperforms equal weights
#     check that optimal weights beat equal weights
#     """


