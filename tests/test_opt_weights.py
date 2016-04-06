import unittest

import numpy as np
import pandas as pd

import opt_weights as ow
from gen_data import *


class TestCalcOptWeights(unittest.TestCase):

    def setUp(self):
        self.ret = (pd.read_csv('tests/test_data/returns.csv', index_col=0, parse_dates=['Date',])
                    .applymap(lambda x: x - 1.))

    def test_selects_opt_puts_all_weight_on_top_performer_in_single_period(self):
        opt_weights = ow.calc_opt_weights(self.ret.ix[:1], alpha=0, norm_type=2)    
        single_per_returns = list(self.ret.ix[:1].values[0])
        max_return_ind = single_per_returns.index(max(single_per_returns))
        self.assertTrue(1 - opt_weights[max_return_ind] < 1e-3)

    def test_opt_weights_are_positive_and_sum_to_one_in_every_period(self):
        weights = ow.calc_opt_weights(self.ret, alpha=.1, norm_type=2)
        self.assertTrue(1. - sum(weights) < 1e-3)
        self.assertTrue(all(weights.map(lambda x: x > 0)))

    def test_high_l2_regularization_leads_to_equal_weight(self):
        weights = ow.calc_opt_weights(self.ret, alpha=1000, norm_type=2)
        self.assertTrue(all(weights.sub(weights.mean()).abs().map(lambda x: x < 1e-3)))
    
    def test_results_are_same_for_both_norm_types_when_alpha_is_zero(self):
        weights_1 = ow.calc_opt_weights(self.ret, alpha=0, norm_type=1)
        weights_2 = ow.calc_opt_weights(self.ret, alpha=0, norm_type=2)
        self.assertTrue(weights_1.equals(weights_2))

    def test_raises_exception_if_contains_nan(self):
        ret_n = self.ret
        ret_n.iloc[50,:] = np.nan
        with self.assertRaises(ValueError):
            weights = ow.calc_opt_weights(ret_n, alpha=.1, norm_type=2)


class TestRollingFitOptWeights(unittest.TestCase):

    def setUp(self):
        self.ret = (pd.read_csv('tests/test_data/returns.csv', index_col=0, parse_dates=['Date',])
                    .applymap(lambda x: x - 1.))
        self.owf = lambda x: ow.calc_opt_weights(x, alpha=.1, norm_type=2)

    def test_look_ahead_per_0_gives_opt_weights_for_each_date(self):
        inp = self.ret.iloc[:15]
        opt_weights = ow.rolling_fit_opt_weights(inp, self.owf, 0)
        self.assertEqual(opt_weights.shape[0], inp.shape[0])
    
    def test_look_ahead_per_reduces_opt_weights_index_from_end(self):
        look_ahead_per = 5
        inp = self.ret.iloc[:15]
        opt_weights = ow.rolling_fit_opt_weights(inp, self.owf, look_ahead_per)
        self.assertEqual(opt_weights.shape[0] + look_ahead_per, inp.shape[0])
        self.assertEqual(inp.index[0], opt_weights.index[0])
        self.assertNotEqual(inp.index[-1], opt_weights.index[-1])