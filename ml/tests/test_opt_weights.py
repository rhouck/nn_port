import os
import unittest

import numpy as np
import pandas as pd

import ml.opt_weights as ow
import ml.utils as ut
from gen_data import *


class TestCalcOptWeights(unittest.TestCase):

    def setUp(self):
        TEST_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'
        self.ret = (pd.read_csv(TEST_DATA_DIR + 'returns.csv', index_col=0, parse_dates=['Date',])
                    .applymap(lambda x: x - 1.))

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

    def test_results_are_diff_for_both_norm_types_when_alpha_is_non_zero(self):
        weights_1 = ow.calc_opt_weights(self.ret, alpha=.1, norm_type=1)
        weights_2 = ow.calc_opt_weights(self.ret, alpha=.1, norm_type=2)
        self.assertFalse(weights_1.equals(weights_2))

    def test_raises_exception_if_contains_nan(self):
        ret_n = self.ret
        ret_n.iloc[50,:] = np.nan
        with self.assertRaises(ValueError):
            weights = ow.calc_opt_weights(ret_n, alpha=.1, norm_type=2)
    
    def test_raises_exception_if_contains_nan(self):
        rets_plus = self.ret.applymap(lambda x: x + 1)
        with self.assertRaises(ValueError):
            weights = ow.calc_opt_weights(rets_plus, alpha=.1, norm_type=2)

    def test_raises_exception_if_too_few_dates_passed(self):
        with self.assertRaises(ValueError):
            weights = ow.calc_opt_weights(self.ret.iloc[:2], alpha=.1, norm_type=2)

    def test_opt_weights_always_result_in_higher_ir_in_sample(self):

        def compare_portfolio_irs(df, look_ahead_per, equal_weights):
            opt_weights = ow.calc_opt_weights(df, alpha=.3, norm_type=2)  
            perf = pd.DataFrame({'equal_weights': df.mul(equal_weights).sum(axis=1),
                                 'opt_weights': df.mul(opt_weights).sum(axis=1),})
            return ut.get_ir(perf).to_dict()
        
        look_ahead_pers = [5, 20, 50, 200]
        equal_weights = pd.Series({k: 1./self.ret.shape[1] for k in self.ret.columns})
        for look_ahead_per in look_ahead_pers:
            df = self.ret.iloc[:look_ahead_per]  
            irs = compare_portfolio_irs(df, look_ahead_per, equal_weights)
            self.assertTrue(irs['opt_weights'] > irs['equal_weights'])


class TestRollingFitOptWeights(unittest.TestCase):

    def setUp(self):
        TEST_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'
        self.ret = (pd.read_csv(TEST_DATA_DIR + 'returns.csv', index_col=0, parse_dates=['Date',])
                    .applymap(lambda x: x - 1.))
        self.owf = lambda x: ow.calc_opt_weights(x, alpha=.1, norm_type=2)
    
    def test_look_ahead_per_reduces_opt_weights_index_from_end(self):
        look_ahead_per = 5
        inp = self.ret.iloc[:15]
        opt_weights = ow.rolling_fit_opt_weights(inp, self.owf, look_ahead_per)
        self.assertEqual(opt_weights.shape[0] + look_ahead_per, inp.shape[0])
        self.assertEqual(inp.index[0], opt_weights.index[0])
        self.assertNotEqual(inp.index[-1], opt_weights.index[-1])