import unittest

import numpy as np
import pandas as pd

from gen_data import *


class TestDataGeneration(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)

    def test_kernel_has_perfect_pred_power_with_no_noise(self):
        Xs, ys, tw = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, 15, 10, 0.)
        self.assertEquals(check_kernel_predictive_accuracy(Xs, ys, tw), 1.)

    def test_adding_noise_reduces_kernel_predicted_power(self):    
        accs = []
        for i in (100., 10., 1., 0.):
            Xs, ys, tw = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, 15, 10, i)
            accs.append(check_kernel_predictive_accuracy(Xs, ys, tw))
        self.assertEquals(sorted(accs), accs)

    def test_gen_correlated_series_works_correctly(self):
        s = synth = pd.Series(np.random.randn(self.dti.shape[0]), index=self.dti)
        for r in (.05, .25, .75):
            cs = gen_correlated_series(s, r)
            self.assertTrue(abs(r - pd.concat([s, cs], axis=1).corr().iloc[0,1]) < 5e-2)
