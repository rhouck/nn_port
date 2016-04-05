import unittest
import datetime

import numpy as np
import pandas as pd

import model_inputs as mi
import model as md
import tensorflow as tf
from gen_data import *


class TestModelInputs(unittest.TestCase):

    def test_validation_doesnt_remove_or_distord_data_unnecessarilly(self):
        self.assertTrue(False)

    def test_buffer_splits_dates_properly(self):
        # buffer should work on piror date, not after
        pass
        #split_inputs_by_date(Xs, ys, datetime.date(2003,1,1), buffer_days=0)


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
