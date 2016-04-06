import unittest
import datetime

import numpy as np
import pandas as pd

import model_inputs as mi
from gen_data import *


class TestModelInputs(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        self.ys_labels = gen_random_onehot(self.dti, 10)
        self.Xs = gen_random_normal(self.dti, 20)
        
    def test_validation_doesnt_remove_or_distort_data_unnecessarilly(self):
        Xs = self.Xs.astype(np.float32)
        ys = self.ys_labels
        Xs_v, ys_v = mi.validate_and_format_Xs_ys(Xs, ys)
        self.assertTrue(Xs_v.equals(Xs))
        self.assertTrue(ys_v.equals(ys))

    def test_validate_raises_exception_if_inputs_not_df_with_dt_index(self):
        with self.assertRaises(Exception):
            _, _ = mi.validate_and_format_Xs_ys(self.Xs.values, self.ys_labels.values)
        Xs = self.Xs
        Xs.index = Xs.index.map(lambda x: str(x))
        with self.assertRaises(Exception):
            _, _ = mi.validate_and_format_Xs_ys(Xs, self.ys_labels)

    def test_splits_dates_doesnt_create_train_test_overlap(self):
        split_date = datetime.date(2002,1,1)
        train, test = mi.split_inputs_by_date(self.Xs, self.ys_labels, split_date, buffer_periods=0)

        def get_len_overlap(a, b):
            return len(set(a) & set(b))

        self.assertEquals(test[0].index[0], pd.to_datetime(split_date))
        self.assertEquals(get_len_overlap(train[0].index, test[0].index), 0)
        self.assertEquals(get_len_overlap(train[0].index, train[1].index), train[0].index.shape[0])
        self.assertEquals(get_len_overlap(test[0].index, test[1].index), test[0].index.shape[0])
        self.assertEquals(len(train[0].index) + len(test[0].index), len(self.Xs.index))
        
    def test_split_dates_buffer_reduces_train_ts_from_end(self):
        buffer_periods = 5
        split_date = datetime.date(2002,1,1)
        train, test = mi.split_inputs_by_date(self.Xs, self.ys_labels, split_date, buffer_periods=buffer_periods)
        self.assertEquals(train[0].index[0], self.Xs.index[0])
        ind = self.Xs.index.to_series()
        self.assertEquals(test[0].index[0], ind.ix[split_date:][0])
        self.assertNotEquals(train[0].index[-1], ind.ix[:split_date][-2])
        self.assertEquals(len(train[0].index) + len(test[0].index), len(self.Xs.index) - buffer_periods)

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
