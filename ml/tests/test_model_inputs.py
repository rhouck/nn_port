import unittest
import datetime

import numpy as np
import pandas as pd

import ml.model_inputs as mi
from gen_data import *


class TestModelInputs(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.dti = pd.DatetimeIndex(start='2000-1-1', freq='B', periods=1000)
        self.split_date = datetime.date(2002,1,1)
        self.ys_labels = gen_random_onehot(self.dti, 10)
        self.Xs = gen_random_normal(self.dti, 20)
        self.Xs_pn, _, _ = gen_2d_random_Xs_onehot_ys_from_random_kernel(self.dti, num_classes=10, num_features=10, noise_sigma=10.)
        
    def test_validation_doesnt_remove_or_distort_data_unnecessarilly(self):
        Xs = self.Xs.astype(np.float32)
        ys = self.ys_labels.astype(np.float32)
        Xs_v, ys_v = mi.validate_and_format_inputs(Xs, ys)
        self.assertTrue(Xs_v.equals(Xs))
        self.assertTrue(ys_v.equals(ys))

    def test_validate_raises_exception_if_inputs_not_df_with_dt_index(self):
        with self.assertRaises(Exception):
            _, _ = mi.validate_and_format_inputs(self.Xs.values, self.ys_labels.values)
        Xs = self.Xs
        Xs.index = Xs.index.map(lambda x: str(x))
        with self.assertRaises(Exception):
            _, _ = mi.validate_and_format_inputs(Xs, self.ys_labels)

    def test_splits_dates_doesnt_create_train_test_overlap(self):
        train, test = mi.split_inputs_by_date([self.Xs, self.ys_labels], self.split_date, 0)

        def get_len_overlap(a, b):
            return len(set(a) & set(b))

        self.assertEquals(test[0].index[0], pd.to_datetime(self.split_date))
        self.assertEquals(get_len_overlap(train[0].index, test[0].index), 0)
        self.assertEquals(get_len_overlap(train[0].index, train[1].index), train[0].index.shape[0])
        self.assertEquals(get_len_overlap(test[0].index, test[1].index), test[0].index.shape[0])
        self.assertEquals(len(train[0].index) + len(test[0].index), len(self.Xs.index))
        
    def test_split_dates_buffer_reduces_train_ts_from_end(self):
        buffer_periods = 5
        train, test = mi.split_inputs_by_date([self.Xs, self.ys_labels], self.split_date, buffer_periods)
        self.assertEquals(train[0].index[0], self.Xs.index[0])
        ind = self.Xs.index.to_series()
        self.assertEquals(test[0].index[0], ind.ix[self.split_date:][0])
        self.assertNotEquals(train[0].index[-1], ind.ix[:self.split_date][-2])
        self.assertEquals(len(train[0].index) + len(test[0].index), len(self.Xs.index) - buffer_periods)

    def test_split_dates_works_on_panels(self):
        try:
            mi.split_inputs_by_date([self.Xs_pn, self.ys_labels], self.split_date, 0)
        except Exception as err:
            raise Exception('split_inputs_by_date failed with panel input: {0}'.format(err))

    def test_split_dates_accepts_future_dates_by_returning_all_data_to_train_set(self):
        Xs_inp = self.Xs_pn
        train, test = mi.split_inputs_by_date([Xs_inp, self.ys_labels], datetime.date(2050,1,1), 0)
        inp_ind = mi.get_date_index(Xs_inp)
        train_ind =  mi.get_date_index(train[0])
        test_ind = mi.get_date_index(test[0])
        self.assertEquals(len(train_ind), len(inp_ind))
        self.assertEquals(len(test_ind), 0)

    def test_panel_Xs_and_ys_must_have_same_num_classes(self):
        with self.assertRaises(ValueError):
            mi.validate_and_format_inputs(self.Xs_pn, self.ys_labels.iloc[:,:5])