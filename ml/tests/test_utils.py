import os
import datetime
import unittest

import pandas as pd

import ml.utils as ut
from constructions import ts_score


class TestUtils(unittest.TestCase):

    def test_get_num_per_p_year_is_accurate(self):
        inps = (('d', 365.), ('w', 52.), ('m', 12.), ('q', 4.),)
        for i in inps:
            s = pd.DatetimeIndex(start=datetime.date(2000,1,1), end=datetime.date(2005,1,1), freq=i[0]).to_series()
            self.assertEqual(ut.get_num_per_p_year(s), i[1]) 

    def test_calc_annual_turnover_is_accurate(self):
        ind = pd.DatetimeIndex(start=datetime.date(2000,1,1), end=datetime.date(2001,1,1), freq='q')
        df = pd.DataFrame({'a': [0, 1, 0, 1], 'b': [1, 0, 1, 0,]}, index=ind)
        self.assertEquals(ut.calc_annual_turnover(df), 4)

    def test_convert_to_and_from_sigmoid_doesnt_change_original_value(self):
        TEST_DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/test_data/'
        ret = (pd.read_csv(TEST_DATA_DIR + 'returns.csv', index_col=0, parse_dates=['Date',])
               .applymap(lambda x: x - 1.))
        ret = ts_score(ret)
        scaler = 1
        ret_as_sigmoid = ret.applymap(lambda x: ut.sigmoid(x * scaler))
        ret_from_sigmoid = ret_as_sigmoid.applymap(lambda x: ut.from_sigmoid(x) / scaler)
        self.assertTrue(ret.corrwith(ret_from_sigmoid).mean() > 1 - 1e-3)
        self.assertTrue(ret.sub(ret_from_sigmoid).stack().sum() < 1e-3)