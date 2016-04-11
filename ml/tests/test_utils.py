import datetime
import unittest

import pandas as pd

import ml.utils as ut


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
