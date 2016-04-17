import math

import numpy as np
import pandas as pd
from toolz.curried import pipe, map

sigmoid = lambda x: 1. / (1. + math.exp(-x))
from_sigmoid = lambda x: -1. * math.log((1 / x) - 1)

def get_num_per_p_year(df):
    df = df.dropna(how='all')
    pers = df.shape[0]
    years_to_final_per = (df.index[-1] - df.index[0]).days / 365.
    years = years_to_final_per * (pers / (pers - 1.))
    return int(round(pers / years))

def get_ir(df):
    q = get_num_per_p_year(df)
    an_ret = df.mean() * q
    an_std = df.std() * math.sqrt(q)
    return an_ret / an_std

def calc_turnover(df):
    delta = df.sub(df.shift()).abs().sum(axis=1) / 2.
    lev = df.abs().sum(axis=1)
    return delta / lev

def calc_annual_turnover(df):
    to = calc_turnover(df)
    return to.mean() * get_num_per_p_year(df)

def get_mean_var_tilt_holdings(df, halflife=24*22):
    alphas = np.ones(df.shape[1])
    cov_pn = pd.ewmcov(df, halflife=halflife, min_periods=halflife / 4).dropna(how='all')
    holdings = pd.DataFrame({date: cov_pn[date].dot(alphas) for date in cov_pn.items}).T
    scaler = 1 / holdings.sum(axis=1)
    return holdings.mul(scaler, axis=0)

def calc_ratio_above_zero(df):
    pos_val = df.applymap(lambda x: 1 if x >= 0 else 0)
    pos_val_sum = pos_val.sum(axis=0)
    return pos_val_sum / float(df.shape[0])

def map_to_date(returns, start_date, func):
    """iteratively apply function to dataframe accross expanding window
    return dataframe of func results by date 
    (func results should be single dim vector)
    """
    date_index = returns[start_date:].index
    p = dict(pipe(date_index,
                  map(lambda x: (x, func(returns[:x])))))
    try:
        return pd.DataFrame(p).T
    except:
        return pd.Panel(p).transpose(0, 2, 1)
