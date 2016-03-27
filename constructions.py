import pandas as pd
import numpy as np
from toolz.curried import pipe, map


def map_to_date(returns, start_date, func):
    """iteratively apply function to dataframe accross expanding window
    return dataframe of func results by date 
    (func results should be single dim vector)
    """
    date_index = returns[start_date:].index
    p = pipe(date_index,
             map(lambda x: (x, func(returns[:x]))))
    return pd.DataFrame(dict(p)).T

def xs_score(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return (df.sub(mean, axis=0)).div(std, axis=0)

def get_peak_ahead_returns(returns, per):
    """returns demeaned peaked ahead returns"""
    cum_rets = returns.cumprod()
    fwd_rets = cum_rets.shift(-per).div(cum_rets)
    fwd_rets = fwd_rets.dropna(how='all')
    return fwd_rets.sub(fwd_rets.mean(axis=1), axis=0)
    #fwd_rets = pd.rolling_mean(returns, per).shift(-per)

def get_flat_cov_matrix(returns):
    """returns vector representing covariance of all assets
    (e.g. one trianglge of the covariance matrix, includign diagonals)
    """
    cov = returns.cov().values
    return (pd.DataFrame(np.tril(cov))
            .replace(0, np.nan)
            .stack()
            .dropna()
            .values)

def get_momentum(returns, halflife):
    """calculates z-scored two-twelve momentum"""
    cum_rets = returns.cumprod()
    two = cum_rets.shift(2)
    twelve = cum_rets.shift(12)
    momentum = two.div(twelve).dropna(how='all')
    mean = pd.ewma(momentum, halflife=halflife, min_periods=float(halflife)/2)
    std = pd.ewmstd(momentum, halflife=halflife, min_periods=float(halflife)/2)
    score = (momentum.sub(mean)).div(std)
    return score.dropna(how='all')

def get_value(returns, halflife):
    """calculates - z-scored price"""
    cum_rets = returns.cumprod()
    mean = pd.ewma(cum_rets, halflife=halflife, min_periods=float(halflife)/2)
    std = pd.ewmstd(cum_rets, halflife=halflife, min_periods=float(halflife)/2)
    score = (cum_rets.sub(mean)).div(std)
    return score.dropna(how='all')
