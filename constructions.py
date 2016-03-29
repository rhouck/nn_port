import pandas as pd
import numpy as np
from toolz.curried import pipe, map


def gen_random_normal(date_index, width):
    length = date_index.shape[0]
    return pd.DataFrame(np.random.randn(length, width), index=date_index) 

def gen_random_probs(date_index, width):
    length = date_index.shape[0]
    df = pd.DataFrame(np.random.rand(length, width), index=date_index) 
    return df.div(df.sum(axis=1), axis=0)

def gen_random_onehot(date_index, width):
    df = gen_random_probs(date_index, width)
    df_maxs = df.apply(lambda x: list(x).index(max(x)), axis=1)
    return pd.get_dummies(df_maxs)

flatten_df = lambda x: x.values.flatten()

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

def xs_score(df):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return (df.sub(mean, axis=0)).div(std, axis=0)

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

def get_peak_ahead_returns(returns, per):
    """returns demeaned peaked ahead returns"""
    cum_rets = returns.cumprod()
    fwd_rets = cum_rets.shift(-per).div(cum_rets)
    fwd_rets = fwd_rets.dropna(how='all')
    return fwd_rets.sub(fwd_rets.mean(axis=1), axis=0)
    #fwd_rets = pd.rolling_mean(returns, per).shift(-per)

def get_multi_freq_historical_returns(returns, per):
    """create df of returns data for each column over multiple frequencies
    Args:
      per: integer - number of returns periods to return
    """
    if returns.isnull().values.any():
        raise ValueError("returns dataframe cannot contain nans")

    cum_rets = returns.cumprod()

    returns_vectors = lambda df, per: df.div(df.shift()).ix[-per:]
    daily = returns_vectors(cum_rets, per)
    weekly = returns_vectors(cum_rets.resample('W', how='last'), per)
    monthly = returns_vectors(cum_rets.resample('M', how='last'), per)
    quarterly = returns_vectors(cum_rets.resample('Q', how='last'), per)
    
    df = pd.concat([daily, weekly, monthly, quarterly])
    ind_name = df.index.name
    return df.reset_index().drop(ind_name, 1)


def build_Xs_from_returns(returns, per):
    """builds df containing returns data for all input ids in each row
    output 
    """
        
    p = pipe(returns.index,
             map(lambda x: (x, returns[:x])),                          # filter lagged returns by date      
             map(lambda x: (x[0], get_flat_returns_vecs(x[1], per))),  # create input rows
             filter(lambda x: np.isnan(x[1]).any()==False),            # drop vectors with nan
             map(lambda x: {x[0]: pd.Series(x[1], name=x[0])}))        # convert to Series
    return pd.DataFrame(merge(p)).T
