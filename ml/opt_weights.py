import pandas as pd
import cvxpy as cv
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge
from itertools import product

import utils as ut


def calc_opt_weights(df, alpha=0, norm_type=2, long_only=True):    
    """returns optimal weights of returns df with optional regularization
    inputs:
        df: a dataframe of returns
        alpha: float defaults to 0, represents scalar applied to regularization term
        norm_type: int 1 or 2, defaults to 2 (for l2 penalty)
    returns:
        pd.Series of optimal weights
    """
    if df.isnull().values.any():
        raise ValueError("dataframe cannot contain nans")
    if df.shape[0] < 3:
        raise ValueError("dataframe must contain at least 3 rows")
    if abs(df.stack().mean() - 1) < .1:
        raise ValueError("returns data must be centered at 0 not 1")

    X = df.values
    y = np.ones(df.shape[0])

    w = cv.Variable(df.shape[1])
    reg = cv.norm(w, norm_type)
    objective = cv.Minimize(cv.sum_squares(X*w - y) + alpha*reg)
    
    if long_only:
        constraints = [0 <= w, sum(w) == 1]
    else: 
        constraints = [sum(w) == 0]
    
    prob = cv.Problem(objective, constraints)
    result = prob.solve()
    
    weights = pd.Series(np.asarray(w.value).flatten(), index=df.columns)
    if not long_only:
        weights = weights / weights[weights > 0].sum()
    
    return weights

def rolling_fit_opt_weights(df, opt_weights_func, look_ahead_per):
    """applies opt_weights_func to rolling window on pandas df"""
    num_rows = df.shape[0]
    p = pipe(xrange(num_rows),
             filter(lambda x: x + look_ahead_per < num_rows),
             map(lambda x: {df.index[x]: opt_weights_func(df.iloc[x:x+look_ahead_per+1])}))
    return pd.DataFrame(merge(p)).T

def calc_opt_weight_portfolio_ir(df, alpha, norm_type, look_ahead_per, long_only=True, tilt_weights=None):
    """calculates ir for optimal portfolio 
    determined by alpha, norm_type, look_ahead_per"""
    opt_weights_func = lambda x: calc_opt_weights(x, alpha=alpha, norm_type=norm_type, long_only=long_only)
    weights = rolling_fit_opt_weights(df, opt_weights_func, look_ahead_per=look_ahead_per)
    try:
        weights -= tilt_weights
    except:
        pass
    return ut.get_ir((df * weights).sum(axis=1))

def opt_weight_ir_grid(df, alphas, look_ahead_pers, long_only=True, tilt_weights=None):
    """exhaustive grid search over alphas, look_ahead_per, norm_types 
    returning dataframe of cumulative returns for each optimal portfolio construction"""
    norm_types = [2,]
    end_date = df.index[-(look_ahead_pers[-1] + 1)]
    p = pipe(product(alphas, norm_types, look_ahead_pers),
             map(lambda x: list(x) + [calc_opt_weight_portfolio_ir(df, x[0], x[1], x[2], long_only, tilt_weights)]),
             map(lambda x: dict(zip(['alpha', 'norm_type', 'look_ahead_per', 'ir'], x))))
    return pd.DataFrame(list(p))

