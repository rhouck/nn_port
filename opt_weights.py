import pandas as pd
import cvxpy as cv
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge
from itertools import product


def calc_opt_weights(df, alpha=0, norm_type=2):    
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
    constraints = [0 <= w, sum(w) == 1]
    prob = cv.Problem(objective, constraints)

    result = prob.solve()
    return pd.Series(np.asarray(w.value).flatten(), index=df.columns)

def rolling_fit_opt_weights(df, opt_weights_func, look_ahead_per):
    """applies opt_weights_func to rolling window on pandas df"""
    num_rows = df.shape[0]
    p = pipe(xrange(num_rows),
             filter(lambda x: x + look_ahead_per < num_rows),
             map(lambda x: {df.index[x]: opt_weights_func(df.iloc[x:x+look_ahead_per+1])}))
    return pd.DataFrame(merge(p)).T

def calc_cum_rets(df, alpha, norm_type, look_ahead_per):
    """calculates cumulative returns for optimal portfolio 
    determined by alpha, norm_type, look_ahead_per"""
    opt_weights_func = lambda x: calc_opt_weights(x, alpha=alpha, norm_type=norm_type)
    weights = rolling_fit_opt_weights(df, opt_weights_func, look_ahead_per=look_ahead_per)
    return (df * weights).sum(axis=1).cumprod()

def cum_prod_grid(df, alphas=np.exp(np.linspace(-10, 2, 10)), 
                  look_ahead_pers=xrange(1,30,5)):
    """exhaustive grid search over alphas, look_ahead_per, norm_types 
    returning dataframe of cumulative returns for each optimal portfolio construction"""
    norm_types = [1,2]
    end_date = df.index[-(look_ahead_pers[-1] + 1)]
    p = pipe(product(alphas, norm_types, look_ahead_pers),
             map(lambda x: list(x) + [calc_cum_rets(df, x[0], x[1], x[2])]),
             map(lambda x: x[:-1] + [x[-1][:end_date].tail(1).values[0]]),
             map(lambda x: dict(zip(['alpha', 'norm_type', 'look_ahead_per', 'cum_ret'], x))))
    return pd.DataFrame(list(p))