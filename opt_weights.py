import pandas as pd
from cvxpy import *
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge


"""
tests:
weights sum to 1, are all positive and <= 1
equal weight == highly regularized
optimal weight == barely regularized
"""

def calc_opt_weights(df, alpha=0, norm_type=2):    
    """returns optimal weights of returns df with optional regularization
    inputs:
        df: a dataframe of returns
        alpha: float defaults to 0, represents scalar applied to regularization term
        norm_type: int 1 or 2, defaults to 2 (for l2 penalty)
    returns:
        pd.Series of optimal weights
    """
    X = df.values
    y = np.ones(df.shape[0])

    w = Variable(df.shape[1])
    reg = norm(w, 2)
    objective = Minimize(sum_squares(X*w - y) + alpha*reg)
    constraints = [0 <= w, sum(w) == 1]
    prob = Problem(objective, constraints)

    result = prob.solve()
    return pd.Series(np.asarray(w.value).flatten(), index=df.columns)

"""
tests:
for each date, never looks back in time
first date in first iteration is same as frist date in input data frame
look exits when fewer additional rows than look ahead input 
(i.e. final date in return df is look_ahead_per less than len of input df)
applied weights with minimum look ahead outperforms equal weights
"""
def rolling_fit_opt_weights(df, opt_weights_func, look_ahead_per):
    """applies opt_weights_func to rolling window on pandas df"""
    num_rows = df.shape[0]
    p = pipe(xrange(num_rows),
             filter(lambda x: x + look_ahead_per < num_rows),
             map(lambda x: {df.index[x]: opt_weights_func(df.iloc[x:x+look_ahead_per])}))
    return pd.DataFrame(merge(p)).T