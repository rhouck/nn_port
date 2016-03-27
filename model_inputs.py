import os
import shutil
import datetime

import pandas as pd
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge

def clear_path(path):
    """deletes all files and sub dirs in dir (for clearing tensorboard logdir)"""
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception, e:
            print e

def multi_freq_returns(returns, per):
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

"""
test that no peak ahead exists - the row for a partcular date should only 
contain returns data from dates prior

test that no outputs contain nan and inputs with nan are caught
"""
def build_Xs_from_returns(returns, per):
    """builds df containing returns data for all input ids in each row
    output 
    """
    def get_flat_returns_vecs(returns, per):
        return (multi_freq_returns(returns, per)
                .values.flatten())
    
    #lagged_returns = returns.shift().dropna(how='all')
    p = pipe(returns.index,
             map(lambda x: (x, returns[:x])),                          # filter lagged returns by date      
             map(lambda x: (x[0], get_flat_returns_vecs(x[1], per))),  # create input rows
             filter(lambda x: np.isnan(x[1]).any()==False),            # drop vectors with nan
             map(lambda x: {x[0]: pd.Series(x[1], name=x[0])}))        # convert to Series
    return pd.DataFrame(merge(p)).T

def validate_and_format_Xs_ys(Xs, ys):
    for i in (Xs, ys):
        if i.isnull().values.any():
            raise ValueError("model inputs cannot contain nans")
    ind = pd.DatetimeIndex(sorted(set(Xs.index) & set(ys.index)))
    return Xs.ix[ind], ys.ix[ind]

def split_inputs_by_date(Xs, ys, split_date, buffer_days=0):
    """splits Xs and ys by 'split_date' + 'buffer_days'"""
    split_date_shifted = split_date + datetime.timedelta(days=buffer_days)
    Xs_a = Xs[:split_date]
    ys_a = ys[:split_date]
    Xs_b = Xs[split_date_shifted:]
    ys_b = ys[split_date_shifted:]
    return ((Xs_a, ys_a), (Xs_b, ys_b))

