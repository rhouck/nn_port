import os
import shutil
import datetime

import pandas as pd
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge


def flatten_panel(pn):
    df = pn.to_frame().T
    df.columns = range(df.shape[1])
    return df

flatten_df = lambda x: x.values.flatten()

def get_date_index(ob):
    try:
        return ob.index
    except:
        return ob.items

def date_in_each_index(date, obs):
    return all((date in get_date_index(ob) for ob in obs))

def get_by_date(date, ob):
    try:
        return ob.ix[date]
    except:
        return ob[date]       

def concat_dfs_by_date(date, obs):
    df = pd.concat([get_by_date(date, ob) for ob in obs], axis=1)
    df.columns = range(len(df.columns))
    return df

def create_2d_features(date_index, obs):
    p = pipe(date_index,
             filter(lambda x: date_in_each_index(x, obs)),
             map(lambda x: (x, concat_dfs_by_date(x, obs))))
    return pd.Panel(dict(p))

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

def validate_and_format_Xs_ys(Xs, ys):
    inps = (Xs, ys)
    for i in inps:
        if i.isnull().values.any():
            raise ValueError("model inputs cannot contain nans")
        if not isinstance(get_date_index(i), pd.tseries.index.DatetimeIndex):
            raise ValueError("model inputs must contain datetime index")
    if len(Xs.shape) == 3:
        if Xs.shape[1] != ys.shape[1]:
            raise ValueError("model inputs dimension mismatch (num classes must be equal)")
    
    inds = [get_date_index(i) for i in inps]
    ind = pd.DatetimeIndex(sorted(set(inds[0]) & set(inds[1])))
    Xs = get_by_date(ind, Xs).astype(np.float32)
    ys = get_by_date(ind, ys)
    
    return Xs, ys

def split_inputs_by_date(Xs, ys, split_date, buffer_periods=0):
    """splits Xs and ys by 'split_date' - 'buffer_periods'"""    
    date_ind = get_date_index(Xs).to_series()
    try:
        test_split_date = date_ind[split_date:][0]
        test_split_ind = date_ind.tolist().index(test_split_date)
    except:
        test_split_ind = date_ind.shape[0]
        
    train_split_ind = test_split_ind - buffer_periods
    Xs_train = Xs.iloc[:train_split_ind]
    ys_train = ys.iloc[:train_split_ind]
    Xs_test = Xs.iloc[test_split_ind:]
    ys_test = ys.iloc[test_split_ind:]
    
    return ((Xs_train, ys_train), (Xs_test, ys_test))