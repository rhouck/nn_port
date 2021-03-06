import os
import shutil
import datetime

import pandas as pd
import numpy as np
from toolz.curried import pipe, filter
from toolz.curried import map as cmap
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
             cmap(lambda x: (x, concat_dfs_by_date(x, obs))))
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

def validate_and_format_inputs(*inps):
    """checks model inputs are formatted properly"""
    if len(inps) == 1 and isinstance(inps[0], (tuple, list)):
        inps = inps[0]

    shapes = []
    for i in inps:
        if i.isnull().values.any():
            raise ValueError("model inputs cannot contain nans")
        if not isinstance(get_date_index(i), pd.tseries.index.DatetimeIndex):
            raise ValueError("model inputs must contain datetime index")
        shapes.append(i.shape)

    # # validate num classes in panel Xs matches num classes ys
    # a = any([len(i)==3 for i in shapes])
    # b = all([i[1]==shapes[0][1] for i in shapes])
    # if a and not b:
    #     raise ValueError("model inputs dimension mismatch (num classes must be equal)")

    inds = [set(get_date_index(i)) for i in inps]
    ind = pd.DatetimeIndex(sorted(set.intersection(*inds)))
    inps = [get_by_date(ind, i).astype(np.float32) for i in inps]   
    return inps

def split_inputs_by_date(inps, split_date, buffer_periods):
    """splits Xs and ys by 'split_date' - 'buffer_periods'"""
    dates = sorted(set(get_date_index(inps[0])))
    date_ind = pd.Series(dates, index=dates)
   
    try:
        test_split_date = date_ind[split_date:][0]
        test_split_ind = date_ind.tolist().index(test_split_date)
        train_split_ind = test_split_ind - buffer_periods -1 
        train_split_date = date_ind.iloc[train_split_ind]
    except:
        train_split_date = split_date        
        test_split_date = split_date       
    
    inps_train = [i[:train_split_date] for i in inps]
    inps_test = [i[test_split_date:] for i in inps]
    inps = zip(inps_train, inps_test)
    inps = map(lambda x: list(x), inps)

    return inps