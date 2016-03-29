import os
import shutil
import datetime

import pandas as pd
import numpy as np
from toolz.curried import pipe, map, filter
from toolz.dicttoolz import merge


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
    inds = [get_date_index(i) for i in inps]
    ind = pd.DatetimeIndex(sorted(set(inds[0]) & set(inds[1])))
    return get_by_date(ind, Xs), get_by_date(ind, ys)

def split_inputs_by_date(Xs, ys, split_date, buffer_days=0):
    """splits Xs and ys by 'split_date' + 'buffer_days'"""
    split_date_shifted = split_date + datetime.timedelta(days=buffer_days)
    Xs_a = Xs[:split_date]
    ys_a = ys[:split_date]
    Xs_b = Xs[split_date_shifted:]
    ys_b = ys[split_date_shifted:]
    return ((Xs_a, ys_a), (Xs_b, ys_b))

