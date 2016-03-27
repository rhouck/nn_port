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

