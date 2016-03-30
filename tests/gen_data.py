import pandas as pd
import numpy as np


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

def gen_2d_random_Xs_onehot_ys_from_random_kernel(date_index, width, classes):
    true_weights = [np.random.randint(-5,6) for i in range(width)]
    Xs = [gen_random_normal(date_index, len(true_weights)) for i in range(classes)]
    ys = pd.DataFrame([(i * true_weights).sum(axis=1).values for i in Xs], columns=date_index).T
    ys = pd.get_dummies(ys.apply(lambda x: list(x).index(max(x)), axis=1))
    Xs = pd.Panel(dict((ind, pd.DataFrame(i)) for ind, i in enumerate(Xs))).transpose(1,0,2)
    return Xs, ys