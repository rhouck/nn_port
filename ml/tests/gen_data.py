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

def add_noise_to_df(df, noise_sigma):
    noise = np.random.randn(df.shape[0], df.shape[1]) * noise_sigma
    noise = pd.DataFrame(noise, columns=df.columns, index=df.index)
    return df + noise

def gen_2d_random_Xs_onehot_ys_from_random_kernel(date_index, num_classes, num_features, noise_sigma=3.):
    true_weights = [np.random.randint(-5,6) for i in range(num_features)]
    Xs = [gen_random_normal(date_index, len(true_weights)) for i in range(num_classes)]
    ys = pd.DataFrame([(i * true_weights).sum(axis=1).values for i in Xs], columns=date_index).T
    ys = add_noise_to_df(ys, noise_sigma)
    ys = pd.get_dummies(ys.apply(lambda x: list(x).index(max(x)), axis=1))
    Xs = pd.Panel(dict((ind, pd.DataFrame(i)) for ind, i in enumerate(Xs))).transpose(1,0,2)
    return Xs, ys, true_weights

def check_kernel_predictive_accuracy(Xs, ys, true_weights):
    """apply kernel to Xs and return correct prediction accuracy"""
    weights_df = pd.DataFrame(np.array([true_weights,]*Xs.shape[1]), index=Xs.major_axis, columns=Xs.minor_axis)
    scaled = Xs.multiply(weights_df)
    
    def get_max_ind(df):
        x = list(df.sum(axis=1).values)
        return x.index(max(x))

    df = pd.DataFrame({'xs': pd.Series([get_max_ind(scaled[item]) for item in scaled.items], index=ys.index),
                       'ys': ys.apply(lambda x: list(x).index(max(x)), axis=1)})
    matches = df.apply(lambda x: x[0]==x[1], axis=1)
    return float(matches.sum()) / matches.shape[0]

def gen_correlated_series(s, r):
    mu = s.mean()
    sig = s.std()
    synth = pd.Series(sig * np.random.randn(s.shape[0]) + mu, index=s.index)
    draw = pd.Series(np.random.rand(s.shape[0]), index=s.index)
    df = pd.concat([s, synth, draw], axis=1)
    df.columns = ['real', 'synth', 'draw']
    return df.apply(lambda x: x['real'] if x['draw'] <= r else x['synth'], axis=1)
