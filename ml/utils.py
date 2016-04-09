import math

def get_num_per_p_year(df):
    df = df.dropna(how='all')
    T = float(df.shape[0])
    t = (df.index[-1] - df.index[0]).days / 365.
    return T / t

def get_ir(df):
    q = get_num_per_p_year(df)
    an_ret = df.mean() * q
    an_std = df.std() * math.sqrt(q)
    return an_ret / an_std