import pandas as pd
import numpy as np

from ml.tests.gen_data import gen_correlated_series
from ml.utils import ts_score

def load_returns(fn, anti_signals=False):
    ret = pd.read_csv(fn, parse_dates=['DATE'], index_col=0)
    ret.loc['2008-01-14'] = np.nan
    ret = (ret.dropna(how='all').replace(np.nan,0))
    if not anti_signals:
        ret = ret[[c for c in ret.columns if 'anti' not in c]]
    return ret

def get_fwd_ret(df, look_ahead_per):
    df_1 = df.applymap(lambda x: x + 1.)
    return pd.rolling_apply(df_1[::-1], look_ahead_per, lambda x : x.prod())[::-1]

def df_to_corr_panel(Xs):
    return (pd.Panel({i: Xs.apply(lambda x: gen_correlated_series(x, .05)) for i in range(5)})
            .swapaxes('items', 'major_axis')
            .swapaxes('major_axis', 'minor_axis'))

def panel_ts_score(pn, scaler):
    for i in pn.minor_axis:
        pn.loc[:,:,i] = ts_score(pn.loc[:,:,i].T).T * scaler
    return pn

def select_final_data_set(df):
    df.index.name = 'DATE'
    return (df.reset_index()
              .drop_duplicates(subset='DATE', keep='last')
              .set_index('DATE'))

def add_cash(df):
    cash = np.random.randn(df.shape[0]) * 1e-10
    cash =  pd.Series(cash, name='cash', index=df.index)
    return pd.concat([df, cash], axis=1)
