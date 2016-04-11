import math

from toolz.curried import pipe, map


def get_num_per_p_year(df):
    df = df.dropna(how='all')
    pers = df.shape[0]
    years_to_final_per = (df.index[-1] - df.index[0]).days / 365.
    years = years_to_final_per * (pers / (pers - 1.))
    return int(round(pers / years))

def get_ir(df):
    q = get_num_per_p_year(df)
    an_ret = df.mean() * q
    an_std = df.std() * math.sqrt(q)
    return an_ret / an_std

def calc_turnover(df):
    delta = df.sub(df.shift()).abs().sum(axis=1) / 2.
    lev = df.abs().sum(axis=1)
    return delta / lev

def calc_annual_turnover(df):
    to = calc_turnover(df)
    return to.mean() * get_num_per_p_year(df)

def map_to_date(returns, start_date, func):
    """iteratively apply function to dataframe accross expanding window
    return dataframe of func results by date 
    (func results should be single dim vector)
    """
    date_index = returns[start_date:].index
    p = dict(pipe(date_index,
                  map(lambda x: (x, func(returns[:x])))))
    try:
        return pd.DataFrame(p).T
    except:
        return pd.Panel(p).transpose(0, 2, 1)
