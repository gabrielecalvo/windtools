import pandas as pd
import numpy as np
from scipy.stats import linregress


def mcp(df):
    assert {'ws_site', 'wd_site', 'ws_ref', 'wd_ref'} <= set(df.columns)
    df['wd_bin'] = pd.cut(df['wd_ref'], np.arange(0, 360, 30))

    df_regression = pd.DataFrame(columns=['m', 'c', 'r', 'p', 'err'])
    for wd_bin in df['wd_bin'].cat.categories:
        data_in_bin = df[df['wd_bin'] == wd_bin]
        if len(data_in_bin) >= 2:
            m, c, r, p, err = linregress(x=data_in_bin['ws_ref'], y=data_in_bin['ws_site'])
            new_row = pd.DataFrame(dict(m=m, c=c, r=r, p=p, err=err), index=[wd_bin])
            df_regression = df_regression.append(new_row)
        else:
            new_row = pd.DataFrame(dict(m=np.nan, c=np.nan, r=np.nan, p=np.nan, err=np.nan), index=[wd_bin])
            df_regression = df_regression.append(new_row)

    print(df_regression)

if __name__ == "__main__":
    df = pd.DataFrame({
        'ws_site': [1, 2, 3],
        'wd_site': [50, 70, 90],
        'ws_ref': [1.1, 2.1, 3.0],
        'wd_ref': [55, 64, 89],
    })
    mcp(df)