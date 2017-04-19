import pandas as pd
import numpy as np
from scipy.stats import linregress


def mcp(df, wd_bin_size=30):
    assert {'ws_site', 'wd_site', 'ws_ref', 'wd_ref'} <= set(df.columns)
    df['wd_bin'] = pd.cut(df['wd_ref'], np.arange(0, 360+wd_bin_size/2, wd_bin_size), right=False)

    regression_params = {}
    for wd_bin, sub_df in df.groupby('wd_bin'):
        n_points = sub_df.shape[0]
        if n_points >= 2:
            m, c, r, p, err = linregress(x=sub_df['ws_ref'], y=sub_df['ws_site'])
            regression_params[wd_bin] = dict(m=m, c=c, r=r, p=p, err=err, n=n_points)
        else:
            regression_params[wd_bin] = {'n': n_points}

    df_regression = pd.DataFrame(regression_params)
    print(df_regression)
    return df_regression

if __name__ == "__main__":
    df = pd.DataFrame({
        'ws_site': [1, 2, 3],
        'wd_site': [50, 70, 90],
        'ws_ref': [1.1, 2.1, 3.0],
        'wd_ref': [55, 64, 89],
    })
    mcp(df)
