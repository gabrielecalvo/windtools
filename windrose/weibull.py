import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import windrose.util as util


class Weibull(object):
    def __init__(self, fpath=None, ws_field='ws', wd_field='wd', **loading_options):
        self.raw_data = pd.DataFrame()
        if fpath:
            field_map = {ws_field: 'ws', wd_field: 'wd'}
            self.load_raw_data(fpath, field_map, loading_options)

    def load_raw_data(self, fpath, field_map, loading_options):
        self.raw_data = util.load_data(fpath=fpath, field_map=field_map, loading_options=loading_options, dropna='any')
        self.raw_data.ix[self.raw_data['wd'] == 360] = 0

    def fit_distribution(self):
        pass


def weibull_fit(dfb, plot=False, savefig=False):
    # get wd binning index sorted
    wd_bin_names = [x for x in dfb['wd_bin'].unique() if pd.notnull(x)]
    wd_bin_names = sorted(wd_bin_names, key=lambda x: float(x[1:].split(",")[0]))
    sp_n = len(wd_bin_names)
    sp_rows = int(np.sqrt(sp_n))
    sp_cols = np.ceil(sp_n / sp_rows)
    lab_fsize = int(-4 / 5 * sp_n + 20)
    weibParam = []

    if plot: fig = plt.figure(figsize=(15, 12), dpi=80)

    for i, i_bin in enumerate(wd_bin_names):
        data = dfb[dfb['wd_bin'] == i_bin][['ws', 'ws_bin']]
        k, mu, lam = stats.weibull_min.fit(data['ws'], floc=0)  # weib fitting
        weibParam.append([i_bin, k, lam])

        if plot:
            ax = fig.add_subplot(sp_rows, sp_cols, i + 1)
            pt = pd.pivot_table(data, values=['ws'], index=['ws_bin'], aggfunc='count').fillna(0)
            bar_x = [float(x[1:].split(',')[0]) for x in pt.index]
            bar_y = [x / sum(pt['ws']) for x in pt['ws']]
            weib_x = np.linspace(0, max(data['ws']), 1000)
            weib_y = stats.weibull_min(k, mu, lam).pdf(weib_x)

            plt.bar(bar_x, bar_y, width=1, label="data")
            plt.plot(weib_x, weib_y, 'r--', linewidth=2, label="weib fit")
            plt.xlabel('wind speed [m/s]', fontsize=lab_fsize)
            plt.ylabel('frequency', fontsize=lab_fsize)
            plt.title('WD={} A={} k={} u={}'.format(i_bin, round(lam, 2), round(k, 2),
                                                    round(np.mean(data['ws']), 2)), fontsize=lab_fsize)
            plt.legend(fontsize=lab_fsize)

    if plot:
        fig.suptitle('Weibull fit', fontsize=21)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.91, wspace=0.5, hspace=0.5)
        if savefig: plt.savefig('weib_fit.png', transparent=True)
        plt.show()

    return weibParam


if __name__ == "__main__":
    df = pd.read_csv("ncdc_Turin.txt", header=1)
    weibull_fit(df, plot=True, savefig=False)
