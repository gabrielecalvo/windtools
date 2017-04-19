import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import windtools.util as util


class Weibull(object):
    def __init__(self, data, ws_field='ws', wd_field='wd', wd_bin_size=30, ws_bin_size=1, prepare_data=True):
        self.data = pd.DataFrame(data)
        self.data.rename(columns={ws_field: 'ws', wd_field: 'wd'})
        self.param = None

        if prepare_data:
            self.prepare_data(wd_bin_size, ws_bin_size)

    @classmethod
    def load_raw_data(cls, fpath, ws_field='ws', wd_field='wd', wd_bin_size=30, **loading_options):
        field_map = {ws_field: 'ws', wd_field: 'wd'}
        df = util.load_data(fpath=fpath, field_map=field_map, loading_options=loading_options, dropna='any')
        return cls(data=df, wd_bin_size=wd_bin_size)

    def prepare_data(self, wd_bin_size, ws_bin_size):
        max_ws = self.data['ws'].max()
        self.data.ix[self.data['wd'] == 360] = 0
        self.data['wd_bin'] = pd.cut(self.data['wd'], bins=np.arange(0, 360.1, wd_bin_size))
        self.data['ws_bin'] = pd.cut(self.data['ws'], bins=np.arange(0, max_ws+0.1, ws_bin_size))
        self.data.dropna(inplace=True)

    def fit_distribution(self):
        result_dict = {}
        for bin_name, sub_df in self.data.groupby('wd_bin'):
            k, mu, lam = stats.weibull_min.fit(sub_df['ws'], floc=0)
            result_dict[bin_name] = {'k': k, 'mu': mu, 'lam': lam}
        self.param = pd.DataFrame(result_dict).T
        return self.param

    def create_plots(self, savefig=False):
        fig = plt.figure(figsize=(15, 12), dpi=80)

        sp_n = len(self.param.shape[0])
        sp_rows = int(np.sqrt(sp_n))
        sp_cols = np.ceil(sp_n / sp_rows)
        lab_fsize = int(-4 / 5 * sp_n + 20)

        for i, (bin_name, sub_df) in enumerate(self.data.groupby('wd_bin')):
            ax = fig.add_subplot(sp_rows, sp_cols, i + 1)

            k, = self.param.loc[bin_name, 'k']
            mu = self.param.loc[bin_name, 'mu']
            lam = self.param.loc[bin_name, 'lam']

            weib_x = np.linspace(0, max(sub_df['ws']), 1000)
            weib_y = stats.weibull_min(k, mu, lam).pdf(weib_x)

            # pt = pd.pivot_table(self.data, values=['ws'], index=['ws_bin'], aggfunc='count').fillna(0)
            # bar_x = [float(x[1:].split(',')[0]) for x in pt.index]
            # bar_y = [x / sum(pt['ws']) for x in pt['ws']]
            # plt.bar(bar_x, bar_y, width=1, label="data")

            plt.plot(weib_x, weib_y, 'r--', linewidth=2, label="weib fit")
            plt.xlabel('wind speed [m/s]', fontsize=lab_fsize)
            plt.ylabel('frequency', fontsize=lab_fsize)
            plt.title('WD={} A={} k={} u={}'.format(bin_name, round(lam, 2), round(k, 2),
                                                    round(np.mean(sub_df['ws']), 2)), fontsize=lab_fsize)
            plt.legend(fontsize=lab_fsize)
            fig.suptitle('Weibull fit', fontsize=21)
            fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.91, wspace=0.5, hspace=0.5)

        if savefig:
            plt.savefig('weib_fit.png', transparent=True)
        plt.show()

# todo: remove below after done refactoring
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
    import os
    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'samples', 'sample_data.csv')
    w = Weibull.load_raw_data(fpath)
    w.fit_distribution()
    w.create_plots()

    print(w.data.head())
    #weibull_fit(df, plot=True, savefig=False)
