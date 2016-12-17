import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from windrose import util


class WindRose(object):
    def __init__(self, fpath=None, ws_field='ws', wd_field='wd', **loading_options):
        self.raw_data = pd.DataFrame()
        self.freq_table = pd.DataFrame()
        if fpath:
            self.field_map = {ws_field: 'ws', wd_field: 'wd'}
            self._load_raw_data(fpath, self.field_map, loading_options)

    def __repr__(self):
        inv_map = {v: k for k, v in self.field_map.items()}
        return 'WindRose ["{}" vs "{}"]'.format(inv_map['ws'], inv_map['wd'])

    def _load_raw_data(self, fpath, field_map, loading_options):
        self.raw_data = util.load_data(fpath=fpath, field_map=field_map, loading_options=loading_options, dropna='any')
        self.raw_data.ix[self.raw_data['wd'] == 360, 'wd'] = 0

    def calc_freq_table(self, ws_bin_step=6, ws_bin_centre=3, wd_bin_step=10, wd_bin_centre=0):
        self.raw_data['ws_bin'] = util.bin_data(data=self.raw_data['ws'], first_centre=ws_bin_centre, step=ws_bin_step)
        self.raw_data['wd_bin'] = util.bin_data(data=self.raw_data['wd'], first_centre=wd_bin_centre, step=wd_bin_step,
                                                periodicity=360)
        n_data = self.raw_data.shape[0]
        count_pivot = pd.pivot_table(data=self.raw_data, values='ws', index=['ws_bin'], columns='wd_bin',
                                     aggfunc='count', fill_value=0)
        self.freq_table = count_pivot/n_data
        return self.freq_table

    def plot_windrose(self, overlay_sector=None, overlay_sector_name='sector', legend=True, labels='degrees',
                      save_fig=None):
        """
        Plots the wind rose according to configurations. The legend can be removed.
        An overlay sector can be added and the labels can be customized.
        The resulting plot can be saved to file or shown.

        :param overlay_sector: list of start and end of the sector to overlay
        (e.g. from 350 to 10 and 30 to 40: [350, 10, 30, 40])
        :param legend: boolean to determine if legend should be displayed
        :param labels: list or string to determine what labels to put around the windrose.
        Default 'degrees' will put the angle in degrees every 30 degrees;
        'cardinal' will put the cardinal direction (e.g. N, SW etc.) every 30 degrees;
        A list will space the elements in the list evenly around the windrose.
        :param save_fig: It will save the plot to the specified file path unless set to None.
        None (default) will show the plot instead.
        """
        n_row, n_col = self.freq_table.shape
        offset = np.pi / n_col
        width = 2 * np.pi / n_col

        # calculate xs (thetas) and ys (radii) of each bar to plot
        theta_wr = np.linspace(0. + offset, n_row * 2 * np.pi + offset, n_col * n_row, endpoint=False)
        theta = -theta_wr + np.pi / 2
        radii = []
        for i in range(n_row, 0, -1):
            radii.extend(self.freq_table.ix[0:i].sum())

        # determines labes (xticklabels)
        if labels == 'degrees':
            xticklabels = [90, 45, 0, 315, 270, 225, 180, 135]
        elif labels == 'cardinal':
            xticklabels = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        else:
            xticklabels = labels

        # create wind rose
        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='polar')
        bars = ax.bar(theta, radii, width=width, bottom=0.)
        ax.set_xticklabels(xticklabels)
        for i, bar in enumerate(bars):
            bar.set_facecolor(plt.cm.jet(1 - int(i / n_col) / n_row))
            if i % n_col == 0:
                idx = int(-i / n_col + (n_row - 1))
                bar.set_label(self.freq_table.index[idx])

        # add overlay sector
        if overlay_sector:
            overlay_label = overlay_sector_name
            for pair in util.pair_list(overlay_sector):
                theta_fss = (90 - pair[1]) * (np.pi / 180)
                width_fss = (pair[1] - pair[0] + 360) % 360 * (np.pi / 180)
                bar_fss = ax.bar(theta_fss, 1.05 * max(radii), width=width_fss, bottom=0.0, label=overlay_label)
                bar_fss[0].set_facecolor('y')
                bar_fss[0].set_alpha(0.5)
                overlay_label = None  # so only one label will be displayed

        if legend:
            ax.legend(bbox_to_anchor=(1.3, 1.1))

        if isinstance(save_fig, str):
            plt.savefig(save_fig, transparent=True, bbox_inches='tight')
        else:
            plt.show()

    def export_frequency_table(self, destination):
        self.freq_table.to_csv(destination)
