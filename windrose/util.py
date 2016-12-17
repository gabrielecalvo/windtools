import pandas as pd
import numpy as np
import os
from itertools import zip_longest


def load_data(fpath, field_map=None, loading_options=None, dropna=None):
    if loading_options is None:
        loading_options = {}
    df = read_file(fpath, **loading_options)

    if field_map:
        df = df[list(field_map.keys())]
        df.columns = field_map.values()
    if dropna:
        df.dropna(how=dropna, inplace=True)
    return df


def read_file(fpath, **kwargs):
    assert isinstance(fpath, str)
    df = pd.DataFrame()
    fname, ext = os.path.splitext(fpath)
    if ext in ['.xls', '.xlsx', '.xlsm']:
        df = pd.read_excel(fpath, **kwargs)
    else:
        encodings = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        for encoding in encodings:
            try:
                df = pd.read_csv(fpath, encoding=encoding, **kwargs)
                break
            except UnicodeDecodeError:
                pass
    return df


def bin_data(data, first_centre, step, max_val=None, periodicity=None):
    if periodicity:
        max_val = periodicity
    elif not max_val:
        max_val = max(data)

    bins_edges = define_bins(first_centre=first_centre, step=step, max_val=max_val)
    bins = pd.cut(data, bins=bins_edges, right=False)

    if periodicity:
        first_cat = bins.cat.categories[0]
        last_cat = bins.cat.categories[-1]
        first_cat_edges = bin_label_to_bin_edges(first_cat)
        last_cat_edges = bin_label_to_bin_edges(last_cat)
        if (first_cat_edges[0] % periodicity == last_cat_edges[0] % periodicity) and \
           (first_cat_edges[1] % periodicity == last_cat_edges[1] % periodicity):
            bins.ix[bins == last_cat] = first_cat
            bins = bins.cat.remove_categories([last_cat])
    return bins


def define_bins(first_centre, step, max_val):
    assert first_centre <= step / 2
    start = first_centre - step / 2
    num = np.ceil((max_val - start) / step) + 1
    stop = start + step * (num-1)
    return np.linspace(start=start, stop=stop, num=num)


def pair_list(t, size=2, fillvalue=None):
    it = iter(t)
    return zip_longest(*[it] * size, fillvalue=fillvalue)


def bin_label_to_bin_edges(label):
    right = label[-1] == ']'
    edges = [eval(i.strip()) for i in label[1:-1].split(',')]
    return edges[0], edges[1], right



if __name__ == '__main__':
    fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests', 'samples', '160590-99999-2015.csv')
    df = load_data(fpath, loading_options=dict(index_col=0, na_values=[999]))
    df.dropna(inplace=True)
    r = bin_data(df['wd'], first_centre=0, step=90, periodicity=360)
    print(r)


#
# ## ======================================== ##
# ## CREATING FAKE DATA
# def create_fake_data():
#     data = np.random.rand(100, 2)
#     data[:, 0] = data[:, 0] * 15
#     data[:, 1] = data[:, 1] * 360
#     with open("ncdc.txt", "w") as f:
#         f.write("ws,wd\n")
#         for ws, wd in data:
#             f.write("{},{}\n".format(round(ws, 2), round(wd)))
#
#
# ## SIMPLE NCDC PARSING
# def ncdc_simple_parser(f_in, f_out):
#     parsed = []
#     with open(f_in, "r") as f:
#         for i, l in enumerate(f):
#             if i == 0:
#                 parsed.append(["station:" + f.name, "latlong:" + l[28:41], "elev:" + l[46:51]])
#                 parsed.append(["ts", "ws", "wd"])
#             dt = "{}-{}-{} {}:{}".format(l[15:19], l[19:21], l[21:23], l[23:25], l[25:27])
#             parsed.append([dt, str(float(l[65:69]) / 10), l[60:63]])
#     with open(f_out, "w") as f:
#         for i in parsed:
#             f.write(','.join(i) + '\n')
#
#
# ## ======================================== ##
#
#
# # def add_data_bins(df, wsb_size=4, wdb_size=30, ws_midbin=False):
# #     # setting the bins
#     ofst = wsb_size / 2 if ws_midbin else 0
#     wsb = [x - ofst for x in range(0, int(max(df['ws']) + 1) + wsb_size, wsb_size)]
#     wdb = [x - wdb_size / 2 for x in range(0, 360 + 2 * wdb_size, wdb_size)]
#     df['ws_bin'] = pd.cut(df['ws'], wsb, include_lowest=True, right=False)
#     df['wd_bin'] = pd.cut(df['wd'], wdb, include_lowest=True, right=False)
#
#     # merge first and last wd bins (around the 0°)
#     wd_bin_names = [x for x in df['wd_bin'].unique() if pd.notnull(x)]
#     wd_bin_names = sorted(wd_bin_names, key=lambda x: float(x[1:].split(",")[0]))
#     df['wd_bin'].replace(wd_bin_names[-1], wd_bin_names[0], inplace=True)
#
#     return df
#
#
# def get_freq_table(df, scaling=None):
#     # create pivot table
#     dfb = pd.pivot_table(df, values=["ws"], index=["ws_bin"], columns=["wd_bin"], aggfunc='count').fillna(0)
#
#     # scaling (to make it equal to the Windfarm tool scaling=8760)
#     if scaling: dfb = dfb / dfb.sum().sum() * scaling
#
#     # reorder rows based on bin and return
#     row_order = sorted(dfb.index, key=lambda x: float(x[1:].split(",")[0]))
#     col_order = sorted(list(dfb['ws']), key=lambda x: float(x[1:].split(",")[0]))
#     return dfb['ws'][col_order].loc[row_order, :]
#
#
# def plot_WR(ft, fss=None, legend=True, savefig=False):
#     n_row, n_col = ft.shape
#     offset = np.pi / n_col
#     width = 2 * np.pi / n_col
#     radii = []
#
#     theta_wr = np.linspace(0.0 + offset, n_row * 2 * np.pi + offset, n_col * n_row, endpoint=False)
#     theta = -theta_wr + np.pi / 2
#     for i in range(n_row, 0, -1):
#         radii.extend(ft.ix[0:i].sum())
#
#     # wind rose
#     ax = plt.subplot(111, polar=True)
#     bars = ax.bar(theta, radii, width=width, bottom=0.0)
#     ax.set_xticklabels([90, 45, 0, 315, 270, 225, 180, 135])
#     for i, bar in enumerate(bars):
#         bar.set_facecolor(plt.cm.jet(1 - int(i / n_col) / (n_row)))
#         if i % n_col == 0:
#             idx = int(-i / n_col + (n_row - 1))
#             bar.set_label(ft.index[idx])
#
#     # free stream sector
#     if fss:
#         theta_fss = (90 - fss[1]) * (np.pi / 180)
#         width_fss = (fss[1] - fss[0] + 360) % 360 * (np.pi / 180)
#         bar_fss = ax.bar(theta_fss, 1.05 * max(radii), width=width_fss,
#                          bottom=0.0, label='fss')
#         bar_fss[0].set_facecolor('y')
#         bar_fss[0].set_alpha(0.5)
#
#     if legend: ax.legend(bbox_to_anchor=(1.3, 1.1))
#     if savefig: plt.savefig('demo.png', transparent=True)
#     plt.show()
#
#
# if __name__ == "__main__":
#     # create_fake_data()
#     # ncdc_simple_parser(".//tests//160610-99999-2015","ncdc_Turin.txt")
#     df = pd.read_csv("ncdc_Turin.txt", header=1)  # reading the data in
#     dfb = add_data_bins(df, wsb_size=4, wdb_size=30, ws_midbin=False)
#     ft = get_freq_table(dfb, scaling=8760)
#     print(ft)
#     plot_WR(ft, fss=[235, 7], legend=True, savefig=False)
#
