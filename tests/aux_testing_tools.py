import os
import shutil
import pandas as pd

tmp_fld = lambda x: os.path.join(os.path.dirname(__file__), 'tmp', x)
samples_fld = lambda x: os.path.join(os.path.dirname(__file__), 'samples', x)


def set_tmp_as_valid():
    tmp_names = ['windrose0.png', 'windrose1.png', 'windrose2.png', 'windrose3.jpg']
    test_names = ['valid_{}'.format(i) for i in tmp_names]
    tmp_paths = [tmp_fld(i) for i in tmp_names]
    test_paths = [samples_fld(i) for i in test_names]

    for fp_tmp, fp_valid in zip(tmp_paths, test_paths):
        print(fp_tmp, fp_valid)
        shutil.copy(src=fp_tmp, dst=fp_valid)


class TestExcel(object):
    def __init__(self, fpath):
        self.path = fpath

    def get_df_from_sheet(self, sheetname=0, **loading_options):
        return pd.read_excel(io=self.path, sheetname=sheetname, **loading_options)


if __name__ == '__main__':
    set_tmp_as_valid()
