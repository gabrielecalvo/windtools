import pytest
import os
import pandas as pd
import numpy as np
import filecmp
from windrose.windrose import WindRose

from tests.aux_testing_tools import samples_fld, tmp_fld, TestExcel
test_excel = TestExcel(fpath=samples_fld('testing.xlsx'))

@pytest.fixture()
def sample_wr():
    fpath = samples_fld('160590-99999-2015.csv')
    wr = WindRose(fpath=fpath, na_values=[999, 999.9])
    wr.calc_freq_table(ws_bin_step=6, ws_bin_centre=3,
                       wd_bin_step=30, wd_bin_centre=0)
    return wr


def test_acceptance_from_raw_data(sample_wr):
    df_expected = test_excel.get_df_from_sheet(sheetname='acceptance_freq_table', index_col=0)
    np.testing.assert_almost_equal(sample_wr.freq_table.as_matrix(), df_expected.as_matrix())


def test_plot_windrose(sample_wr):
    tmp_names = ['windrose0.png', 'windrose1.png', 'windrose2.png', 'windrose3.jpg']
    test_names = ['valid_{}'.format(i) for i in tmp_names]
    tmp_paths = [tmp_fld(i) for i in tmp_names]
    test_paths = [samples_fld(i) for i in test_names]

    # defaults
    sample_wr.plot_windrose(overlay_sector=None, legend=True, save_fig=tmp_paths[0])
    # with fss
    sample_wr.plot_windrose(overlay_sector=[350, 10], legend=False, save_fig=tmp_paths[1])
    # file formats
    sample_wr.plot_windrose(overlay_sector=[350, 10, 40, 55], legend=True, save_fig=tmp_paths[2])
    sample_wr.plot_windrose(overlay_sector=[350, 10, 40, 55], legend=True, save_fig=tmp_paths[3])

    # check file created and clean up
    for fname, testname in zip(tmp_paths, test_paths):
        assert os.path.isfile(fname)
        assert os.path.isfile(testname)
        assert filecmp.cmp(fname, testname)

    # no save but show
    sample_wr.plot_windrose(overlay_sector=[350, 10], legend=True, save_fig=False)

