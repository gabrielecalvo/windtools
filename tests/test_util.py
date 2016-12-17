import pytest
import pandas as pd
import numpy as np
import pandas.util.testing as pdt

from windrose.util import define_bins, pair_list, load_data, bin_label_to_bin_edges, bin_data
from tests.aux_testing_tools import samples_fld, TestExcel

test_excel = TestExcel(fpath=samples_fld('testing.xlsx'))

@pytest.mark.parametrize('centre,step,max_val,exp', [
    (-2, 1, 2, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
    (-1.5, 1, 2, [-2, -1, 0, 1, 2]),
    (-2, 1, 2.5, [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]),
    (-1.5, 1, 2.5, [-2, -1, 0, 1, 2, 3]),
    (-2, 2, 2, [-3, -1, 1, 3]),
    (-1.5, 2, 2, [-2.5, -0.5, 1.5, 3.5]),
    (-2, 2, 2.5, [-3, -1, 1, 3]),
    (-1.5, 2, 2.5, [-2.5, -0.5, 1.5, 3.5]),
])
def test_define_bins_ends(centre, step, max_val, exp):
    np.testing.assert_equal(define_bins(centre, step, max_val), exp)


def test_pair_list_fail():
    target = [1, 2, 3, 4]
    assert not list(pair_list(target)) == target


@pytest.mark.parametrize('target, exp', [
    ([1, 2, 3, 4], [(1, 2), (3, 4)]),
    ([1, 2, 3, 4, 5], [(1, 2), (3, 4), (5, None)]),
    ([], []),
    ([1], [(1, None)]),
])
def test_pair_list_normal(target, exp):
    actual = pair_list(target)
    assert list(actual) == exp


@pytest.mark.parametrize('target, size, fillvalue, exp', [
    ([], 3, None, []),
    ([1], 3, None, [(1, None, None)]),
    ([1, 2], 3, None, [(1, 2, None)]),
    ([1, 2, 3], 3, None, [(1, 2, 3)]),
    ([1, 2, 3, 4], 3, None, [(1, 2, 3), (4, None, None)]),
    ([1, 2, 3, 4], 3, '_', [(1, 2, 3), (4, '_', '_')]),
])
def test_pair_list_other_options(target, size, fillvalue, exp):
    actual = pair_list(target, size=size, fillvalue=fillvalue)
    assert list(actual) == exp


def test_load_raw_data():
    fpath = samples_fld('160590-99999-2015.csv')

    df = load_data(fpath)
    assert list(df.columns) == ['ts', 'ws', 'wd', 'ws2', 'wd2']
    actual = pd.Series(['2015-01-01 00:00', np.nan, 240, -0.2431753934, 238.2065799763])
    pdt.assert_almost_equal(df.iloc[0, :].tolist(), actual)

    df = load_data(fpath, dropna='any')
    assert list(df.columns) == ['ts', 'ws', 'wd', 'ws2', 'wd2']
    actual = pd.Series(['2015-01-01 01:50', 1.5, 190, 1.0175993219, 189.5036829752])
    assert pdt.assert_almost_equal(df.iloc[0, :].tolist(), actual)

    df = load_data(fpath, loading_options={'index_col': 0}, field_map={'ws2': 'ws', 'wd': 'wd'}, dropna='any')
    assert sorted(list(df.columns)) == ['wd', 'ws']
    assert df['ws'].iloc[0] == -0.2431753934
    assert df['wd'].iloc[0] == 240


@pytest.mark.parametrize('label, exp', [
    ('[-10, 10)', (-10, 10, False)),
    ('[350, 10)', (350, 10, False)),
    ('[45, 135)', (45, 135, False)),
    ('(45, 135]', (45, 135, True)),
])
def test_bin_label_to_bin_edges(label, exp):
    assert bin_label_to_bin_edges(label) == exp


def test_bin_data():
    d_linear = pd.Series([1.1, 3.5, 4.0, 5.9])
    r = bin_data(data=d_linear, first_centre=0.5, step=1)
    assert len(r.cat.categories) == 6
    assert r.tolist() == ['[1, 2)', '[3, 4)', '[4, 5)', '[5, 6)']

    d_periodic = pd.Series([0, 0.1, 180, 350, 359.9])
    r = bin_data(data=d_periodic, first_centre=0, step=30, periodicity=360)
    assert len(r.cat.categories) == 12
    assert r.tolist() == ['[-15, 15)', '[-15, 15)', '[165, 195)', '[-15, 15)', '[-15, 15)']
