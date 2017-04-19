import pytest
import os
import sys
import shutil
from datetime import datetime
import pandas as pd
import numpy as np
import pandas.util.testing as pdt

from windtools.util import bin_data, bin_label_to_bin_edges, define_bins, download_file, ensure_folder, get_path, \
    load_data, pair_list, read_file
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
    fpath = samples_fld('sample_data.csv')

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


def test_ensure_folder():
    root = os.path.dirname(os.path.abspath(__file__))
    new_folder = os.path.join(root, 'tmp', 'test_ensure_folder')
    assert not os.path.isdir(new_folder)
    ensure_folder(new_folder)
    assert os.path.isdir(new_folder)
    shutil.rmtree(new_folder)


def test_read_file():
    root = os.path.dirname(os.path.abspath(__file__))
    expected = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [datetime(2015, 1, 7), datetime(2015, 1, 8), datetime(2015, 1, 9)],
        'c': ['A', 'B', 'C']
    })

    for fname in ['test_read.xls', 'test_read.xlsx', 'test_read.xlsm']:
        fpath = os.path.join(root, 'samples', fname)
        df = read_file(fpath)
        assert df.equals(expected)

    for fname in ['test_read_1.csv', 'test_read_2.csv', 'test_read_3.csv']:
        fpath = os.path.join(root, 'samples', fname)
        df = read_file(fpath, parse_dates=['b'], dayfirst=True)
        assert df.equals(expected)


def test_download_file():
    root = get_path(__file__)
    url_test = 'http://example.org/'

    r = download_file(url_test)
    assert sys.getsizeof(r) == 1287

    save_to = os.path.join(root, 'tmp', 't')
    download_file(url_test, save_to=save_to)
    assert os.stat(save_to).st_size == 1270
    os.remove(save_to)