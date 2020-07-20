# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import pytest
import os
import pyuvdata.tests as uvtest
from hera_qm import utils
from hera_qm.data import DATA_PATH
from hera_qm.ant_metrics import get_ant_metrics_dict
from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
from hera_qm.omnical_metrics import get_omnical_metrics_dict
from hera_qm.utils import get_metrics_dict
import numpy as np


def test_get_metrics_ArgumentParser_ant_metrics():
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.crossCut == 5.0
    assert args.metrics_path == ''
    assert args.verbose is True
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    assert args.extension == 'foo'


def test_get_metrics_ArgumentParser_firstcal_metrics():
    a = utils.get_metrics_ArgumentParser('firstcal_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.std_cut == 0.5
    assert args.extension == '.firstcal_metrics.hdf5'
    assert args.metrics_path == ''
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    assert args.extension == 'foo'


def test_get_metrics_ArgumentParser_omnical_metrics():
    a = utils.get_metrics_ArgumentParser('omnical_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.fc_files is None
    assert args.phs_std_cut == 0.3
    assert args.extension == '.omni_metrics.json'
    assert args.metrics_path == ''
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    assert args.extension, 'foo'


def test_get_metrics_ArgumentParser_xrfi_h1c_run():
    a = utils.get_metrics_ArgumentParser('xrfi_h1c_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.summary_ext == 'flag_summary.h5'
    assert args.algorithm == 'xrfi_simple'
    assert args.nsig_df == 6.0
    assert args.px_threshold == 0.2
    # try to set something
    args = a.parse_args(['--px_threshold', '4.0'])
    assert args.px_threshold == 4.0


def test_get_metrics_ArgumentParser_delay_xrfi_h1c_idr2_1_run():
    a = utils.get_metrics_ArgumentParser('delay_xrfi_h1c_idr2_1_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.algorithm == 'xrfi_simple'
    assert args.nsig_dt == 6.0
    assert args.px_threshold == 0.2
    assert args.filename is None
    assert args.tol == 1e-7
    assert args.waterfalls is None
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    assert args.waterfalls == 'a,g'


def test_get_metrics_ArgumentParser_xrfi_run():
    a = utils.get_metrics_ArgumentParser('xrfi_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init == 6.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj', '3.0'])
    assert args.sig_adj == 3.0


def test_get_metrics_ArgumentParser_xrfi_run_data_only():
    a = utils.get_metrics_ArgumentParser('xrfi_run_data_only')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init == 6.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj', '3.0'])
    assert args.sig_adj == 3.0


def test_get_metrics_ArgumentParser_day_threshold_run():
    a = utils.get_metrics_ArgumentParser('day_threshold_run')
    # First try defaults - test a few of them
    args = a.parse_args(['fooey'])
    assert args.nsig_f_adj == 3.0
    assert args.nsig_f == 7.0
    assert args.data_files == ['fooey']
    # try to set something
    args = a.parse_args(['--nsig_t', '3.0', 'fooey'])
    assert args.nsig_t == 3.0


def test_get_metrics_ArgumentParser_xrfi_apply():
    a = utils.get_metrics_ArgumentParser('xrfi_apply')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.extension == 'R'
    assert args.flag_file is None
    assert args.output_uvflag is True
    assert args.output_uvflag_ext == 'flags.h5'
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    assert args.waterfalls == 'a,g'


def test_get_metrics_ArgumentParser_error():
    # raise error for requesting unknown type of parser
    pytest.raises(AssertionError, utils.get_metrics_ArgumentParser, 'fake_method')


def test_get_metrics_ArgumentParser_xrfi_h3c_idr2_1_run():
    a = utils.get_metrics_ArgumentParser('xrfi_h3c_idr2_1_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init == 6.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj', '3.0', '--ocalfits_files', 'foo', 'boo'])
    assert args.sig_adj == 3.0
    assert len(args.ocalfits_files) == 2


def test_metrics2mc():
    # test ant metrics
    filename = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    d = utils.metrics2mc(filename, ftype='ant')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    assert len(d['array_metrics']) == 0
    ant_metrics_list = get_ant_metrics_dict()
    assert set(d['ant_metrics'].keys()) == set(ant_metrics_list.keys())

    # test firstcal metrics
    filename = os.path.join(DATA_PATH, 'example_firstcal_metrics.json')
    d = utils.metrics2mc(filename, ftype='firstcal')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    firstcal_array_metrics = set(['firstcal_metrics_agg_std_y',
                                  'firstcal_metrics_good_sol_y',
                                  'firstcal_metrics_max_std_y'])
    assert set(d['array_metrics'].keys()) == firstcal_array_metrics
    firstcal_metrics_list = get_firstcal_metrics_dict()
    firstcal_ant_metrics = set(firstcal_metrics_list.keys()) - firstcal_array_metrics
    # remove others not in this data file
    firstcal_ant_metrics -= {'firstcal_metrics_good_sol_x', 'firstcal_metrics_good_sol',
                             'firstcal_metrics_agg_std_x', 'firstcal_metrics_agg_std',
                             'firstcal_metrics_max_std_x'}
    assert set(d['ant_metrics']) == firstcal_ant_metrics

    # test omnical metrics
    filename = os.path.join(DATA_PATH, 'example_omnical_metrics.json')
    d = utils.metrics2mc(filename, ftype='omnical')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    om = 'omnical_metrics_'
    assert set(d['array_metrics'].keys()) == set([om + 'chisq_tot_avg_XX',
                                                  om + 'chisq_good_sol_XX',
                                                  om + 'chisq_tot_avg_YY',
                                                  om + 'chisq_good_sol_YY',
                                                  om + 'ant_phs_std_max_XX',
                                                  om + 'ant_phs_std_good_sol_XX',
                                                  om + 'ant_phs_std_max_YY',
                                                  om + 'ant_phs_std_good_sol_YY'])
    assert set(d['ant_metrics'].keys()) == set([om + 'chisq_ant_avg',
                                                om + 'chisq_ant_std',
                                                om + 'ant_phs_std'])
    assert len(d['ant_metrics'][om + 'chisq_ant_avg']) == 32

    # Hit the exceptions
    pytest.raises(ValueError, utils.metrics2mc, filename, ftype='foo')


def test_get_metrics_dict():
    ant_metrics_dict = get_ant_metrics_dict()
    firstcal_metrics_dict = get_firstcal_metrics_dict()
    omnical_metrics_dict = get_omnical_metrics_dict()
    metrics_dict = get_metrics_dict()
    for key in ant_metrics_dict:
        assert ant_metrics_dict[key] == metrics_dict[key]
    for key in firstcal_metrics_dict:
        assert firstcal_metrics_dict[key] == metrics_dict[key]
    for key in omnical_metrics_dict:
        assert omnical_metrics_dict[key] == metrics_dict[key]


def test_dynamic_slice():
    a = np.arange(10).reshape(2, 5)
    b = utils.dynamic_slice(a, slice(1, 3))
    assert b.shape == (2, 2)
    np.testing.assert_array_equal(b, np.array([[1, 2], [6, 7]]))
    b = utils.dynamic_slice(a, slice(1, 3), axis=1)
    assert b.shape == (2, 2)
    b = utils.dynamic_slice(a, slice(1, None), axis=0)
    assert b.shape == (1, 5)

    pytest.raises(ValueError, utils.dynamic_slice, 'foo', slice(0, None))


def test_strip_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root = utils.strip_extension(path)
    assert root == path[:-4]


def test_strip_extension_return_ext_basename():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    assert root == path[:-4]


def test_strip_extension_return_ext_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    assert ext == path[-3:]
