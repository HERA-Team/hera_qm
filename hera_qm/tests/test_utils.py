# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import nose.tools as nt
import os
import pyuvdata.tests as uvtest
from pyuvdata import UVCal
from pyuvdata import UVData
from hera_qm import utils
from hera_qm.data import DATA_PATH
from hera_qm.ant_metrics import get_ant_metrics_dict
from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
from hera_qm.omnical_metrics import get_omnical_metrics_dict
from hera_qm.utils import get_metrics_dict
import numpy as np


def test_get_pol():
    filename = 'zen.2457698.40355.xx.HH.uvcA'
    nt.assert_equal(utils.get_pol(filename), 'xx')


def test_gen_full_returns_input_for_full_pol_files_uvh5():
    pol_list = ['xx', 'xy', 'yx', 'yy']
    full_pol_file = os.path.join(DATA_PATH,
                                 'zen.2457698.40355.full_pol_test.uvh5')
    full_pol_file_list = [full_pol_file]
    expected_output_list = [[full_pol_file]]
    output_files = utils.generate_fullpol_file_list(full_pol_file_list, pol_list)
    nt.assert_equal(sorted(expected_output_list), sorted(output_files))


def test_gen_full_returns_input_for_full_pol_files_uvfits():
    pol_list = ['xx', 'xy', 'yx', 'yy']
    full_pol_file = os.path.join(DATA_PATH,
                                 'zen.2457698.40355.full_pol_test.uvfits')
    full_pol_file_list = [full_pol_file]
    expected_output_list = [[full_pol_file]]
    output_files = utils.generate_fullpol_file_list(full_pol_file_list, pol_list)
    nt.assert_equal(sorted(expected_output_list), sorted(output_files))


def test_gen_full_returns_fullpol_for_partial_pol_files():
    pol_list = ['xx', 'xy', 'yx', 'yy']
    partial_pol_file = os.path.join(DATA_PATH, 'zen.2457698.40355.partial_pol_test.uvh5')
    partial_pol_file_list = [partial_pol_file]
    nt.assert_raises(ValueError, utils.generate_fullpol_file_list,
                     partial_pol_file_list, pol_list)


def test_generate_fullpol_file_list():
    pol_list = ['xx', 'xy', 'yx', 'yy']
    xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
    xy_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xy.HH.uvcA')
    yx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.yx.HH.uvcA')
    yy_file = os.path.join(DATA_PATH, 'zen.2457698.40355.yy.HH.uvcA')
    file_list = [xx_file, xy_file, yx_file, yy_file]

    # feed in one file at a time
    fullpol_file_list = utils.generate_fullpol_file_list([xx_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([xy_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([yx_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([yy_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))

    # feed in all four files
    fullpol_file_list = utils.generate_fullpol_file_list(file_list, pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))

    # checks that we have a list of lists with outer length of 1, and inner length of 4
    nt.assert_equal(len(fullpol_file_list), 1)
    nt.assert_equal(len(fullpol_file_list[0]), 4)

    # try to pass in a file that doesn't have all pols present
    lone_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
    fullpol_file_list = uvtest.checkWarnings(utils.generate_fullpol_file_list,
                                             [[lone_file], pol_list], nwarnings=1,
                                             message='Could not find')
    nt.assert_equal(fullpol_file_list, [])


def test_get_metrics_ArgumentParser_ant_metrics():
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.crossCut, 5.0)
    nt.assert_equal(args.alwaysDeadCut, 10.0)
    nt.assert_equal(args.metrics_path, '')
    nt.assert_equal(args.verbose, True)
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    nt.assert_equal(args.extension, 'foo')


def test_get_metrics_ArgumentParser_firstcal_metrics():
    a = utils.get_metrics_ArgumentParser('firstcal_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.std_cut, 0.5)
    nt.assert_equal(args.extension, '.firstcal_metrics.hdf5')
    nt.assert_equal(args.metrics_path, '')
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    nt.assert_equal(args.extension, 'foo')


def test_get_metrics_ArgumentParser_omnical_metrics():
    a = utils.get_metrics_ArgumentParser('omnical_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.fc_files, None)
    nt.assert_equal(args.phs_std_cut, 0.3)
    nt.assert_equal(args.extension, '.omni_metrics.json')
    nt.assert_equal(args.metrics_path, '')
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    nt.assert_equal(args.extension, 'foo')


def test_get_metrics_ArgumentParser_xrfi_h1c_run():
    a = utils.get_metrics_ArgumentParser('xrfi_h1c_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.infile_format, 'miriad')
    nt.assert_equal(args.summary_ext, 'flag_summary.h5')
    nt.assert_equal(args.algorithm, 'xrfi_simple')
    nt.assert_equal(args.nsig_df, 6.0)
    nt.assert_equal(args.px_threshold, 0.2)
    # try to set something
    args = a.parse_args(['--px_threshold', '4.0'])
    nt.assert_equal(args.px_threshold, 4.0)


def test_get_metrics_ArgumentParser_delay_xrfi_h1c_idr2_1_run():
    a = utils.get_metrics_ArgumentParser('delay_xrfi_h1c_idr2_1_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.infile_format, 'miriad')
    nt.assert_equal(args.algorithm, 'xrfi_simple')
    nt.assert_equal(args.nsig_dt, 6.0)
    nt.assert_equal(args.px_threshold, 0.2)
    nt.assert_equal(args.filename, None)
    nt.assert_equal(args.tol, 1e-7)
    nt.assert_equal(args.waterfalls, None)
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    nt.assert_equal(args.waterfalls, 'a,g')


def test_get_metrics_ArgumentParser_xrfi_run():
    a = utils.get_metrics_ArgumentParser('xrfi_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.init_metrics_ext, 'init_xrfi_metrics.h5')
    nt.assert_equal(args.kt_size, 8)
    nt.assert_equal(args.sig_init, 6.0)
    nt.assert_equal(args.freq_threshold, 0.35)
    nt.assert_equal(args.ex_ants, None)
    # try to set something
    args = a.parse_args(['--time_threshold', '0.4'])
    nt.assert_equal(args.time_threshold, 0.4)


def test_get_metrics_ArgumentParser_xrfi_apply():
    a = utils.get_metrics_ArgumentParser('xrfi_apply')
    # First try defaults - test a few of them
    args = a.parse_args('')
    nt.assert_equal(args.infile_format, 'miriad')
    nt.assert_equal(args.extension, 'R')
    nt.assert_equal(args.flag_file, None)
    nt.assert_equal(args.output_uvflag, True)
    nt.assert_equal(args.output_uvflag_ext, 'flags.h5')
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    nt.assert_equal(args.waterfalls, 'a,g')


def test_get_metrics_ArgumentParser_error():
    # raise error for requesting unknown type of parser
    nt.assert_raises(AssertionError, utils.get_metrics_ArgumentParser, 'fake_method')


def test_metrics2mc():
    # test ant metrics
    filename = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    d = utils.metrics2mc(filename, ftype='ant')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    nt.assert_equal(len(d['array_metrics']), 0)
    ant_metrics_list = get_ant_metrics_dict()
    nt.assert_equal(set(d['ant_metrics'].keys()), set(ant_metrics_list.keys()))

    # test firstcal metrics
    filename = os.path.join(DATA_PATH, 'example_firstcal_metrics.json')
    d = utils.metrics2mc(filename, ftype='firstcal')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    firstcal_array_metrics = set(['firstcal_metrics_agg_std_y',
                                  'firstcal_metrics_good_sol_y',
                                  'firstcal_metrics_max_std_y'])
    nt.assert_equal(set(d['array_metrics'].keys()), firstcal_array_metrics)
    firstcal_metrics_list = get_firstcal_metrics_dict()
    firstcal_ant_metrics = set(firstcal_metrics_list.keys()) - firstcal_array_metrics
    # remove others not in this data file
    firstcal_ant_metrics -= {'firstcal_metrics_good_sol_x', 'firstcal_metrics_good_sol',
                             'firstcal_metrics_agg_std_x', 'firstcal_metrics_agg_std',
                             'firstcal_metrics_max_std_x'}
    nt.assert_equal(set(d['ant_metrics']), firstcal_ant_metrics)

    # test omnical metrics
    filename = os.path.join(DATA_PATH, 'example_omnical_metrics.json')
    d = utils.metrics2mc(filename, ftype='omnical')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    om = 'omnical_metrics_'
    nt.assert_equal(set(d['array_metrics'].keys()),
                    set([om + 'chisq_tot_avg_XX', om + 'chisq_good_sol_XX',
                         om + 'chisq_tot_avg_YY', om + 'chisq_good_sol_YY',
                         om + 'ant_phs_std_max_XX', om + 'ant_phs_std_good_sol_XX',
                         om + 'ant_phs_std_max_YY', om + 'ant_phs_std_good_sol_YY']))
    nt.assert_equal(set(d['ant_metrics'].keys()),
                    set([om + 'chisq_ant_avg', om + 'chisq_ant_std', om + 'ant_phs_std']))
    nt.assert_equal(len(d['ant_metrics'][om + 'chisq_ant_avg']), 32)

    # Hit the exceptions
    nt.assert_raises(ValueError, utils.metrics2mc, filename, ftype='foo')


def test_get_metrics_dict():
    ant_metrics_dict = get_ant_metrics_dict()
    firstcal_metrics_dict = get_firstcal_metrics_dict()
    omnical_metrics_dict = get_omnical_metrics_dict()
    metrics_dict = get_metrics_dict()
    for key in ant_metrics_dict:
        nt.assert_equal(ant_metrics_dict[key], metrics_dict[key])
    for key in firstcal_metrics_dict:
        nt.assert_equal(firstcal_metrics_dict[key], metrics_dict[key])
    for key in omnical_metrics_dict:
        nt.assert_equal(omnical_metrics_dict[key], metrics_dict[key])


def test_dynamic_slice():
    a = np.arange(10).reshape(2, 5)
    b = utils.dynamic_slice(a, slice(1, 3))
    nt.assert_equal(b.shape, (2, 2))
    np.testing.assert_array_equal(b, np.array([[1, 2], [6, 7]]))
    b = utils.dynamic_slice(a, slice(1, 3), axis=1)
    nt.assert_equal(b.shape, (2, 2))
    b = utils.dynamic_slice(a, slice(1, None), axis=0)
    nt.assert_equal(b.shape, (1, 5))

    nt.assert_raises(ValueError, utils.dynamic_slice, 'foo', slice(0, None))


def test_strip_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root = utils.strip_extension(path)
    nt.assert_equal(root, path[:-4])


def test_strip_extension_return_ext_basename():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    nt.assert_equal(root, path[:-4])


def test_strip_extension_return_ext_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    nt.assert_equal(ext, path[-3:])
