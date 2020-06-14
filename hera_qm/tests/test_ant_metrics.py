# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
import os
import sys
import pyuvdata.tests as uvtest
from hera_qm import utils
from hera_qm import ant_metrics
from hera_qm import metrics_io
from hera_qm.data import DATA_PATH
import hera_qm.tests as qmtest


class fake_data():
    """Generate fake data with known values."""

    def __init__(self):
        self.data = {}
        for bl in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]:
            # data[bl] = {}
            for poli, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
                # Give each bl different data
                np.random.seed(bl[0] * 10 + bl[1] + 100 * poli)
                self.data[bl[0], bl[1], pol] = np.random.randn(2, 3)

    def get_data(self, *key):
        return self.data[key]

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()


@pytest.fixture(scope='function')
def lowlevel_data():
    data = fake_data()
    ants = [0, 1, 2, 3]
    pols = ['xx', 'xy', 'yx', 'yy']
    antpols = ['x', 'y']
    bls = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]

    class DataHolder(object):
        def __init__(self, data, ants, pols, antpols, bls):
            self.data = data
            self.ants = ants
            self.pols = pols
            self.antpols = antpols
            self.bls = bls

    lowlevel_data = DataHolder(data, ants, pols, antpols, bls)

    # yield returns the data we need but lets us continue after for cleanup
    yield lowlevel_data

    # post-test cleanup
    del(lowlevel_data)


def test_mean_Vij_metrics(lowlevel_data):
    mean_Vij = ant_metrics.mean_Vij_metrics(lowlevel_data.data,
                                            lowlevel_data.pols,
                                            lowlevel_data.antpols,
                                            lowlevel_data.ants,
                                            lowlevel_data.bls,
                                            rawMetric=True)
    # The reference dictionaries here
    # and in other functions were determined
    # by running the metrics by hand with the
    # random seeds defined in fake_data()
    ref = {(0, 'x'): 1.009, (0, 'y'): 0.938, (1, 'x'): 0.788,
           (1, 'y'): 0.797, (2, 'x'): 0.846, (2, 'y'): 0.774,
           (3, 'x'): 0.667, (3, 'y'): 0.755}
    for key, val in ref.items():
        assert np.allclose(val, mean_Vij[key], atol=1e-3)
    zs = ant_metrics.mean_Vij_metrics(lowlevel_data.data, lowlevel_data.pols,
                                      lowlevel_data.antpols, lowlevel_data.ants,
                                      lowlevel_data.bls)
    ref = {(0, 'x'): 1.443, (0, 'y'): 4.970, (1, 'x'): -0.218,
           (1, 'y'): 0.373, (2, 'x'): 0.218, (2, 'y'): -0.373,
           (3, 'x'): -1.131, (3, 'y'): -0.976}
    for key, val in ref.items():
        assert np.allclose(val, zs[key], atol=1e-2)


def test_mean_Vij_cross_pol_metrics(lowlevel_data):
    mean_Vij_cross_pol = ant_metrics.mean_Vij_cross_pol_metrics(lowlevel_data.data,
                                                                lowlevel_data.pols,
                                                                lowlevel_data.antpols,
                                                                lowlevel_data.ants,
                                                                lowlevel_data.bls,
                                                                rawMetric=True)
    ref = {(0, 'x'): 0.746, (0, 'y'): 0.746, (1, 'x'): 0.811,
           (1, 'y'): 0.811, (2, 'x'): 0.907, (2, 'y'): 0.907,
           (3, 'x'): 1.091, (3, 'y'): 1.091}
    for key, val in ref.items():
        assert np.isclose(val, mean_Vij_cross_pol[key], atol=1e-3)
    zs = ant_metrics.mean_Vij_cross_pol_metrics(lowlevel_data.data,
                                                lowlevel_data.pols,
                                                lowlevel_data.antpols,
                                                lowlevel_data.ants,
                                                lowlevel_data.bls)
    ref = {(0, 'x'): -0.948, (0, 'y'): -0.948, (1, 'x'): -0.401,
           (1, 'y'): -0.401, (2, 'x'): 0.401, (2, 'y'): 0.401,
           (3, 'x'): 1.944, (3, 'y'): 1.944}
    for key, val in ref.items():
        assert np.isclose(val, zs[key], atol=1e-3)


def test_per_antenna_modified_z_scores():
    metric = {(0, 'x'): 1, (50, 'x'): 0, (2, 'x'): 2,
              (2, 'y'): 2000, (0, 'y'): -300}
    zscores = ant_metrics.per_antenna_modified_z_scores(metric)
    np.testing.assert_almost_equal(zscores[0, 'x'], 0, 10)
    np.testing.assert_almost_equal(zscores[50, 'x'], -0.6745, 10)
    np.testing.assert_almost_equal(zscores[2, 'x'], 0.6745, 10)


def test_exclude_partially_excluded_ants():
    before_xants = [(0, 'x'), (0, 'y'), (1, 'x'), (2, 'y')]
    after_xants = ant_metrics.exclude_partially_excluded_ants(['x', 'y'], before_xants)
    after_xants_truth = [(0, 'x'), (0, 'y'), (1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]
    assert set(after_xants) == set(after_xants_truth)


def test_antpol_metric_sum_ratio():
    crossMetrics = {(0, 'x'): 1.0, (0, 'y'): 1.0, (1, 'x'): 1.0}
    sameMetrics = {(0, 'x'): 2.0, (0, 'y'): 2.0}
    xants = [(1, 'y')]
    crossPolRatio = ant_metrics.antpol_metric_sum_ratio([0, 1], ['x', 'y'],
                                                        crossMetrics,
                                                        sameMetrics,
                                                        xants=xants)
    assert crossPolRatio == {(0, 'x'): .5, (0, 'y'): .5}


def test_load_antenna_metrics():
    # load a metrics file and check some values
    metrics_file = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    metrics = ant_metrics.load_antenna_metrics(metrics_file)

    assert np.isclose(metrics['final_mod_z_scores']['meanVijXPol'][(72, 'x')], 0.17529333517595402)
    assert np.isclose(metrics['final_mod_z_scores']['meanVijXPol'][(72, 'y')], 0.17529333517595402)
    assert np.isclose(metrics['final_mod_z_scores']['meanVijXPol'][(31, 'y')], 0.7012786080508268)

    # change some values to FPE values, and write it out
    metrics['final_mod_z_scores']['meanVijXPol'][(72, 'x')] = np.nan
    metrics['final_mod_z_scores']['meanVijXPol'][(72, 'y')] = np.inf
    metrics['final_mod_z_scores']['meanVijXPol'][(31, 'y')] = -np.inf

    outpath = os.path.join(DATA_PATH, 'test_output',
                           'ant_metrics_output.hdf5')
    metrics_io.write_metric_file(outpath, metrics, overwrite=True)

    # test reading it back in, and that the values agree
    metrics_new = ant_metrics.load_antenna_metrics(outpath)
    assert np.isnan(metrics_new['final_mod_z_scores']['meanVijXPol'][(72, 'x')])
    assert np.isinf(metrics_new['final_mod_z_scores']['meanVijXPol'][(72, 'y')])
    assert np.isneginf(metrics_new['final_mod_z_scores']['meanVijXPol'][(31, 'y')])

    # clean up after ourselves
    os.remove(outpath)


def test_load_ant_metrics_json():
    json_file = os.path.join(DATA_PATH, 'example_ant_metrics.json')
    hdf5_file = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    warn_message = ["JSON-type files can still be read but are no longer "
                    "written by default.\n"
                    "Write to HDF5 format for future compatibility."]
    json_dict = uvtest.checkWarnings(ant_metrics.load_antenna_metrics,
                                     func_args=[json_file],
                                     category=PendingDeprecationWarning,
                                     nwarnings=1,
                                     message=warn_message)
    hdf5_dict = ant_metrics.load_antenna_metrics(hdf5_file)

    # The written hdf5 may have these keys that differ by design
    # so ignore them.
    json_dict.pop('history', None)
    json_dict.pop('version', None)
    hdf5_dict.pop('history', None)
    hdf5_dict.pop('version', None)

    # This function recursively walks dictionary and compares
    # data types together with asserts or np.allclose
    assert qmtest.recursive_compare_dicts(hdf5_dict, json_dict)


@pytest.fixture(scope='function')
def antmetrics_data():
    dataFileList = [DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA',
                    DATA_PATH + '/zen.2457698.40355.yy.HH.uvcA',
                    DATA_PATH + '/zen.2457698.40355.xy.HH.uvcA',
                    DATA_PATH + '/zen.2457698.40355.yx.HH.uvcA']
    if not os.path.exists(DATA_PATH + '/test_output/'):
        os.makedirs(DATA_PATH + '/test_output/')
    # internal names for summary statistics
    summaryStats = ['xants', 'crossedAntsRemoved', 'deadAntsRemoved',
                    'removalIter', 'finalMetrics', 'allMetrics',
                    'finalModzScores', 'allModzScores', 'crossCut',
                    'deadCut', 'dataFileList']

    class DataHolder():
        def __init__(self, dataFileList, summaryStats):
            self.dataFileList = dataFileList
            self.summaryStats = summaryStats
    antmetrics_data = DataHolder(dataFileList, summaryStats)

    # yield returns the data we need but lets us continue after for cleanup
    yield antmetrics_data

    # post-test cleanup
    del(antmetrics_data)


def test_load_errors(antmetrics_data):
    with pytest.raises(ValueError):
        uvtest.checkWarnings(ant_metrics.AntennaMetrics,
                             [[DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA']],
                             {"fileformat": 'miriad'}, nwarnings=1,
                             message='antenna_diameters is not set')
    with pytest.raises(IOError):
        ant_metrics.AntennaMetrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                   fileformat='uvfits')
    with pytest.raises(NotImplementedError):
        ant_metrics.AntennaMetrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                   fileformat='fhd')
    with pytest.raises(NotImplementedError):
        ant_metrics.AntennaMetrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                   fileformat='not_a_format')


def test_init(antmetrics_data):
    am = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                    fileformat='miriad')
    assert len(am.ants) == 19
    assert set(am.pols) == set(['xx', 'yy', 'xy', 'yx'])
    assert set(am.antpols) == set(['x', 'y'])
    assert len(am.bls) == 19 * 18 / 2 + 19


def test_iterative_antenna_metrics_and_flagging_and_saving_and_loading(antmetrics_data):
    am = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                    fileformat='miriad')
    with pytest.raises(KeyError):
        filename = os.path.join(DATA_PATH, 'test_output',
                                'ant_metrics_output.hdf5')
        am.save_antenna_metrics(filename)

    am.iterative_antenna_metrics_and_flagging()
    for stat in antmetrics_data.summaryStats:
        assert hasattr(am, stat)
    assert (81, 'x') in am.xants
    assert (81, 'y') in am.xants
    assert (81, 'x') in am.deadAntsRemoved
    assert (81, 'y') in am.deadAntsRemoved

    outfile = os.path.join(DATA_PATH, 'test_output',
                           'ant_metrics_output.hdf5')
    am.save_antenna_metrics(outfile)
    loaded = ant_metrics.load_antenna_metrics(outfile)
    # json names for summary statistics
    jsonStats = ['xants', 'crossed_ants', 'dead_ants', 'removal_iteration',
                 'final_metrics', 'all_metrics', 'final_mod_z_scores',
                 'all_mod_z_scores', 'cross_pol_z_cut', 'dead_ant_z_cut',
                 'datafile_list', 'version']
    for stat, jsonStat in zip(antmetrics_data.summaryStats, jsonStats):
        assert np.array_equal(loaded[jsonStat],
                              getattr(am, stat))
    os.remove(outfile)


def test_save_json(antmetrics_data):
    am = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                    fileformat='miriad')
    am.iterative_antenna_metrics_and_flagging()
    for stat in antmetrics_data.summaryStats:
        assert hasattr(am, stat)
    assert (81, 'x') in am.xants
    assert (81, 'y') in am.xants
    assert (81, 'x') in am.deadAntsRemoved
    assert (81, 'y') in am.deadAntsRemoved

    outfile = os.path.join(DATA_PATH, 'test_output',
                           'ant_metrics_output.json')
    warn_message = ["JSON-type files can still be written "
                    "but are no longer written by default.\n"
                    "Write to HDF5 format for future compatibility."]
    uvtest.checkWarnings(am.save_antenna_metrics,
                         func_args=[outfile], func_kwargs={'overwrite': True},
                         category=PendingDeprecationWarning, nwarnings=1,
                         message=warn_message)

    # am.save_antenna_metrics(json_file)
    warn_message = ["JSON-type files can still be read but are no longer "
                    "written by default.\n"
                    "Write to HDF5 format for future compatibility."]
    loaded = uvtest.checkWarnings(ant_metrics.load_antenna_metrics,
                                  func_args=[outfile],
                                  category=PendingDeprecationWarning,
                                  nwarnings=1,
                                  message=warn_message)
    _ = loaded.pop('history', '')

    jsonStats = ['xants', 'crossed_ants', 'dead_ants', 'removal_iteration',
                 'final_metrics', 'all_metrics', 'final_mod_z_scores',
                 'all_mod_z_scores', 'cross_pol_z_cut', 'dead_ant_z_cut',
                 'datafile_list', 'version']

    for stat, jsonStat in zip(antmetrics_data.summaryStats, jsonStats):
        file_val = loaded[jsonStat]
        obj_val = getattr(am, stat)
        if isinstance(file_val, dict):
            assert qmtest.recursive_compare_dicts(file_val, obj_val)
        else:
            assert file_val == obj_val
    os.remove(outfile)


def test_add_file_appellation(antmetrics_data):
    am = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                    fileformat='miriad')
    am.iterative_antenna_metrics_and_flagging()
    for stat in antmetrics_data.summaryStats:
        assert hasattr(am, stat)
    assert (81, 'x') in am.xants
    assert (81, 'y') in am.xants
    assert (81, 'x') in am.deadAntsRemoved
    assert (81, 'y') in am.deadAntsRemoved

    outfile = os.path.join(DATA_PATH, 'test_output',
                           'ant_metrics_output')

    am.save_antenna_metrics(outfile, overwrite=True)
    outname = os.path.join(DATA_PATH, 'test_output',
                           'ant_metrics_output.hdf5')
    assert os.path.isfile(outname)
    os.remove(outname)


def test_cross_detection(antmetrics_data):
    am2 = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                     fileformat='miriad')
    am2.iterative_antenna_metrics_and_flagging(crossCut=3, deadCut=10)
    for stat in antmetrics_data.summaryStats:
        assert hasattr(am2, stat)
    assert (81, 'x') in am2.xants
    assert (81, 'y') in am2.xants
    assert (81, 'x') in am2.crossedAntsRemoved
    assert (81, 'y') in am2.crossedAntsRemoved


def test_totally_dead_ants(antmetrics_data):
    am2 = ant_metrics.AntennaMetrics(antmetrics_data.dataFileList,
                                     fileformat='miriad')
    deadant = 9
    for ant1, ant2 in am2.bls:
        if deadant in (ant1, ant2):
            for pol in am2.pols:
                am2.data[ant1, ant2, pol][:] = 0.0
    am2.reset_summary_stats()
    am2.find_totally_dead_ants()
    for antpol in am2.antpols:
        assert (deadant, antpol) in am2.xants
        assert (deadant, antpol) in am2.deadAntsRemoved
        assert am2.removalIter[(deadant, antpol)] == -1


def test_reds_from_file_read_file():
    from hera_cal.io import HERAData
    from hera_cal.redcal import get_pos_reds

    # Miriad file will need to be read in
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
    reds = ant_metrics.reds_from_file(testfile, vis_format='miriad')
    assert len(reds) > 1
    hd = HERAData(testfile, filetype='miriad')
    reds_check = get_pos_reds(hd.read()[0].antpos)
    assert reds == reds_check


def test_reds_from_file_no_read_file():
    from hera_cal.io import HERAData
    from hera_cal.redcal import get_pos_reds

    # uvh5 file will not need to be read in
    testfile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvh5')
    reds = ant_metrics.reds_from_file(testfile, vis_format='uvh5')
    assert len(reds) > 1
    hd = HERAData(testfile, filetype='uvh5')
    reds_check = get_pos_reds(hd.antpos)
    assert reds == reds_check


def test_run_ant_metrics_no_files():
    # get argument object
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_mean_vij"
    arg8 = "--run_red_corr"
    arg9 = "--run_cross_pols"
    arguments = ' '.join([arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    # test running with no files
    cmd = ' '.join([arguments, ''])
    args = a.parse_args(cmd.split())
    pols = list(args.pol.split(','))

    history = cmd

    pytest.raises(AssertionError, ant_metrics.ant_metrics_run,
                  args.files, pols, args.crossCut, args.deadCut,
                  args.alwaysDeadCut, args.metrics_path,
                  args.extension, args.vis_format,
                  args.verbose, history, args.run_mean_vij,
                  args.run_red_corr, args.run_cross_pols)


def test_run_ant_metrics_one_file():
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_mean_vij"
    arg8 = "--run_red_corr"
    arg9 = "--run_cross_pols"
    arguments = ' '.join([arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    # test running with a lone file
    lone_file = os.path.join(DATA_PATH,
                             'zen.2457698.40355.xx.HH.uvcAA')
    cmd = ' '.join([arguments, lone_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))

    # this test raises a warning, then fails...
    uvtest.checkWarnings(pytest.raises,
                         [AssertionError, ant_metrics.ant_metrics_run,
                          args.files, pols, args.crossCut,
                          args.deadCut, args.alwaysDeadCut,
                          args.metrics_path,
                          args.extension, args.vis_format,
                          args.verbose, history, args.run_mean_vij,
                          args.run_red_corr, args.run_cross_pols],
                         nwarnings=1,
                         message='Could not find')


def test_ant_metrics_run_no_metrics():
    """Test an argument is raised if no metrics are set to True."""
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--skip_mean_vij"
    arg8 = "--skip_red_corr"
    arg9 = "--skip_cross_pols"
    arguments = ' '.join([arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.uvA.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    pytest.raises(AssertionError, ant_metrics.ant_metrics_run,
                  args.files, pols, args.crossCut,
                  args.deadCut, args.alwaysDeadCut,
                  args.metrics_path,
                  args.extension, args.vis_format,
                  args.verbose, history, args.run_mean_vij,
                  args.run_red_corr, args.run_cross_pols)


def test_ant_metrics_run_all_metrics():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_mean_vij"
    arg8 = "--run_red_corr"
    arg9 = "--run_cross_pols"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_mean_vij=args.run_mean_vij,
                                run_red_corr=args.run_red_corr,
                                run_cross_pols=args.run_cross_pols)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_ant_metrics_run_only_mean_vij():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_mean_vij"
    arg8 = "--skip_red_corr"
    arg9 = "--skip_cross_pols"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_mean_vij=args.run_mean_vij,
                                run_red_corr=args.run_red_corr,
                                run_cross_pols=args.run_cross_pols)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_ant_metrics_run_only_red_corr():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--skip_mean_vij"
    arg8 = "--run_red_corr"
    arg9 = "--skip_cross_pols"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6,
                          arg7, arg8, arg9])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_mean_vij=args.run_mean_vij,
                                run_red_corr=args.run_red_corr,
                                run_cross_pols=args.run_cross_pols)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_ant_metrics_run_only_cross_pols():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_cross_pols_only"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_cross_pols_only=args.run_cross_pols_only)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_ant_metrics_run_cross_pols_mean_vij_only():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_cross_pols_only"
    arg8 = "--skip_red_corr"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_red_corr=args.run_red_corr,
                                run_cross_pols_only=args.run_cross_pols_only)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_ant_metrics_run_cross_pols_red_corr_only():
    # get arguments
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)
    arg0 = "-p xx,yy,xy,yx"
    arg1 = "--crossCut=5"
    arg2 = "--deadCut=5"
    arg3 = "--extension=.ant_metrics.hdf5"
    arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH,
                                                   'test_output'))
    arg5 = "--vis_format=miriad"
    arg6 = "--alwaysDeadCut=10"
    arg7 = "--run_cross_pols_only"
    arg8 = "--skip_mean_vij"
    arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8])

    xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2458002.47754.HH.ant_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, xx_file])
    args = a.parse_args(cmd.split())
    history = cmd
    pols = list(args.pol.split(','))
    if os.path.exists(dest_file):
        os.remove(dest_file)
    ant_metrics.ant_metrics_run(args.files, pols, args.crossCut,
                                args.deadCut, args.alwaysDeadCut,
                                args.metrics_path,
                                args.extension, args.vis_format,
                                args.verbose, history=history,
                                run_mean_vij=args.run_mean_vij,
                                run_cross_pols_only=args.run_cross_pols_only)
    assert os.path.exists(dest_file)
    os.remove(dest_file)
