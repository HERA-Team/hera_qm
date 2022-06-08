# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
import os
from hera_qm import ant_metrics
from hera_qm import metrics_io
from hera_qm.data import DATA_PATH
import hera_qm.tests as qmtest

pytestmark = pytest.mark.filterwarnings(
    "ignore:The uvw_array does not match the expected values given the antenna positions.",
)


def test_calc_corr_stats():
    data_sum = {(0, 1, 'ee'): np.array([[20, 10.0 + 10.0j], [10.0, 30.0]]),
                (0, 1, 'nn'): np.array([[10.0j, 10.0], [20.0 + 100j, 10.0]])}
    data_diff = {(0, 1, 'ee'): np.array([[0.0, 1.0 + 1.0j], [1.0, 3.0]]),
                (0, 1, 'nn'): np.array([[3.0, 1.0j], [0.0, 2.0]])}
    flags = {(0, 1, 'ee'): np.array([[False, False], [False, False]]),
                (0, 1, 'nn'): np.array([[False, False], [False, False]])}

    # test normal operation
    corr_stats = ant_metrics.calc_corr_stats(data_sum,data_diff,flags)
    assert corr_stats[(0,1,'ee')] == 1
    assert corr_stats[(0,1,'nn')] == pytest.approx(0.9578,abs=1e-2)

    # test with flags
    flags[(0,1,'nn')][1,0] = True
    corr_stats = ant_metrics.calc_corr_stats(data_sum,data_diff,flags)
    assert corr_stats[(0,1,'nn')] == pytest.approx(0.9457,abs=1e-2)

    # test no diff data
    corr_stats = ant_metrics.calc_corr_stats(data_sum)
    assert corr_stats[(0,1,'nn')] == pytest.approx(1,abs=1e-2)
    assert corr_stats[(0,1,'ee')] == pytest.approx(0.9238,abs=1e-2)

    # test no diff, odd number of times
    data_sum = {(0, 1, 'ee'): np.array([[20, 10.0 + 10.0j], [10.0, 30.0], [30, 200+ 50j]]),
                (0, 1, 'nn'): np.array([[10.0j, 10.0], [20.0 + 100j, 10.0], [10j, 20 + 30j]])}
    corr_stats = ant_metrics.calc_corr_stats(data_sum)
    assert corr_stats[(0,1,'nn')] == pytest.approx(1,abs=1e-2)
    assert corr_stats[(0,1,'ee')] == pytest.approx(0.9238,abs=1e-2)


# def test_corr_metrics():
#     corr_stats = {(0, 1, 'ee'): 1.0,
#                      (0, 2, 'ee'): 1.0,
#                      (0, 3, 'ee'): 0.4,
#                      (1, 2, 'ee'): 0.8,
#                      (1, 3, 'ee'): 0.3,
#                      (2, 3, 'ee'): 0.15,
#                      (1, 0, 'nn'): 1.0,
#                      (0, 2, 'nn'): 1.0,
#                      (0, 3, 'nn'): 0.4,
#                      (1, 2, 'nn'): 0.8,
#                      (1, 3, 'nn'): 0.3,
#                      (2, 3, 'nn'): 0.15}

# # test normal operation
#     corr = ant_metrics.corr_metrics(corr_stats)
#     for ant in corr:
#         assert ant[0] in [0,1,2,3]
#         assert ant[1] in ['Jee','Jnn']
#         if ant[0] == 3:
#             assert corr[ant] < 0.4
#         else:
#             assert corr[ant] >= 0.4

#     # test xants
#     corr = ant_metrics.corr_metrics(corr_stats, xants=[3])
#     for ant in corr:
#         assert ant[0] in [0,1,2]
#         assert ant[1] in ['Jee','Jnn']
#         assert ant != (3,'Jnn') and ant != (3,'Jee')
#         assert corr[ant] == {0: 1, 1: 0.9, 2: 0.9}[ant[0]]

#     # test pols
#     corr = ant_metrics.corr_metrics(corr_stats, pols=['ee'])
#     for ant in corr:
#         assert ant[0] in [0,1,2,3]
#         assert ant[1] in ['Jee']
#         assert corr[ant] == pytest.approx({0: 0.8, 1: 0.7, 2: 0.65, 3: 0.85/3}[ant[0]])


# def test_corr_cross_pol_metrics():
#     # Set up such that antenna 1 is cross-polarized
#     corr_stats = {(0, 1, 'ee'): 0.6, (1, 0, 'nn'): 0.6,
#                      (0, 2, 'ee'): 1.0, (0, 2, 'nn'): 1.0,
#                      (0, 3, 'ee'): 0.9, (0, 3, 'nn'): 0.9,
#                      (1, 2, 'ee'): 0.7, (1, 2, 'nn'): 0.5,
#                      (1, 3, 'ee'): 0.45, (1, 3, 'nn'): 0.45,
#                      (2, 3, 'ee'): 0.8, (2, 3, 'nn'): 0.8,
#                      (0, 1, 'en'): 1.0, (1, 0, 'ne'): 1.0,
#                      (0, 2, 'en'): 0.5, (0, 2, 'ne'): 0.5,
#                      (0, 3, 'en'): 0.4, (0, 3, 'ne'): 0.4,
#                      (1, 2, 'en'): 0.8, (1, 2, 'ne'): 0.9,
#                      (1, 3, 'en'): 0.85, (1, 3, 'ne'): 0.85,
#                      (2, 3, 'en'): 0.6, (2, 3, 'ne'): 0.6}

#     # test normal operation
#     xpol_metric = ant_metrics.corr_cross_pol_metrics(corr_stats)
#     for ant in xpol_metric:
#         assert ant[0] in [0,1,2,3]
#         assert ant[1] in ['Jnn','Jee']
#         if ant[0] == 1:
#             assert xpol_metric[ant] < 0
#         else:
#             assert xpol_metric[ant] > 0

#     # test xants
#     xpol_metric = ant_metrics.corr_cross_pol_metrics(corr_stats, xants=[1])
#     for ant in xpol_metric:
#         assert ant[0] in [0,2,3]
#         assert ant[1] in ['Jnn','Jee']
#         assert xpol_metric[ant] > 0

#     xpol_metric = ant_metrics.corr_cross_pol_metrics(corr_stats, xants=[2])
#     for ant in xpol_metric:
#         assert ant[0] in [0,1,3]
#         assert ant[1] in ['Jnn','Jee']
#         if ant[0] == 1:
#             assert xpol_metric[ant] < 0
#         else:
#             assert xpol_metric[ant] > 0

#     # test error
#     corr_stats = {(0, 1, 'ee'): 1.0,
#                      (0, 2, 'ee'): 1.0,
#                      (1, 2, 'ee'): 0.8,
#                      (1, 0, 'nn'): 1.0,
#                      (0, 2, 'nn'): 1.0,
#                      (1, 2, 'nn'): 0.8}

#     with pytest.raises(ValueError):
#         xpol_metric = ant_metrics.corr_cross_pol_metrics(corr_stats)


def test_load_antenna_metrics():
    # N.B. This test operates on an old json with an old polarization convetion.
    # Appears to work OK for now, but may not be worth maintaining.

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
    # N.B. This test operates on an old json with an old polarization convetion.
    # Appears to work OK for now, but may not be worth maintaining.

    json_file = os.path.join(DATA_PATH, 'example_ant_metrics.json')
    hdf5_file = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    warn_message = ["JSON-type files can still be read but are no longer "
                    "written by default.\n"
                    "Write to HDF5 format for future compatibility."]
    with pytest.warns(PendingDeprecationWarning, match=warn_message[0]):
        json_dict = ant_metrics.load_antenna_metrics(json_file)
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


def test_init():
    # try with both sum files only and both
    files = [{'sum_files': DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5', 'diff_files': None},
             {'sum_files': DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5',
              'diff_files': DATA_PATH + '/zen.2459122.49827.diff.downselected.uvh5'}]

    for to_load in files:
        # load data
        am = ant_metrics.AntennaMetrics(**to_load, apriori_xants=[51, (116, 'Jnn'), (93, 'jnn')])

        # test metadata
        assert am.datafile_list_sum == [to_load['sum_files']]
        assert am.hd_sum is not None
        if to_load['diff_files'] is not None:
            assert am.datafile_list_diff == [to_load['diff_files']]
            assert am.hd_sum is not None
        else:
            assert am.datafile_list_diff is None
            assert am.hd_diff is None
        assert am.history == ''

        # test antennas and baselines
        true_antnums = [51, 87, 116, 93, 65, 85, 53, 160, 36, 83, 135, 157, 98, 117, 68]
        assert len(am.bls) == len(true_antnums) * (len(true_antnums) - 1) / 2 * 4 + len(true_antnums) * 4
        for antpol in ['Jnn', 'Jee']:
            assert antpol in am.antpols
            for antnum in true_antnums:
                assert antnum in am.antnums
                assert (antnum, antpol) in am.ants

        # test apriori xants
        assert (51, 'Jnn') in am.apriori_xants
        assert (51, 'Jee') in am.apriori_xants
        assert 51 not in am.apriori_xants
        assert (116, 'Jnn') in am.apriori_xants
        assert (116, 'Jee') not in am.apriori_xants
        assert (93, 'Jnn') in am.apriori_xants
        assert (116, 'Jee') not in am.apriori_xants

        # test errors in parsing apriori xants
        with pytest.raises(ValueError):
            ant_metrics.AntennaMetrics(**to_load, apriori_xants=(9, 'Jnn'))
        with pytest.raises(ValueError):
            ant_metrics.AntennaMetrics(**to_load, apriori_xants=[(9, 10, 'nn')])
        with pytest.raises(ValueError):
            ant_metrics.AntennaMetrics(**to_load, apriori_xants=[1.0])

        # test _reset_summary_stats
        for ant in am.apriori_xants:
            assert am.removal_iteration[ant] == -1
            assert ant in am.xants
        assert am.crossed_ants == []
        assert am.dead_ants == []
        assert am.iter == 0
        assert am.all_metrics == {}
        assert am.final_metrics == {}

        # test Nbls_per_load and Nfiles_per_load
        am2 = ant_metrics.AntennaMetrics(**to_load, Nbls_per_load=100, Nfiles_per_load=1)
        for pol in am.corr_matrices:
            np.testing.assert_array_equal(am.corr_matrices[pol], am2.corr_matrices[pol])


def test_iterative_antenna_metrics_and_flagging():
    files = {'sum_files': DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5',
             'diff_files': DATA_PATH + '/zen.2459122.49827.diff.downselected.uvh5'}
    am = ant_metrics.AntennaMetrics(**files)

    # try normal operation
    am.iterative_antenna_metrics_and_flagging(verbose=True, crossCut=0., deadCut=.4, )
    for ap in ['Jnn', 'Jee']:
        assert (93, ap) in am.dead_ants
        assert (93, ap) in am.xants
        assert (65, ap) in am.dead_ants
        assert (65, ap) in am.xants
        assert (116, ap) in am.dead_ants
        assert (116, ap) in am.xants
        assert (51, ap) in am.crossed_ants
        assert (87, ap) in am.crossed_ants
        assert (51, ap) in am.xants
        assert (87, ap) in am.xants

    assert list(am.all_metrics.keys()) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for metric in am.all_metrics[1]:
        for ant in am.all_metrics[1][metric]:
            assert (am.all_metrics[1][metric][ant] <= 1) or np.isnan(am.all_metrics[1][metric][ant])
    assert am.final_metrics['corrXPol'][(87, 'Jnn')] == am.final_metrics['corrXPol'][(87, 'Jee')]
    assert am.final_metrics['corrXPol'][(87, 'Jnn')] == am.all_metrics[3]['corrXPol'][(87, 'Jnn')]
    assert am.final_metrics['corrXPol'][(87, 'Jee')] == am.all_metrics[3]['corrXPol'][(87, 'Jee')]

    # test _find_totally_dead_ants
    # for bl in am.corr_stats:
    #     if 68 in bl:
    #         am.corr_stats[bl] = 0.0
    # am.iterative_antenna_metrics_and_flagging(verbose=True)
    # assert (68, 'Jnn') in am.xants
    # assert (68, 'Jee') in am.xants
    # assert (68, 'Jnn') in am.dead_ants
    # assert (68, 'Jee') in am.dead_ants
    # assert am.removal_iteration[68, 'Jnn'] == -1
    # assert am.removal_iteration[68, 'Jee'] == -1


def test_ant_metrics_run_and_load_antenna_metrics():
    files = {'sum_files': DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5',
             'diff_files': DATA_PATH + '/zen.2459122.49827.diff.downselected.uvh5'}
    am = ant_metrics.AntennaMetrics(**files)
    am.iterative_antenna_metrics_and_flagging()

    ant_metrics.ant_metrics_run(**files, overwrite=True, history='test_history_string', verbose=True)
    am_hdf5 = ant_metrics.load_antenna_metrics(files['sum_files'].replace('.uvh5', '.ant_metrics.hdf5'))

    assert 'test_history_string' in am_hdf5['history']
    assert am.version_str == am_hdf5['version']
    assert am.crossCut == am_hdf5['cross_pol_cut']
    assert am.deadCut == am_hdf5['dead_ant_cut']
    assert set(am.xants) == set(am_hdf5['xants'])
    assert set(am.crossed_ants) == set(am_hdf5['crossed_ants'])
    assert set(am.dead_ants) == set(am_hdf5['dead_ants'])
    assert set(am.datafile_list_sum) == set(am_hdf5['datafile_list_sum'])
    assert set(am.datafile_list_diff) == set(am_hdf5['datafile_list_diff'])

    assert qmtest.recursive_compare_dicts(am.removal_iteration, am_hdf5['removal_iteration'])
    assert qmtest.recursive_compare_dicts(am.final_metrics, am_hdf5['final_metrics'])
    assert qmtest.recursive_compare_dicts(am.all_metrics, am_hdf5['all_metrics'])

    os.remove(files['sum_files'].replace('.uvh5', '.ant_metrics.hdf5'))

    # test a priori flagging via YAML
    four_pol_uvh5 = DATA_PATH + '/zen.2457698.40355.full_pol_test.uvh5'
    apf_yaml = os.path.join(DATA_PATH, 'a_priori_flags_old_pols.yaml')
    am = ant_metrics.ant_metrics_run(four_pol_uvh5, diff_files=four_pol_uvh5, overwrite=True, a_priori_xants_yaml=apf_yaml, verbose=True)
    am_hdf5 = ant_metrics.load_antenna_metrics(four_pol_uvh5.replace('.uvh5', '.ant_metrics.hdf5'))
    for ant in [(0, 'Jxx'), (0, 'Jyy'), (10, 'Jxx'), (10, 'Jyy'), (1, 'Jxx'), (3, 'Jyy')]:
        assert ant in am_hdf5['xants']
        assert am_hdf5['removal_iteration'][ant] == -1

    os.remove(four_pol_uvh5.replace('.uvh5', '.ant_metrics.hdf5'))
