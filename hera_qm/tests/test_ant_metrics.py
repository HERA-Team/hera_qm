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


def test_per_antenna_modified_z_scores():
    metric = {(0, 'Jnn'): 1, (50, 'Jnn'): 0, (2, 'Jnn'): 2,
              (2, 'Jee'): 2000, (0, 'Jee'): -300}
    zscores = ant_metrics.per_antenna_modified_z_scores(metric)
    np.testing.assert_almost_equal(zscores[0, 'Jnn'], 0, 10)
    np.testing.assert_almost_equal(zscores[50, 'Jnn'], -0.6745, 10)
    np.testing.assert_almost_equal(zscores[2, 'Jnn'], 0.6745, 10)


def test_time_freq_abs_vis_stats():
    data = {(0, 1, 'ee'): np.array([[0.0, 1.0], [1.0, 3.0]])}
    flags = {(0, 1, 'ee'): np.array([[False, False], [False, True]])}

    # test normal operation
    assert ant_metrics.time_freq_abs_vis_stats(data)[(0, 1, 'ee')] == 1
    assert ant_metrics.time_freq_abs_vis_stats(data, freq_alg=np.nanmean)[(0, 1, 'ee')] == 1.25
    assert ant_metrics.time_freq_abs_vis_stats(data, time_alg=np.nanmean, freq_alg=np.nanmean)[(0, 1, 'ee')] == 1.25

    # test with lots of zeros
    data2 = {(0, 1, 'ee'): np.array([[0.0, 0.0], [0.0, 3.0]])}
    assert ant_metrics.time_freq_abs_vis_stats(data2)[(0, 1, 'ee')] == 0
    assert ant_metrics.time_freq_abs_vis_stats(data2, freq_alg=np.nanmean)[(0, 1, 'ee')] == 0
    assert ant_metrics.time_freq_abs_vis_stats(data2, time_alg=np.nanmean, freq_alg=np.nanmean)[(0, 1, 'ee')] == 0

    # test with flags
    assert ant_metrics.time_freq_abs_vis_stats(data, flags=flags)[(0, 1, 'ee')] == 1
    assert ant_metrics.time_freq_abs_vis_stats(data, flags=flags, freq_alg=np.nanmean)[(0, 1, 'ee')] == .75

    # test with nans and infs
    data3 = {(0, 1, 'ee'): np.array([[0.0, np.nan], [np.inf, 3.0]])}
    assert ant_metrics.time_freq_abs_vis_stats(data3)[(0, 1, 'ee')] == 1.5


def test_mean_Vij_metrics():
    abs_vis_stats = {(0, 1, 'ee'): 1.0,
                     (0, 2, 'ee'): 3.0,
                     (0, 3, 'ee'): 11.0,
                     (1, 2, 'ee'): 2.0,
                     (1, 3, 'ee'): 9.0,
                     (2, 3, 'ee'): 10.0,
                     (0, 1, 'nn'): 1.0,
                     (0, 2, 'nn'): 3.0,
                     (0, 3, 'nn'): 11.0,
                     (1, 2, 'nn'): 2.0,
                     (1, 3, 'nn'): 9.0,
                     (2, 3, 'nn'): 10.0}

    # test normal operation
    mean_Vij = ant_metrics.mean_Vij_metrics(abs_vis_stats)
    for ant in mean_Vij:
        assert ant[0] in [0, 1, 2, 3]
        assert ant[1] in ['Jee', 'Jnn']
        if 3 in ant:
            assert np.abs(mean_Vij[ant]) > 5
        else:
            assert np.abs(mean_Vij[ant]) < 2

    # test rawMetric
    mean_Vij = ant_metrics.mean_Vij_metrics(abs_vis_stats, rawMetric=True)
    for ant in mean_Vij:
        assert ant[0] in [0, 1, 2, 3]
        assert ant[1] in ['Jee', 'Jnn']
        assert mean_Vij[ant] == {0: 5, 1: 4, 2: 5, 3: 10}[ant[0]]

    # test pols
    mean_Vij = ant_metrics.mean_Vij_metrics(abs_vis_stats, pols=['ee'], rawMetric=True)
    for ant in mean_Vij:
        assert ant[0] in [0, 1, 2, 3]
        assert ant[1] in ['Jee']
        assert mean_Vij[ant] == {0: 5, 1: 4, 2: 5, 3: 10}[ant[0]]

    # test xants
    mean_Vij = ant_metrics.mean_Vij_metrics(abs_vis_stats, xants=[3, (1, 'Jee')], rawMetric=True)
    for ant in mean_Vij:
        assert ant[0] in [0, 1, 2]
        assert ant[1] in ['Jee', 'Jnn']
        assert ant != (1, 'Jee')
        assert mean_Vij[ant] == {0: 5, 1: 4, 2: 5}[ant[0]]

    # test error
    abs_vis_stats = {(0, 1, 'ee'): 1.0}
    with pytest.raises(ValueError):
        mean_Vij = ant_metrics.mean_Vij_metrics(abs_vis_stats)


def test_antpol_metric_sum_ratio():
    crossMetrics = {(0, 'Jnn'): 1.0, (0, 'Jee'): 1.0}
    sameMetrics = {(0, 'Jnn'): 2.0, (0, 'Jee'): 2.0}
    crossPolRatio = ant_metrics.antpol_metric_sum_ratio(crossMetrics, sameMetrics)
    assert crossPolRatio == {(0, 'Jnn'): .5, (0, 'Jee'): .5}


def test_mean_Vij_metrics():
    abs_vis_stats = {(0, 1, 'ee'): 10.0,
                     (0, 2, 'ee'): 1.0,
                     (1, 2, 'ee'): 2.0,
                     (0, 1, 'en'): 3.0,
                     (0, 2, 'en'): 9.0,
                     (1, 2, 'en'): 11.0,
                     (0, 1, 'ne'): 3.0,
                     (0, 2, 'ne'): 9.0,
                     (1, 2, 'ne'): 11.0,
                     (0, 1, 'nn'): 10.0,
                     (0, 2, 'nn'): 1.0,
                     (1, 2, 'nn'): 2.0}

    # test normal operation
    mean_Vij_cross = ant_metrics.mean_Vij_cross_pol_metrics(abs_vis_stats)
    for ant in mean_Vij_cross:
        assert ant[0] in [0, 1, 2]
        assert ant[1] in ['Jee', 'Jnn']
        if 2 in ant:
            assert np.abs(mean_Vij_cross[ant]) > 5
        else:
            assert np.abs(mean_Vij_cross[ant]) < 2

    # test rawMetric
    mean_Vij_cross = ant_metrics.mean_Vij_cross_pol_metrics(abs_vis_stats, rawMetric=True)
    for ant in mean_Vij_cross:
        assert ant[0] in [0, 1, 2]
        assert ant[1] in ['Jee', 'Jnn']
        assert mean_Vij_cross[ant] == {0: 12 / 11, 1: 7 / 6, 2: 20 / 3}[ant[0]]

    # test xants
    mean_Vij_cross1 = ant_metrics.mean_Vij_cross_pol_metrics(abs_vis_stats, xants=[(1, 'Jee')], rawMetric=True)
    mean_Vij_cross2 = ant_metrics.mean_Vij_cross_pol_metrics(abs_vis_stats, xants=[1], rawMetric=True)
    for mean_Vij_cross in [mean_Vij_cross1, mean_Vij_cross2]:
        for ant in mean_Vij_cross:
            assert ant[0] in [0, 2]
            assert ant[1] in ['Jee', 'Jnn']
            assert mean_Vij_cross[ant] == {0: 12 / 11, 2: 20 / 3}[ant[0]]


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
    # load data
    four_pol_uvh5 = DATA_PATH + '/zen.2457698.40355.full_pol_test.uvh5'
    am = ant_metrics.AntennaMetrics(four_pol_uvh5, apriori_xants=[9, (10, 'Jxx'), (20, 'jxx')])

    # test metadata
    assert am.datafile_list == [four_pol_uvh5]
    assert am.hd is not None
    assert am.history == ''
    assert 'Git hash' in am.version_str

    # test antennas and baselines
    true_antnums = [9, 10, 20, 22, 31, 43, 53, 64, 65, 72, 80, 81, 88, 89, 96, 97, 104, 105, 112]
    assert len(am.bls) == len(true_antnums) * (len(true_antnums) - 1) / 2 * 4 + len(true_antnums) * 4
    for antpol in ['Jxx', 'Jyy']:
        assert antpol in am.antpols
        for antnum in true_antnums:
            assert antnum in am.antnums
            assert (antnum, antpol) in am.ants

    # test apriori xants
    assert (9, 'Jxx') in am.apriori_xants
    assert (9, 'Jyy') in am.apriori_xants
    assert 9 not in am.apriori_xants
    assert (10, 'Jxx') in am.apriori_xants
    assert (10, 'Jyy') not in am.apriori_xants
    assert (20, 'Jxx') in am.apriori_xants
    assert (20, 'Jyy') not in am.apriori_xants

    # test errors in parsing apriori xants
    with pytest.raises(ValueError):
        ant_metrics.AntennaMetrics(four_pol_uvh5, apriori_xants=(9, 'Jxx'))
    with pytest.raises(ValueError):
        ant_metrics.AntennaMetrics(four_pol_uvh5, apriori_xants=[(9, 10, 'xx')])
    with pytest.raises(ValueError):
        ant_metrics.AntennaMetrics(four_pol_uvh5, apriori_xants=[1.0])

    # test _reset_summary_stats
    for ant in am.apriori_xants:
        assert am.removal_iteration[ant] == -1
        assert ant in am.xants
    assert am.crossed_ants == []
    assert am.dead_ants == []
    assert am.iter == 0
    assert am.all_metrics == {}
    assert am.all_mod_z_scores == {}
    assert am.final_metrics == {}
    assert am.final_mod_z_scores == {}

    # test _load_time_freq_abs_vis_stats
    for bl in am.bls:
        assert bl in am.abs_vis_stats
        assert np.real(am.abs_vis_stats[bl]) >= 0
        assert np.imag(am.abs_vis_stats[bl]) == 0

    # test Nbls_per_load
    am2 = ant_metrics.AntennaMetrics(four_pol_uvh5, Nbls_per_load=100)
    for bl in am.bls:
        assert am.abs_vis_stats[bl] == am2.abs_vis_stats[bl]


def test_iterative_antenna_metrics_and_flagging():
    four_pol_uvh5 = DATA_PATH + '/zen.2457698.40355.full_pol_test.uvh5'
    am = ant_metrics.AntennaMetrics(four_pol_uvh5)

    # try normal operation
    am.iterative_antenna_metrics_and_flagging(verbose=True, crossCut=5, deadCut=5)
    assert (81, 'Jxx') in am.crossed_ants
    assert (81, 'Jyy') in am.crossed_ants
    assert (81, 'Jxx') in am.xants
    assert (81, 'Jyy') in am.xants
    assert list(am.all_mod_z_scores.keys()) == [0, 1]
    assert list(am.all_metrics.keys()) == [0, 1]
    assert am.all_mod_z_scores[0]['meanVijXPol'][(81, 'Jxx')] >= 5
    assert am.all_mod_z_scores[0]['meanVijXPol'][(81, 'Jyy')] >= 5
    for metric in am.all_metrics[1]:
        for ant in am.all_metrics[1][metric]:
            assert am.all_metrics[1][metric][ant] < 5
    assert am.final_mod_z_scores['meanVijXPol'][(81, 'Jxx')] == am.final_mod_z_scores['meanVijXPol'][(81, 'Jyy')]
    assert am.final_mod_z_scores['meanVijXPol'][(81, 'Jxx')] == am.all_mod_z_scores[0]['meanVijXPol'][(81, 'Jxx')]
    assert am.final_mod_z_scores['meanVijXPol'][(81, 'Jyy')] == am.all_mod_z_scores[0]['meanVijXPol'][(81, 'Jyy')]
    assert am.final_metrics['meanVijXPol'][(81, 'Jxx')] == am.final_metrics['meanVijXPol'][(81, 'Jyy')]
    assert am.final_metrics['meanVijXPol'][(81, 'Jxx')] == am.all_metrics[0]['meanVijXPol'][(81, 'Jxx')]
    assert am.final_metrics['meanVijXPol'][(81, 'Jyy')] == am.all_metrics[0]['meanVijXPol'][(81, 'Jyy')]

    # try run_cross_pols=False
    am.iterative_antenna_metrics_and_flagging(verbose=True, deadCut=4, run_cross_pols=False)
    assert (81, 'Jxx') in am.dead_ants
    assert (81, 'Jyy') in am.dead_ants
    assert (81, 'Jxx') in am.xants
    assert (81, 'Jyy') in am.xants
    assert am.crossed_ants == []

    # try run_cross_pols_only=True
    am.iterative_antenna_metrics_and_flagging(verbose=True, run_cross_pols_only=True)
    assert (81, 'Jxx') in am.crossed_ants
    assert (81, 'Jyy') in am.crossed_ants
    assert (81, 'Jxx') in am.xants
    assert (81, 'Jyy') in am.xants
    assert am.dead_ants == []

    # test _find_totally_dead_ants
    for bl in am.abs_vis_stats:
        if 9 in bl:
            am.abs_vis_stats[bl] = 0.0
    am.iterative_antenna_metrics_and_flagging(verbose=True)
    assert (9, 'Jxx') in am.xants
    assert (9, 'Jyy') in am.xants
    assert (9, 'Jxx') in am.dead_ants
    assert (9, 'Jyy') in am.dead_ants
    assert am.removal_iteration[9, 'Jxx'] == -1
    assert am.removal_iteration[9, 'Jyy'] == -1

    # test error
    with pytest.raises(ValueError):
        am.iterative_antenna_metrics_and_flagging(verbose=True, run_cross_pols=False, run_cross_pols_only=True)


def test_ant_metrics_run_and_load_antenna_metrics():
    four_pol_uvh5 = DATA_PATH + '/zen.2457698.40355.full_pol_test.uvh5'
    am = ant_metrics.AntennaMetrics(four_pol_uvh5)
    am.iterative_antenna_metrics_and_flagging()

    ant_metrics.ant_metrics_run(four_pol_uvh5, overwrite=True, history='test_history_string', verbose=True)
    am_hdf5 = ant_metrics.load_antenna_metrics(four_pol_uvh5.replace('.uvh5', '.ant_metrics.hdf5'))

    assert 'test_history_string' in am_hdf5['history']
    assert am.version_str == am_hdf5['version']
    assert am.crossCut == am_hdf5['cross_pol_z_cut']
    assert am.deadCut == am_hdf5['dead_ant_z_cut']
    assert set(am.xants) == set(am_hdf5['xants'])
    assert set(am.crossed_ants) == set(am_hdf5['crossed_ants'])
    assert set(am.dead_ants) == set(am_hdf5['dead_ants'])
    assert set(am.datafile_list) == set(am_hdf5['datafile_list'])

    assert qmtest.recursive_compare_dicts(am.removal_iteration, am_hdf5['removal_iteration'])
    assert qmtest.recursive_compare_dicts(am.final_metrics, am_hdf5['final_metrics'])
    assert qmtest.recursive_compare_dicts(am.all_metrics, am_hdf5['all_metrics'])
    assert qmtest.recursive_compare_dicts(am.final_mod_z_scores, am_hdf5['final_mod_z_scores'])
    assert qmtest.recursive_compare_dicts(am.all_mod_z_scores, am_hdf5['all_mod_z_scores'])

    os.remove(four_pol_uvh5.replace('.uvh5', '.ant_metrics.hdf5'))
