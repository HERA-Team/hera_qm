# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
import os
import glob
from hera_qm import auto_metrics
from hera_qm import metrics_io
from hera_qm.data import DATA_PATH
import hera_qm.tests as qmtest


def test_nanmad():
    # test 1D
    test = np.arange(10.)
    test[0:4] = np.nan
    assert auto_metrics.nanmad(test) == 1.5

    # test 2D
    test2 = np.outer(np.arange(10.), np.ones(2))
    test2[0:4] = np.nan
    np.testing.assert_array_equal(auto_metrics.nanmad(test2, axis=0), np.array([1.5, 1.5]))


def test_nanmedian_abs_diff():
    test = np.outer(np.arange(10.), np.ones(5))
    test[0,0] = np.nan
    test[1,0] = 2
    np.testing.assert_array_equal(auto_metrics.nanmedian_abs_diff(test), np.ones(5))


def test_nanmean_abs_diff():
    test = np.outer(np.arange(10.), np.ones(5))
    test[0,0] = np.nan
    test[1,0] = 2
    assert auto_metrics.nanmean_abs_diff(test)[0] == 7. / 8
    np.testing.assert_array_equal(auto_metrics.nanmean_abs_diff(test)[1:], np.ones(4))


def test_check_only_auto_keys():
    good_data = {(0, 0, 'ee'): np.zeros((10, 10)), (2, 2, 'nn'): np.zeros((10, 10))}
    auto_metrics._check_only_auto_keys(good_data)

    # test cross-correlation keys
    bad_data = {(0, 0, 'ee'): np.zeros((10, 10)), (0, 2, 'nn'): np.zeros((10, 10))}
    with pytest.raises(ValueError):
        auto_metrics._check_only_auto_keys(bad_data)

    # test cross-pol keys
    bad_data = {(0, 0, 'ee'): np.zeros((10, 10)), (0, 0, 'ne'): np.zeros((10, 10))}
    with pytest.raises(ValueError):
        auto_metrics._check_only_auto_keys(bad_data)


def test_get_auto_spectra():
    autos = {(0, 0, 'ee'): 10 * np.ones((10, 5)), (1, 1, 'ee'): 10 * np.ones((10, 5))}
    autos[0, 0, 'ee'][:, 1] = 20
    autos[0, 0, 'ee'][0, 4] = 20

    # test basic operation
    spectra = auto_metrics.get_auto_spectra(autos)
    np.testing.assert_array_equal(spectra[0, 0, 'ee'], np.array([1., 2., 1., 1., 1.]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], np.array([1., 1., 1., 1., 1.]))

    # test time_avg_func=np.nanmean
    spectra = auto_metrics.get_auto_spectra(autos, time_avg_func=np.nanmean)
    np.testing.assert_array_equal(spectra[0, 0, 'ee'], np.array([1., 2., 1., 1., 1.1]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], np.array([1., 1., 1., 1., 1.]))

    # test turning off scalar_norm
    spectra = auto_metrics.get_auto_spectra(autos, time_avg_func=np.nanmean, scalar_norm=False)
    np.testing.assert_array_equal(spectra[0, 0, 'ee'], 10 * np.array([1., 2., 1., 1., 1.1]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], 10 * np.array([1., 1., 1., 1., 1.]))

    # test replacing norm_func
    spectra = auto_metrics.get_auto_spectra(autos, time_avg_func=np.nanmean, norm_func=np.nanmean)
    np.testing.assert_array_almost_equal(spectra[0, 0, 'ee'], 50 / 61 * np.array([1., 2., 1., 1., 1.1]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], np.array([1., 1., 1., 1., 1.]))

    # test turning on waterfall norm with ex_ants
    spectra = auto_metrics.get_auto_spectra(autos, time_avg_func=np.nanmean, norm_func=np.nanmean,
                                            scalar_norm=False, waterfall_norm=True, ex_ants=[0])
    np.testing.assert_array_almost_equal(spectra[0, 0, 'ee'], np.array([1., 2., 1., 1., 1.1]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], np.array([1., 1., 1., 1., 1.]))

    # test turning on waterfall norm without ex_ants
    spectra = auto_metrics.get_auto_spectra(autos, time_avg_func=np.nanmean, norm_func=np.nanmean,
                                            scalar_norm=False, waterfall_norm=True)
    np.testing.assert_array_almost_equal(spectra[0, 0, 'ee'], np.array([1., 4. / 3, 1., 1., 31. / 30]))
    np.testing.assert_array_almost_equal(spectra[1, 1, 'ee'], np.array([1., 2. / 3, 1., 1., 29. / 30]))

    # test with flags
    flags = np.zeros((10, 5), dtype=bool)
    flags[0, 4] = True
    spectra = auto_metrics.get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanmean, scalar_norm=False)
    np.testing.assert_array_equal(spectra[0, 0, 'ee'], 10 * np.array([1., 2., 1., 1., 1.]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], 10 * np.array([1., 1., 1., 1., 1.]))

    # test with totally flagged channel
    flags[:, 1] = True
    spectra = auto_metrics.get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanmean, scalar_norm=False)
    np.testing.assert_array_equal(spectra[0, 0, 'ee'], 10 * np.array([1., np.nan, 1., 1., 1.]))
    np.testing.assert_array_equal(spectra[1, 1, 'ee'], 10 * np.array([1., np.nan, 1., 1., 1.]))


def test_spectrum_modz_scores():
    np.random.seed(21)
    bls = [(a, a, 'ee') for a in np.arange(100)]

    # test normal operation with one high power antenna
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] += 10
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean)
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 5000
        else:
            assert np.abs(modzs[bl]) < 5
            
    # test with a nan
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] += 10
    auto_spectra[(1, 1, 'ee')][30] = np.nan
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean)
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 5000
        else:
            assert np.abs(modzs[bl]) < 5

    # test with single spectral channel outlier in median mode
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')][0] += 10
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmedian, metric_func=np.nanmedian)
    for bl in auto_spectra:
        assert np.abs(modzs[bl]) < 10
        
    # test with single spectral channel outlier in mean mode
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')][0] += 10
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean, metric_func=np.nanmean)
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 50
        else:
            assert np.abs(modzs[bl]) < 5

    # test abs_diff=False
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] -= 10
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean, abs_diff=False)
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] < -5000
        else:
            assert np.abs(modzs[bl]) < 5

    # test metric_log=True
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] *= 10
    auto_spectra[(1, 1, 'ee')] /= 10
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean, metric_log=True, ex_ants=[0, 1])
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 500
        elif bl == (1, 1, 'ee'):
            assert modzs[bl] > 500
        else:
            assert np.abs(modzs[bl]) < 5

    # test both metric_log=True and abs_diff=False
    modzs = auto_metrics.spectrum_modz_scores(auto_spectra, overall_spec_func=np.nanmean, metric_log=True, abs_diff=False, ex_ants=[0, 1])
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 500
        elif bl == (1, 1, 'ee'):
            assert modzs[bl] < -500, modzs[bl]
        else:
            assert np.abs(modzs[bl]) < 5


def test_iterative_spectrum_modz():
    bls = [(a, a, 'ee') for a in np.arange(100)]

    # test that antennas get excluded in order of badness
    np.random.seed(21)
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] += 1
    auto_spectra[(1, 1, 'ee')] += 10
    ex_ants, modzs = auto_metrics.iterative_spectrum_modz(auto_spectra, modz_cut=10., overall_spec_func=np.nanmean, metric_func=np.nanmean)
    assert ex_ants == [1, 0]
    for bl in auto_spectra:
        if bl == (0, 0, 'ee'):
            assert modzs[bl] > 1000
        elif bl == (1, 1, 'ee'):
            assert modzs[bl] > 10000
        else:
            assert np.abs(modzs[bl]) < 10

    # test that order of exclusion doesn't matter for final modified Z scores
    np.random.seed(21)
    auto_spectra = {bl: .01 * np.random.randn(100) + 1 for bl in bls}
    auto_spectra[(0, 0, 'ee')] += 1
    auto_spectra[(1, 1, 'ee')] += 10
    ex_ants, modzs_2 = auto_metrics.iterative_spectrum_modz(auto_spectra, modz_cut=10., prior_ex_ants=[0], overall_spec_func=np.nanmean, metric_func=np.nanmean)
    assert ex_ants == [0, 1]
    for bl in auto_spectra:
        assert modzs_2[bl] == modzs[bl]


def test_auto_metrics_run():
    # run main function on downselected H4C data, then read results
    auto_files = glob.glob(os.path.join(DATA_PATH, 'zen.2459122.*.sum.autos.downselected.uvh5'))
    metrics_outfile = os.path.join(DATA_PATH, 'unittest_auto_metrics.h5')
    ex_ants, modzs, spectra, flags = auto_metrics.auto_metrics_run(metrics_outfile, auto_files,
                                                                   median_round_modz_cut=75., mean_round_modz_cut=5.,
                                                                   edge_cut=100, Kt=8, Kf=8, sig_init=5.0, sig_adj=2.0, 
                                                                   chan_thresh_frac=.05, history='unittest', overwrite=True)
    metrics_in = metrics_io.load_metric_file(metrics_outfile)

    # what to expect
    pols = ['ee', 'nn']
    bad_ants = [123, 90, 93]
    good_ants = [36, 50, 66, 83, 85, 91, 99, 109, 117, 118, 120, 127, 128, 129, 141, 143, 144, 160, 162, 163, 165, 176, 185]

    # assert that all bad antennas are in the data and that all good antennas aren't
    for bad_ant in bad_ants:
        if bad_ant != 123:  # median_round_modz_cut=150. was picked so this would only get cut on round 2
            assert bad_ant in ex_ants['r1_ex_ants']
        else:
            assert bad_ant not in ex_ants['r1_ex_ants']
        assert bad_ant in ex_ants['r2_ex_ants']
    for good_ant in good_ants:
        assert good_ant not in ex_ants['r1_ex_ants']
        assert good_ant not in ex_ants['r2_ex_ants']

    # check that all results have sensible shapes and types and bls
    for ant in (bad_ants + good_ants):
        for pol in pols:
            bl = (ant, ant, pol)
            for spec_type in spectra:
                assert bl in spectra[spec_type]
                assert spectra[spec_type][bl].shape == (1536,)
                assert spectra[spec_type][bl].dtype == np.dtype('float64')
            for metric in modzs:
                assert bl in modzs[metric]
                assert isinstance(modzs[metric][bl], float)

    # test edge_cut and that not all flags are True
    assert np.all(flags[:, 0:100])
    assert np.all(flags[:, -100:])
    assert not np.all(flags)

    # test that file outputs match function outputs
    qmtest.recursive_compare_dicts(ex_ants, metrics_in['ex_ants'])
    qmtest.recursive_compare_dicts(modzs, metrics_in['modzs'])
    qmtest.recursive_compare_dicts(spectra, metrics_in['spectra'])
    np.testing.assert_array_equal(flags, metrics_in['flags'])

    # test history
    assert metrics_in['history'] == 'unittest'

    # test saved parameters
    assert metrics_in['parameters']
    assert metrics_in['parameters']['median_round_modz_cut'] == 75.
    assert metrics_in['parameters']['mean_round_modz_cut'] == 5.
    assert metrics_in['parameters']['edge_cut'] == 100
    assert metrics_in['parameters']['Kt'] == 8 
    assert metrics_in['parameters']['Kf'] == 8
    assert metrics_in['parameters']['sig_init'] == 5.0
    assert metrics_in['parameters']['sig_adj'] == 2.0
    assert metrics_in['parameters']['chan_thresh_frac'] == .05

    # remove outfile
    os.remove(metrics_outfile)
