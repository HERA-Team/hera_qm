# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
import os
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


