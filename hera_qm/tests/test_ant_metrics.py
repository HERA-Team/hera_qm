# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
# import os
# import sys
# import pyuvdata.tests as uvtest
# from hera_qm import utils
from hera_qm import ant_metrics


class fake_data():
# from hera_qm import metrics_io
# from hera_qm.data import DATA_PATH
# import hera_qm.tests as qmtest


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


