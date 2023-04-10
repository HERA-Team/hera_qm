# -*- coding: utf-8 -*-
# Copyright (c) 2023 the HERA Project
# Licensed under the MIT License
"""Tests for the time_series_metrics module."""

import pytest
import numpy as np
from hera_qm import time_series_metrics


def test_true_stretches():
    assert time_series_metrics.true_stretches(np.array([True, False, False])) == [slice(0, 1, None)]
    assert time_series_metrics.true_stretches(np.array([True, False, False, True])) == [slice(0, 1, None), slice(3, 4, None)]
    assert time_series_metrics.true_stretches(np.array([False, True, True, False])) == [slice(1, 3, None)]


def test_impose_max_flag_gap():
    np.testing.assert_array_equal(time_series_metrics.impose_max_flag_gap(np.array([False, True, False, True, False, False]), max_flag_gap=0),
                                  np.array([True, True, True, True, False, False]))
    np.testing.assert_array_equal(time_series_metrics.impose_max_flag_gap(np.array([False, False, True, False, True, False]), max_flag_gap=0),
                                  np.array([False, False, True, True, True, True]))
    np.testing.assert_array_equal(time_series_metrics.impose_max_flag_gap(np.array([False, False, True, False, True, False]), max_flag_gap=1),
                                  np.array([False, False, True, False, True, False]))
    np.testing.assert_array_equal(time_series_metrics.impose_max_flag_gap(np.array([False, False, True, True, True, False]), max_flag_gap=2),
                                  np.array([False, False, True, True, True, True]))


def test_metric_convolution_flagging():
    metric = np.concatenate([[0] * 10, [15, 3, 15], [0] * 10, [3, 15] * 10, [0]]).astype(float)
    starting_flags = metric > 12
    new_flags = time_series_metrics.metric_convolution_flagging(metric, starting_flags, [0, 4], sigma=1)
    assert np.all(new_flags[10:13])
    assert new_flags[-1]
    new_flags = time_series_metrics.metric_convolution_flagging(metric, starting_flags, [0, 4], sigma=.1)
    assert ~new_flags[11]
    assert ~new_flags[-1]
    new_flags = time_series_metrics.metric_convolution_flagging(metric, starting_flags, [0, 4], sigma=.1, max_flag_gap=0)
    assert np.all(new_flags[0:13])
    assert np.all(~new_flags[14:24])
    assert np.all(new_flags[24:])
