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