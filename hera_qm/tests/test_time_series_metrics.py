# -*- coding: utf-8 -*-
# Copyright (c) 2023 the HERA Project
# Licensed under the MIT License
"""Tests for the time_series_metrics module."""

import pytest
import numpy as np
from hera_qm import time_series_metrics


def test_true_stretches():
    assert time_series_metricstrue_stretches(np.array([True, False, False])) == [slice(0, 1, None)]
    assert time_series_metricstrue_stretches(np.array([True, False, False, True])) == [slice(0, 1, None), slice(3, 4, None)]
    assert time_series_metricstrue_stretches(np.array([False, True, True, False])) == [slice(1, 3, None)]

