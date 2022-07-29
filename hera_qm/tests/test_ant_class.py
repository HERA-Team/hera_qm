# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
from hera_qm import ant_class


def test_check_antpol():
    ant_class._check_antpol((1, 'Jnn'))
    ant_class._check_antpol((1, 'ee'))
    ant_class._check_antpol((1, 'x'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1.0, 'Jee'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1, 'not a pol'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1, 1, 'Jee'))

