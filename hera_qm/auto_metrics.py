# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics using day-long autocorrelations."""
import numpy as np
from copy import deepcopy
from .version import hera_qm_version_str


def nanmad(a, axis=None):
    '''Analogous to np.nanmedian, but for median absolute deviation.'''
    nanmed = np.nanmedian(a, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(a - nanmed), axis=axis)

def nanmedian_abs_diff(a, axis=0):
    '''Computes the absolute difference between neightbors along axis, then collaposes 
    along that axis with the nanmedian. Useful for studying temporal variability, e.g.'''
    return np.nanmedian(np.abs(np.diff(a, axis=axis)), axis=axis)

def nanmean_abs_diff(a, axis=0):
    '''Computes the absolute difference between neightbors along axis, then collaposes 
    along that axis with the nanmean. Useful for studying temporal variability, e.g.'''
    return np.nanmean(np.abs(np.diff(a, axis=axis)), axis=axis)

def _check_only_auto_keys(data):
    '''Verify that keys in data are only autocorrelation keys of the form (ant, ant, pol).'''
    for bl in data:
        ap0, ap1 = utils.split_pol(bl[2])
        if (bl[0] != bl[1]) or (ap0 != ap1):
            raise ValueError(f'{bl} is not an autocorrelation key.')