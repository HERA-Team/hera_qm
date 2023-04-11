# -*- coding: utf-8 -*-
# Copyright (c) 2023 the HERA Project
# Licensed under the MIT License

"""Utility functions for flagging and metrics on time series data."""
import numpy as np
from scipy.ndimage import convolve


def true_stretches(bool_arr):
    '''Returns a list of slices corresponding to contiguous sequences where bool_arr is True.'''
    # find the indices where bool_arr changes from True to False or vice versa
    ba = np.array(bool_arr)
    diff = np.diff(ba.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    
    # handle first and last values
    if ba[0]:
        starts = np.insert(starts, 0, 0)
    if ba[-1]:
        ends = np.append(ends, len(bool_arr) - 1)

    stretches = [slice(starts[i], ends[i] + 1) for i in range(len(starts))]
    return stretches


def impose_max_flag_gap(flags, max_flag_gap=30):
    '''Adds flags to limit the largest possible number of flagged times between unflagged times.
    
    Arguments:
        flags_in: 1D boolean numpy array of starting flags (modified in place)
        max_flag_gap: integer maximum allowed sequential flags (default 30)
    '''
    bad_stretches = sorted(true_stretches(flags), key=lambda s: s.stop - s.start)[::-1]
    for bs in bad_stretches:
        if bs.stop - bs.start > max_flag_gap:
            # figure out whether to flag everything before or after this gap
            if np.sum(~flags[bs.stop:]) >= np.sum(~flags[:bs.start]):
                flags[:bs.start] = True
            else:
                flags[bs.stop:] = True
                
    return flags


def metric_convolution_flagging(metric, starting_flags, ok_range, sigma=30, max_flag_gap=30):
    '''Grows flags by looking at whether the given metric returns to some OK range in the 
    gap between flags. Also imposes a max_flag_gap (see impose_max_flag_gap()).
    
    Arguments:
        metric: 1D numpy array of floats containing the per-file metric values used for flagging.
        starting_flags: 1D numpy array of booleans of initial flags
        ok_range: length-2 tuple of convolved metric range outside of which flags are considered bad
        sigma: standard deviation of metric Gaussian smoothing scale (in units of integrations)
        max_flag_gap: integer maximum allowed sequential flags (default 30)
    Returns:
        new_flags: starting flags, but with possible additional flags
    '''
    
    # compute convolved metric
    kernel = np.exp(-np.arange(-len(metric) // 2, len(metric) // 2 + 1)**2 / 2 / sigma**2)
    kernel /= np.sum(kernel)
    convolved_metric = convolve(metric, kernel, mode='reflect')
    
    new_flags = np.array(starting_flags)
    nflags = np.sum(new_flags)
    while True:
        # figure out which bad stretches are "persistant" and which are just blips
        bad_stretches = true_stretches(new_flags)
        persistant_bad_stretches = [bs for bs in bad_stretches if 
                                    np.any((convolved_metric[bs] > ok_range[1]) | (convolved_metric[bs] < ok_range[0]))]

        # figure out which stretches that aren't "persistant" bad eventually go back to OK... don't flag these 
        not_persistant_bad = np.ones_like(new_flags)
        for bs in bad_stretches:
            if np.any(convolved_metric[bs] > ok_range[1]):
                not_persistant_bad[bs] = False
    
        for stretch in true_stretches(not_persistant_bad):
            if not np.any((convolved_metric[stretch] >= ok_range[0]) & (convolved_metric[stretch] <= ok_range[1])):
                new_flags[stretch] = True
                
        impose_max_flag_gap(new_flags, max_flag_gap=max_flag_gap)
        # check if flags haven't grown and if so, break
        if np.sum(new_flags) == nflags:
            return new_flags
        else:
            nflags = np.sum(new_flags)
