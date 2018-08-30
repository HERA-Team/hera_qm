# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from .uvflag import UVFlag
from hera_qm import utils as qm_utils
from hera_qm.version import hera_qm_version_str
import warnings
import copy
import collections


#############################################################################
# Utility functions
#############################################################################

def flag_xants(uv, xants, inplace=True):
    """Flag visibilities containing specified antennas.
    Args:
        uv (UVData, UVCal, or UVFlag object): Data to be flagged
        xants (list of ints): antennas to flag
        inplace (bool): Apply flags to uv (Default). If False, returns UVFlag object
                        with only xants flags.
    Returns:
        uvo: if inplace, applies flags to input uv. If not inplace,
                uvo is a new UVFlag object with only xants flags.
    """
    # check that we got an appropriate object
    if not issubclass(uv.__class__, (UVData, UVCal, UVFlag)):
        raise ValueError('First argument to flag_xants must be a UVData, UVCal, '
                         ' or UVFlag object.')
    if isinstance(uv, UVFlag) and uv.type == 'waterfall':
        raise ValueError('Cannot flag antennas on UVFlag obejct of type "waterfall".')

    if not inplace:
        if isinstance(uv, UVFlag):
            uvo = uv.copy()
            uvo.to_flag()
        else:
            uvo = UVFlag(uv, mode='flag')
    else:
        uvo = uv

    if isinstance(uvo, UVFlag) and uvo.mode != 'flag':
        raise ValueError('Cannot flag antennas on UVFlag obejct in mode ' + uvo.mode)

    if not isinstance(xants, collections.Iterable):
        xants = [xants]
    if issubclass(uvo.__class__, UVData) or (isinstance(uvo, UVFlag) and uvo.type == 'baseline'):
        all_ants = np.unique(np.append(uvo.ant_1_array, uvo.ant_2_array))
        for ant in all_ants:
            for xant in xants:
                blts = uvo.antpair2ind(ant, xant)
                uvo.flag_array[blts, :, :, :] = True
                blts = uvo.antpair2ind(xant, ant)
                uvo.flag_array[blts, :, :, :] = True
    elif issubclass(uvo.__class__, UVCal) or (isinstance(uvo, UVFlag) and uvo.type == 'antenna'):
        for xant in xants:
            ai = np.where(uvo.ant_array == xant)[0]
            uvo.flag_array[ai, :, :, :, :] = True

    if not inplace:
        return uvo


#############################################################################
# Functions for preprocessing data prior to RFI flagging
#############################################################################

def medmin(d):
    '''Calculate the median minus minimum statistic of array.
    Args:
        d (array): 2D data array of the shape (time,frequency).
    Returns:
        (float): medmin statistic.

    Notes:
        The statistic first computes the minimum value of the array along the
        first axis (the time axis, if the array is passed in as (time, frequency,
        so that a single spectrum is returned). The median of these values is
        computed, multiplied by 2, and then the minimum value is subtracted off.
        The goal is to get a proxy for the "noise" in the 2d array.
    '''
    if d.ndim != 2:
        raise ValueError('Input to medmin must be 2D array.')
    mn = np.min(d, axis=0)
    return 2 * np.median(mn) - np.min(mn)


def medminfilt(d, Kt=8, Kf=8):
    '''Filter an array on scales of Kt,Kf indexes with medmin.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        Kt (int, optional): integer representing box dimension in time to apply statistic.
        Kf (int, optional): integer representing box dimension in frequency to apply statistic.
    Returns:
        array: filtered array with same shape as input array.
    '''
    if d.ndim != 2:
        raise ValueError('Input to medminfilt must be 2D array.')
    if Kt > d.shape[0]:
        warnings.warn("Kt value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kt, d.shape[0]))
        Kt = d.shape[0]
    if Kf > d.shape[1]:
        warnings.warn("Kf value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kf, d.shape[1]))
        Kf = d.shape[1]
    d_sm = np.empty_like(d)
    for i in xrange(d.shape[0]):
        for j in xrange(d.shape[1]):
            i0, j0 = max(0, i - Kt), max(0, j - Kf)
            i1, j1 = min(d.shape[0], i + Kt), min(d.shape[1], j + Kf)
            d_sm[i, j] = medmin(d[i0:i1, j0:j1])
    return d_sm


def detrend_deriv(d, dt=True, df=True):
    ''' Detrend array by taking the derivative in either time, frequency
        or both. When taking the derivative of both, the derivative in
        frequency is performed first, then in time.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        dt (bool, optional): derivative across time bins.
        df (bool, optional): derivative across frequency bins.
    Returns:
        array: detrended array with same shape as input array.
    '''
    if d.ndim != 2:
        raise ValueError('Input to detrend_deriv must be 2D array.')
    if not (dt or df):
        raise ValueError("dt and df cannot both be False when calling detrend_deriv")
    if df:
        # take gradient along frequency
        d_df = np.gradient(d, axis=1)
    else:
        d_df = d
    if dt:
        # take gradient along time
        d_dtdf = np.gradient(d_df, axis=0)
    else:
        d_dtdf = d_df

    d2 = np.abs(d_dtdf)**2
    # model sig as separable function of 2 axes
    sig_f = np.median(d2, axis=0)
    sig_f.shape = (1, -1)
    sig_t = np.median(d2, axis=1)
    sig_t.shape = (-1, 1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    # don't divide by zero, instead turn those entries into +inf
    f = np.true_divide(d_dtdf, sig, where=(np.abs(sig) > 1e-7))
    f = np.where(np.abs(sig) > 1e-7, f, np.inf)
    return f


def detrend_medminfilt(d, Kt=8, Kf=8):
    """Detrend array using medminfilt statistic. See medminfilt.
    Args:
        d (array): 2D data array of the shape (time, frequency) to detrend
        Kt (int): size in time to apply medminfilter over
        Kf (int): size in frequency to apply medminfilter over
    Returns:
        float array: float array of outlier significance metric
    """
    if d.ndim != 2:
        raise ValueError('Input to detrend_medminfilt must be 2D array.')
    d_sm = medminfilt(np.abs(d), 2 * Kt + 1, 2 * Kf + 1)
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts minmed on same scale as average
    sig = np.sqrt(medminfilt(d_sq, 2 * Kt + 1, 2 * Kf + 1)) * (np.sqrt(Kt**2 + Kf**2) / .64)
    # don't divide by zero, instead turn those entries into +inf
    f = np.true_divide(d_rs, sig, where=(np.abs(sig) > 1e-7))
    f = np.where(np.abs(sig) > 1e-7, f, np.inf)
    return f


def detrend_medfilt(d, Kt=8, Kf=8):
    """Detrend array using a median filter.
    Args:
        d (array): 2D data array to detrend.
        K (int, optional): box size to apply medminfilt over
    Returns:
        f: array of outlier significance metric. Same type and size as d.
    """
    # Delay import so scipy is not required for any use of hera_qm
    from scipy.signal import medfilt2d

    if d.ndim != 2:
        raise ValueError('Input to detrend_medfilt must be 2D array.')
    if Kt > d.shape[0]:
        warnings.warn("Kt value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kt, d.shape[0]))
        Kt = d.shape[0]
    if Kf > d.shape[1]:
        warnings.warn("Kf value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kf, d.shape[1]))
        Kf = d.shape[1]
    d = np.concatenate([d[Kt - 1::-1], d, d[:-Kt - 1:-1]], axis=0)
    d = np.concatenate([d[:, Kf - 1::-1], d, d[:, :-Kf - 1:-1]], axis=1)
    if np.iscomplexobj(d):
        d_sm_r = medfilt2d(d.real, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm_i = medfilt2d(d.imag, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm = d_sm_r + 1j * d_sm_i
    else:
        d_sm = medfilt2d(d, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average
    sig = np.sqrt(medfilt2d(d_sq, kernel_size=(2 * Kt + 1, 2 * Kf + 1)) / .456)
    # don't divide by zero, instead turn those entries into +inf
    f = np.true_divide(d_rs, sig, where=(np.abs(sig) > 1e-8))
    f = np.where(np.abs(sig) > 1e-8, f, np.inf)
    return f[Kt:-Kt, Kf:-Kf]


# Update algorithm_dict whenever new metric algorithm is created.
algorithm_dict = {'medmin': medmin, 'medminfilt': medminfilt, 'detrend_deriv': detrend_deriv,
                  'detrend_medminfilt': detrend_medminfilt, 'detrend_medfilt': detrend_medfilt}

#############################################################################
# RFI flagging algorithms
#############################################################################


def watershed_flag(uvf_m, uvf_f, nsig_p=2., nsig_f=None, nsig_t=None, avg_method='quadmean',
                   inplace=True):
    '''Expands a set of flags using a watershed algorithm.
    Uses a UVFlag object in 'metric' mode (i.e. how many sigma the data point is
    from the center) and a set of flags to grow the flags using defined thresholds.

    Args:
        uvf_m: UVFlag object in 'metric' mode
        uvf_f: UVFlag object in 'flag' mode
        nsig_p: Number of sigma above which to flag pixels which are near
               previously flagged pixels. Default is 2.0.
        nsig_f: Number of sigma above which to flag channels which are near
               fully flagged channels. Bypassed if None (Default).
        nsig_t: Number of sigma above which to flag integrations which are near
               fully flagged integrations. Bypassed if None (Default)
        avg_method: Method to average metric data for frequency and time watershedding.
                    Options are 'mean', 'absmean', and 'quadmean' (Default).
        inplace: Whether to update uvf_f or create a new flag object. Default is True.

    Returns:
        uvf: UVFlag object in 'flag' mode with flags after watershed.
    '''
    # Check inputs
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')
    if (not isinstance(uvf_f, UVFlag)) or (uvf_f.mode != 'flag'):
        raise ValueError('uvf_f must be UVFlag instance with mode == "flag."')
    if uvf_m.metric_array.shape != uvf_f.flag_array.shape:
        raise ValueError('uvf_m and uvf_f must have data of same shape. Shapes '
                         'are: ' + str(uvf_m.metric_array.shape) + ' and '
                         + str(uvf_f.flag_array.shape))
    # Handle in place
    if inplace:
        uvf = uvf_f
    else:
        uvf = copy.deepcopy(uvf_f)

    try:
        avg_f = qm_utils.averaging_dict[avg_method]
    except KeyError:
        raise KeyError('avg_method must be one of: "mean", "absmean", or "quadmean".')

    # Convenience
    farr = uvf.flag_array
    marr = uvf_m.metric_array
    warr = uvf_m.weights_array

    if uvf_m.type == 'baseline':
        # Pixel watershed
        # TODO: bypass pixel-based if none
        for b in np.unique(uvf.baseline_array):
            i = np.where(uvf.baseline_array == b)[0]
            for pi in range(uvf.polarization_array.size):
                farr[i, 0, :, pi] += _ws_flag_waterfall(marr[i, 0, :, pi],
                                                        farr[i, 0, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            d = avg_f(marr, axis=(0, 1, 3), weights=warr)
            f = np.all(farr, axis=(0, 1, 3))
            farr[:, :, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, 1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            ts = np.unique(uvf.time_array)
            d = np.zeros(ts.size)
            f = np.zeros(ts.size, dtype=np.bool)
            for i, t in enumerate(ts):
                d[i] = avg_f(marr[uvf.time_array == t, 0, :, :],
                             weights=warr[uvf.time_array == t, 0, :, :])
                f[i] = np.all(farr[uvf.time_array == t, 0, :, :])
            f = _ws_flag_waterfall(d, f, nsig_t)
            for i, t in enumerate(ts):
                farr[uvf.time_array == t, :, :, :] += f[i]
    elif uvf_m.type == 'antenna':
        # Pixel watershed
        for ai in range(uvf.ant_array.size):
            for pi in range(uvf.polarization_array.size):
                farr[ai, 0, :, :, pi] += _ws_flag_waterfall(marr[ai, 0, :, :, pi].T,
                                                            farr[ai, 0, :, :, pi].T, nsig_p).T
        if nsig_f is not None:
            # Channel watershed
            d = avg_f(marr, axis=(0, 1, 3, 4), weights=warr)
            f = np.all(farr, axis=(0, 1, 3, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, 1, -1, 1, 1)
        if nsig_t is not None:
            # Time watershed
            d = avg_f(marr, axis=(0, 1, 2, 4), weights=warr)
            f = np.all(farr, axis=(0, 1, 2, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(d, f, nsig_t).reshape(1, 1, 1, -1, 1)
    elif uvf_m.type == 'waterfall':
        # Pixel watershed
        for pi in range(uvf.polarization_array.size):
            farr[:, :, pi] += _ws_flag_waterfall(marr[:, :, pi], farr[:, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            d = avg_f(marr, axis=(0, 2), weights=warr)
            f = np.all(farr, axis=(0, 2))
            farr[:, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            d = avg_f(marr, axis=(1, 2), weights=warr)
            f = np.all(farr, axis=(1, 2))
            farr[:, :, :] += _ws_flag_waterfall(d, f, nsig_t).reshape(-1, 1, 1)
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf


def _ws_flag_waterfall(d, fin, nsig=2.):
    ''' Performs watershed algorithm on 1D or 2D arrays of metric and input flags.
    This is a helper function for watershed_flag, but not usually called
    by end users.

    Args:
        d: 2D or 1D array. Should be in units of standard deviations.
        fin: input (boolean) flags used as seed of watershed. Same size as d.
        nsig: number of sigma to flag above for point near flagged points.
    Returns:
        f: boolean array matching size of d and fin, with watershedded flags.
    '''

    if d.shape != fin.shape:
        raise ValueError('d and f must match in shape. Shapes are: ' + str(d.shape)
                         + ' and ' + str(fin.shape))
    f = copy.deepcopy(fin)
    # There may be an elegant way to combine these... for the future.
    if d.ndim == 1:
        prevn = 0
        x = np.where(f)[0]
        while x.size != prevn:
            prevn = x.size
            for dx in [-1, 1]:
                xp = (x + dx).clip(0, f.size - 1)
                i = np.where(d[xp] > nsig)[0]  # if our metric > sig
                f[xp[i]] = 1
                x = np.where(f)[0]
    elif d.ndim == 2:
        prevx, prevy = 0, 0
        x, y = np.where(f)
        while x.size != prevx and y.size != prevy:
            prevx, prevy = x.size, y.size
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                xp, yp = (x + dx).clip(0, f.shape[0] - 1), (y + dy).clip(0, f.shape[1] - 1)
                i = np.where(d[xp, yp] > nsig)[0]  # if our metric > sig
                f[xp[i], yp[i]] = 1
                x, y = np.where(f)
    else:
        raise ValueError('Data must be 1D or 2D.')
    return f


def flag(uvf_m, nsig_p=6., nsig_f=None, nsig_t=None, avg_method='quadmean'):
    '''Creates a set of flags based on a "metric" type UVFlag object.
    Args:
        uvf_m: UVFlag object in 'metric' mode (ie. number of sigma data is from middle)
        nsig_p: Number of sigma above which to flag pixels. Default is 6.
                Bypassed if None.
        nsig_f: Number of sigma above which to flag channels. Bypassed if None (Default).
        nsig_t: Number of sigma above which to flag integrations. Bypassed if None (Default).
        avg_method: Method to average metric data for frequency and time flagging.
                    Options are 'mean', 'absmean', and 'quadmean' (Default).

    Returns:
        uvf_f: UVFlag object in 'flag' mode with flags determined from uvm.
    '''
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')

    try:
        avg_f = qm_utils.averaging_dict[avg_method]
    except KeyError:
        raise KeyError('avg_method must be one of: "mean", "absmean", or "quadmean".')

    # initialize
    uvf_f = copy.deepcopy(uvf_m)
    uvf_f.to_flag()

    # Pixel flagging
    if nsig_p is not None:
        uvf_f.flag_array[uvf_m.metric_array > nsig_p] = True

    if uvf_m.type == 'baseline':
        if nsig_f is not None:
            # Channel flagging
            d = avg_f(uvf_m.metric_array, axis=(0, 1, 3), weights=uvf_m.weights_array)
            indf = np.where(d > nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :] = True
        if nsig_t is not None:
            # Time flagging
            ts = np.unique(uvf_m.time_array)
            d = np.zeros(ts.size)
            for i, t in enumerate(ts):
                d[i] = avg_f(uvf_m.metric_array[uvf_m.time_array == t, 0, :, :],
                             weights=uvf_m.weights_array[uvf_m.time_array == t, 0, :, :])
            indf = np.where(d > nsig_t)[0]
            for t in ts[indf]:
                uvf_f.flag_array[uvf_f.time_array == t, :, :, :] = True
    elif uvf_m.type == 'antenna':
        if nsig_f is not None:
            # Channel flag
            d = avg_f(uvf_m.metric_array, axis=(0, 1, 3, 4), weights=uvf_m.weights_array)
            indf = np.where(d > nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :, :] = True
        if nsig_t is not None:
            # Time watershed
            d = avg_f(uvf_m.metric_array, axis=(0, 1, 2, 4), weights=uvf_m.weights_array)
            indt = np.where(d > nsig_t)[0]
            uvf_f.flag_array[:, :, :, indt, :] = True
    elif uvf_m.type == 'waterfall':
        if nsig_f is not None:
            # Channel flag
            d = avg_f(uvf_m.metric_array, axis=(0, 2), weights=uvf_m.weights_array)
            indf = np.where(d > nsig_f)[0]
            uvf_f.flag_array[:, indf, :] = True
        if nsig_t is not None:
            # Time watershed
            d = avg_f(uvf_m.metric_array, axis=(1, 2), weights=uvf_m.weights_array)
            indt = np.where(d > nsig_t)[0]
            uvf_f.flag_array[indt, :, :] = True
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf_f


def flag_apply(uvf, uv, keep_existing=True, force_pol=False, history='',
               return_net_flags=False):
    '''Apply flags from UVFlag or list of UVFlag objects to UVData or UVCal.
    Args:
        uvf: UVFlag, path to UVFlag file, or list of these. Must be in 'flag' mode, and either
             match uv argument, or be a waterfall that can be made to match.
        uv:  UVData or UVCal object to apply flags to.
        keep_existing: If True (default), add flags to existing flags in uv.
                       If False, replace existing flags in uv.
        force_pol: If True, will use 1 pol to broadcast to any other pol.
                   Otherwise, will require polarizations match (default).
        history: history string to be added to uv.history
        return_net_flags: If True, return a UVFlag object with net flags applied.
                          If False (default) do not return net flags.
    Returns:
        net_flags: (if return_net_flags is set) returns UVFlag object with net flags.
    '''
    if issubclass(uv.__class__, UVData):
        expected_type = 'baseline'
    elif issubclass(uv.__class__, UVCal):
        expected_type = 'antenna'
    else:
        raise ValueError('Flags can only be applied to UVData or UVCal objects.')
    if not isinstance(uvf, (list, tuple, np.ndarray)):
        uvf = [uvf]
    net_flags = UVFlag(uv, mode='flag', copy_flags=keep_existing, history=history)
    for f in uvf:
        if isinstance(f, str):
            f = UVFlag(f)  # Read file
        elif not isinstance(f, UVFlag):
            raise ValueError('Input to apply_flag must be UVFlag or path to UVFlag file.')
        if f.mode != 'flag':
            raise ValueError('UVFlag objects must be in mode "flag" to apply to data.')
        if f.type == 'waterfall':
            if expected_type == 'baseline':
                f.to_baseline(uv, force_pol=force_pol)
            else:
                f.to_antenna(uv, force_pol=force_pol)
        # Use built-in or function
        net_flags |= f
    uv.flag_array += net_flags.flag_array
    uv.history += 'FLAGGING HISTORY: ' + history + ' END OF FLAGGING HISTORY.'

    if return_net_flags:
        return net_flags


#############################################################################
# Higher level functions that loop through data to calculate metrics
#############################################################################

def calculate_metric(uv, algorithm, gains=True, chisq=False, **kwargs):
    """
    Iterate over waterfalls in a UVData or UVCal object and generate a UVFlag object
    of mode 'metric'.

    Args:
        uv: UVData or UVCal object to calculate metrics on.
        algorithm: (str) metric algorithm name. Must be defined in algorithm_dict.
        gains: (bool) If True, and uv is UVCal, calculate metric based on gains.
               Supersedes chisq.
        chisq: (bool) If True, and gains==False, calculate metric based on chisq.
        **kwargs: Keyword arguments that are passed to algorithm.
    Returns:
        uvf: UVFlag object of mode 'metric' corresponding to the uv object.
    """
    if not issubclass(uv.__class__, (UVData, UVCal)):
        raise ValueError('uv must be a UVData or UVCal object.')
    try:
        alg_func = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    uvf = UVFlag(uv)
    if issubclass(uv.__class__, UVData):
        uvf.weights_array = uv.nsample_array * np.logical_not(uv.flag_array).astype(np.float)
    else:
        uvf.weights_array = np.logical_not(uv.flag_array).astype(np.float)
    if issubclass(uv.__class__, UVData):
        for key, d in uv.antpairpol_iter():
            ind1, ind2, pol = uv._key2inds(key)
            for ind, ipol in zip((ind1, ind2), pol):
                if len(ind) == 0:
                    continue
                uvf.metric_array[ind, 0, :, ipol] = alg_func(np.abs(d), **kwargs)
    elif issubclass(uv.__class__, UVCal):
        for ai in range(uv.Nants_data):
            for pi in range(uv.Njones):
                # Note transposes are due to freq, time dimensions rather than the
                # expected time, freq
                if gains:
                    d = np.abs(uv.gain_array[ai, 0, :, :, pi].T)
                elif chisq:
                    d = np.abs(uv.quality_array[ai, 0, :, :, pi].T)
                else:
                    raise ValueError('When calculating metric for UVCal object, '
                                     'gains or chisq must be set to True.')
                uvf.metric_array[ai, 0, :, :, pi] = alg_func(d, **kwargs).T
    return uvf


#############################################################################
# "Pipelines" -- these routines define the flagging strategy for some data
#############################################################################

def xrfi_h1c_pipe(uv, Kt=8, Kf=8, sig_init=6., sig_adj=2., px_threshold=0.2,
                  freq_threshold=0.5, time_threshold=0.05, return_summary=False,
                  gains=True, chisq=False):
    """xrfi excision pipeline we used for H1C. Uses detrending and watershed algorithms above.
    Args:
        uv: UVData or UVCal object to flag
        Kt (int): time size for detrending box. Default is 8.
        Kf (int): frequency size for detrending box. Default is 8.
        sig_init (float): initial sigma to flag.
        sig_adj (float): number of sigma to flag adjacent to flagged data (sig_init)
        px_threshold: Fraction of flags required to trigger a broadcast across baselines
                      for a given (time, frequency) pixel. Default is 0.2.
        freq_threshold: Fraction of channels required to trigger broadcast across
                        frequency (single time). Default is 0.5.
        time_threshold: Fraction of times required to trigger broadcast across
                        time (single frequency). Default is 0.05.
        return_summary: Return UVFlag object with fraction of baselines/antennas
                        that were flagged in initial flag/watershed (before broadcasting)
        gains (bool): If True (Default) and uv is UVCal, calculate flagging based on gains
        chisq (bool): If True and uv is UVCal, calculate flagging based on chisquared.
                      Note gains overrides chisq
    Returns:
        uvf_f: UVFlag object of initial flags (initial flag + watershed)
        uvf_wf: UVFlag object of waterfall type after thresholding in time/freq
        uvf_w (if return_summary): UVFlag object with fraction of flags in uvf_f
    """
    uvf = calculate_metric(uv, 'detrend_medfilt', Kt=Kt, Kf=Kf, gains=gains, chisq=chisq)
    uvf_f = flag(uvf, nsig_p=sig_init, nsig_f=None, nsig_t=None)
    uvf_f = watershed_flag(uvf, uvf_f, nsig_p=sig_adj, nsig_f=None, nsig_t=None)
    uvf_w = copy.deepcopy(uvf_f)
    uvf_w.to_waterfall()
    # I realize the naming convention has flipped, which results in nsig_f=time_threshold.
    # time_threshold is defined as fraction of time flagged to flag a given channel.
    # nsig_f is defined as significance required to flag a channel.
    uvf_wf = flag(uvf_w, nsig_p=px_threshold, nsig_f=time_threshold,
                  nsig_t=freq_threshold)

    if return_summary:
        return uvf_f, uvf_wf, uvf_w
    else:
        return uvf_f, uvf_wf

#############################################################################
# Wrappers -- Interact with input and output files
#############################################################################


def xrfi_h1c_run(indata, history, infile_format='miriad', extension='.flags.h5',
                 summary=False, summary_ext='.flag_summary.h5', xrfi_path='',
                 model_file=None, model_file_format='uvfits',
                 calfits_file=None, kt_size=8, kf_size=8, sig_init=6.0, sig_adj=2.0,
                 px_threshold=0.2, freq_threshold=0.5, time_threshold=0.05,
                 ex_ants='', metrics_json='', filename=None):
    """
    Run RFI-flagging algorithm from H1C on a single data file, and optionally calibration files,
    and store results in npz files.

    Args:
        indata -- Either UVData object or data file to run RFI flagging on.
        history -- history string to include in files
        infile_format -- File format for input files. Default is miriad.
        extension -- Extension to be appended to input file name. Default is ".flags.h5"
        summary -- Run summary of RFI flags and store in h5 file. Default is False.
        summary_ext -- Extension for summary file. Default is ".flag_summary.h5"
        xrfi_path -- Path to save flag files to. Default is same directory as input file.
        model_file -- Model visibility file to flag on.
        model_file_format -- File format for input model file. Default is uvfits.
        calfits_file -- Calfits file to use to flag on gains and/or chisquared values.
        kt_size -- Size of kernel in time dimension for detrend in xrfi algorithm. Default is 8.
        kf_size -- Size of kernel in frequency dimension for detrend in xrfi. Default is 8.
        sig_init -- Starting number of sigmas to flag on. Default is 6.
        sig_adj -- Number of sigmas to flag on for data adjacent to a flag. Default is 2.
        px_threshold -- Fraction of flags required to trigger a broadcast across baselines
                        for a given (time, frequency) pixel. Default is 0.2.
        freq_threshold -- Fraction of channels required to trigger broadcast across
                          frequency (single time). Default is 0.5.
        time_threshold -- Fraction of times required to trigger broadcast across
                          time (single frequency). Default is 0.05.
        ex_ants -- Comma-separated list of antennas to exclude. Flags of visibilities
                   formed with these antennas will be set to True.
        metrics_json -- Metrics file that contains a list of excluded antennas. Flags of
                        visibilities formed with these antennas will be set to True.
        filename -- File for which to flag RFI (only one file allowed).
    Return:
       None

    This function will take in a UVData object or  data file and optionally a cal file and
    model visibility file, and run an RFI-flagging algorithm to identify contaminated
    observations. Each set of flagging will be stored, as well as compressed versions.
    """
    if indata is None:
        if (model_file is None) and (calfits_file is None):
            raise AssertionError('Must provide at least one of: indata, '
                                 'model_file, or calfits_file.')
        warnings.warn('indata is None, not flagging on any data visibilities.')
    elif issubclass(indata.__class__, UVData):
        uvd = indata
        if filename is None:
            raise AssertionError('Please provide a filename to go with UVData object. '
                                 'The filename is used in conjunction with "extension" '
                                 'to determine the output filename.')
        else:
            if not isinstance(filename, str):
                raise ValueError('filename must be string path to file.')
    else:
        filename = indata
        uvd = UVData()
        if infile_format == 'miriad':
            uvd.read_miriad(filename)
        elif infile_format == 'uvfits':
            uvd.read_uvfits(filename)
        else:
            raise ValueError('Unrecognized input file format ' + str(infile_format))

    # Compute list of excluded antennas
    if ex_ants != '' or metrics_json != '':
        # import function from hera_cal
        from hera_cal.omni import process_ex_ants
        xants = process_ex_ants(ex_ants, metrics_json)

        # Flag the visibilities corresponding to the specified antennas
        flag_xants(uvd, xants)

    # append to history
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str

    if xrfi_path != '':
        # If explicitly given output path, use it. Otherwise use path from data.
        dirname = xrfi_path

    # Flag on data
    if indata is not None:
        uvf_f, uvf_wf, uvf_w = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                             sig_adj=sig_adj, px_threshold=px_threshold,
                                             freq_threshold=freq_threshold, time_threshold=time_threshold,
                                             return_summary=True)
        if xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(filename))
        basename = os.path.basename(filename)
        # Save watersheded flags
        outfile = ''.join([basename, extension])
        outpath = os.path.join(dirname, outfile)
        uvf_f.history += history
        uvf_f.write(outpath, clobber=True)
        # Save thresholded waterfall
        outfile = ''.join([basename, '.waterfall', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        if summary:
            sum_file = ''.join([basename, summary_ext])
            sum_path = os.path.join(dirname, sum_file)
            uvf_w.history += history
            uvf_w.write(sum_path, clobber=True)

    # Flag on model visibilities
    if model_file is not None:
        uvm = UVData()
        if model_file_format == 'miriad':
            uvm.read_miriad(model_file)
        elif model_file_format == 'uvfits':
            uvm.read_uvfits(model_file)
        else:
            raise ValueError('Unrecognized input file format ' + str(model_file_format))
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvm.time_array),
                                atol=1e-5, rtol=0)
                    and np.allclose(uvd.freq_array, uvm.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of model vis file must match'
                                 'the data file.')
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvm, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold)
        if xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(model_file))
        # Only save thresholded waterfall
        outfile = ''.join([os.path.basename(model_file), extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)

    # Flag on gain solutions and chisquared values
    if calfits_file is not None:
        uvc = UVCal()
        uvc.read_calfits(calfits_file)
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvc.time_array),
                                atol=1e-5, rtol=0)
                    and np.allclose(uvd.freq_array, uvc.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of calfits file must match'
                                 'the data file.')
        # By default, runs on gains
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold)
        if xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(calfits_file))
        outfile = ''.join([os.path.basename(calfits_file), '.g', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        # repeat for chisquared
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold,
                                      gains=False, chisq=True)
        outfile = ''.join([os.path.basename(calfits_file), '.x', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath)

    return


def xrfi_h1c_apply(filename, history, infile_format='miriad', xrfi_path='',
                   outfile_format='miriad', extension='R', overwrite=False,
                   flag_file=None, waterfalls=None, output_uvflag=True,
                   output_uvflag_ext='.flags.h5'):
    """
    Apply flags in the fashion of H1C.
    Read in a flag array and optionally several waterfall flags, and insert into
    a data file.

    Args:
        filename -- Data file in which update flag array.
        history -- history string to include in files
        infile_format -- File format for input files. Default is miriad.
        xrfi_path -- Path to save output to. Default is same directory as input file.
        outfile_format -- File format for output files. Default is miriad.
        extension -- Extension to be appended to input file name. Default is "R".
        overwrite -- Option to overwrite output file if it already exists.
        flag_file -- npz file containing full flag array to insert into data file.
        waterfalls -- list or comma separated list of npz files containing waterfalls of flags
                      to broadcast to full flag array and union with flag array in flag_file.
        output_uvflag -- Whether to save uvflag with the final flag array.
                      The flag array will be identical to what is stored in the data.
        output_uvflag_ext -- Extension to be appended to input file name. Default is ".flags.h5".
    Return:
        None
    """
    # make sure we were given files to process
    if len(filename) == 0:
        raise AssertionError('Please provide a visibility file')
    if isinstance(filename, (list, np.ndarray, tuple)) and len(filename) > 1:
        raise AssertionError('xrfi_apply currently only takes a single data file.')
    if isinstance(filename, (list, np.ndarray, tuple)):
        filename = filename[0]
    uvd = UVData()
    if infile_format == 'miriad':
        uvd.read_miriad(filename)
    elif infile_format == 'uvfits':
        uvd.read_uvfits(filename)
    elif infile_format == 'fhd':
        uvd.read_fhd(filename)
    else:
        raise ValueError('Unrecognized input file format ' + str(infile_format))

    full_list = []
    # Read in flag file
    if flag_file is not None:
        full_list += [flag_file]

    # Read in waterfalls
    if waterfalls is not None:
        if not isinstance(waterfalls, list):
            # Assume comma separated list
            waterfalls = waterfalls.split(',')
        full_list += waterfalls

    uvf = flag_apply(full_list, uvd, force_pol=True, return_net_flags=True)

    # save output when we're done
    if xrfi_path == '':
        # default to the same directory
        abspath = os.path.abspath(filename)
        dirname = os.path.dirname(abspath)
    else:
        dirname = xrfi_path
    basename = os.path.basename(filename)
    outfile = ''.join([basename, extension])
    outpath = os.path.join(dirname, outfile)
    if outfile_format == 'miriad':
        uvd.write_miriad(outpath, clobber=overwrite)
    elif outfile_format == 'uvfits':
        if os.path.exists(outpath) and not overwrite:
            raise ValueError('File exists: skipping')
        uvd.write_uvfits(outpath, force_phase=True, spoof_nonessential=True)
    else:
        raise ValueError('Unrecognized output file format ' + str(outfile_format))
    if output_uvflag:
        # Save uvflag with the final flag array and relevant metadata
        outpath = outpath + output_uvflag_ext
        uvf.write(outpath)
