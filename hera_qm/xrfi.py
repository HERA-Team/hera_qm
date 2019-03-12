# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from .uvflag import UVFlag
from . import utils as qm_utils
from .version import hera_qm_version_str
from .metrics_io import process_ex_ants
import warnings
import copy
import collections
from six.moves import range


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


def resolve_xrfi_path(xrfi_path, fname):
    """ Determine xrfi_path based on given directory or default to dirname of given file.
    Args:
        xrfi_path (str): Directory to write xrfi outputs.
        fname (str): Filename to determine backup directory if xrfi_path == ''
    Returns:
        dirname (str): If xrfi_path != '', returns xrfi_path. Otherwise returns
            directory of file.
    """
    if (xrfi_path != '') and (os.path.exists(xrfi_path)):
        dirname = xrfi_path
    else:
        dirname = os.path.dirname(os.path.abspath(fname))
    return dirname


def robust_divide(a, b):
    '''Prevent division by zero by setting values to infinity when the denominator
    is small for the given data type.
    Args:
        a (array): Numerator
        b (array): Denominator
    Returns:
        f (array): Division a / b. Elements where b is small (or zero) are set to infinity.
    '''
    thresh = np.finfo(b.dtype).eps
    f = np.true_divide(a, b, where=(np.abs(b) > thresh))
    f = np.where(np.abs(b) > thresh, f, np.inf)
    return f


#############################################################################
# Functions for preprocessing data prior to RFI flagging
#############################################################################

def medmin(d, flags=None):
    '''Calculate the median minus minimum statistic of array.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        flags (array, optional): 2D flag array to be interpretted as mask for d.
            NOT USED in this function.
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


def medminfilt(d, flags=None, Kt=8, Kf=8):
    '''Filter an array on scales of Kt,Kf indexes with medmin.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        flags (array, optional): 2D flag array to be interpretted as mask for d.
            NOT USED in this function.
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
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            i0, j0 = max(0, i - Kt), max(0, j - Kf)
            i1, j1 = min(d.shape[0], i + Kt), min(d.shape[1], j + Kf)
            d_sm[i, j] = medmin(d[i0:i1, j0:j1])
    return d_sm


def detrend_deriv(d, flags=None, dt=True, df=True):
    ''' Detrend array by taking the derivative in either time, frequency
        or both. When taking the derivative of both, the derivative in
        frequency is performed first, then in time.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        flags (array, optional): 2D flag array to be interpretted as mask for d.
            NOT USED in this function.
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
    f = robust_divide(d_dtdf, sig)
    return f


def detrend_medminfilt(d, flags=None, Kt=8, Kf=8):
    """Detrend array using medminfilt statistic. See medminfilt.
    Args:
        d (array): 2D data array of the shape (time, frequency) to detrend
        flags (array, optional): 2D flag array to be interpretted as mask for d.
            NOT USED in this function.
        Kt (int): size in time to apply medminfilter over
        Kf (int): size in frequency to apply medminfilter over
    Returns:
        float array: float array of outlier significance metric
    """
    if d.ndim != 2:
        raise ValueError('Input to detrend_medminfilt must be 2D array.')
    d_sm = medminfilt(np.abs(d), Kt=2 * Kt + 1, Kf=2 * Kf + 1)
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts minmed on same scale as average
    sig = np.sqrt(medminfilt(d_sq, Kt=2 * Kt + 1, Kf=2 * Kf + 1)) * (np.sqrt(Kt**2 + Kf**2) / .64)
    # don't divide by zero, instead turn those entries into +inf
    f = robust_divide(d_rs, sig)
    return f


def detrend_medfilt(d, flags=None, Kt=8, Kf=8):
    """Detrend array using a median filter.
    Args:
        d (array): 2D data array to detrend.
        flags (array, optional): 2D flag array to be interpretted as mask for d.
            NOT USED in this function.
        Kt (int, optional): box size in time (first) dimension to apply medfilt
            over. Default is 8 pixels.
        Kf (int, optional): box size in frequency (second) dimension to apply medfilt
            over. Default is 8 pixels.
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
    f = robust_divide(d_rs, sig)
    return f[Kt:-Kt, Kf:-Kf]


def detrend_meanfilt(d, flags=None, Kt=8, Kf=8):
    """Detrend array using a mean filter.
    Args:
        d (array): 2D data array to detrend.
        flags (array, optional): 2D flag array to be interpretted as mask for d.
        Kt (int, optional): box size in time (first) dimension to apply medfilt
            over. Default is 8 pixels.
        Kf (int, optional): box size in frequency (second) dimension to apply medfilt
            over. Default is 8 pixels.
    Returns:
        f: array of outlier significance metric. Same type and size as d.
    """
    # Delay import so astropy is not required for any use of hera_qm
    from astropy.convolution import convolve

    if d.ndim != 2:
        raise ValueError('Input to detrend_meanfilt must be 2D array.')
    if Kt > d.shape[0]:
        warnings.warn("Kt value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kt, d.shape[0]))
        Kt = d.shape[0]
    if Kf > d.shape[1]:
        warnings.warn("Kf value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(Kf, d.shape[1]))
        Kf = d.shape[1]
    kernel = np.ones((2 * Kt + 1, 2 * Kf + 1))
    d = np.concatenate([d[Kt - 1::-1], d, d[:-Kt - 1:-1]], axis=0)
    d = np.concatenate([d[:, Kf - 1::-1], d, d[:, :-Kf - 1:-1]], axis=1)
    if flags is not None:
        flags = np.concatenate([flags[Kt - 1::-1], flags, flags[:-Kt - 1:-1]], axis=0)
        flags = np.concatenate([flags[:, Kf - 1::-1], flags, flags[:, :-Kf - 1:-1]], axis=1)
    d_sm = convolve(d, kernel, mask=flags, boundary='extend')
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average
    sig = np.sqrt(convolve(d_sq, kernel, mask=flags))
    # don't divide by zero, instead turn those entries into +inf
    f = robust_divide(d_rs, sig)
    return f[Kt:-Kt, Kf:-Kf]


# Update algorithm_dict whenever new metric algorithm is created.
algorithm_dict = {'medmin': medmin, 'medminfilt': medminfilt, 'detrend_deriv': detrend_deriv,
                  'detrend_medminfilt': detrend_medminfilt, 'detrend_medfilt': detrend_medfilt,
                  'detrend_meanfilt': detrend_meanfilt}

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
            d = qm_utils.collapse(marr, avg_method, axis=(0, 1, 3), weights=warr)
            f = np.all(farr, axis=(0, 1, 3))
            farr[:, :, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, 1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            ts = np.unique(uvf.time_array)
            d = np.zeros(ts.size)
            f = np.zeros(ts.size, dtype=np.bool)
            for i, t in enumerate(ts):
                d[i] = qm_utils.collapse(marr[uvf.time_array == t, 0, :, :], avg_method,
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
            d = qm_utils.collapse(marr, avg_method, axis=(0, 1, 3, 4), weights=warr)
            f = np.all(farr, axis=(0, 1, 3, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, 1, -1, 1, 1)
        if nsig_t is not None:
            # Time watershed
            d = qm_utils.collapse(marr, avg_method, axis=(0, 1, 2, 4), weights=warr)
            f = np.all(farr, axis=(0, 1, 2, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(d, f, nsig_t).reshape(1, 1, 1, -1, 1)
    elif uvf_m.type == 'waterfall':
        # Pixel watershed
        for pi in range(uvf.polarization_array.size):
            farr[:, :, pi] += _ws_flag_waterfall(marr[:, :, pi], farr[:, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            d = qm_utils.collapse(marr, avg_method, axis=(0, 2), weights=warr)
            f = np.all(farr, axis=(0, 2))
            farr[:, :, :] += _ws_flag_waterfall(d, f, nsig_f).reshape(1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            d = qm_utils.collapse(marr, avg_method, axis=(1, 2), weights=warr)
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
        uvf_f: UVFlag object in 'flag' mode with flags determined from uvf_m.
    '''
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')

    # initialize
    uvf_f = copy.deepcopy(uvf_m)
    uvf_f.to_flag()

    # Pixel flagging
    if nsig_p is not None:
        uvf_f.flag_array[uvf_m.metric_array >= nsig_p] = True

    if uvf_m.type == 'baseline':
        if nsig_f is not None:
            # Channel flagging
            d = qm_utils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3),
                                  weights=uvf_m.weights_array)
            indf = np.where(d >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :] = True
        if nsig_t is not None:
            # Time flagging
            ts = np.unique(uvf_m.time_array)
            d = np.zeros(ts.size)
            for i, t in enumerate(ts):
                d[i] = qm_utils.collapse(uvf_m.metric_array[uvf_m.time_array == t, 0, :, :],
                                         avg_method,
                                         weights=uvf_m.weights_array[uvf_m.time_array == t, 0, :, :])
            indf = np.where(d >= nsig_t)[0]
            for t in ts[indf]:
                uvf_f.flag_array[uvf_f.time_array == t, :, :, :] = True
    elif uvf_m.type == 'antenna':
        if nsig_f is not None:
            # Channel flag
            d = qm_utils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3, 4),
                                  weights=uvf_m.weights_array)
            indf = np.where(d >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :, :] = True
        if nsig_t is not None:
            # Time watershed
            d = qm_utils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 2, 4),
                                  weights=uvf_m.weights_array)
            indt = np.where(d >= nsig_t)[0]
            uvf_f.flag_array[:, :, :, indt, :] = True
    elif uvf_m.type == 'waterfall':
        if nsig_f is not None:
            # Channel flag
            d = qm_utils.collapse(uvf_m.metric_array, avg_method, axis=(0, 2),
                                  weights=uvf_m.weights_array)
            indf = np.where(d >= nsig_f)[0]
            uvf_f.flag_array[:, indf, :] = True
        if nsig_t is not None:
            # Time watershed
            d = qm_utils.collapse(uvf_m.metric_array, avg_method, axis=(1, 2),
                                  weights=uvf_m.weights_array)
            indt = np.where(d >= nsig_t)[0]
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
            f = f.copy()  # don't change the input object
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

def calculate_metric(uv, algorithm, cal_mode='gain', **kwargs):
    """
    Iterate over waterfalls in a UVData or UVCal object and generate a UVFlag object
    of mode 'metric'.

    Args:
        uv: UVData or UVCal object to calculate metrics on.
        algorithm: (str) metric algorithm name. Must be defined in algorithm_dict.
        cal_mode: (str) Mode to calculate metric if uv is UVCal. Options are
                  'gain', 'chisq', and 'tot_chisq' to use the gain_array,
                  'quality_array', and 'total_quality_array', respectively.
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
                flags = uv.flag_array[ind, 0, :, ipol]
                uvf.metric_array[ind, 0, :, ipol] = alg_func(np.abs(d), flags=flags, **kwargs)
    elif issubclass(uv.__class__, UVCal):
        if cal_mode == 'tot_chisq':
            uvf.to_waterfall()
            for pi in range(uv.Njones):
                d = np.abs(uv.total_quality_array[0, :, :, pi].T)
                flags = np.all(uv.flag_array[:, 0, :, :, pi], axis=0).T
                uvf.metric_array[:, :, pi] = alg_func(d, flags=flags, **kwargs)
        else:
            for ai in range(uv.Nants_data):
                for pi in range(uv.Njones):
                    # Note transposes are due to freq, time dimensions rather than the
                    # expected time, freq
                    flags = uv.flag_array[ai, 0, :, :, pi].T
                    if cal_mode == 'gain':
                        d = np.abs(uv.gain_array[ai, 0, :, :, pi].T)
                    elif cal_mode == 'chisq':
                        d = np.abs(uv.quality_array[ai, 0, :, :, pi].T)
                    else:
                        raise ValueError('When calculating metric for UVCal object, '
                                         'cal_mode must be "gain", "chisq", or "tot_chisq".')
                    uvf.metric_array[ai, 0, :, :, pi] = alg_func(d, flags=flags, **kwargs).T
    return uvf


#############################################################################
# "Pipelines" -- these routines define the flagging strategy for some data
#   Note: "current" pipes should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################

def xrfi_h1c_pipe(uv, Kt=8, Kf=8, sig_init=6., sig_adj=2., px_threshold=0.2,
                  freq_threshold=0.5, time_threshold=0.05, return_summary=False,
                  cal_mode='gain'):
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
        cal_mode: (str) Mode to calculate metric if uv is UVCal. Options are
                  'gain', 'chisq', and 'tot_chisq' to use the gain_array,
                  'quality_array', and 'total_quality_array', respectively.
    Returns:
        uvf_f: UVFlag object of initial flags (initial flag + watershed)
        uvf_wf: UVFlag object of waterfall type after thresholding in time/freq
        uvf_w (if return_summary): UVFlag object with fraction of flags in uvf_f
    """
    uvf = calculate_metric(uv, 'detrend_medfilt', Kt=Kt, Kf=Kf, cal_mode=cal_mode)
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


def xrfi_pipe(uv, alg='detrend_medfilt', Kt=8, Kf=8, xants=[], cal_mode='gain',
              sig_init=6.0, sig_adj=2.0):
    """xrfi excision pipeline used for H1C IDR2.2. Uses detrending and watershed algorithms above.
    Args:
        uv (UVData or UVCal): Object to calculate metric.
        alg (str, optional): Algorithm for calculating metric. Default is
            'detrend_medfilt'.
        Kt (int, optional): Size of kernel in time dimension for detrend in
            xrfi algorithm. Default is 8.
        Kf (int, optional): Size of kernel in frequency dimension for detrend
            in xrfi algorithm. Default is 8.
        xants (list): List of antennas to flag. Default is empty list.
        cal_mode: (str) Mode to calculate metric if uv is UVCal. Options are
            'gain', 'chisq', and 'tot_chisq' to use the gain_array,
            'quality_array', and 'total_quality_array', respectively.
        sig_init (float, optional): Starting number of sigmas to flag on. Default is 6.
        sig_adj (float, optional): Number of sigmas to flag on for data adjacent
            to a flag. Default is 2.0.
    Returns:
        uvf_m (UVFlag): UVFlag object with metric after collapsing to waterfall and to
            single pol. Weights array is set to ones.
        uvf_fws (UVFlag): UVFlag object with flags after watershed.
    """
    flag_xants(uv, xants)
    uvf_m = calculate_metric(uv, alg, Kt=Kt, Kf=Kf, cal_mode=cal_mode)
    uvf_m.to_waterfall(keep_pol=False)
    uvf_m.weights_array = uvf_m.weights_array.astype(np.bool).astype(np.float)
    alg_func = algorithm_dict[alg]
    uvf_m.metric_array[:, :, 0] = alg_func(uvf_m.metric_array[:, :, 0],
                                           flags=~uvf_m.weights_array[:, :, 0].astype(np.bool),
                                           Kt=Kt, Kf=Kf)
    uvf_f = flag(uvf_m, nsig_p=sig_init)
    uvf_fws = watershed_flag(uvf_m, uvf_f, nsig_p=sig_adj, inplace=False)
    return uvf_m, uvf_fws

#############################################################################
# Wrappers -- Interact with input and output files
#   Note: "current" wrappers should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################


def xrfi_run(ocalfits_file, acalfits_file, model_file, data_file, history,
             init_metrics_ext='init_xrfi_metrics.h5', init_flags_ext='init_flags.h5',
             final_metrics_ext='final_xrfi_metrics.h5', final_flags_ext='final_flags.h5',
             xrfi_path='', kt_size=8, kf_size=8, sig_init=5.0, sig_adj=2.0,
             freq_threshold=0.35, time_threshold=0.5, ex_ants=None, metrics_file=None,
             cal_ext='flagged_abs'):
    """xrfi excision pipeline used for H1C IDR2.2. Uses detrending and watershed algorithms above.
    Args:
        ocalfits_file (str): Omnical calfits file to use to flag on gains and
            chisquared values.
        acalfits_file (str): Abscal calfits file to use to flag on gains and
            chisquared values.
        model_file (str): Model visibility file to flag on.
        data_file (str): Raw visibility data file to flag.
        history (str): History string to include in files
        init_metrics_ext (str, optional): Extension to be appended to input file name
            for initial metric object. Default is "init_xrfi_metrics.h5".
        init_flags_ext (str, optional): Extension to be appended to input file name
            for initial flag object. Default is "init_flags.h5".
        final_metrics_ext (str, optional): Extension to be appended to input file name
            for final metric object. Default is "final_xrfi_metrics.h5".
        final_flags_ext (str, optional): Extension to be appended to input file name
            for final flag object. Default is "final_flags.h5".
        xrfi_path (str, optional): Path to save xrfi files to. Default is same
            directory as data_file.
        kt_size (int, optional): Size of kernel in time dimension for detrend in
            xrfi algorithm. Default is 8.
        kf_size (int, optional): Size of kernel in frequency dimension for detrend
            in xrfi algorithm. Default is 8.
        sig_init (float, optional): Starting number of sigmas to flag on. Default is 5.
        sig_adj (float, optional): Number of sigmas to flag on for data adjacent
            to a flag. Default is 2.0.
        freq_threshold (float, optional): Fraction of times required to trigger
            broadcast across times (single freq). Default is 0.35.
        time_threshold (float, optional): Fraction of channels required to trigger
            broadcast across frequency (single time). Default is 0.5.
        ex_ants (str, optional): Comma-separated list of antennas to exclude.
            Flags of visibilities formed with these antennas will be set to True.
        metrics_file (str, optional): Metrics file that contains a list of excluded
            antennas. Flags of visibilities formed with these antennas will be set to True.
        cal_ext (str, optional): Extension to replace penultimate extension in
            calfits file for output calibration including flags. Defaults is "flagged_abs".
            For example, an input_cal of "foo.goo.calfits" would result in
            "foo.flagged_abs.calfits".
    Returns:
        None
    """
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str
    dirname = resolve_xrfi_path(xrfi_path, data_file)
    xants = process_ex_ants(ex_ants=ex_ants, metrics_file=metrics_file)

    # Initial run on cal data products
    alg = 'detrend_medfilt'
    # Calculate metric on abscal data
    uvc_a = UVCal()
    uvc_a.read_calfits(acalfits_file)
    uvf_apriori = UVFlag(uvc_a, mode='flag', copy_flags=True)
    uvf_ag, uvf_agf = xrfi_pipe(uvc_a, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ax, uvf_axf = xrfi_pipe(uvc_a, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on omnical data
    uvc_o = UVCal()
    uvc_o.read_calfits(ocalfits_file)
    flag_apply(uvf_apriori, uvc_o, keep_existing=True)
    uvf_og, uvf_ogf = xrfi_pipe(uvc_o, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ox, uvf_oxf = xrfi_pipe(uvc_o, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on model vis
    uv_v = UVData()
    uv_v.read(model_file)
    uvf_v, uvf_vf = xrfi_pipe(uv_v, alg=alg, xants=[], Kt=kt_size, Kf=kf_size,
                              sig_init=sig_init, sig_adj=sig_adj)

    # Combine the metrics together
    uvf_metrics = uvf_v.combine_metrics([uvf_og, uvf_ox, uvf_ag, uvf_ax],
                                        method='quadmean', inplace=False)
    alg_func = algorithm_dict[alg]
    uvf_metrics.metric_array[:, :, 0] = alg_func(uvf_metrics.metric_array[:, :, 0],
                                                 flags=~uvf_metrics.weights_array[:, :, 0].astype(np.bool),
                                                 Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f = flag(uvf_metrics, nsig_p=sig_init)
    uvf_fws = watershed_flag(uvf_metrics, uvf_f, nsig_p=sig_adj, inplace=False)
    # OR everything together for initial flags
    uvf_apriori.to_waterfall(method='and', keep_pol=False)
    uvf_init = uvf_fws | uvf_ogf | uvf_oxf | uvf_agf | uvf_axf | uvf_vf | uvf_apriori

    # Write out initial (combined) metrics and flags
    basename = qm_utils.strip_extension(os.path.basename(data_file))
    outfile = '.'.join([basename, init_metrics_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_metrics.write(outpath, clobber=True)
    outfile = '.'.join([basename, init_flags_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_init.write(outpath, clobber=True)

    # Second round -- use init flags to mask and recalculate everything
    # Read in data file
    uv_d = UVData()
    uv_d.read(data_file)
    for uv in [uvc_o, uvc_a, uv_v, uv_d]:
        flag_apply(uvf_init, uv, keep_existing=True, force_pol=True)

    # Do next round of metrics
    alg = 'detrend_meanfilt'  # Change to meanfilt because it can mask flagged pixels
    # Calculate metric on abscal data
    uvf_ag2, uvf_agf2 = xrfi_pipe(uvc_a, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ax2, uvf_axf2 = xrfi_pipe(uvc_a, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on omnical data
    uvf_og2, uvf_ogf2 = xrfi_pipe(uvc_o, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj)
    uvf_ox2, uvf_oxf2 = xrfi_pipe(uvc_o, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on model vis
    uvf_v2, uvf_vf2 = xrfi_pipe(uv_v, alg=alg, xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj)

    # Calculate metric on data file
    uvf_d2, uvf_df2 = xrfi_pipe(uv_d, alg=alg, xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj)

    # Combine the metrics together
    uvf_metrics2 = uvf_d2.combine_metrics([uvf_og2, uvf_ox2, uvf_ag2, uvf_ax2, uvf_v2, uvf_d2],
                                          method='quadmean', inplace=False)
    alg_func = algorithm_dict[alg]
    uvf_metrics2.metric_array[:, :, 0] = alg_func(uvf_metrics2.metric_array[:, :, 0],
                                                  flags=uvf_init.flag_array[:, :, 0],
                                                  Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f2 = flag(uvf_metrics2, nsig_p=sig_init)
    uvf_fws2 = watershed_flag(uvf_metrics2, uvf_f2, nsig_p=sig_adj, inplace=False)
    uvf_combined2 = (uvf_fws2 | uvf_ogf2 | uvf_oxf2 | uvf_agf2 | uvf_axf2
                     | uvf_vf2 | uvf_df2 | uvf_init)

    # Threshold
    uvf_temp = uvf_combined2.copy()
    uvf_temp.to_metric(convert_wgts=True)
    uvf_final = flag(uvf_temp, nsig_p=1.0, nsig_f=freq_threshold, nsig_t=time_threshold)

    # Write out final metrics and flags
    outfile = '.'.join([basename, final_metrics_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_metrics2.write(outpath, clobber=True)
    outfile = '.'.join([basename, final_flags_ext])
    outpath = os.path.join(dirname, outfile)
    uvf_final.write(outpath, clobber=True)

    # Save calfits with new flags
    flag_apply(uvf_final, uvc_a, force_pol=True, history=history)
    basename = qm_utils.strip_extension(os.path.basename(acalfits_file))
    basename = qm_utils.strip_extension(basename)  # Also get rid of .abs
    outfile = '.'.join([basename, cal_ext, 'calfits'])
    outpath = os.path.join(dirname, outfile)
    uvc_a.write_calfits(outpath, clobber=True)


def xrfi_h1c_run(indata, history, infile_format='miriad', extension='flags.h5',
                 summary=False, summary_ext='flag_summary.h5', xrfi_path='',
                 model_file=None, model_file_format='uvfits',
                 calfits_file=None, kt_size=8, kf_size=8, sig_init=6.0, sig_adj=2.0,
                 px_threshold=0.2, freq_threshold=0.5, time_threshold=0.05,
                 ex_ants=None, metrics_file=None, filename=None):
    """
    Run RFI-flagging algorithm from H1C on a single data file, and optionally calibration files,
    and store results in npz files.

    Args:
        indata -- Either UVData object or data file to run RFI flagging on.
        history -- history string to include in files
        infile_format -- File format for input files. Not currently used while
                         we use pyuvdata's generic read function, But will
                         be implemented for partial io.
        extension -- Extension to be appended to input file name. Default is "flags.h5"
        summary -- Run summary of RFI flags and store in h5 file. Default is False.
        summary_ext -- Extension for summary file. Default is "flag_summary.h5"
        xrfi_path -- Path to save flag files to. Default is same directory as input file.
        model_file -- Model visibility file to flag on.
        model_file_format -- File format for input model file. Not currently used while
                         we use pyuvdata's generic read function, But will
                         be implemented for partial io.
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
        metrics_file -- Metrics file that contains a list of excluded antennas. Flags of
                        visibilities formed with these antennas will be set to True.
        filename -- File for which to flag RFI (only one file allowed).
    Return:
       None

    This function will take in a UVData object or data file and optionally a cal file and
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
        if filename is None or filename == '':
            filename = indata
        elif not isinstance(filename, str):
            raise ValueError('filename must be string path to file.')
        uvd = UVData()
        uvd.read(filename)

    # append to history
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str

    # Flag on data
    if indata is not None:
        # Flag visibilities corresponding to specified antennas
        xants = process_ex_ants(ex_ants=ex_ants, metrics_file=metrics_file)
        flag_xants(uvd, xants)
        uvf_f, uvf_wf, uvf_w = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                             sig_adj=sig_adj, px_threshold=px_threshold,
                                             freq_threshold=freq_threshold, time_threshold=time_threshold,
                                             return_summary=True)
        dirname = resolve_xrfi_path(xrfi_path, filename)
        basename = qm_utils.strip_extension(os.path.basename(filename))
        # Save watersheded flags
        outfile = '.'.join([basename, extension])
        outpath = os.path.join(dirname, outfile)
        uvf_f.history += history
        uvf_f.write(outpath, clobber=True)
        # Save thresholded waterfall
        outfile = '.'.join([basename, 'waterfall', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        if summary:
            sum_file = '.'.join([basename, summary_ext])
            sum_path = os.path.join(dirname, sum_file)
            uvf_w.history += history
            uvf_w.write(sum_path, clobber=True)

    # Flag on model visibilities
    if model_file is not None:
        uvm = UVData()
        uvm.read(model_file)
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvm.time_array),
                                atol=1e-5, rtol=0)
                    and np.allclose(uvd.freq_array, uvm.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of model vis file must match'
                                 'the data file.')
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvm, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold)
        dirname = resolve_xrfi_path(xrfi_path, model_file)
        # Only save thresholded waterfall
        basename = qm_utils.strip_extension(os.path.basename(model_file))
        outfile = '.'.join([basename, extension])
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
        dirname = resolve_xrfi_path(xrfi_path, calfits_file)
        basename = qm_utils.strip_extension(os.path.basename(calfits_file))
        outfile = '.'.join([basename, 'g', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath, clobber=True)
        # repeat for chisquared
        uvf_f, uvf_wf = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                      sig_adj=sig_adj, px_threshold=px_threshold,
                                      freq_threshold=freq_threshold, time_threshold=time_threshold,
                                      cal_mode='chisq')
        outfile = '.'.join([basename, 'x', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath)

    return


def xrfi_h1c_apply(filename, history, infile_format='miriad', xrfi_path='',
                   outfile_format='miriad', extension='R', overwrite=False,
                   flag_file=None, waterfalls=None, output_uvflag=True,
                   output_uvflag_ext='flags.h5'):
    """
    Apply flags in the fashion of H1C.
    Read in a flag array and optionally several waterfall flags, and insert into
    a data file.

    Args:
        filename -- Data file in which update flag array.
        history -- history string to include in files
        infile_format -- File format for input files. Not currently used while
                         we use pyuvdata's generic read function, But will
                         be implemented for partial io.
        xrfi_path -- Path to save output to. Default is same directory as input file.
        outfile_format -- File format for output files. Default is miriad.
        extension -- Extension to be appended to input file name. Default is "R".
        overwrite -- Option to overwrite output file if it already exists.
        flag_file -- npz file containing full flag array to insert into data file.
        waterfalls -- list or comma separated list of npz files containing waterfalls of flags
                      to broadcast to full flag array and union with flag array in flag_file.
        output_uvflag -- Whether to save uvflag with the final flag array.
                      The flag array will be identical to what is stored in the data.
        output_uvflag_ext -- Extension to be appended to input file name. Default is "flags.h5".
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
    uvd.read(filename)

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
    dirname = resolve_xrfi_path(xrfi_path, filename)
    basename = qm_utils.strip_extension(os.path.basename(filename))
    outfile = '.'.join([basename, extension])
    outpath = os.path.join(dirname, outfile)
    extension_dict = {'miriad': '.uv', 'uvfits': '.uvfits', 'uvh5': '.uvh5'}
    try:
        outpath += extension_dict[outfile_format]
    except KeyError:
        raise ValueError('Unrecognized output file format ' + str(outfile_format))
    if outfile_format == 'miriad':
        uvd.write_miriad(outpath, clobber=overwrite)
    elif outfile_format == 'uvfits':
        if os.path.exists(outpath) and not overwrite:
            raise ValueError('File ' + outpath + ' exists: skipping')
        uvd.write_uvfits(outpath, force_phase=True, spoof_nonessential=True)
    elif outfile_format == 'uvh5':
        uvd.write_uvh5(outpath, clobber=overwrite)
    if output_uvflag:
        # Save uvflag with the final flag array and relevant metadata
        outfile = '.'.join([basename, extension, output_uvflag_ext])
        outpath = os.path.join(dirname, outfile)
        uvf.write(outpath)
