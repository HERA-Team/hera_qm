# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for performing RFI identification and excision."""

import numpy as np
import os
from collections.abc import Iterable
from pyuvdata import UVData
from pyuvdata import UVCal
from pyuvdata import UVFlag
from . import utils as qm_utils
from pyuvdata import utils as uvutils
from . import __version__
from .metrics_io import process_ex_ants
from . import metrics_io
import warnings
import glob
import re
import copy

#############################################################################
# Utility functions
#############################################################################

def flag_xants(uv, xants, inplace=True, run_check=True,
               check_extra=True, run_check_acceptability=True):
    """Flag visibilities containing specified antennas.

    Parameters
    ----------
    uv : UVData or UVCal or UVFlag
        Object containing data to be flagged. Should be a UVData, UVCal, or
        UVFlag object.
    xants : list of ints
        List of antenna numbers to completely flag.
    inplace : bool, optional
        If True, apply flags to the uv object. If False, return a UVFlag object
        with only xants flaged. Default is True.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvo : UVData or UVCal or UVFlag
        If inplace is True, uvo is a reference to the input uv object, but with
        the flags specified in xants flagged. If inplace is False, uvo is a new
        UVFlag object with only xants flaged.

    Raises
    ------
    ValueError:
        If uv is not a UVData, UVCal, UVFlag, or subclassed object, a ValueError
        is raised. If a UVFlag of a "waterfall" type is passed in, a ValueError
        is also raised. If a UVFlag object that is not in "flag" mode is passed
        in, a ValueError is raised.

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
            uvo.to_flag(run_check=run_check, check_extra=check_extra,
                        run_check_acceptability=run_check_acceptability)
        else:
            uvo = UVFlag(uv, mode='flag')
    else:
        uvo = uv

    if isinstance(uvo, UVFlag) and uvo.mode != 'flag':
        raise ValueError('Cannot flag antennas on UVFlag obejct in mode ' + uvo.mode)

    if not isinstance(xants, Iterable):
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


def resolve_xrfi_path(xrfi_path, fname, jd_subdir=False):
    """Determine xrfi_path based on given directory or default to dirname of given file.

    Parameters
    ----------
    xrfi_path : str
        Directory to which to write xrfi outputs.
    fname : str
        Filename to determine backup directory if xrfi_path == ''.
    jd_subdir : bool, optional
        Whether to append the filename directory with a subdirectory with
        {JD}_xrfi (when xrfi_path is ''). Default is False.
        This option assumes the standard HERA naming scheme: zen.{JD}.{JD_decimal}.HH.uvh5

    Returns
    -------
    dirname : str
        If xrfi_path is not '', dirname is xrfi_path. Otherwise it returns the
        directory of the file.
    """
    if (xrfi_path != '') and (os.path.exists(xrfi_path)):
        dirname = xrfi_path
    else:
        dirname = os.path.dirname(os.path.abspath(fname))
        if jd_subdir:
            # Get JD string
            xrfi_subfolder = '.'.join(os.path.basename(fname).split('.')[0:3]) + '.xrfi'
            dirname = os.path.join(dirname, xrfi_subfolder)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    return dirname


def _check_convolve_dims(data, K1=None, K2=None):
    """Check the kernel sizes to be used in various convolution-like operations.

    If the kernel sizes are too big, replace them with the largest allowable size
    and issue a warning to the user.

    Parameters
    ----------
    data : array
        1- or 2-D array that will undergo convolution-like operations.
    K1 : int, optional
        Integer representing box dimension in first dimension to apply statistic.
        Defaults to None (see Returns)
    K2 : int, optional
        Integer representing box dimension in second dimension to apply statistic.
        Only used if data is two dimensional

    Returns
    -------
    K1 : int
        Input K1 or data.shape[0] if K1 is larger than first dim of arr.
        If K1 is not provided, will return data.shape[0].
    K2 : int (only if data is two dimensional)
        Input K2 or data.shape[1] if K2 is larger than second dim of arr.
        If data is 2D but K2 is not provided, will return data.shape[1].

    Raises
    ------
    ValueError:
        If the number of dimensions of the arr array is not 1 or 2, a ValueError is raised;
        If K1 < 1, or if data is 2D and K2 < 1.
    """
    if data.ndim not in (1, 2):
        raise ValueError('Input to filter must be 1- or 2-D array.')
    if K1 is None:
        warnings.warn("No K1 input provided. Using the size of the data for the "
                      "kernel size.")
        K1 = data.shape[0]
    elif K1 > data.shape[0]:
        warnings.warn("K1 value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(K1, data.shape[0]))
        K1 = data.shape[0]
    elif K1 < 1:
        raise ValueError('K1 must be greater than or equal to 1.')
    if (data.ndim == 2) and (K2 is None):
        warnings.warn("No K2 input provided. Using the size of the data for the "
                      "kernel size.")
        K2 = data.shape[1]
    elif (data.ndim == 2) and (K2 > data.shape[1]):
        warnings.warn("K2 value {0:d} is larger than the data of dimension {1:d}; "
                      "using the size of the data for the kernel size".format(K2, data.shape[1]))
        K2 = data.shape[1]
    elif (data.ndim == 2) and (K2 < 1):
        raise ValueError('K2 must be greater than or equal to 1.')
    if data.ndim == 1:
        return K1
    else:
        return K1, K2


def robust_divide(num, den):
    """Prevent division by zero.

    This function will compute division between two array-like objects by setting
    values to infinity when the denominator is small for the given data type. This
    avoids floating point exception warnings that may hide genuine problems
    in the data.

    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.

    Returns
    -------
    out : array
        The result of dividing num / den. Elements where b is small (or zero) are set
        to infinity.

    """
    thresh = np.finfo(den.dtype).eps
    out = np.true_divide(num, den, where=(np.abs(den) > thresh))
    out = np.where(np.abs(den) > thresh, out, np.inf)
    return out


#############################################################################
# Functions for preprocessing data prior to RFI flagging
#############################################################################

def medmin(data, flags=None):
    """Calculate the median minus minimum statistic of array.

    Note
    ----
    The statistic first computes the minimum value of the array along the
    first axis (the time axis, if the array is passed in as (time, frequency,
    so that a single spectrum is returned). The median of these values is
    computed, multiplied by 2, and then the minimum value is subtracted off.
    The goal is to get a proxy for the "noise" in the 2d array.

    Parameters
    ----------
    data : array
        2D data array of the shape (time,frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.

    Returns
    -------
    medmin : array
        The result of the medmin statistic.

    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dims
    mn = np.min(data, axis=0)
    return 2 * np.median(mn) - np.min(mn)


def medminfilt(data, flags=None, Kt=8, Kf=8):
    """Filter an array on scales of Kt,Kf indexes with medmin.

    Parameters
    ----------
    data : array
        2D data array of the shape (time, frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        An integer representing box dimension in time to apply statistic. Default
        is 8 pixels.
    Kf : int, optional
        An integer representing box dimension in frequency to apply statistic.
        Default is 8 pixels.

    Returns
    -------
    d_sm : array
        The filtered array with the same shape as input array.

    """
    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    d_sm = np.empty_like(data)
    for ind1 in range(data.shape[0]):
        for ind2 in range(data.shape[1]):
            i0, j0 = max(0, ind1 - Kt), max(0, ind2 - Kf)
            i1, j1 = min(data.shape[0], ind1 + Kt), min(data.shape[1], ind2 + Kf)
            d_sm[ind1, ind2] = medmin(data[i0:i1, j0:j1])
    return d_sm


def detrend_deriv(data, flags=None, dt=True, df=True):
    """Detrend array by taking the derivative in either time, frequency, or both.

    Note
    ----
    When taking the derivative of both, the derivative in frequency is performed
    first, then in time.

    Parameters
    ----------
    data : array
        2D data array of the shape (time,frequency).
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    dt : bool, optional
        The derivative across time bins. Default is True.
    df : bool, optional
        The derivative across frequency bins. Default is True.

    Returns
    -------
    out : array
        A detrended array with same shape as input array.

    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dims
    if not (dt or df):
        raise ValueError("dt and df cannot both be False when calling detrend_deriv")
    if df:
        # take gradient along frequency
        d_df = np.gradient(data, axis=1)
    else:
        d_df = data
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
    out = robust_divide(d_dtdf, sig)
    return out


def detrend_medminfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using medminfilt statistic. See medminfilt.

    Parameters
    ----------
    data : array
        2D data array of the shape (time, frequency) to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        An integer representing box dimension in time to apply statistic. Default
        is 8 pixels.
    Kf : int, optional
        An integer representing box dimension in frequency to apply statistic.
        Default is 8 pixels.

    Returns
    -------
    out : array
        An array of outlier significance metric.

    """
    _ = _check_convolve_dims(data, 1, 1)  # Just check data dimensions
    d_sm = medminfilt(np.abs(data), Kt=(2 * Kt + 1), Kf=(2 * Kf + 1))
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    # puts minmed on same scale as average
    sig = np.sqrt(medminfilt(d_sq, Kt=(2 * Kt + 1), Kf=(2 * Kf + 1))) * (np.sqrt(Kt**2 + Kf**2) / .64)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out


def detrend_medfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using a median filter of surrounding pixels (but not the center one).

    Parameters
    ----------
    data : array
        2D data array to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over. Default
        is 8 pixels.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.

    """
    # Delay import so scipy is not required for any use of hera_qm
    from scipy.ndimage import median_filter

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    footprint = np.ones((2 * Kt + 1, 2 * Kf + 1))
    footprint[Kt, Kf] = 0    
    if np.iscomplexobj(data):
        d_sm_r = median_filter(data.real, footprint=footprint, mode='reflect')
        d_sm_i = median_filter(data.imag, footprint=footprint, mode='reflect')
        d_sm = d_sm_r + 1j * d_sm_i
    else:
        d_sm = median_filter(data, footprint=footprint, mode='reflect')
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    # Factor of .456 is to put mod-z scores on same scale as standard deviation.
    sig = np.sqrt(median_filter(d_sq, footprint=footprint, mode='reflect') / .456)
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out


def detrend_meanfilt(data, flags=None, Kt=8, Kf=8):
    """Detrend array using a mean filter.

    Parameters
    ----------
    data : array
        2D data array to detrend.
    flags : array, optional
        2D flag array to be interpretted as mask for d.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over.
        Default is 8 pixels.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.
    """
    # Delay import so astropy is not required for any use of hera_qm
    # Using astropy instead of scipy for treatement of Nan: http://docs.astropy.org/en/stable/convolution/
    from astropy.convolution import convolve

    Kt, Kf = _check_convolve_dims(data, Kt, Kf)
    kernel = np.ones((2 * Kt + 1, 2 * Kf + 1))
    # do a mirror extend, like in scipy's convolve, which astropy doesn't support
    data = np.concatenate([data[Kt - 1::-1], data, data[:-Kt - 1:-1]], axis=0)
    data = np.concatenate([data[:, Kf - 1::-1], data, data[:, :-Kf - 1:-1]], axis=1)
    if flags is not None:
        flags = np.concatenate([flags[Kt - 1::-1], flags, flags[:-Kt - 1:-1]], axis=0)
        flags = np.concatenate([flags[:, Kf - 1::-1], flags, flags[:, :-Kf - 1:-1]], axis=1)
    d_sm = convolve(data, kernel, mask=flags, boundary='extend')
    d_rs = data - d_sm
    d_sq = np.abs(d_rs)**2
    sig = np.sqrt(convolve(d_sq, kernel, mask=flags))
    # don't divide by zero, instead turn those entries into +inf
    out = robust_divide(d_rs, sig)
    return out[Kt:-Kt, Kf:-Kf]


def zscore_full_array(data, flags=None, modified=False):
    """Calculate the z-score for full array, rather than a defined kernel size.

    This is a special case of
    detrend_medfilt/detrend_meanfilt, but is a separate function so it only
    takes the median/mean once for efficiency. It also doesn't introduce edge
    effects that would be very drastic if one were to call detrend_medfilt with
    large kernel size.

    Parameters
    ----------
    data : array
        2D data array to process.
    flags : array, optional
        2D flag array to be interpretted as mask for d. ONLY used for the regular
        zscore (not modified).
    modified : bool, optional
        Whether to calculate the modified z-scores. Default is False.

    Returns
    -------
    out : array
        An array containing the outlier significance metric. Same type and size as d.

    """
    data = np.array(data)  # makes a copy of the data
    if flags is not None:
        data[flags] = np.nan
    if modified:
        if np.any(np.iscomplex(data)):
            med_r = np.nanmedian(data).real
            med_i = np.nanmedian(data).imag
            mad_r = np.nanmedian(np.abs(data.real - med_r))
            mad_i = np.nanmedian(np.abs(data.imag - med_i))
            mad = np.sqrt(mad_r**2 + mad_i**2)
            d_rs = data - med_r - 1j * med_i
        else:
            med = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - med))
            d_rs = data - med
        # don't divide by zero, instead turn those entries into +inf
        out = robust_divide(d_rs, np.array([1.486 * mad]))
    else:
        d_rs = data - np.nanmean(data)
        out = robust_divide(d_rs, np.array([np.nanstd(data)]))
    out[np.isnan(out)] = np.inf  # turn all nans into infs
    return out


def modzscore_1d(data, flags=None, kern=8, detrend=True):
    """Calculate modified zscores in 1d.

    Parameters
    ----------
    data : array
        1D data array to detrend.
    flags : array, optional
        1D flag array to be interpretted as mask for d. NOT USED in this function,
        but kept for symmetry with other preprocessing functions.
    kern : int, optional
        The box size to apply medfilt over. Default is 8 pixels. Center pixel excluded.
    detrend : bool, optional
        Whether to detrend the data before calculating zscores. Default is True.
        Setting to False is equivalent to an infinite kernel, but the function
        does it more efficiently.

    Returns
    -------
    zscore : array
        An array containing the outlier significance metric. Same type and size as data.
    """
    if detrend:
        # Delay import so scipy is not required for use of hera_qm
        from scipy.ndimage import median_filter

        footprint = np.ones(2 * kern + 1)
        footprint[kern] = 0
        data = np.concatenate([data[kern - 1::-1], data, data[:-kern - 1:-1]])
        # detrend in 1D. Do real/imag regardless of whether data are complex because it's cheap.
        d_sm_r = median_filter(data.real, footprint=footprint)
        d_sm_i = median_filter(data.imag, footprint=footprint)
        d_sm = d_sm_r + 1j * d_sm_i
        d_rs = data - d_sm
        d_sq = np.abs(d_rs)**2
        # Factor of .456 is to put mod-z scores on same scale as standard deviation.
        sig = np.sqrt(median_filter(d_sq, footprint=footprint) / .456)
        zscore = robust_divide(d_rs, sig)[kern:-kern]
    else:
        d_rs = data - np.nanmedian(data.real) - 1j * np.nanmedian(data.imag)
        d_sq = np.abs(d_rs)**2
        # Factor of .456 is to put mod-z scores on same scale as standard deviation.
        sig = np.sqrt(np.nanmedian(d_sq) / .456)
        zscore = robust_divide(d_rs, np.array([sig]))
    return zscore.astype(data.dtype)


# Update algorithm_dict whenever new metric algorithm is created.
algorithm_dict = {'medmin': medmin, 'medminfilt': medminfilt, 'detrend_deriv': detrend_deriv,
                  'detrend_medminfilt': detrend_medminfilt, 'detrend_medfilt': detrend_medfilt,
                  'detrend_meanfilt': detrend_meanfilt, 'zscore_full_array': zscore_full_array,
                  'modzscore_1d': modzscore_1d}

#############################################################################
# RFI flagging algorithms
#############################################################################


def watershed_flag(uvf_m, uvf_f, nsig_p=2., nsig_f=None, nsig_t=None, avg_method='quadmean',
                   inplace=True, run_check=True, check_extra=True,
                   run_check_acceptability=True):
    """Expand a set of flags using a watershed algorithm.

    This function uses a UVFlag object in 'metric' mode (i.e. how many sigma the data
    point is from the center) and a set of flags to grow the flags using defined
    thresholds.

    Parameters
    ----------
    uvf_m : UVFlag object
        A UVFlag object in 'metric' mode.
    uvf_f : UVFlag object
        A UVFlag object in 'flag' mode.
    nsig_p : float, optional
        The Number of sigma above which to flag pixels which are near previously
        flagged pixels. Default is 2.0.
    nsig_f : float, optional
        The Number of sigma above which to flag channels which are near fully
        flagged frequency channels. Bypassed if None (Default).
    nsig_t : float, optional
        Number of sigma above which to flag integrations which are near fully
        flagged integrations. Bypassed if None (Default)
    avg_method : {"mean", "absmean", "quadmean"}, optional
        Method to average metric data for frequency and time watershedding.
        Default is "quadmean".
    inplace : bool, optional
        If True, update uvf_f. If False, create a new flag object. Default is True.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf : UVFlag object
       A UVFlag object in 'flag' mode with flags after watershed.

    Raises
    ------
    ValueError:
        If uvf_m is not in "metric" mode, if uvf_f is not in "flag" mode, if
        uvf_m and uvf_f do not have the same shape, or if uvf_m has an unknown
        type, then a ValueError is raised.

    """
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
        uvf = uvf_f.copy()

    # Convenience
    farr = uvf.flag_array
    marr = uvf_m.metric_array
    warr = uvf_m.weights_array

    if uvf_m.type == 'baseline':
        # Pixel watershed
        # TODO: bypass pixel-based if none
        for bl in np.unique(uvf.baseline_array):
            ind = np.where(uvf.baseline_array == bl)[0]
            for pi in range(uvf.polarization_array.size):
                farr[ind, 0, :, pi] += _ws_flag_waterfall(marr[ind, 0, :, pi],
                                                          farr[ind, 0, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 3), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 3))
            farr[:, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, 1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            ts = np.unique(uvf.time_array)
            tempd = np.zeros(ts.size)
            tempf = np.zeros(ts.size, dtype=np.bool_)
            for ti, time in enumerate(ts):
                tempd[ti] = uvutils.collapse(marr[uvf.time_array == time, 0, :, :], avg_method,
                                             weights=warr[uvf.time_array == time, 0, :, :])
                tempf[ti] = np.all(farr[uvf.time_array == time, 0, :, :])
            tempf = _ws_flag_waterfall(tempd, tempf, nsig_t)
            for ti, time in enumerate(ts):
                farr[uvf.time_array == time, :, :, :] += tempf[ti]
    elif uvf_m.type == 'antenna':
        # Pixel watershed
        for ai in range(uvf.ant_array.size):
            for pi in range(uvf.polarization_array.size):
                farr[ai, 0, :, :, pi] += _ws_flag_waterfall(marr[ai, 0, :, :, pi].T,
                                                            farr[ai, 0, :, :, pi].T, nsig_p).T
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 3, 4), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 3, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, 1, -1, 1, 1)
        if nsig_t is not None:
            # Time watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 1, 2, 4), weights=warr)
            tempf = np.all(farr, axis=(0, 1, 2, 4))
            farr[:, :, :, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_t).reshape(1, 1, 1, -1, 1)
    elif uvf_m.type == 'waterfall':
        # Pixel watershed
        for pi in range(uvf.polarization_array.size):
            farr[:, :, pi] += _ws_flag_waterfall(marr[:, :, pi], farr[:, :, pi], nsig_p)
        if nsig_f is not None:
            # Channel watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(0, 2), weights=warr)
            tempf = np.all(farr, axis=(0, 2))
            farr[:, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_f).reshape(1, -1, 1)
        if nsig_t is not None:
            # Time watershed
            tempd = uvutils.collapse(marr, avg_method, axis=(1, 2), weights=warr)
            tempf = np.all(farr, axis=(1, 2))
            farr[:, :, :] += _ws_flag_waterfall(tempd, tempf, nsig_t).reshape(-1, 1, 1)
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)

    if run_check:
        uvf.check(check_extra=check_extra,
                  run_check_acceptability=run_check_acceptability)
    return uvf


def _ws_flag_waterfall(metric, fin, nsig=2.):
    """Perform the watershed algorithm on 1D or 2D arrays of metric and input flags.

    This is a helper function for watershed_flag, but not usually called
    by end users.

    Parameters
    ----------
    metric : array
        A 2D or 1D array. Should be in units of standard deviations.
    fin : array
        The input (boolean) flags used as the seed of the watershed. Same size as metric.
    nsig : float, optional
        The number of sigma to flag above for points near flagged points. Default is 2.

    Returns
    -------
    fout : array
        A boolean array matching size of metric and fin, with watershedded flags.

    Raises
    ------
    ValueError:
        If the shapes of metric and fin do not match, or if the number of dimensions is not
        equal to 1 or 2, a ValueError is raised.

    """
    # Delay import so scipy is not required for use of hera_qm
    from scipy.signal import convolve
    if metric.shape != fin.shape:
        raise ValueError('metric and fin must match in shape. Shapes are: ' + str(metric.shape)
                         + ' and ' + str(fin.shape))
    fout = fin.copy()
    while True:
        nflags = np.sum(fout)
        try:
            kernel = {1: [1, 0, 1], 2: [[0, 1, 0], [1, 0, 1], [0, 1, 0]]}[metric.ndim]
        except KeyError:
            raise ValueError('Data must be 1D or 2D.')
        is_neighbor_flagged = convolve(fout, kernel, mode='same').astype(bool)
        fout |= (is_neighbor_flagged & (metric >= nsig))
        if np.sum(fout) == nflags:
            break
    return fout


def xrfi_waterfall(data, flags=None, Kt=8, Kf=8, nsig_init=6., nsig_adj=2.,
                   algorithm='detrend_medfilt'):
    """Compute metrics, flag, and then watershed on a single waterfall.

    Parameters
    ----------
    data : array
        2D data array (Ntimes, Nfreqs) to use in flagging.
    flags : array, optional
        2D flag array to be interpretted as mask for data. Ignored for many algorithms
        for calculating metrics of "outlierness" (e.g. detrend_medfilt) but always
        ORed with the intial flags from the metrics before the watershed is applied.
    Kt : int, optional
        The box size in time (first) dimension to apply medfilt over. Default is
        8 pixels.
    Kf : int, optional
        The box size in frequency (second) dimension to apply medfilt over. Default
        is 8 pixels.
    nsig_init : float, optional
        The number of sigma in the metric above which to flag pixels. Default is 6.
    nsig_adj : float, optional
        The number of sigma to flag above for points near flagged points. Default is 2.
    algorithm : str
        The metric algorithm name. Must be defined in algorithm_dict.

    Returns
    -------
    new_flags : array
        A final boolean array of flags matching the size of data.
    """
    try:
        alg_func = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    metrics = alg_func(data, flags=flags, Kt=Kt, Kf=Kf)
    init_flags = (metrics >= nsig_init)
    if flags is not None:
        init_flags |= flags
    new_flags = _ws_flag_waterfall(metrics, init_flags, nsig=nsig_adj)
    return new_flags


def flag(uvf_m, nsig_p=6., nsig_f=None, nsig_t=None, avg_method='quadmean',
         run_check=True, check_extra=True, run_check_acceptability=True):
    """Create a set of flags based on a "metric" type UVFlag object.

    Parameters
    ----------
    uvf_m : UVFlag object
        A UVFlag object in 'metric' mode (i.e., number of sigma data is from middle).
    nsig_p : float, optional
        The number of sigma above which to flag pixels. Default is 6.0. Bypassed
        if None.
    nsig_f : float, optional
        The number of sigma above which to flag channels. Bypassed if None (Default).
    nsig_t : float, optional
        The number of sigma above which to flag integrations. Bypassed if None (Default).
    avg_method : {"mean", "absmean", "quadmean"}, optional
        Method to average metric data for frequency and time flagging. Default is
        "quadmean".
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf_f : UVFlag object
        A UVFlag object in 'flag' mode with flags determined from uvf_m.


    Raises
    ------
    ValueError:
        If uvf_m is not a UVFlag object in metric mode, or if the type of uvf_m
        is not recognized, a ValueError is raised.

    """
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')

    # initialize
    uvf_f = uvf_m.copy()
    uvf_f.to_flag(run_check=run_check, check_extra=check_extra,
                  run_check_acceptability=run_check_acceptability)

    # Pixel flagging
    if nsig_p is not None:
        uvf_f.flag_array[np.abs(uvf_m.metric_array) >= nsig_p] = True

    if uvf_m.type == 'baseline':
        if nsig_f is not None:
            # Channel flagging
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3),
                                    weights=uvf_m.weights_array)
            indf = np.where(np.abs(data) >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :] = True
        if nsig_t is not None:
            # Time flagging
            ts = np.unique(uvf_m.time_array)
            data = np.zeros(ts.size)
            for ti, time in enumerate(ts):
                data[ti] = uvutils.collapse(uvf_m.metric_array[uvf_m.time_array == time, 0, :, :],
                                            avg_method,
                                            weights=uvf_m.weights_array[uvf_m.time_array == time, 0, :, :])
            indf = np.where(np.abs(data) >= nsig_t)[0]
            for time in ts[indf]:
                uvf_f.flag_array[uvf_f.time_array == time, :, :, :] = True
    elif uvf_m.type == 'antenna':
        if nsig_f is not None:
            # Channel flag
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 3, 4),
                                    weights=uvf_m.weights_array)
            indf = np.where(np.abs(data) >= nsig_f)[0]
            uvf_f.flag_array[:, :, indf, :, :] = True
        if nsig_t is not None:
            # Time watershed
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 1, 2, 4),
                                    weights=uvf_m.weights_array)
            indt = np.where(np.abs(data) >= nsig_t)[0]
            uvf_f.flag_array[:, :, :, indt, :] = True
    elif uvf_m.type == 'waterfall':
        if nsig_f is not None:
            # Channel flag
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(0, 2),
                                    weights=uvf_m.weights_array)
            indf = np.where(np.abs(data) >= nsig_f)[0]
            uvf_f.flag_array[:, indf, :] = True
        if nsig_t is not None:
            # Time watershed
            data = uvutils.collapse(uvf_m.metric_array, avg_method, axis=(1, 2),
                                    weights=uvf_m.weights_array)
            indt = np.where(np.abs(data) >= nsig_t)[0]
            uvf_f.flag_array[indt, :, :] = True
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf_f


def threshold_wf(uvf_m, nsig_f=7., nsig_t=7., nsig_f_adj=3., nsig_t_adj=3.,
                 detrend=False, run_check=True, check_extra=True,
                 run_check_acceptability=True):
    """Flag on a "waterfall" type UVFlag in "metric" mode.

    Use median to collapses to one dimension, thresholds, then broadcasts back to waterfall.
    Uses the 1D watershed algorithm to look for moderate outliers adjacent to stronger outliers.
    Non-finite metric values (e.g. from a priori flags) are excluded using np.nanmedian from
    the collapse to 1D, but treated as flagged for the purpose of adjacency-based flagging.

    Parameters
    ----------
    uvf_m : UVFlag object
        A UVFlag object in 'metric' mode (i.e., number of sigma data is from middle),
        and type 'waterfall'.
    nsig_f : float, optional
        The number of sigma above which to flag channels. Default is 7.0.
    nsig_t : float, optional
        The number of sigma above which to flag integrations. Default is 7.0.
    nsig_f_adj : float, optional
        The number of sigma above which to flag channels if they neighbor flagged channels.
        Default is 3.0.
    nsig_t_adj : float, optional
        The number of sigma above which to flag integrations if they neighbor flagged integrations.
        Default is 3.0.
    detrend : bool, optional
        Whether to detrend the 1D data before calculating zscores. Default is False.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf_f : UVFlag object
        A UVFlag object in 'flag' mode with flags determined from uvf_m.

    Raises
    ------
    ValueError:
        If uvf_m is not a UVFlag object in metric mode, or if the type of uvf_m
        is not "waterfall", a ValueError is raised.
    """
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode != 'metric') or (uvf_m.type != 'waterfall'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric" and'
                         + ' type == "waterfall."')

    # initialize
    uvf_f = uvf_m.copy()
    uvf_f.to_flag(run_check=run_check, check_extra=check_extra,
                  run_check_acceptability=run_check_acceptability)

    # handle a priori flags as nans and use nanmedians
    data = copy.deepcopy(uvf_m.metric_array)
    data[~np.isfinite(data)] = np.nan

    # Collapse to 1D and calculate z scores
    spec = np.nanmedian(data, axis=(0, 2))
    zspec = modzscore_1d(spec, detrend=detrend)
    tseries = np.nanmedian(data, axis=(1, 2))
    ztseries = modzscore_1d(tseries, detrend=detrend)

    # Flag based on zscores and thresholds, treating a priori flags as triggering nsig_f/nsig_t
    zspec[~np.isfinite(zspec)] = nsig_f
    f_flags = _ws_flag_waterfall(np.abs(zspec), np.abs(zspec) >= nsig_f, nsig=nsig_f_adj)
    uvf_f.flag_array[:, f_flags, :] = True

    tseries[~np.isfinite(tseries)] = nsig_t
    t_flags = _ws_flag_waterfall(np.abs(ztseries), np.abs(ztseries) >= nsig_t, nsig=nsig_t_adj)
    uvf_f.flag_array[t_flags, :, :] = True

    return uvf_f


def flag_apply(uvf, uv, keep_existing=True, force_pol=False, history='',
               return_net_flags=False, run_check=True,
               check_extra=True, run_check_acceptability=True):
    """Apply flags from UVFlag or list of UVFlag objects to UVData or UVCal.

    Parameters
    ----------
    uvf : UVFlag or str or list
        A UVFlag object, path to UVFlag file, or list of these. These must be in
        'flag' mode, and either match the uv argument, or be a waterfall that can
        be made to match it.
    uv : UVData or UVCal
        A UVData or UVCal object to which to apply flags.
    keep_existing : bool, optional
        If True, add flags to existing flags in uv. If False, replace existing
        flags in uv. Default is True.
    force_pol : bool, optional
        If True, will use 1 pol to broadcast to any other pol. If False, will
        require polarizations to match. Default is False.
    history : str, optional
        The history string to be added to uv.history. Default is empty string.
    return_net_flags : bool, optional
        If True, return a UVFlag object with net flags applied. If False, do not
        return net flags. Default is False.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    net_flags : UVFlag object
        If return_net_flags is True, returns UVFlag object with net flags.

    Raises
    ------
    ValueError:
        If uv is not a UVData or UVCal object, if uvf is not a string or
        UVFlag object, or if the UVFlag objects in uvf are not in "flag" mode,
        a ValueError is raised.

    """
    if issubclass(uv.__class__, UVData):
        expected_type = 'baseline'
    elif issubclass(uv.__class__, UVCal):
        expected_type = 'antenna'
    else:
        raise ValueError('Flags can only be applied to UVData or UVCal objects.')
    if not isinstance(uvf, (list, tuple, np.ndarray)):
        uvf = [uvf]
    net_flags = UVFlag(uv, mode='flag', copy_flags=keep_existing, history=history)
    for uvf_i in uvf:
        if isinstance(uvf_i, str):
            uvf_i = UVFlag(uvf_i)  # Read file
        elif not isinstance(uvf_i, UVFlag):
            raise ValueError('Input to apply_flag must be UVFlag or path to UVFlag file.')
        if uvf_i.mode != 'flag':
            raise ValueError('UVFlag objects must be in mode "flag" to apply to data.')
        if uvf_i.type == 'waterfall':
            uvf_i = uvf_i.copy()  # don't change the input object
            if expected_type == 'baseline':
                uvf_i.to_baseline(uv, force_pol=force_pol, run_check=run_check,
                                  check_extra=check_extra,
                                  run_check_acceptability=run_check_acceptability)
            else:
                uvf_i.to_antenna(uv, force_pol=force_pol, run_check=run_check,
                                 check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
        # Use built-in or function
        net_flags |= uvf_i
    uv.flag_array += net_flags.flag_array
    uv.history += 'FLAGGING HISTORY: ' + history + ' END OF FLAGGING HISTORY.'

    if return_net_flags:
        return net_flags


def simple_flag_waterfall(data,  Kt=8, Kf=8, sig_init=5.0, sig_adj=2.0, edge_cut=0, chan_thresh_frac=1.0):
    '''XRFI-lite: performs median and mean filtering on a single waterfall, with 
    watershed expansion of flags, and spectral and temporal thresholding.
    
    Parameters
    ----------
    data : 2D numpy array of floats or complex numbers
        Waterfall to use to find and flag outliers.
    Kt : int
        Time half-width of kernel for med/meanfilt.
    Kf : int
        Frequency half-width of kernel for med/meanfilt.
    nsig_init : float
        The number of sigma in the metric above which to flag pixels. Default is 5.
    nsig_adj : float
        The number of sigma to flag above for points near flagged points. Default is 2.
    edge_cut : integer
        Number of channels at each band edge to flag automatically.
    chan_thresh_frac : float
        Fraction of times flagged (excluding completely flagged integrations) above which
        to flag an entire channel. Default 1.0 means no additional flags.

    Returns
    -------
    flags : 2D numpy array of booleans
        Final waterfall of flags with the same shape as the data.
        
    '''
    # Perform medfilt-based RFI excision with watershed growth of flags
    medfilt_metric = detrend_medfilt(data, Kt=Kt, Kf=Kf)
    medfilt_flags = (np.abs(medfilt_metric) > sig_init)
    medfilt_flags |= _ws_flag_waterfall(medfilt_metric, medfilt_flags, nsig=sig_adj)
    
    # Perform meanfilt-based RFI excision with watershed growth of flags
    meanfilt_metric = detrend_meanfilt(data, flags=medfilt_flags, Kt=Kt, Kf=Kf)
    flags = medfilt_flags | (np.abs(meanfilt_metric) > sig_init) 
    flags |= _ws_flag_waterfall(meanfilt_metric, flags, nsig=sig_adj)
    
    # Perform spectral thresholding 
    combined_metric = np.where(medfilt_flags, medfilt_metric, meanfilt_metric)
    spec = np.nanmedian(combined_metric, axis=0)
    zspec = modzscore_1d(spec, detrend=False)
    zspec[~np.isfinite(zspec)] = sig_init
    flags[:, _ws_flag_waterfall(np.abs(zspec), np.abs(zspec) >= sig_init, nsig=sig_adj)] = True

    # Perform temporal thresholding
    tseries = np.nanmedian(combined_metric, axis=1)
    ztseries = modzscore_1d(tseries, detrend=False)
    tseries[~np.isfinite(tseries)] = sig_init
    flags[_ws_flag_waterfall(np.abs(ztseries), np.abs(ztseries) >= sig_init, nsig=sig_adj), :] = True

    # Flag edge channels
    if edge_cut > 0:
        flags[:, :edge_cut] = True
        flags[:, -edge_cut:] = True
    
    # Flag channels that are flagged more than chan_thresh_frac (excluding completely flagged times)
    min_flags_per_chan = np.min(np.sum(flags, axis=0))
    chan_thresh = min_flags_per_chan + chan_thresh_frac * (flags.shape[0] - min_flags_per_chan)
    flags[:, np.sum(flags, axis=0) > chan_thresh] = True

    return flags


#############################################################################
# Higher level functions that loop through data to calculate metrics
#############################################################################

def calculate_metric(uv, algorithm, cal_mode='gain', run_check=True,
                     check_extra=True, run_check_acceptability=True, **kwargs):
    """Make a UVFlag object of mode 'metric' from a UVData or UVCal object.

    Parameters
    ----------
    uv : UVData or UVCal
        A UVData or UVCal object to calculate metrics on.
    algorithm : str
        The metric algorithm name. Must be defined in algorithm_dict.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.
    **kwargs : dict
        A dictionary of Keyword arguments that are passed to algorithm.

    Returns
    -------
    uvf : UVFlag object
        A UVFlag of mode 'metric' corresponding to the uv object.

    Raises
    ------
    ValueError:
        If uv is not a UVData or UVCal object, or if "cal_mode" is not in the list
        above, then a ValueError is raised.

    KeyError:
        If "algorithm" is not in the list of known algorithm options, a KeyError
        is raised.

    """
    if not issubclass(uv.__class__, (UVData, UVCal)):
        raise ValueError('uv must be a UVData or UVCal object.')
    try:
        alg_func = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    uvf = UVFlag(uv)
    if issubclass(uv.__class__, UVData):
        uvf.weights_array = uv.nsample_array * np.logical_not(uv.flag_array).astype(np.float64)
    else:
        uvf.weights_array = np.logical_not(uv.flag_array).astype(np.float64)
    if issubclass(uv.__class__, UVData):
        for key, data in uv.antpairpol_iter():
            ind1, ind2, pol = uv._key2inds(key)
            for ind, ipol in zip((ind1, ind2), pol):
                if len(ind) == 0:
                    continue
                flags = uv.flag_array[ind, 0, :, ipol]
                uvf.metric_array[ind, 0, :, ipol] = alg_func(np.abs(data), flags=flags, **kwargs)

    elif issubclass(uv.__class__, UVCal):
        if cal_mode == 'tot_chisq':
            uvf.to_waterfall(run_check=run_check,
                             check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
            for pi in range(uv.Njones):
                data = np.abs(uv.total_quality_array[0, :, :, pi].T)
                flags = np.all(uv.flag_array[:, 0, :, :, pi], axis=0).T
                uvf.metric_array[:, :, pi] = alg_func(data, flags=flags, **kwargs)
        else:
            for ai in range(uv.Nants_data):
                for pi in range(uv.Njones):
                    # Note transposes are due to freq, time dimensions rather than the
                    # expected time, freq
                    flags = uv.flag_array[ai, 0, :, :, pi].T
                    if cal_mode == 'gain':
                        data = np.abs(uv.gain_array[ai, 0, :, :, pi].T)
                    elif cal_mode == 'chisq':
                        data = np.abs(uv.quality_array[ai, 0, :, :, pi].T)
                    else:
                        raise ValueError('When calculating metric for UVCal object, '
                                         'cal_mode must be "gain", "chisq", or "tot_chisq".')
                    uvf.metric_array[ai, 0, :, :, pi] = alg_func(data, flags=flags, **kwargs).T
    if run_check:
        uvf.check(check_extra=check_extra,
                  run_check_acceptability=run_check_acceptability)
    return uvf


#############################################################################
# "Pipelines" -- these routines define the flagging strategy for some data
#   Note: "current" pipes should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################

def xrfi_h1c_pipe(uv, Kt=8, Kf=8, sig_init=6., sig_adj=2., px_threshold=0.2,
                  freq_threshold=0.5, time_threshold=0.05, return_summary=False,
                  cal_mode='gain', run_check=True, check_extra=True,
                  run_check_acceptability=True):
    """Run the xrfi excision pipeline we used for H1C.

    This pipeline uses the detrending and watershed algorithms above.

    Parameters
    ----------
    uv : UVData or UVCal
        The UVData or UVCal object to flag.
    Kt : int, optional
        The time size for detrending box. Default is 8.
    Kf : int, optional
        The frequency size for detrending box. Default is 8.
    sig_init : float, optional
        The initial sigma to flag. Default is 6.0.
    sig_adj : float, optional
        The number of sigma to flag adjacent to flagged data. Default is 2.0.
    px_threshold : float, optional
        The fraction of flags required to trigger a broadcast across baselines
        for a given (time, frequency) pixel. Default is 0.2.
    freq_threshold : float, optional
        The fraction of channels required to trigger broadcast across frequency
        (for a single time). Default is 0.5.
    time_threshold : float, optional
        The fraction of times required to trigger broadcast across
        time (for a single frequency). Default is 0.05.
    return_summary : bool, optional
        If True, return a UVFlag object with the fraction of baselines/antennas
        that were flagged in the initial flag/watershed (before broadcasting).
        Default is False.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf_f : UVFlag object
        A UVFlag object of the "initial flags" (initial flag + watershed).
    uvf_wf : UVFlag object
        A UVFlag object of waterfall type after thresholding in time/freq.
    uvf_w : UVFlag object
        If return_summary is True, a UVFlag object with fraction of flags in uvf_f.

    """
    uvf = calculate_metric(uv, 'detrend_medfilt', Kt=Kt, Kf=Kf, cal_mode=cal_mode,
                           run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)
    uvf_f = flag(uvf, nsig_p=sig_init, nsig_f=None, nsig_t=None,
                 run_check=run_check, check_extra=check_extra,
                 run_check_acceptability=run_check_acceptability)
    uvf_f = watershed_flag(uvf, uvf_f, nsig_p=sig_adj, nsig_f=None, nsig_t=None,
                           run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)
    uvf_w = uvf_f.copy()
    uvf_w.to_waterfall()
    # I realize the naming convention has flipped, which results in nsig_f=time_threshold.
    # time_threshold is defined as fraction of time flagged to flag a given channel.
    # nsig_f is defined as significance required to flag a channel.
    uvf_wf = flag(uvf_w, nsig_p=px_threshold, nsig_f=time_threshold,
                  nsig_t=freq_threshold, run_check=run_check,
                  check_extra=check_extra,
                  run_check_acceptability=run_check_acceptability)

    if return_summary:
        return uvf_f, uvf_wf, uvf_w
    else:
        return uvf_f, uvf_wf


def xrfi_pipe(uv, alg='detrend_medfilt', Kt=8, Kf=8, xants=[], cal_mode='gain',
              skip_flags=False,
              wf_method='quadmean', reset_weights=True,
              sig_init=6.0, sig_adj=2.0, label='', center_metric=True,
              run_check=True, check_extra=True, run_check_acceptability=True):
    """Run the xrfi excision pipeline originally designed for H1C IDR2.2.

    This pipeline uses the detrending and watershed algorithms above.

    Parameters
    ----------
    uv : UVData or UVCal
        A UVData or UVCal object on which to calculate the metric.
    alg : str, optional
        The algorithm for calculating the metric. Default is "detrend_medfilt".
    Kt : int, optional
        The size of kernel in time dimension for detrending in the xrfi algorithm.
        Default is 8.
    Kf : int, optional
        The size of kernel in frequency dimension for detrending in the xrfi
        algorithm. Default is 8.
    xants : list, optional
        A list of antennas to flag. Default is an empty list.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    skip_flags : bool, optional
        If True, skip flagging steps (only compute the metric).
        Default is False.
    wf_method :  str, {"quadmean", "absmean", "mean", "or", "and"}
        How to collapse the dimension(s) to form a single waterfall.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    label: str, optional
        Label to be added to UVFlag objects.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf_m : UVFlag object
        A UVFlag object with metric after collapsing to waterfall and to single pol.
        The weights array is set to ones.
    uvf_fws : UVFlag object
        A UVFlag object with flags after watershed.

    """
    alg_func = algorithm_dict[alg]
    if not issubclass(uv.__class__, UVFlag):
        flag_xants(uv, xants, run_check=run_check,
                   check_extra=check_extra,
                   run_check_acceptability=run_check_acceptability)
        uvf_m = calculate_metric(uv, alg, Kt=Kt, Kf=Kf, cal_mode=cal_mode,
                                 run_check=run_check, check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
        uvf_m.label = label
        uvf_m.to_waterfall(method=wf_method, keep_pol=False,
                           run_check=run_check, check_extra=check_extra,
                           run_check_acceptability=run_check_acceptability)
        # This next line resets the weights to 1 (with data) or 0 (no data) to equally
        # combine with the other metrics.
        # Flag and watershed on each data product individually.
        # That is, on each complete file (e.g. calibration gains), not on individual
        # antennas/baselines. We don't broadcast until the very end.
    else:
        uvf_m = uv
    if center_metric:
        # Pass the z-scores through the filter again to get a zero-centered, width-of-one distribution.
        uvf_m.metric_array[:, :, 0] = alg_func(uvf_m.metric_array[:, :, 0],
                                               flags=~(uvf_m.weights_array[:, :, 0].astype(np.bool_)),
                                               Kt=Kt, Kf=Kf)
    if reset_weights:
        uvf_m.weights_array = uvf_m.weights_array.astype(np.bool_).astype(np.float64)
    if not skip_flags:
        uvf_f = flag(uvf_m, nsig_p=sig_init, run_check=run_check,
                     check_extra=check_extra,
                     run_check_acceptability=run_check_acceptability)
        uvf_fws = watershed_flag(uvf_m, uvf_f, nsig_p=sig_adj, inplace=False,
                                 run_check=run_check, check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
        uvf_fws.label += ' Flags.'
    else:
        uvf_fws = None
    return uvf_m, uvf_fws


def chi_sq_pipe(uv, alg='zscore_full_array', modified=False, sig_init=6.0,
                sig_adj=2.0, label='', run_check=True,
                check_extra=True, run_check_acceptability=True):
    """Zero-center and normalize the full total chi squared array, flag, and watershed.

    Parameters
    ----------
    uv : UVCal
        A UVCal object on which to calculate the metric.
    alg : str, optional
        The algorithm for calculating the metric. Default is "modz_full_array".
    modified : bool, optional
        Whether to calculate the modified z-scores. Default is False.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    label: str, optional
        Label to be added to UVFlag objects.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uvf_m : UVFlag object
        A UVFlag object with metric after collapsing to single pol.
        The weights array is set to ones.
    uvf_fws : UVFlag object
        A UVFlag object with flags after watershed.

    """
    uvf_m = calculate_metric(uv, alg, cal_mode='tot_chisq', modified=modified,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
    uvf_m.label = label
    uvf_m.to_waterfall(keep_pol=False, run_check=run_check,
                       check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)
    # This next line resets the weights to 1 (with data) or 0 (no data) to equally
    # combine with the other metrics.
    uvf_m.weights_array = uvf_m.weights_array.astype(np.bool_).astype(np.float64)
    alg_func = algorithm_dict[alg]
    # Pass the z-scores through the filter again to get a zero-centered, width-of-one distribution.
    uvf_m.metric_array[:, :, 0] = alg_func(uvf_m.metric_array[:, :, 0], modified=modified,
                                           flags=~(uvf_m.weights_array[:, :, 0].astype(np.bool_)))
    # Flag and watershed on waterfall
    uvf_f = flag(uvf_m, nsig_p=sig_init, run_check=run_check,
                 check_extra=check_extra,
                 run_check_acceptability=run_check_acceptability)
    uvf_fws = watershed_flag(uvf_m, uvf_f, nsig_p=sig_adj, inplace=False,
                             run_check=run_check, check_extra=check_extra,
                             run_check_acceptability=run_check_acceptability)
    uvf_fws.label += ' Flags.'
    return uvf_m, uvf_fws

def xrfi_run_step(uv_files=None, uv=None, uvf_apriori=None,
                  alg='detrend_medfilt', kt_size=8, kf_size=8,
                  xants=None, cal_mode='gain', correlations='cross',
                  wf_method='quadmean', sig_init=5.0, sig_adj=2.0, label='',
                  calculate_uvf_apriori=False, reinitialize=True,
                  Nwf_per_load=None, apply_uvf_apriori=True,
                  dtype='uvcal', run_filter=True,
                  metrics=None, flags=None, modified_z_score=False,
                  a_priori_flag_yaml=None,
                  a_priori_ants_only=False,
                  ignore_xants_override=False,
                  use_cross_pol_vis=True,
                  run_check=True,
                  check_extra=True,
                  run_check_acceptability=True):
    """Helper functin for xrfi run.

    This function contains the repeated pattern in xrfi run in which a uv_files is supplied
    If it has not yet been loaded, then we load it. If it has, we can supply it as uv
    If apriori flags exist, then we apply them.
    If not, we can optionally compute apriori flags from the uv_file
    If we want to run filter on the data, then we do can compute metrics and
    flag according to arguments specifying metric algorithms and their parameters.
    Any computed metrics and flags are appended to user supplied lists
    that can contain previously computed metrics / flags, and returned.

    Parameters
    ----------
    uv_file : string, optional
        name of file to load (which is a uvdata or uvcal calfits file)
        default is None. If none is provided, then metric calculations
        and flagging will be conducted object provided in uv arg (see below).
        If uv_files but not uv is provided, must specify the type of data
        (uvdata or uvcal) using the dtype arg below.
    uv : uvdata or uvcal object, optional
        uvdata or uvcal object to flag on.
        default is none. if none is provided, then will attempt to load in
        and initialize file specified by uv_file.
    uvf_apriori : uvflag object, optional.
        uvflag containing apriori flags to apply to uv / file loaded in uv_file.
        default is none. This should be a waterfall mode uvflag object.
    alg : str, optional
        The algorithm for calculating the metric. Default is "detrend_medfilt".
    kt_size : int, optional
        The size of kernel in time dimension for detrending in the xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        The size of kernel in frequency dimension for detrending in the xrfi
        algorithm. Default is 8.
    xants : list, optional
        A list of antennas to flag. Default is an empty list.
    cal_mode : {"gain", "chisq", "tot_chisq"}, optional
        The mode to calculate metric if uv is a UVCal object. The options use
        the gain_array, quality_array, and total_quality_array attributes,
        respectively. Default is "gain".
    correlations : {"auto", "cross", "both"}, optional
        The data correlations to use in  metric.
        "auto" means use only auto-correlations.
        "cross" means use only cross-correlations.
        "both" means use both.
    wf_method :  str, {"quadmean", "absmean", "mean", "or", "and"}
        How to collapse the dimension(s) to form a single waterfall.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    label : str, optional
        Label to be added to UVFlag objects.
    calculate_uvf_apriori : bool, optional
        if true, and uvf_apriori provided is None, then calculate a uvf_apriori
        from the flag arrays provided in the uv / uv_files inputs.
        the computed uvf_apriori will be a waterfall.
        default is False.
    reinitialize : bool, optional
        Reinitialize uv object if both uv and uv_files are provided.
        This is necessary for instances where different sets of baselines
        are operated on successively in a uvdata object and we pass the uvdata
        object in multiple calls to xrfi_run_step.
        default is True (only happens if both uv and uv_files are specified).
    Nwf_per_load : int, optional
        number of waterfalls to load simultaneously for flagging. This provides
        support for partial i/o whereby a net waterfall metric is computed by
        loading in small numbers of waterfalls from uv, computing a waterfall
        metric for each chunk, and combining them to form the waterfall for the
        entire dataset.
        CURRENTLY ONLY WORKS FOR UVDATA OBJECTS. UVCAL OBJECTS STILL HAVE ALL
        WATERFALLS LOADED SIMULTANEOUSLY.
        default (none) sets Nwf_per_load = Number of waterfalls in uvdata that
        are of the type specified by the correlation argument above.
    apply_uvf_apriori : bool, optional
        If false, don't apply the supplied uvf_apriori to uv before computing
        metrics and flagging.
        default is True.
    dtype : string optional
        specifies the type of data in uv_files (needed if uv is not provided).
        set to "uvdata" if uv_files contains a uvdata object. "uvcal" if uv_file
        contains a uvcal object.
        default is 'uvcal'.
    run_filter : bool, optional
        if false, don't calculate any metrics or flags. occassions for setting
        this to false are for using xrfi_run_step to initialize a uvcal object
        and to obtain (or apply) uvf_apriori without running calculating metrics
        or flagging.
    metrics : list, optional
        optional list to append metrics too.
        default is None
    flags : list, optional
        optional list to append flags too.
        default is None.
    modified_z_score : bool, optional
        if True, calculate modified_z_score when computing overall z-score
        (only used if alg='overall_z_score').
        Default is False.
    a_priori_flag_yaml : str, optional
        Path to file containing a priori frequency, time, or antenna flags.
        Antenna flags will include all polarizations, even if only one pol is listed.
        See hera_qm.metrics_io.read_a_priori_[chan/int/ant]_flags() for details.
    a_priori_ants_only : bool, optional
        if True, only apply antenna flags from apriori yaml.
    ignore_xants_override : bool, optional
        If True, will not apply xants or get xants from the a_priori_flag_yaml. 
        Used for redundantly averaged data and/or omnical visibility solutions where
        baseline keys represent multiple redundant antnena pairs and it is presumed
        that bad antenna exclusion was already performed.
    use_cross_pol_vis : bool, optional
        If True (default), also load and flag on cross-polarized visibilities (e.g. 'ne'/'en')
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    uv : UVData or UVCal object (if specified)
        the provided uvdata or uvflag object. If apriori_flags where applied
        then uv will differ from the uv provided in args in that the flags are now applied
    uvf : UVFlag object
        UVFlag containing metrics if any filters were run. Is None if run_filter=False
        or uv = None and uv_files = None
    uvf_f : UVFlag object
        UFlag containing flags if any filters were run. Is None if run_filter=False or
        uv = None and uv_files = None
    uvf_apriori : UVFlag object
        UVFlag containing apriori flags provided in args or the apriori flags
        derived from uv. Can be None if uv / uv_files were None or calculate_uvf_apriori
        = False.
    metrics : list
        list equal to the metrics list provided as a parameter plus uvf appended
        if uvf was calculated (is not None).
    flags : list
        list equal to the flags list provided as a parameter plus uvf_f appended
        if uvf_f was caculated (is not None)


    """
    # flags and metrics are lists
    # that computed flags and metrics can be appended too.
    if flags is None:
        flags = []
    if metrics is None:
        metrics = []
    if (xants is None) or (ignore_xants_override):
        xants_here = []
    else:
        xants_here = copy.deepcopy(xants)
    if uv is None:
        # if no uv is provided, we should try loading it from the file specified
        # by uv_file.
        if uv_files is not None:
            # initialize appropriately if a string was provided.
            if dtype=='uvcal':
                uv = UVCal()
                # No partial i/o for uvcal yet.
                uv.read_calfits(uv_files)
                if a_priori_flag_yaml is not None:
                    uv = qm_utils.apply_yaml_flags(uv, a_priori_flag_yaml,
                                                   flag_ants=not(ignore_xants_override),
                                                   flag_times=not(a_priori_ants_only),
                                                   flag_freqs=not(a_priori_ants_only))
            elif dtype=='uvdata':
                uv = UVData()
                uv.read(uv_files, read_data=False)
            else:
                raise ValueError("%s is an invalid dtype. Must be 'uvcal' or 'uvdata'."%dtype)
    no_uvf_apriori = (uvf_apriori is None)
    # now, assuming uv was either provided or successfully loaded.
    if uv is not None:
        # if we want, we can reinitialize uv
        if reinitialize:
            if uv_files is not None:
                if issubclass(uv.__class__, UVData):
                    uv.read(uv_files, read_data=False)
                else:
                    uv.read_calfits(uv_files)
                    if a_priori_flag_yaml is not None:
                        uv = qm_utils.apply_yaml_flags(uv, a_priori_flag_yaml,
                                                       flag_ants=not(ignore_xants_override),
                                                       flag_times=not(a_priori_ants_only),
                                                       flag_freqs=not(a_priori_ants_only))
        # The following code applies if uv is a UVData object.
        if issubclass(uv.__class__, UVData):
            bls = uv.get_antpairpols()
            if not use_cross_pol_vis:
                # cut baselines whose polarization is not the same as its conjugate (e.g. 'ne')
                bls = [app for app in bls if (app[2] == uvutils.conj_pol(app[2]))]
            if correlations == 'cross':
                bls = [app for app in bls if app[1] != app[0]]
            elif correlations == 'auto':
                bls = [app for app in bls if app[1] == app[0]]

            nbls = len(bls)
            # figure out how many baseline chunks
            # we need to iterate over.
            if Nwf_per_load is None:
                Nwf_per_load = nbls
            nloads = int(np.ceil(nbls / Nwf_per_load))
            # iterate over baseline chunks
            for loadnum in range(nloads):
                # read in chunk
                uv.read(uv_files, bls=bls[loadnum * Nwf_per_load:(loadnum + 1) * Nwf_per_load], axis='blt')
                if a_priori_flag_yaml is not None:
                    uv = qm_utils.apply_yaml_flags(uv, a_priori_flag_yaml,
                                                   flag_times=not(a_priori_ants_only),
                                                   flag_freqs=not(a_priori_ants_only))
                # if no uvf apriori was provided.
                if no_uvf_apriori:
                    # and we want to calculate it
                    if calculate_uvf_apriori:
                        # then extract the flags for the chunk of baselines we are on
                        uvf_apriori_chunk = UVFlag(uv, mode='flag', copy_flags=True, label='A priori flags.')
                        # waterfall them
                        uvf_apriori_chunk.to_waterfall(method='and', keep_pol=False, run_check=run_check,
                                                check_extra=check_extra,
                                                run_check_acceptability=run_check_acceptability)
                        # and if this is the first chunk, initialize uvf_apriori
                        if loadnum == 0:
                            uvf_apriori = uvf_apriori_chunk
                        # if this isn't the first chunk, and them with uvf_apriori.
                        else:
                            uvf_apriori = uvf_apriori and uvf_apriori_chunk
                # if uvf_apriori was supplied and we want to apply it, then apply it to the current
                # data chunk.
                elif apply_uvf_apriori:
                    flag_apply(uvf_apriori, uv, keep_existing=True, run_check=run_check, run_check_acceptability=run_check_acceptability, force_pol=True)
                if run_filter:
                    # We can compute individual metrics for each baseline and then collapse them
                    # onto a running average metric. Some slight modifications to xrfi_pipe were necessary to make
                    # this work. First, each individual chunk cannot be translated to a z-score centered at zero so
                    # we disable this step (per chunk) with the center_metric keyword. We also don't want to flag or
                    # reset the weights which we deactivate with the reset_weights and skip_flags keyword
                    uvft, _ = xrfi_pipe(uv, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants_here, skip_flags=True,
                                             cal_mode=cal_mode, sig_init=sig_init, sig_adj=sig_adj,
                                             reset_weights=False, center_metric=False, wf_method=wf_method,
                                             label=label, run_check=run_check, check_extra=check_extra,
                                             run_check_acceptability=run_check_acceptability)
                    # if this is the first chunk, set uvf (metrics) equal to metrics chunk.
                    if loadnum == 0:
                        uvf = uvft
                    # otherwise, combine metrics chunk with uvf aggragate.
                    else:
                        uvf.combine_metrics(uvft, method=wf_method, run_check=run_check,
                                           check_extra=check_extra, run_check_acceptability=run_check_acceptability)
            if run_filter:
                # now that we have a uvf that includes the combined metric of all the baselines, we can
                # run one last round of xrfi_pipe with flagging enabled, centering the metric enabled, and resetting
                # the weights enabled to perform these final steps which are run on the full collapsed metric.
                # note that we pass uvf as an arg instead of uv. xrfi_pipe has been modified so that if a uvflag
                # is passed in place of uvdata or uvcal, it skips the metric calculation /waterfalling
                # steps and goes straight to steps performed on combined metrics, i.e.
                # flagging, normalizing, and weights reseting.
                uvf, uvf_f = xrfi_pipe(uvf, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants_here, skip_flags=False,
                                         cal_mode=cal_mode, sig_init=sig_init, sig_adj=sig_adj, wf_method=wf_method,
                                         center_metric=True, reset_weights=True,
                                         label=label, run_check=run_check, check_extra=check_extra,
                                         run_check_acceptability=run_check_acceptability)
        # the following code is for when uv is a UVCal object.
        elif issubclass(uv.__class__, UVCal):
            # if uvf_apriori is not provided and we wish to derive it from uv
            # do so here.
            if uvf_apriori is None:
                if calculate_uvf_apriori:
                     uvf_apriori = UVFlag(uv, mode='flag', copy_flags=True, label='A priori flags.')
                     uvf_apriori.to_waterfall(method='and', keep_pol=False, run_check=run_check,
                                             check_extra=check_extra,
                                             run_check_acceptability=run_check_acceptability)
            # if uvf_apriori is not None and we wish to apply it to uv, do so here.
            elif apply_uvf_apriori:
                flag_apply(uvf_apriori, uv, keep_existing=True, run_check=run_check,
                           check_extra=check_extra, force_pol=True,
                           run_check_acceptability=run_check_acceptability)
            if run_filter:
                # if run_filter is true, perform chi_sq_pipe or xrfi_pipe
                if alg in ['zscore_full_array']:
                    uvf, uvf_f = chi_sq_pipe(uv, alg=alg, modified=modified_z_score, sig_init=sig_init, sig_adj=sig_adj,
                                             label=label, run_check=run_check, check_extra=check_extra, run_check_acceptability=run_check_acceptability)
                else:
                    uvf, uvf_f = xrfi_pipe(uv, alg=alg, Kt=kt_size, Kf=kf_size, xants=xants_here,
                                           cal_mode=cal_mode, sig_init=sig_init, sig_adj=sig_adj,
                                           wf_method=wf_method,
                                           label=label, run_check=run_check, check_extra=check_extra,
                                           run_check_acceptability=run_check_acceptability)
        # append uvf and flags if calculated.
        if run_filter:
            metrics += [uvf]
            flags += [uvf_f]
        # otherwise instantiate them to None
        else:
            uvf = None; uvf_f = None
        # if uvf_apriori was derived from uv, append it to flags
        if uvf_apriori is not None and calculate_uvf_apriori:
            flags += [uvf_apriori]
    else:
        uvf = None; uvf_f = None

    return uv, uvf, uvf_f, uvf_apriori, metrics, flags




#############################################################################
# Wrappers -- Interact with input and output files
#   Note: "current" wrappers should have simple names, but when replaced,
#         they should stick around with more descriptive names.
#############################################################################


def xrfi_run(ocalfits_files=None, acalfits_files=None, model_files=None,
             data_files=None, a_priori_flag_yaml=None,
             a_priori_ants_only=False, use_cross_pol_vis=True,
             omnical_median_filter=True, omnical_mean_filter=True,
             omnical_chi2_median_filter=True, omnical_chi2_mean_filter=True,
             omnical_zscore_filter=True,
             abscal_median_filter=True, abscal_mean_filter=True,
             abscal_chi2_median_filter=True, abscal_chi2_mean_filter=True,
             abscal_zscore_filter=True,
             omnivis_median_filter=True, omnivis_mean_filter=True,
             auto_median_filter=True, auto_mean_filter=True,
             cross_median_filter=False, cross_mean_filter=True,
             history=None, wf_method='quadmean', Nwf_per_load=None,
             xrfi_path='', kt_size=8, kf_size=8, 
             sig_init_med=10.0, sig_adj_med=4.0, sig_init_mean=5.0, sig_adj_mean=2.0,
             ex_ants=None, metrics_files=[],
             output_prefixes=None, throw_away_edges=True, clobber=False,
             run_check=True, check_extra=True, run_check_acceptability=True):
    """Run the xrfi excision pipeline used for H1C IDR2.2.
    This pipeline uses the detrending and watershed algorithms above.
    The algorithm is run on several data products: omnical gains, omnical chisq,
    abscal gains, abscal chisq, omnical visibility solutions, renormalized chisq,
    and the raw data. All of these, except the data, are run twice - first to
    get an initial estimate of heavily contaminated data, and a second time
    to get better estimate. The metrics and flags from each data product and both
    rounds are stored in the xrfi_path (which defaults to a subdirectory, see
    xrfi_path below). Also stored are the a priori flags and combined metrics/flags.
    User must provide at least one of ocalfits_files, acalfits_files, model_files,
    or data_files.
    Parameters
    ----------
    ocalfits_files : str or list of strings, optional
        The omnical calfits file to use to flag on gains and chisquared values.
    acalfits_files : str or list of strings, optional
        The abscal calfits file to use to flag on gains and chisquared values.
    model_files : str or list of strings, optional
        THe model visibility file to flag on.
    data_files : str or list of strings, optional
        The raw visibility data file to flag.
    a_priori_flag_yaml : str, optional
        Path to file containing a priori frequency, time, or antenna flags.
        Antenna flags will include all polarizations, even if only one pol is listed.
        See hera_qm.metrics_io.read_a_priori_[chan/int/ant]_flags() for details.
    a_priori_ants_only : bool, optional
        If True, only apply apriori flags from ants but don't apply apriori
        time and frequency flags.
        Default is False.
        WARNING: Default can cause excess flags in the vicinity the apriori regions
        due to edge effects.
    use_cross_pol_vis : bool, optional
        If True (default), also load and flag on cross-polarized visibilities (e.g. 'ne'/'en')
    omnical_median_filter : bool, optional
        If true, run a median filter on omnical gains.
        Mean filters are run after median filters.
        Default is True.
        If no omnical calfits files are provided
        filter is not run.
    omnical_mean_filter : bool, optional
        If True, run a mean filter on omnical gain solutions.
        Mean filters are run after median filters.
        Default is True.
        If no omnical calfits files are provided
        filter is not run.
    omnical_chi2_median_filter : bool, optional
        If True, run a median filter on omnical chisquare statistics.
        Mean filters are run after median filters.
        Default is True.
        If no omnical calfits files are provided
        filter is not run.
    omnical_chi2_mean_filter : bool, optional
        If True, run mean filter on abscal chisquare statistics.
        Mean filters are run after median filters.
        Default is True.
        If no omnical calfits files are provided
        filter is not run.
    omnical_zscore_filter : bool, optional
        If True, flag on omnical total z-score statistic.
        Default is True.
        If no omnical calfits files are provided
        filter is not run.
    abscal_median_filter : bool, optional
        If true, run a median filter on abscal gains.
        Mean filters are run after median filters.
        Default is True.
        If no abscal calfits files are provided
        filter is not run.
    abscal_mean_filter : bool, optional
        If True, run mean filter on abscal gain solutions.
        Mean filters are run after median filters.
        Default is True.
        If no abscal calfits files are provided
        filter is not run.
    abscal_chi2_median_filter : bool, optional
        If True, run a median filter on abscal chisquare statistics.
        Mean filters are run after median filters.
        Default is True.
        If no abscal calfits files are provided
        filter is not run.
    abscal_chi2_mean_filter : bool, optional
        If True, run mean filter on abscal chisquare statistics.
        Mean filters are run after median filters.
        Default is True.
        If no abscal calfits files are provided
        filter is not run.
    abscal_zscore_filter : bool, optional
        If True, flag on abscal total z-score statistic.
        Default is True.
        If no abscal calfits files are provided
        filter is not run.
    omnivis_median_filter : bool, optional
        If True, flag on omnivis median filter statistic.
        Mean filters are run after median filters.
        Default is True.
    omnivis_mean_filter : bool, optional
        If True, flag on omnivis mean filter statistic.
        Mean filters are run after median filters.
        Default is True.
    auto_median_filter : bool, optional
        If True, flag on autocorr median filter statistic.
        Mean filters are run after median filters.
        Default is True.
    auto_mean_filter : bool, optional
        If True, flag on autocorrelations mean filter statistics.
        Mean filters are run after median filters.
        Default is True.
    cross_median_filter : bool, optional
        If True, flag on data median filter statistic.
        Mean filters are run after median filters.
        Default is False. N.B. This option is generally quite expensive.
    cross_mean_filter : bool, optional
        If True, flag on data mean filter statistic.
        Mean filters are run after median filters.
        Default is True.
    Nwf_per_load : int, optional
        Specify the number of waterfalls to load simultaneously when computing
        metrics.
    wf_method :  str, {"quadmean", "absmean", "mean", "or", "and"}
        How to collapse the dimension(s) to form a single waterfall.
    history : str
        The history string to include in files.
    xrfi_path : str, optional
        Path to save xrfi files to. Default is a subdirectory "{JD}/" inside
        the same directory as data_file.
    kt_size : int, optional
        The size of kernel in time dimension for detrend in xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        Size of kernel in frequency dimension for detrend in xrfi algorithm.
        Default is 8.
    sig_init_med : float, optional
        The starting number of sigmas to flag on during the medfilt round. Default is 5.0.
    sig_adj_med : float, optional
        The number of sigmas to flag on for data adjacent to a flag during the medfilt round.
        Default is 2.0.
    sig_init_mean : float, optional
        The starting number of sigmas to flag on during the meanfilt round. Default is 5.0.
    sig_adj_mean : float, optional
        The number of sigmas to flag on for data adjacent to a flag during the meanfilt round.
        Default is 2.0.
    ex_ants : str, optional
        A comma-separated list of antennas to exclude. Flags of visibilities formed
        with these antennas will be set to True. Default is None (i.e., no antennas
        will be excluded).
    metrics_files : str or list of strings, optional
        path or list of paths to file(s) containing ant_metrics or auto_metrics readable by 
        metrics_io.load_metric_file. Used for finding ex_ants and is combined with antennas
        excluded via ex_ants. Flags of visibilities formed with these antennas will be set
        to True. Default is [] (no metrics files used, no antennas excluded).
    output_prefixes : str or list of strings, optional
        Optional output prefix. If none is provided, use data_file.
        Required of data_file is None.
        Provide output_prefixes in the same format as a data_file with an extension
        (should have a .uvh5 at the end). Output products will replace extension
        with various output labels. For example, output_prefixes='filename.uvh5'
        will result in products with names like 'filename.cross_flags1.h5'.
    throw_away_edges : bool, optional
        avoids writing out files at the edges where there is overlap with the time
        deconvolution kernel. Used in a chunked analysis where stride length is
        shorter than the chunk size (i.e. overlapping times to avoid edge effects).
        If True, files at the beginning and end of the night that might exhibit edge
        effects are written, but fully flagged. The "night" is determined by looking
        for all files in the same folder as the inputs that have the same decimal JD.
        (This might cause problems with observations that span noon UTC, since
        that's when JD increments.)
    clobber : bool, optional
        If True, overwrite existing files. Default is False.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.
    Returns
    -------
    None
    """
    if ocalfits_files is None and acalfits_files is None and model_files is None and data_files is None:
        raise ValueError("Must provide at least one of the following; ocalfits_files, acalfits_files, model_files, data_files")

    # recast strings as lists of strings
    if isinstance(acalfits_files, (str, np.string_)):
        acalfits_files = [acalfits_files]
    if isinstance(ocalfits_files, (str, np.string_)):
        ocalfits_files = [ocalfits_files]
    if isinstance(model_files, (str, np.string_)):
        model_files = [model_files]
    if isinstance(data_files, (str, np.string_)):
        data_files = [data_files]
    if isinstance(output_prefixes, (str, np.string_)):
        output_prefixes = [output_prefixes]

    # ensure that an optional output prefix if no data file is provided.
    if output_prefixes is None:
        if data_files is not None:
            output_prefixes = data_files
        else:
            raise ValueError("Must provide either output_prefixes or data_files!")

    # construct history
    if history is None:
        history = ''
    history = 'Flagging command: "' + history + '", Using ' + __version__

    # Combine excluded antenna indices from ex_ants, metrics_file, and a_priori_flag_yaml
    xants = process_ex_ants(ex_ants=ex_ants, metrics_files=metrics_files)
    if a_priori_flag_yaml is not None:
        xants = list(set(list(xants) + metrics_io.read_a_priori_ant_flags(a_priori_flag_yaml, ant_indices_only=True)))

    # build dictionary of common kwargs for xrfi_run_step
    xrfi_run_step_kwargs = {'kt_size': kt_size, 'kf_size': kf_size, 'xants': xants, 
                            'wf_method': wf_method, 'Nwf_per_load': Nwf_per_load,  'a_priori_flag_yaml': a_priori_flag_yaml, 
                            'a_priori_ants_only': a_priori_ants_only, 'use_cross_pol_vis': use_cross_pol_vis, 
                            'run_check': run_check, 'check_extra': check_extra, 'run_check_acceptability': run_check_acceptability}
        
    # vdict stores all of the outputs (metrics, flags etc...) All possible products
    # are ultimately referenced by vdict but could be set to None. Outputs set to None are simply
    # not written and can be used to determine whether steps depending on them should be run.
    vdict={'uvf_apriori': None,
    'uvf_init': None, 'uvf_fws': None, 'uvf_f': None, 'uvf_metrics': None,
    'uvf_metrics2': None, 'uvf_f2': None, 'uvf_fws2': None,
    'uvf_combined2': None, 'uvc_o':None, 'uvc_a':None, 'uvc_v':None, 'uv_v':None, 'uv_d':None} # keep all the variables here.

    def _run_all_filters(median_round, 
                         omnical_filter, omnical_chi2_filter, omnical_zscore_filter, 
                         abscal_filter, abscal_chi2_filter, abscal_zscore_filter,
                         omnivis_filter, cross_filter, auto_filter):
        '''This function runs all possible filters, updating vdict as appropriate'''
        
        # Start with empty list of flags and metrics for later combination
        metrics = []
        flags = []
        
        # Modify these labels and parameters based on whether we're doing mean- or median-based statistics
        label = {True: 'median', False: 'mean'}[median_round]
        rnd = {True: '', False: '2'}[median_round]
        rndnum = {True: '1', False: '2'}[median_round]
        alg = {True: 'detrend_medfilt', False: 'detrend_meanfilt'}[median_round]
        modified = {True: 'modified ', False: ''}[median_round]
        start_flag_name = {True: 'uvf_apriori', False: 'uvf_init'}[median_round]
        final_flag_name = {True: 'uvf_init', False: 'uvf_combined2'}[median_round]

        # Median/mean filter omnical gains and chi^2
        if omnical_filter:
            (vdict['uvc_o'], vdict[f'uvf_og{rnd}'], vdict[f'uvf_ogf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_o'], uv_files=ocalfits_files, alg=alg, cal_mode='gain', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Omnical gains, {label} filter.', apply_uvf_apriori=True, **xrfi_run_step_kwargs)
        if omnical_chi2_filter:
            (vdict['uvc_o'], vdict[f'uvf_ox{rnd}'], vdict[f'uvf_oxf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_o'], uv_files=ocalfits_files, alg=alg, cal_mode='tot_chisq', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Omnical chisq, {label} filter.', apply_uvf_apriori=False, **xrfi_run_step_kwargs)
        if omnical_zscore_filter:
            (vdict['uvc_o'], vdict[f'uvf_oz{rnd}'], vdict[f'uvf_ozf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_o'], uv_files=ocalfits_files, alg='zscore_full_array', cal_mode=None, metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Omnical overall {modified}z-score of chisq.', apply_uvf_apriori=False, **xrfi_run_step_kwargs)

        # Median/mean filter abscal gains and chi^2
        if abscal_filter:
            (vdict['uvc_a'], vdict[f'uvf_ag{rnd}'], vdict[f'uvf_agf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_a'], uv_files=acalfits_files, alg=alg, cal_mode='gain', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Abscal gains, {label} filter.', apply_uvf_apriori=True, **xrfi_run_step_kwargs)
        if abscal_chi2_filter:
            (vdict['uvc_a'], vdict[f'uvf_ax{rnd}'], vdict[f'uvf_axf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_a'], uv_files=acalfits_files, alg=alg, cal_mode='tot_chisq', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Abscal chisq, {label} filter.', apply_uvf_apriori=False, **xrfi_run_step_kwargs)
        if abscal_zscore_filter:
            (vdict['uvc_a'], vdict[f'uvf_az{rnd}'], vdict[f'uvf_azf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uvc_a'], uv_files=acalfits_files, alg='zscore_full_array', cal_mode=None, metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=False, label=f'Abscal overall {modified}z-score of chisq.', apply_uvf_apriori=False, **xrfi_run_step_kwargs)

        # Median/mean filter omnical visibility solutions, cross-correlations, and autocorrelations
        if omnivis_filter:
            (vdict['uv_v'], vdict[f'uvf_v{rnd}'], vdict[f'uvf_vf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uv_v'], uv_files=model_files, alg=alg,  dtype='uvdata', correlations='both', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=True, label=f'Omnical visibility solutions, {label} filter.', apply_uvf_apriori=True, ignore_xants_override=True, **xrfi_run_step_kwargs)
        if cross_filter:
            (vdict['uv_d'], vdict[f'uvf_d{rnd}'], vdict[f'uvf_df{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uv_d'], uv_files=data_files, alg=alg,  dtype='uvdata', correlations='cross', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=True, label=f'Crosscorr, {label} filter.', apply_uvf_apriori=True, **xrfi_run_step_kwargs)
        if auto_filter:
            (vdict['uv_d'], vdict[f'uvf_da{rnd}'], vdict[f'uvf_daf{rnd}'], vdict[start_flag_name], metrics, flags) = \
                xrfi_run_step(uv=vdict['uv_d'], uv_files=data_files, alg=alg,  dtype='uvdata', correlations='auto', metrics=metrics, flags=flags, uvf_apriori=vdict[start_flag_name],
                              reinitialize=True, label=f'Autocorr, {label} filter.', apply_uvf_apriori=True, **xrfi_run_step_kwargs)

        # Now that we've had a chance to load in all of the provided data products and run filters when specified, 
        # we combine the metrics computed so far into a combined metrics object, using the metrics list.
        if len(metrics) > 0:
            if len(metrics) > 1:
                vdict[f'uvf_metrics{rnd}'] = metrics[-1].combine_metrics(metrics[:-1], method='quadmean', inplace=False)
            else:
                vdict[f'uvf_metrics{rnd}'] = copy.deepcopy(metrics[-1])
            
            # Prep starting flags for combined metrics    
            if median_round:
                flags_for_combined_metrics = ~vdict[f'uvf_metrics{rnd}'].weights_array[:, :, 0].astype(np.bool_)
            else:
                if vdict['uvf_init'] is None:
                    vdict['uvf_init'] = copy.deepcopy(vdict['uvf_metrics2'])
                    vdict['uvf_init'].to_flag(run_check=run_check, check_extra=check_extra,
                                     run_check_acceptability=run_check_acceptability)
                    vdict['uvf_init'].flag_array[:] = False
                flags_for_combined_metrics = vdict['uvf_init'].flag_array[:, :, 0]

            # Calculate combined metrics and flag on them
            vdict[f'uvf_metrics{rnd}'].label = f'Combined metrics, round {rndnum}.'
            vdict[f'uvf_metrics{rnd}'].metric_array[:, :, 0] = algorithm_dict[alg](vdict[f'uvf_metrics{rnd}'].metric_array[:, :, 0],
                                                                                   flags=flags_for_combined_metrics,
                                                                                   Kt=kt_size, Kf=kf_size)
            vdict[f'uvf_f{rnd}'] = flag(vdict[f'uvf_metrics{rnd}'], nsig_p=xrfi_run_step_kwargs['sig_init'], run_check=run_check,
                                        check_extra=check_extra, run_check_acceptability=run_check_acceptability)
            vdict[f'uvf_fws{rnd}'] = watershed_flag(vdict[f'uvf_metrics{rnd}'], vdict[f'uvf_f{rnd}'], nsig_p=xrfi_run_step_kwargs['sig_adj'], 
                                                    inplace=False, run_check=run_check, check_extra=check_extra,
                                                    run_check_acceptability=run_check_acceptability)
            vdict[f'uvf_fws{rnd}'].label = f'Flags from combined metrics, round {rndnum}.'
            flags += [vdict[f'uvf_fws{rnd}']]

            # OR all flags so far
            vdict[final_flag_name] = copy.deepcopy(flags[0])
            if len(flags) > 1:
                for flg in flags[1:]:
                    vdict[final_flag_name] |= flg
            vdict[final_flag_name].label = f'ORd flags, round {rndnum}.'
                  
    ########################
    # RUN MEDIAN ROUND
    ########################

    # settings for median round
    xrfi_run_step_kwargs['modified_z_score'] =  True 
    xrfi_run_step_kwargs['calculate_uvf_apriori'] = True
    xrfi_run_step_kwargs['sig_init'] = sig_init_med
    xrfi_run_step_kwargs['sig_adj'] = sig_adj_med

    _run_all_filters(True, 
                     omnical_median_filter, omnical_chi2_median_filter, omnical_zscore_filter, 
                     abscal_median_filter, abscal_chi2_median_filter, abscal_zscore_filter,
                     omnivis_median_filter, cross_median_filter, auto_median_filter)
    spoof_init = (vdict['uvf_init'] is None)
    median_round_flags_copy = copy.deepcopy(vdict['uvf_init'])

    ########################
    # RUN MEAN ROUND
    ########################

    # settings for median round
    xrfi_run_step_kwargs['modified_z_score'] =  False 
    xrfi_run_step_kwargs['calculate_uvf_apriori'] = False
    xrfi_run_step_kwargs['sig_init'] = sig_init_mean
    xrfi_run_step_kwargs['sig_adj'] = sig_adj_mean

    # Now perform the mean filtering after median filtering.
    _run_all_filters(False, 
                     omnical_mean_filter, omnical_chi2_mean_filter, omnical_zscore_filter, 
                     abscal_mean_filter, abscal_chi2_mean_filter, abscal_zscore_filter,
                     omnivis_mean_filter, cross_mean_filter, auto_mean_filter)
    vdict['uvf_init'] = median_round_flags_copy  # prevents this from getting modified above

    ########################
    # WRITE EVERYTHING OUT
    ########################

    if spoof_init:
        vdict['uvf_init'] = None
    uvf_dict = {'apriori_flags.h5': 'uvf_apriori',
                'v_metrics1.h5': 'uvf_v', 'v_flags1.h5': 'uvf_vf',
                'og_metrics1.h5': 'uvf_og', 'og_flags1.h5': 'uvf_ogf',
                'ox_metrics1.h5': 'uvf_ox', 'ox_flags1.h5': 'uvf_oxf',
                'ag_metrics1.h5': 'uvf_ag', 'ag_flags1.h5': 'uvf_agf',
                'ax_metrics1.h5': 'uvf_ax', 'ax_flags1.h5': 'uvf_axf',
                'auto_metrics1.h5': 'uvf_da', 'auto_flags1.h5': 'uvf_daf',
                'cross_metrics1.h5': 'uvf_d', 'cross_flags1.h5': 'uvf_df',
                'omnical_chi_sq_renormed_metrics1.h5': 'uvf_oz', 'omnical_chi_sq_flags1.h5': 'uvf_ozf',
                'abscal_chi_sq_renormed_metrics1.h5': 'uvf_az', 'abscal_chi_sq_flags1.h5': 'uvf_azf',
                'combined_metrics1.h5': 'uvf_metrics', 'combined_flags1.h5': 'uvf_fws',
                'flags1.h5': 'uvf_init',
                'v_metrics2.h5': 'uvf_v2', 'v_flags2.h5': 'uvf_vf2',
                'og_metrics2.h5': 'uvf_og2', 'og_flags2.h5': 'uvf_ogf2',
                'ox_metrics2.h5': 'uvf_ox2', 'ox_flags2.h5': 'uvf_oxf2',
                'ag_metrics2.h5': 'uvf_ag2', 'ag_flags2.h5': 'uvf_agf2',
                'ax_metrics2.h5': 'uvf_ax2', 'ax_flags2.h5': 'uvf_axf2',
                'auto_metrics2.h5': 'uvf_da2', 'auto_flags2.h5': 'uvf_daf2',
                'cross_metrics2.h5': 'uvf_d2', 'cross_flags2.h5': 'uvf_df2',
                'omnical_chi_sq_renormed_metrics2.h5': 'uvf_oz2', 'omnical_chi_sq_flags2.h5': 'uvf_ozf2',
                'abscal_chi_sq_renormed_metrics2.h5': 'uvf_az2', 'abscal_chi_sq_flags2.h5': 'uvf_azf2',
                'combined_metrics2.h5': 'uvf_metrics2', 'combined_flags2.h5': 'uvf_fws2',
                'flags2.h5': 'uvf_combined2'}

    # Read metadata from first file to get integrations per file.
    if data_files is not None:
        uvlist = data_files
        uvtemp = UVData()
        uvtemp.read(uvlist[0], read_data=False)
    elif model_files is not None:
        uvlist = model_files
        uvtemp = UVData()
        uvtemp.read(uvlist[0], read_data=False)
    elif ocalfits_files is not None:
        uvlist = ocalfits_files
        uvtemp = UVCal()
        uvtemp.read_calfits(uvlist[0])
    elif acalfits_files is not None:
        uvlist = acalfits_files
        uvtemp = UVCal()
        uvtemp.read_calfits(uvlist[0])
    nintegrations = len(uvlist) * uvtemp.Ntimes
    # Determine the actual files to store
    # We will drop kt_size / (integrations per file) files at the start and
    # end to avoid edge effects from the convolution kernel.
    # If this chunk includes the start or end of the night, we will write
    # output files for those, but flag everything.
    # Calculate number of files to drop on edges, rounding up.
    ndrop = int(np.ceil(kt_size / uvtemp.Ntimes))
    # start_ind and end_ind are the indices in the file list to include
    start_ind = ndrop
    end_ind = len(uvlist) - ndrop
    # If we're the first or last job, store all flags for the edge
    datadir = os.path.dirname(os.path.abspath(uvlist[0]))
    bname = os.path.basename(uvlist[0])
    # Because we don't necessarily know the filename structure, search for
    # files that are the same except integer JD but different decimals
    decimal_JD = re.search('[0-9]+.[0-9]+', bname)[0]
    search_JD = decimal_JD.split('.')[0] + '.' +  len(decimal_JD.split('.')[1]) * '?'
    search_str = os.path.join(datadir, bname.replace(decimal_JD, search_JD))
    all_files = sorted(glob.glob(search_str))
    if os.path.basename(uvlist[0]) == os.path.basename(all_files[0]) or not throw_away_edges:
        # This is the first job, store the early edge.
        start_ind = 0
    if os.path.basename(uvlist[-1]) == os.path.basename(all_files[-1]) or not throw_away_edges:
        # Last job, store the late edge.
        end_ind = len(uvlist)
        ndrop = 0

    # Loop through the files to output, storing all the different data products.
    for ind in range(start_ind, end_ind):
        night_index = all_files.index(os.path.abspath(uvlist[ind]))
        # we need an absolute index for entire night to do edge flagging.
        dirname = resolve_xrfi_path(xrfi_path, output_prefixes[ind], jd_subdir=True)
        basename = qm_utils.strip_extension(os.path.basename(output_prefixes[ind]))
        for ext, uvf in uvf_dict.items():
            if (uvf in vdict) and (vdict[uvf] is not None):
                # This is calculated separately for each uvf because machine
                # precision error was leading to times not found in object.
                this_times = np.unique(vdict[uvf].time_array)
                # This assumes that all files have the same number of time integrations!
                t_ind = ind * uvtemp.Ntimes
                uvf_out = vdict[uvf].select(times=this_times[t_ind:(t_ind + uvtemp.Ntimes)],
                                     inplace=False)
                # Determine indices relative to zero below and above which to flag edges.
                # flag all integrations above Ntimes - (kernel size - ntimes x (number of files to end of night)
                lower_flag_ind = np.max([uvtemp.Ntimes - (kt_size - uvtemp.Ntimes * (len(all_files) - 1 - night_index)), 0])
                # flag all integrations below kernel_size - ntimes x (number of files from beginning of night)
                upper_flag_ind = np.min([np.max([kt_size - uvtemp.Ntimes * night_index, 0]), uvtemp.Ntimes])
                if throw_away_edges:
                    if (ext == 'flags2.h5'):
                        if lower_flag_ind < uvtemp.Ntimes and lower_flag_ind >=  0:
                            uvf_out.flag_array[lower_flag_ind:] = True
                        if upper_flag_ind > 0:
                            uvf_out.flag_array[:upper_flag_ind] = True
                outfile = '.'.join([basename, ext])
                outpath = os.path.join(dirname, outfile)
                uvf_out.history += history
                uvf_out.write(outpath, clobber=clobber)

def xrfi_h3c_idr2_1_run(ocalfits_files, acalfits_files, model_files, data_files,
                        flag_command, xrfi_path='', kt_size=8, kf_size=8,
                        sig_init=5.0, sig_adj=2.0, ex_ants=None, metrics_file=None,
                        clobber=False, run_check=True, check_extra=True,
                        run_check_acceptability=True):
    """Run the xrfi excision pipeline used for H3C IDR2.1.

    This pipeline uses the detrending and watershed algorithms above.
    Several files are concatenated together to perform the detrending,
    and the flags from the inner files are stored*.
    The algorithm is run on several data products: omnical gains, omnical chisq,
    abscal gains, abscal chisq, omnical visibility solutions, renormalized chisq,
    and the raw data. All of these, except the data, are run twice - first to
    get an initial estimate of heavily contaminated data, and a second time
    to get better estimate. The metrics and flags from each data product and both
    rounds are stored in the xrfi_path (which defaults to a subdirectory, see
    xrfi_path below). Also stored are the a priori flags and combined metrics/flags.

    * For a given chunk of files, we do not store output files for the edges,
    determined by the size of the time kernel (kt_size) and the number of
    integrations per file. The exception is the very start and end of a day,
    which are stored, but completely flagged because they will never not be at
    the edge.
    It is up to the user to run overlapping chunks to ensure output is created
    for every input file.

    Parameters
    ----------
    ocalfits_files : str
        The omnical calfits files to use to flag on gains and chisquared values.
    acalfits_files : str
        The abscal calfits files to use to flag on gains and chisquared values.
    model_files : str
        THe model visibility files to flag on.
    data_files : str
        The raw visibility data files to flag.
    flag_command : str
        The flagging command used to call this function. Usually determined
        in the script that invokes this function.
    xrfi_path : str, optional
        Path to save xrfi files to. Default is a subdirectory "{JD}/" inside
        the same directory as data_file.
    kt_size : int, optional
        The size of kernel in time dimension for detrend in xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        Size of kernel in frequency dimension for detrend in xrfi algorithm.
        Default is 8.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 5.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag.
        Default is 2.0.
    ex_ants : str, optional
        A comma-separated list of antennas to exclude. Flags of visibilities formed
        with these antennas will be set to True. Default is None (i.e., no antennas
        will be excluded).
    metrics_file : str, optional
        Metrics file that contains a list of excluded antennas. Flags of visibilities
        formed with these antennas will be set to True. Default is None (i.e.,
        no antennas will be excluded).
    clobber : bool, optional
        If True, overwrite existing files. Default is False.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    None

    """
    history = 'Flagging command: "' + flag_command + '", Using ' + __version__
    xants = process_ex_ants(ex_ants=ex_ants, metrics_files=metrics_file)

    # Make sure input files are sorted
    ocalfits_files = sorted(ocalfits_files)
    acalfits_files = sorted(acalfits_files)
    model_files = sorted(model_files)
    data_files = sorted(data_files)

    # Make keyword dict to save some space on repeated options
    check_kwargs = {'run_check': run_check, 'check_extra': check_extra,
                    'run_check_acceptability': run_check_acceptability}

    # Initial run on cal data products
    # Calculate metric on abscal data
    uvc_a = UVCal()
    uvc_a.read_calfits(acalfits_files)
    uvf_apriori = UVFlag(uvc_a, mode='flag', copy_flags=True, label='A priori flags.')
    uvf_ag, uvf_agf = xrfi_pipe(uvc_a, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj,
                                label='Abscal gains, round 1.', **check_kwargs)
    uvf_ax, uvf_axf = xrfi_pipe(uvc_a, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj,
                                label='Abscal chisq, round 1.', **check_kwargs)

    # Calculate metric on omnical data
    uvc_o = UVCal()
    uvc_o.read_calfits(ocalfits_files)
    flag_apply(uvf_apriori, uvc_o, keep_existing=True, run_check=run_check,
               check_extra=check_extra,
               run_check_acceptability=run_check_acceptability)
    uvf_og, uvf_ogf = xrfi_pipe(uvc_o, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj,
                                label='Omnical gains, round 1.', **check_kwargs)
    uvf_ox, uvf_oxf = xrfi_pipe(uvc_o, alg='detrend_medfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj,
                                label='Omnical chisq, round 1.', **check_kwargs)

    # Calculate metric on model vis
    uv_v = UVData()
    uv_v.read(model_files, axis='blt')
    uvf_v, uvf_vf = xrfi_pipe(uv_v, alg='detrend_medfilt', xants=[], Kt=kt_size, Kf=kf_size,
                              sig_init=sig_init, sig_adj=sig_adj,
                              label='Omnical visibility solutions, round 1.',
                              **check_kwargs)

    # Get the non-detrended total chi-squared values, normalized across the full waterfall.
    uvf_chisq, uvf_chisq_f = chi_sq_pipe(uvc_o, alg='zscore_full_array', modified=True,
                                         sig_init=sig_init, sig_adj=sig_adj,
                                         label='Renormalized chisq, round 1.',
                                         **check_kwargs)

    # Combine the metrics together
    uvf_metrics = uvf_v.combine_metrics([uvf_og, uvf_ox, uvf_ag, uvf_ax, uvf_chisq],
                                        method='quadmean', inplace=False)
    uvf_metrics.label = 'Combined metrics, round 1.'
    alg_func = algorithm_dict['detrend_medfilt']
    uvf_metrics.metric_array[:, :, 0] = alg_func(uvf_metrics.metric_array[:, :, 0],
                                                 flags=~uvf_metrics.weights_array[:, :, 0].astype(np.bool_),
                                                 Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f = flag(uvf_metrics, nsig_p=sig_init, run_check=run_check,
                 check_extra=check_extra,
                 run_check_acceptability=run_check_acceptability)
    uvf_fws = watershed_flag(uvf_metrics, uvf_f, nsig_p=sig_adj, inplace=False,
                             **check_kwargs)
    uvf_fws.label = 'Flags from combined metrics, round 1.'

    # OR everything together for initial flags
    uvf_apriori.to_waterfall(method='and', keep_pol=False, **check_kwargs)
    uvf_init = (uvf_fws | uvf_ogf | uvf_oxf | uvf_agf | uvf_axf | uvf_vf
                | uvf_chisq_f | uvf_apriori)
    uvf_init.label = 'ORd flags, round 1.'

    # Second round -- use init flags to mask and recalculate everything
    # Read in data file
    uv_d = UVData()
    uv_d.read(data_files, axis='blt')
    for uv in [uvc_o, uvc_a, uv_v, uv_d]:
        flag_apply(uvf_init, uv, keep_existing=True, force_pol=True,
                   **check_kwargs)

    # Do next round of metrics
    # Change to meanfilt because it can mask flagged pixels
    # Calculate metric on abscal data
    uvf_ag2, uvf_agf2 = xrfi_pipe(uvc_a, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj,
                                  label='Abscal gains, round 2.', **check_kwargs)
    uvf_ax2, uvf_axf2 = xrfi_pipe(uvc_a, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj,
                                  label='Abscal chisq, round 2.', **check_kwargs)

    # Calculate metric on omnical data
    uvf_og2, uvf_ogf2 = xrfi_pipe(uvc_o, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='gain', sig_init=sig_init, sig_adj=sig_adj,
                                  label='Omnical gains, round 2.', **check_kwargs)
    uvf_ox2, uvf_oxf2 = xrfi_pipe(uvc_o, alg='detrend_meanfilt', Kt=kt_size, Kf=kf_size, xants=xants,
                                  cal_mode='tot_chisq', sig_init=sig_init, sig_adj=sig_adj,
                                  label='Omnical chisq, round 2.', **check_kwargs)

    # Calculate metric on model vis
    uvf_v2, uvf_vf2 = xrfi_pipe(uv_v, alg='detrend_meanfilt', xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj,
                                label='Omnical visibility solutions, round 2.',
                                **check_kwargs)

    # Calculate metric on data file
    uvf_d2, uvf_df2 = xrfi_pipe(uv_d, alg='detrend_meanfilt', xants=[], Kt=kt_size, Kf=kf_size,
                                sig_init=sig_init, sig_adj=sig_adj,
                                label='Data, round 2.', **check_kwargs)

    # Get the non-detrended total chi-squared values, normalized across the full waterfall.
    uvf_chisq2, uvf_chisq_f2 = chi_sq_pipe(uvc_o, alg='zscore_full_array', modified=False,
                                           sig_init=sig_init, sig_adj=sig_adj,
                                           label='Renormalized chisq, round 2.',
                                           **check_kwargs)

    # Combine the metrics together
    uvf_metrics2 = uvf_d2.combine_metrics([uvf_og2, uvf_ox2, uvf_ag2, uvf_ax2,
                                           uvf_v2, uvf_d2, uvf_chisq2],
                                          method='quadmean', inplace=False)
    uvf_metrics2.label = 'Combined metrics, round 2.'
    alg_func = algorithm_dict['detrend_meanfilt']
    uvf_metrics2.metric_array[:, :, 0] = alg_func(uvf_metrics2.metric_array[:, :, 0],
                                                  flags=uvf_init.flag_array[:, :, 0],
                                                  Kt=kt_size, Kf=kf_size)

    # Flag on combined metrics
    uvf_f2 = flag(uvf_metrics2, nsig_p=sig_init, **check_kwargs)
    uvf_fws2 = watershed_flag(uvf_metrics2, uvf_f2, nsig_p=sig_adj,
                              inplace=False, **check_kwargs)
    uvf_fws2.label = 'Flags from combined metrics, round 2.'
    uvf_combined2 = (uvf_fws2 | uvf_ogf2 | uvf_oxf2 | uvf_agf2 | uvf_axf2
                     | uvf_vf2 | uvf_df2 | uvf_chisq_f2 | uvf_init)
    uvf_combined2.label = 'ORd flags, round 2.'

    # Write everything out
    uvf_dict = {'apriori_flags.h5': uvf_apriori,
                'v_metrics1.h5': uvf_v, 'v_flags1.h5': uvf_vf,
                'og_metrics1.h5': uvf_og, 'og_flags1.h5': uvf_ogf,
                'ox_metrics1.h5': uvf_ox, 'ox_flags1.h5': uvf_oxf,
                'ag_metrics1.h5': uvf_ag, 'ag_flags1.h5': uvf_agf,
                'ax_metrics1.h5': uvf_ax, 'ax_flags1.h5': uvf_axf,
                'chi_sq_renormed1.h5': uvf_chisq, 'chi_sq_flags1.h5': uvf_chisq_f,
                'combined_metrics1.h5': uvf_metrics, 'combined_flags1.h5': uvf_fws,
                'flags1.h5': uvf_init,
                'v_metrics2.h5': uvf_v2, 'v_flags2.h5': uvf_vf2,
                'og_metrics2.h5': uvf_og2, 'og_flags2.h5': uvf_ogf2,
                'ox_metrics2.h5': uvf_ox2, 'ox_flags2.h5': uvf_oxf2,
                'ag_metrics2.h5': uvf_ag2, 'ag_flags2.h5': uvf_agf2,
                'ax_metrics2.h5': uvf_ax2, 'ax_flags2.h5': uvf_axf2,
                'data_metrics2.h5': uvf_d2, 'data_flags2.h5': uvf_df2,
                'chi_sq_renormed2.h5': uvf_chisq2, 'chi_sq_flags2.h5': uvf_chisq_f2,
                'combined_metrics2.h5': uvf_metrics2, 'combined_flags2.h5': uvf_fws2,
                'flags2.h5': uvf_combined2}

    # Determine the actual files to store
    # We will drop kt_size / (integrations per file) files at the start and
    # end to avoid edge effects from the convolution kernel.
    # If this chunk includes the start or end of the night, we will write
    # output files for those, but flag everything.

    # Read metadata from first file to get integrations per file.
    uvtemp = UVData()
    uvtemp.read(data_files[0], read_data=False)
    nintegrations = len(data_files) * uvtemp.Ntimes
    # Calculate number of files to drop on edges, rounding up.
    ndrop = int(np.ceil(kt_size / uvtemp.Ntimes))
    # start_ind and end_ind are the indices in the file list to include
    start_ind = ndrop
    end_ind = len(data_files) - ndrop
    # If we're the first or last job, store all flags for the edge
    datadir = os.path.dirname(os.path.abspath(data_files[0]))
    bname = os.path.basename(data_files[0])
    # Because we don't necessarily know the filename structure, search for
    # files that are the same except different numbers (JDs)
    search_str = os.path.join(datadir, re.sub('[0-9]', '?', bname))
    all_files = sorted(glob.glob(search_str))
    if os.path.basename(data_files[0]) == os.path.basename(all_files[0]):
        # This is the first job, store the early edge.
        start_ind = 0
    if os.path.basename(data_files[-1]) == os.path.basename(all_files[-1]):
        # Last job, store the late edge.
        end_ind = len(data_files)

    # Loop through the files to output, storing all the different data products.
    for ind in range(start_ind, end_ind):
        dirname = resolve_xrfi_path(xrfi_path, data_files[ind], jd_subdir=True)
        basename = qm_utils.strip_extension(os.path.basename(data_files[ind]))
        for ext, uvf in uvf_dict.items():
            # This is calculated separately for each uvf because machine
            # precision error was leading to times not found in object.
            this_times = np.unique(uvf.time_array)
            t_ind = ind * uvtemp.Ntimes
            uvf_out = uvf.select(times=this_times[t_ind:(t_ind + uvtemp.Ntimes)],
                                 inplace=False)
            if (ext == 'flags2.h5') and ((ind <= ndrop) or (ind >= nintegrations - ndrop)):
                # Edge file, flag it completely.
                uvf_out.flag_array = np.ones_like(uvf_out.flag_array)
            outfile = '.'.join([basename, ext])
            outpath = os.path.join(dirname, outfile)
            uvf_out.history += history
            uvf_out.write(outpath, clobber=clobber)


def day_threshold_run(data_files, history, nsig_f=7., nsig_t=7.,
                      nsig_f_adj=3., nsig_t_adj=3., flag_abscal=True,
                      clobber=False, a_priori_flag_yaml=None,
                      run_check=True, check_extra=True,
                      run_check_acceptability=True):
    """Apply thresholding across all times/frequencies, using a full day of data.

    This function will write UVFlag files for each data input (omnical gains,
    omnical chisquared, abscal gains, etc.) for the full day. These files will be
    written in the same directory as the first data_file, and have filenames
    "zen.{JD}.{type}_threshold_flags.h5", where {type} describes the input data (e.g.
    "og" for omnical gains).
    This function will also copy the abscal calfits files but with flags defined
    by the union of all flags from xrfi_run and the day thresholding. These files
    will replace "abs" with "flagged_abs" in the filenames, and saved in the same
    directory as each abscal file.

    Parameters
    ----------
    data_files : list of strings
        Paths to the raw data files which have been used to calibrate and rfi flag so far.
    history : str
        The history string to include in files.
    nsig_f : float, optional
        The number of sigma above which to flag channels. Default is 7.0.
    nsig_t : float, optional
        The number of sigma above which to flag integrations. Default is 7.0.
    nsig_f_adj : float, optional
        The number of sigma above which to flag channels if they neighbor flagged channels.
        Default is 3.0.
    nsig_t_adj : float, optional
        The number of sigma above which to flag integrations if they neighbor flagged integrations.
        Default is 3.0.
    flag_abscal : bool, optional
        If True, generate new abscal solutions with day thresholded flags.
    clobber : bool, optional
        If True, overwrite existing files. Default is False.
    a_priori_flag_yaml : str, optional
        string specifying apriori flagging yaml.
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    None

    """
    history = 'Flagging command: "' + history + '", Using ' + __version__
    data_files = sorted(data_files)
    xrfi_dirs = [resolve_xrfi_path('', dfile, jd_subdir=True) for dfile in data_files]
    basename = '.'.join(os.path.basename(data_files[0]).split('.')[0:2])
    outdir = resolve_xrfi_path('', data_files[0])
    # Set up extensions to find the many files
    types = ['og', 'ox', 'ag', 'ax', 'v', 'cross', 'auto', 'omnical_chi_sq_renormed',
             'abscal_chi_sq_renormed', 'combined']
    mexts = ['og_metrics', 'ox_metrics', 'ag_metrics', 'ax_metrics',
             'v_metrics', 'cross_metrics', 'auto_metrics', 'omnical_chi_sq_renormed_metrics',
             'abscal_chi_sq_renormed_metrics', 'combined_metrics']
    # Read in the metrics objects
    filled_metrics = []
    for ext in mexts:
        # Fill in 2nd metrics with 1st metrics where 2nd are not available.
        files1_all = [glob.glob(d + '/*' + ext + '1.h5') for d in xrfi_dirs]
        files2_all = [glob.glob(d + '/*' + ext + '2.h5') for d in xrfi_dirs]
        # only consider flagging products that exist in all observations for thresholding.
        if np.all([len(f) > 0 for f in files1_all]) and np.all([len(f) > 0 for f in files2_all]):
            files1 = [glob.glob(d + '/*' + ext + '1.h5')[0] for d in xrfi_dirs]
            files2 = [glob.glob(d + '/*' + ext + '2.h5')[0] for d in xrfi_dirs]
            uvf1 = UVFlag(files1)
            uvf2 = UVFlag(files2)
            uvf2.metric_array = np.where(np.isinf(uvf2.metric_array), uvf1.metric_array,
                                         uvf2.metric_array)
            filled_metrics.append(uvf2)
        elif np.all([len(f) > 0 for f in files2_all]):
            # some flags only exist in round2 (data for example).
            files = [glob.glob(d + '/*' + ext + '2.h5')[0] for d in xrfi_dirs]
            filled_metrics.append(UVFlag(files))
        elif np.all([len(f) > 0 for f in files1_all]):
            # some flags only exist in round1 (if we chose median filtering only for example).
            files = [glob.glob(d + '/*' + ext + '1.h5')[0] for d in xrfi_dirs]
            filled_metrics.append(UVFlag(files))
        else:
            filled_metrics.append(None)
    filled_metrics_that_exist = [f for f in filled_metrics if f is not None]
    # Threshold each metric and save flag object
    uvf_total = filled_metrics_that_exist[0].copy()
    uvf_total.to_flag(run_check=run_check, check_extra=check_extra,
                      run_check_acceptability=run_check_acceptability)
    for i, uvf_m in enumerate(filled_metrics):
        if uvf_m is not None:
            uvf_f = threshold_wf(uvf_m, nsig_f=nsig_f, nsig_t=nsig_t,
                                 nsig_f_adj=nsig_f_adj, nsig_t_adj=nsig_t_adj,
                                 detrend=False, run_check=run_check,
                                 check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability)
            outfile = '.'.join([basename, types[i] + '_threshold_flags.h5'])
            outpath = os.path.join(outdir, outfile)
            uvf_f.write(outpath, clobber=clobber)
            uvf_total |= uvf_f

    # Read non thresholded flags and combine
    # Include round 1 and 2 flags for potential medain filter only.
    for rnd in [1, 2]:
        for mext in (mexts + ['flags']):
            try:
                ext_here = f'{mext.replace("metrics", "flags")}{rnd}.h5'
                files = [glob.glob(f'{d}/*.{ext_here}')[0] for d in xrfi_dirs]
                uvf_total |= UVFlag(files)
            except IndexError:
                pass

    outfile = '.'.join([basename, 'total_threshold_flags.h5'])
    outpath = os.path.join(outdir, outfile)
    uvf_total.write(outpath, clobber=clobber)
    if a_priori_flag_yaml is not None:
        uvf_total = qm_utils.apply_yaml_flags(uvf_total, a_priori_flag_yaml)
        outfile = '.'.join([basename, 'total_threshold_and_a_priori_flags.h5'])
        outpath = os.path.join(outdir, outfile)
        uvf_total.write(outpath, clobber=clobber)
    if flag_abscal:
        # Apply to abs calfits
        uvc_a = UVCal()
        incal_ext = 'abs'
        outcal_ext = 'flagged_abs'
        for dfile in data_files:
            basename = qm_utils.strip_extension(dfile)
            abs_in = '.'.join([basename, incal_ext, 'calfits'])
            abs_out = '.'.join([basename, outcal_ext, 'calfits'])
            # abscal flagging only happens if the abscal files exist.
            uvc_a.read_calfits(abs_in)

            # select the times from the file we are going to flag
            uvf_file = uvf_total.select(times=uvc_a.time_array, inplace=False)

            flag_apply(uvf_file, uvc_a, force_pol=True, history=history,
                       run_check=run_check, check_extra=check_extra,
                       run_check_acceptability=run_check_acceptability)
            uvc_a.write_calfits(abs_out, clobber=clobber)


def xrfi_h1c_run(indata, history, infile_format='miriad', extension='flags.h5',
                 summary=False, summary_ext='flag_summary.h5', xrfi_path='',
                 model_file=None, model_file_format='uvfits',
                 calfits_file=None, kt_size=8, kf_size=8, sig_init=6.0, sig_adj=2.0,
                 px_threshold=0.2, freq_threshold=0.5, time_threshold=0.05,
                 ex_ants=None, metrics_file=None, filename=None, run_check=True,
                 check_extra=True, run_check_acceptability=True):
    """Run the RFI-flagging algorithm from H1C and store results in npz files.

    This function runs on a single data file, and optionally calibration files, and
    writes the results to npz files.

    Notes
    -----
    This function will take in a UVData object or data file and optionally a cal file and
    model visibility file, and run an RFI-flagging algorithm to identify contaminated
    observations. Each set of flagging will be stored, as well as compressed versions.

    Parameters
    ----------
    indata : UVData or str
        A UVData object or data file on which to run RFI flagging.
    history : str
        The history string to include in files.
    infile_format : str, optional
        The file format for input files. Not currently used because we use pyuvdata's
        generic read function, but will be implemented for partial io.
    extension : str, optional
        The extension to be appended to input file name. Default is "flags.h5".
    summary : bool, optional
        If True, compute a summary of RFI flags and store in a .h5 file.
        Default is False.
    summary_ext : str, optional
        The extension for the summary file. Default is "flag_summary.h5".
    xrfi_path : str, optional
        The path to save flag files to. Default is the same directory as input file.
    model_file : str, optional
        The model visibility file to flag on. This step is skipped if not specified.
    model_file_format : str, optional
        The file format for input model file. Not currently used because we use
        pyuvdata's generic read function, but will be implemented for partial io.
    calfits_file : str, optional
        The calfits file to use to flag on gains and/or chisquared values.
    kt_size : int, optional
        The size of the kernel in time dimension for detrend in xrfi algorithm.
        Default is 8.
    kf_size : int, optional
        The size of the kernel in frequency dimension for detrend in xrfi algorithm.
        Default is 8.
    sig_init : float, optional
        The starting number of sigmas to flag on. Default is 6.0.
    sig_adj : float, optional
        The number of sigmas to flag on for data adjacent to a flag. Default is 2.0.
    px_threshold : float, optional
        The fraction of flags required to trigger a broadcast across baselines for
        a given (time, frequency) pixel. Default is 0.2.
    freq_threshold : float, optional
        The fraction of channels required to trigger a broadcast across frequency
        (for a single time). Default is 0.5.
    time_threshold : float, optional
        The fraction of times required to trigger a broadcast across time
        (for a single frequency). Default is 0.05.
    ex_ants : str, optional
        A comma-separated list of antennas to exclude. Flags of visibilities
        formed with these antennas will be set to True.
    metrics_file : str, optional
        A metrics file that contains a list of excluded antennas. Flags of
        visibilities formed with these antennas will be set to True.
    filename : str, optional
        The file for which to flag RFI (only one file allowed).
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    None

    Raises
    ------
    AssertionError:
        If "indata", "model_file", and "calfits_file" are all not provided,
        or if the filename for the input UVData object is not provided,
        an AssertionError is raised.

    ValueError:
        If filename is not a string, or if there is a mis-match in the time
        or frequency axes between the UVData object and the model visibility/
        calibration solution file, a ValueError is raised.

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
    history = 'Flagging command: "' + history + '", Using ' + __version__

    # Flag on data
    if indata is not None:
        # Flag visibilities corresponding to specified antennas
        xants = process_ex_ants(ex_ants=ex_ants, metrics_files=metrics_file)
        flag_xants(uvd, xants, run_check=run_check, check_extra=check_extra,
                   run_check_acceptability=run_check_acceptability)
        uvf_f, uvf_wf, uvf_w = xrfi_h1c_pipe(uvd, Kt=kt_size, Kf=kf_size, sig_init=sig_init,
                                             sig_adj=sig_adj, px_threshold=px_threshold,
                                             freq_threshold=freq_threshold, time_threshold=time_threshold,
                                             return_summary=True, run_check=run_check,
                                             check_extra=check_extra,
                                             run_check_acceptability=run_check_acceptability)
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
                                      freq_threshold=freq_threshold, time_threshold=time_threshold,
                                      run_check=run_check,
                                      check_extra=check_extra,
                                      run_check_acceptability=run_check_acceptability)
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
                                      freq_threshold=freq_threshold, time_threshold=time_threshold,
                                      run_check=run_check,
                                      check_extra=check_extra,
                                      run_check_acceptability=run_check_acceptability)
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
                                      cal_mode='chisq', run_check=run_check,
                                      check_extra=check_extra,
                                      run_check_acceptability=run_check_acceptability)
        outfile = '.'.join([basename, 'x', extension])
        outpath = os.path.join(dirname, outfile)
        uvf_wf.history += history
        uvf_wf.write(outpath)

    return


def xrfi_h1c_apply(filename, history, infile_format='miriad', xrfi_path='',
                   outfile_format='miriad', extension='R', overwrite=False,
                   flag_file=None, waterfalls=None, output_uvflag=True,
                   output_uvflag_ext='flags.h5', run_check=True,
                   check_extra=True, run_check_acceptability=True):
    """Apply flags in the fashion of H1C.

    Read in a flag array and optionally several waterfall flags, and insert into
    a data file.

    Parameters
    ----------
    filename : str
        Data file in which update flag array.
    history : str
        The history string to include in files.
    infile_format : str, optional
        File format for input files. Not currently used because we use pyuvdata's
        generic read function, but will be implemented for partial io.
    xrfi_path : str, optional
        The path to save output to. Default is same directory as input file.
    outfile_format : {"miriad", "uvfits", "uvh5"}, optional
        The file format for output files. Default is "miriad".
    extension : str, optional
        The extension to be appended to input file name. Default is "R".
    overwrite : bool, optional
        If True, overwrite the output file if it already exists. Default is False.
    flag_file : str, optional
        The path to the npz file containing full flag array to insert into data file.
    waterfalls, optional
        A list or comma separated string of npz file names containing waterfalls
        of flags to broadcast to full flag array and union with flag array in flag_file.
    output_uvflag : bool, optional
        If True, save a UVFlag file with the final flag array. The flag array will
        be identical to what is stored in the data.
    output_uvflag_ext : str, optional
        The extension to be appended to input file name. Default is "flags.h5".
    run_check : bool
        Option to check for the existence and proper shapes of parameters
        on UVFlag Object.
    check_extra : bool
        Option to check optional parameters as well as required ones.
    run_check_acceptability : bool
        Option to check acceptable range of the values of parameters
        on UVFlag Object.

    Returns
    -------
    None

    Raises
    ------
    AssertionError:
        If no input filename is provided, an AssertionError is raised.

    ValueError:
        If outfile_format is not valid, or if the target output file exists and
        overwrite is False, a ValueError is raised.

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

    uvf = flag_apply(full_list, uvd, force_pol=True, return_net_flags=True,
                     run_check=run_check, check_extra=check_extra,
                     run_check_acceptability=run_check_acceptability)

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
