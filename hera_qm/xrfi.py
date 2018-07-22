from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from .uvflag import UVFlag
from hera_qm import utils as qm_utils
from hera_qm.version import hera_qm_version_str
import warnings


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
    if not isinstance(uv, (UVData, UVCal, UVFlag)):
        raise ValueError('First argument to flag_xants must be a UVData, UVCal, '
                         ' or UVFlag object.')
    if isinstance(uv, UVFlag) and uv.type == 'wf':
        raise ValueError('Cannot flag antennas on UVFlag obejct of type "wf".')

    if not inplace:
        if isinstance(uv, UVFlag):
            uvo = copy.deepcopy(uv).to_flag()
        else:
            uvo = UVFlag(uv, mode='flag')
    else:
        uvo = uv

    if isinstance(uvo, UVData) or (isinstance(uvo, UVFlag) and uvo.type == 'baseline'):
        all_ants = np.unique(np.append(uvo.ant_1_array, uvo.ant_2_array))
        for ant in all_ants:
            for xant in xants:
                blts = uvo.antpair2ind(ant, xant)
                uvo.flag_array[blts, :, :, :] = True
                blts = uvo.antpair2ind(xant, ant)
                uvo.flag_array[blts, :, :, :] = True
    elif isinstance(uvo, UVCal) or (isinstance(uvo, UVFlag) and uvo.type == 'antenna'):
        for xant in xants:
            ai = np.where(uvo.ant_array == xant)
            uvo.flag_array[ai, :, :, :, :] = True
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
    '''
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
    if Kt > d.shape[0] or Kf > d.shape[1]:
        raise AssertionError('Kernel size exceeds data.')
    d_sm = np.empty_like(d)
    for i in xrange(d.shape[0]):
        for j in xrange(d.shape[1]):
            i0, j0 = max(0, i - Kt), max(0, j - Kf)
            i1, j1 = min(d.shape[0], i + Kt), min(d.shape[1], j + Kf)
            d_sm[i, j] = medmin(d[i0:i1, j0:j1])
    return d_sm


def detrend_deriv(d, dt=True, df=True):
    ''' Detrend array by taking the derivative in either time, frequency
        or both.
    Args:
        d (array): 2D data array of the shape (time,frequency).
        dt (bool, optional): derivative across time bins.
        df (bool, optional): derivative across frequency bins.
    Returns:
        array: detrended array with same shape as input array.
    '''

    if df:
        d_df = np.empty_like(d)
        d_df[:, 1:-1] = (d[:, 1:-1] - .5 * (d[:, :-2] + d[:, 2:])) / np.sqrt(1.5)
        d_df[:, 0] = (d[:, 0] - d[:, 1]) / np.sqrt(2)
        d_df[:, -1] = (d[:, -1] - d[:, -2]) / np.sqrt(2)
    else:
        d_df = d
    if dt:
        d_dt = np.empty_like(d_df)
        d_dt[1:-1] = (d_df[1:-1] - .5 * (d_df[:-2] + d_df[2:])) / np.sqrt(1.5)
        d_dt[0] = (d_df[0] - d_df[1]) / np.sqrt(2)
        d_dt[-1] = (d_df[-1] - d_df[-2]) / np.sqrt(2)
    else:
        d_dt = d
    d2 = np.abs(d_dt)**2
    # model sig as separable function of 2 axes
    sig_f = np.median(d2, axis=0)
    sig_f.shape = (1, -1)
    sig_t = np.median(d2, axis=1)
    sig_t.shape = (-1, 1)
    sig = np.sqrt(sig_f * sig_t / np.median(sig_t))
    # don't divide by zero, instead turn those entries into +inf
    f = np.true_divide(d_dt, sig, where=(np.abs(sig) > 1e-7))
    f = np.where(np.abs(sig) > 1e-7, f, np.inf)
    return f


def detrend_medminfilt(d, Kt=8, Kf=8):
    """Detrend array using medminfilt statistic. See medminfilt.
    Args:
        d (array): data array of the shape (time, frequency) to detrend
        Kt (int): size in time to apply medminfilter over
        Kf (int): size in frequency to apply medminfilter over
    Returns:
        float array: float array of outlier significance metric
    """
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
        d (array): data array to detrend.
        K (int, optional): box size to apply medminfilt over
    Returns:
        f: array of outlier significance metric. Same type and size as d.
    """
    # Delay import so scipy is not required for any use of hera_qm
    from scipy.signal import medfilt2d

    if Kt > d.shape[0] or Kf > d.shape[1]:
        raise AssertionError('Kernel size exceeds data.')
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
    f = np.true_divide(d_rs, sig, where=(np.abs(sig) > 1e-7))
    f = np.where(np.abs(sig) > 1e-7, f, np.inf)
    return f[Kt:-Kt, Kf:-Kf]


# Update algorithm_dict whenever new metric algorithm is created.
algorithm_dict = {'medmin': medmin, 'medminfilt': medminfilt, 'detrend_deriv': detrend_deriv,
                  'detrend_medminfilt': detrend_medminfilt, 'detrend_medfilt': detrend_medfilt}

#############################################################################
# RFI flagging algorithms
#############################################################################

def watershed_flag(uvf_m, uvf_f, nsig_p=2., nsig_f=2., nsig_t=2., avg_method='quadmean',
                   inplace=True):
    '''Expands a set of flags using a watershed algorithm.
    Uses a UVFlag object in 'metric' mode (i.e. how many sigma the data point is
    from the center) and a set of flags to grow the flags using defined thresholds.

    Args:
        uvf_m: UVFlag object in 'metric' mode
        uvf_f: UVFlag object in 'flag' mode
        nsig_p: Number of sigma above which to flag pixels which are near
               previously flagged pixels. Default is 2.0.
        TODO: option to skip 1D watersheds by setting to None
        nsig_f: Number of sigma above which to flag channels which are near
               fully flagged channels. Default is 2.0.
        nsig_t: Number of sigma above which to flag integrations which are near
               fully flagged integrations. Default is 2.0.
        avg_method: Method to average metric data for frequency and time watershedding.
                    Options are 'mean', 'absmean', and 'quadmean' (Default).
        inplace: Whether to update uvf_f or create a new flag object. Default is True.

    Returns:
        uvf: UVFlag object in 'flag' mode with flags after watershed.
    '''
    # Check inputs
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode == 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')
    if (not isinstance(uvf_f, UVFlag)) or (uvf_f.mode == 'flag'):
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
        for b in np.unique(uvf.baseline_array):
            i = np.where(uvf.baseline_array == b)
            for pi in range(uvf.polarization_array.size):
                farr[i, 0, :, pi] += _ws_flag_wf(marr[i, 0, :, pi],
                                                 farr[i, 0, :, pi], nsig_p)
        # Channel watershed
        d = avg_f(marr, axis=(0, 1, 3), weights=warr)
        f = np.all(farr, axis=(0, 1, 3))
        farr[:, :, :, :] += _ws_flag_wf(d, f, nsig_f).reshape(1, 1, -1, 1)
        # Time watershed
        ts = np.unique(uvf.time_array)
        d = np.zeros(ts.size)
        f = np.zeros(ts.size, dtype=np.bool)
        for i, t in enumerate(ts):
            d[i] = avg_f(marr[uvf.time_array == t, 0, :, :],
                         weights=warr[uvf.time_array == t, 0, :, :])
            f[i] = np.all(farr[uvf.time_array == t, 0, :, :])
        f = _ws_flag_wf(d, f, nsig_t)
        for i, t in enumerate(ts):
            farr[uvf.time_array == t, :, :, :] += f[i]
    elif uvf_m.type == 'antenna':
        # Pixel watershed
        for ai in range(uvf.ant_array.size):
            for pi in range(uvf.polarization_array.size):
                farr[ai, 0, :, :, pi] += _ws_flag_wf(marr[ai, 0, :, :, pi].T,
                                                     farr[ai, 0, :, :, pi].T, nsig_p).T
        # Channel watershed
        d = avg_f(marr, axis=(0, 1, 3, 4), weights=warr)
        f = np.all(farr, axis=(0, 1, 3, 4))
        farr[:, :, :, :, :] += _ws_flag_wf(d, f, nsig_f).reshape(1, 1, -1, 1, 1)
        # Time watershed
        d = avg_f(marr, axis=(0, 1, 2, 4), weights=warr)
        f = np.all(farr, axis=(0, 1, 2, 4))
        farr[:, :, :, :, :] += _ws_flag_wf(d, f, nsig_t).reshape(1, 1, 1, -1, 1)
    elif uvf_m.type == 'wf':
        # Pixel watershed
        for pi in range(uvf.polarization_array.size):
            farr[:, :, pi] += _ws_flag_wf(marr[:, :, pi], farr[:, :, pi], nsig_p)
        # Channel watershed
        d = avg_f(marr, axis=(0, 2), weights=warr)
        f = np.app(farr, axis=(0, 2))
        farr[:, :, :] += _ws_flag_wf(d, f, nsig_f).reshape(1, -1, 1)
        # Time watershed
        d = avg_f(marr, axis=(1, 2), weights=warr)
        f = np.all(farr, axis=(1, 2))
        farr[:, :, :] += _ws_flag_wf(d, f, nsig_t).reshape(-1, 1, 1)
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf


def _ws_flag_wf(d, fin, nsig=2.):
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
                         + ' and ' + str(f.shape))
    f = copy.deepcopy(fin)
    # There may be an elegant way to combine these... for the future.
    if d.ndim == 1:
        prevn = 0
        x = np.where(f)[0]
        while x.size != prevn:
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
                f1.mask[xp[i], yp[i]] = 1
                x, y = np.where(f)
    else:
        raise ValueError('Data must be 1D or 2D.')
    return f


def flag(uvf_m, nsig_p=6., nsig_f=3., nsig_t=3., avg_method='quadmean'):
    '''Creates a set of flags based on a "metric" type UVFlag object.
    Args:
        uvf_m: UVFlag object in 'metric' mode (ie. number of sigma data is from middle)
        nsig_p: Number of sigma above which to flag pixels. Default is 6.0.
        TODO: option to skip 1D flagging by setting to None
        nsig_f: Number of sigma above which to flag channels. Default is 3.0.
        nsig_t: Number of sigma above which to flag integrations. Default is 3.0.
        avg_method: Method to average metric data for frequency and time flagging.
                    Options are 'mean', 'absmean', and 'quadmean' (Default).

    Returns:
        uvf_f: UVFlag object in 'flag' mode with flags determined from uvm.
    '''
    # Check input
    if (not isinstance(uvf_m, UVFlag)) or (uvf_m.mode == 'metric'):
        raise ValueError('uvf_m must be UVFlag instance with mode == "metric."')

    try:
        avg_f = qm_utils.averaging_dict[avg_method]
    except KeyError:
        raise KeyError('avg_method must be one of: "mean", "absmean", or "quadmean".')

    # initialize
    uvf_f = copy.deepcopy(uvf_m).to_flag()

    # Pixel flagging
    uvf_f.flag_array[uvf_m.metric_array > nsig_p] = True

    if uvf_m.type == 'baseline':
        # Channel flagging
        d = avg_f(uvf_m.metric_array, axis=(0, 1, 3), weights=uvf_m.weights_array)
        indf = np.where(d > nsig_f)[0]
        uvf_f.flag_array[:, :, indf, :] = True
        # Time flagging
        ts = np.unique(uvf_m.time_array)
        d = np.zeros(ts.size)
        for i, t in enumerate(ts):
            d[i] = avg_f(marr[uvf.time_array == t, 0, :, :],
                         weights=warr[uvf.time_array == t, 0, :, :])
        indf = np.where(d > nsig_t)[0]
        for t in ts[indf]:
            uvf_f.flag_array[uvf.time_array == t, :, :, :] = True
    elif uvf_m.type == 'antenna':
        # Channel flag
        d = avg_f(uvf_m.metric_array, axis=(0, 1, 3, 4), weights=warr)
        indf = np.where(d > nsig_f)[0]
        uvf_f.flag_array[:, :, indf, :, :] = True
        # Time watershed
        d = avg_f(uvf_m.metric_array, axis=(0, 1, 2, 4), weights=warr)
        indt = np.where(d > nsig_t)[0]
        uvf_f.flag_array[:, :, :, indt, :] = True
    elif uvf_m.type == 'wf':
        # Channel flag
        d = avg_f(uvf_m.metric_array, axis=(0, 2), weights=warr)
        indf = np.where(d > nsig_f)[0]
        uvf_f.flag_array[:, indf, :] = True
        # Time watershed
        d = avg_f(uvf_m.metric_array, axis=(1, 2), weights=warr)
        indt = np.where(d > nsig_t)[0]
        uvf_f.flag_array[indt, :, :] = True
    else:
        raise ValueError('Unknown UVFlag type: ' + uvf_m.type)
    return uvf_f


def xrfi_simple(d, f=None, nsig_df=6, nsig_dt=6, nsig_all=0):
    '''Flag RFI using derivatives in time and frequency.
    Args:
        d (array): 2D data array of the shape (time, frequency) to flag
        f (array, optional): input flags, defaults to zeros
        nsig_df (float, optional): number of sigma above median to flag in frequency direction
        nsig_dt (float, optional): number of sigma above median to flag in time direction
        nsig_all (float, optional): overall flag above some sigma. Skip if 0.
    Returns:
        bool array: mask array for flagging.
    '''
    if f is None:
        f = np.zeros(d.shape, dtype=np.bool)
    if nsig_df > 0:
        d_df = d[:, 1:-1] - .5 * (d[:, :-2] + d[:, 2:])
        d_df2 = np.abs(d_df)**2
        sig2 = np.median(d_df2, axis=1)
        sig2.shape = (-1, 1)
        f[:, 0] = 1
        f[:, -1] = 1
        f[:, 1:-1] = np.where(d_df2 / sig2 > nsig_df**2, 1, f[:, 1:-1])
    if nsig_dt > 0:
        d_dt = d[1:-1, :] - .5 * (d[:-2, :] + d[2:, :])
        d_dt2 = np.abs(d_dt)**2
        sig2 = np.median(d_dt2, axis=0)
        sig2.shape = (1, -1)
        f[0, :] = 1
        f[-1, :] = 1
        f[1:-1, :] = np.where(d_dt2 / sig2 > nsig_dt**2, 1, f[1:-1, :])
    if nsig_all > 0:
        ad = np.abs(d)
        med = np.median(ad)
        sig = np.sqrt(np.median(np.abs(ad - med)**2))
        f = np.where(ad > med + nsig_all * sig, 1, f)
    return f


def xrfi_h1c(d, f=None, Kt=8, Kf=8, sig_init=6, sig_adj=2):
    """xrfi excision algorithm we used for H1C. Uses detrending and watershed algorithms above.
    Args:
        d (array): 2D of data array.
        f (array, optional): input flag array.
        Kt (int, optional): time size for detrending box.
        Kf (int, optional): frequency size for detrending box/
        sig_init (float, optional): initial sigma to flag.
        sig_adj (float, optional): number of sigma to flag adjacent to flagged data (sig_init)

    Returns:
        bool array: array of flags
    """
    try:
        nsig = detrend_medfilt(d, Kt=Kt, Kf=Kf)
        f = watershed_flag(np.abs(nsig), f=f, sig_init=sig_init, sig_adj=sig_adj)
    except AssertionError:
        warnings.warn('Kernel size exceeds data. Flagging all data.')
        f = np.ones_like(d, dtype=np.bool)
    return f


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
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError('uv must be a UVData or UVCal object.')
    try:
        mfunc = algorithm_dict[algorithm]
    except KeyError:
        raise KeyError('Algorithm not found in list of available functions.')
    uvf = UVFlag(uv)
    uvf.weights_array = uv.nsample_array * np.logical_not(uv.flag_array).astype(np.float)
    if isinstance(uv, UVData):
        for key, d in uv.antpairpol_iter():
            ind1, ind2, pol = uv._key2inds(key)
            for ind, ipol in zip((ind1, ind2), pol):
                if len(ind) == 0:
                    continue
                f = uv.flag_array[ind, 0, :, ipol]
                uvf.flag_array[ind, 0, :, ipol] = mfunc(np.abs(d), f=f, **kwargs)
    elif isinstance(uv, UVCal):
        for ai in range(uv.Nants_data):
            for pi in range(uv.Njones):
                # Note transposes are due to freq, time dimensions rather than the
                # expected time, freq
                f = uv.flag_array[ai, 0, :, :, pi].T
                if gains:
                    d = np.abs(uv.gain_array[ai, 0, :, :, pi].T)
                elif chisq:
                    d = np.abs(uv.quality_array[ai, 0, :, :, pi].T)
                else:
                    raise ValueError('When calculating metric for UVCal object, '
                                     'gains or chisq must be set to True.')
                uvf.flag_array[ai, 0, :, :, pi] = mfunc(d, f=f, **kwargs).T
    return uvf


def xrfi_run(indata, args, history):
    """
    Run an RFI-flagging algorithm on a single data file, and optionally calibration files,
    and store results in npz files.

    Args:
       indata -- Either UVData object or data file to run RFI flagging on.
       args -- parsed arguments via argparse.ArgumentParser.parse_args
       history -- history string to include in files
    Return:
       None

    This function will take in a UVData object or  data file and optionally a cal file and
    model visibility file, and run an RFI-flagging algorithm to identify contaminated
    observations. Each set of flagging will be stored, as well as compressed versions.
    """
    if indata is None:
        if (args.model_file is None) and (args.calfits_file is None):
            raise AssertionError('Must provide at least one of: filename, '
                                 'model_file, or calfits_file.')
        warnings.warn('indata is none, not flagging on any data visibilities.')
    elif isinstance(indata, UVData):
        uvd = indata
        if len(args.filename) == 0:
            raise AssertionError('Please provide a filename to go with UVData object. '
                                 'The filename is used in conjunction with "extension" '
                                 'to determine the output filename.')
        else:
            if isinstance(args.filename, str):
                filename = args.filename
            else:
                filename = args.filename[0]
    else:
        # make sure we were given files to process
        if len(indata) == 0:
            if (args.model_file is None) and (args.calfits_file is None):
                raise AssertionError('Must provide at least one of: filename, '
                                     'model_file, or calfits_file.')
            indata = None
            warnings.warn('indata is none, not flagging on any data visibilities.')
        elif len(indata) > 1:
            raise AssertionError('xrfi_run currently only takes a single data file.')
        else:
            filename = indata[0]
            uvd = UVData()
            if args.infile_format == 'miriad':
                uvd.read_miriad(filename)
            elif args.infile_format == 'uvfits':
                uvd.read_uvfits(filename)
            elif args.infile_format == 'fhd':
                uvd.read_fhd(filename)
            else:
                raise ValueError('Unrecognized input file format ' + str(args.infile_format))

    # Compute list of excluded antennas
    if args.ex_ants != '' or args.metrics_json != '':
        # import function from hera_cal
        from hera_cal.omni import process_ex_ants
        xants = process_ex_ants(args.ex_ants, args.metrics_json)

        # Flag the visibilities corresponding to the specified antennas
        uvd = flag_xants(uvd, xants)

    # Flag on full data set
    if indata is not None:
        d_flag_array = vis_flag(uvd, args)

        # Make a "normalized waterfall" to account for data already flagged in file
        d_wf_tot = qm_utils.flags2waterfall(uvd, flag_array=d_flag_array)
        d_wf_prior = qm_utils.flags2waterfall(uvd, flag_array=uvd.flag_array)
        d_wf_norm = normalize_wf(d_wf_tot, d_wf_prior)
        d_wf_t = threshold_flags(d_wf_norm, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)

    # Flag on model visibilities
    if args.model_file is not None:
        uvm = UVData()
        if args.model_file_format == 'miriad':
            uvm.read_miriad(args.model_file)
        elif args.model_file_format == 'uvfits':
            uvm.read_uvfits(args.model_file)
        elif args.model_file_format == 'fhd':
            uvm.read_fhd(args.model_file)
        else:
            raise ValueError('Unrecognized input file format ' + str(args.model_file_format))
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvm.time_array), atol=1e-5, rtol=0) and
                    np.allclose(uvd.freq_array, uvm.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of model vis file must match'
                                 'the data file.')
        m_flag_array = vis_flag(uvm, args)
        m_waterfall = qm_utils.flags2waterfall(uvm, flag_array=m_flag_array)
        m_wf_prior = qm_utils.flags2waterfall(uvm)
        m_wf_norm = normalize_wf(m_waterfall, m_wf_prior)
        m_wf_t = threshold_flags(m_wf_norm, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)

    # Flag on gain solutions and chisquared values
    if args.calfits_file is not None:
        uvc = UVCal()
        uvc.read_calfits(args.calfits_file)
        if indata is not None:
            if not (np.allclose(np.unique(uvd.time_array), np.unique(uvc.time_array), atol=1e-5, rtol=0) and
                    np.allclose(uvd.freq_array, uvc.freq_array, atol=1., rtol=0)):
                raise ValueError('Time and frequency axes of calfits file must match'
                                 'the data file.')
        g_flag_array, x_flag_array = cal_flag(uvc, args)
        g_waterfall = qm_utils.flags2waterfall(uvc, flag_array=g_flag_array)
        x_waterfall = qm_utils.flags2waterfall(uvc, flag_array=x_flag_array)
        c_wf_prior = qm_utils.flags2waterfall(uvc)
        g_wf_norm = normalize_wf(g_waterfall, c_wf_prior)
        x_wf_norm = normalize_wf(x_waterfall, c_wf_prior)
        g_wf_t = threshold_flags(g_wf_norm, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)
        x_wf_t = threshold_flags(x_wf_norm, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)

    # append to history
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str

    # save output when we're done
    if args.xrfi_path != '':
        # If explicitly given output path, use it. Otherwise use path from data.
        dirname = args.xrfi_path
    if indata is not None:
        if args.xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(filename))
        basename = os.path.basename(filename)
        outfile = ''.join([basename, args.extension])
        outpath = os.path.join(dirname, outfile)
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        np.savez(outpath, flag_array=d_flag_array, waterfall=d_wf_t, baseline_array=uvd.baseline_array,
                 antpairs=uvd.get_antpairs(), polarization_array=uvd.polarization_array, freq_array=uvd.freq_array,
                 time_array=uvd.time_array, lst_array=uvd.lst_array, antpos=antpos, ants=ants, history=history)
        if (args.summary):
            sum_file = ''.join([basename, args.summary_ext])
            sum_path = os.path.join(dirname, sum_file)
            # Summarize using one of the raw flag arrays
            summarize_flags(uvd, sum_path, flag_array=d_flag_array)
    if args.model_file is not None:
        if args.xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(args.model_file))
        outfile = ''.join([os.path.basename(args.model_file), args.extension])
        outpath = os.path.join(dirname, outfile)
        antpos, ants = uvm.get_ENU_antpos(center=True, pick_data_ants=True)
        np.savez(outpath, flag_array=m_flag_array, waterfall=m_wf_t, baseline_array=uvm.baseline_array,
                 antpairs=uvm.get_antpairs(), polarization_array=uvm.polarization_array, freq_array=uvm.freq_array,
                 time_array=uvm.time_array, lst_array=uvm.lst_array, antpos=antpos, ants=ants, history=history)
    if args.calfits_file is not None:
        # Save flags from gains and chisquareds in separate files
        if args.xrfi_path == '':
            dirname = os.path.dirname(os.path.abspath(args.calfits_file))
        outfile = ''.join([os.path.basename(args.calfits_file), '.g', args.extension])
        outpath = os.path.join(dirname, outfile)
        np.savez(outpath, flag_array=g_flag_array, waterfall=g_wf_t, ants=uvc.ant_array,
                 jones_array=uvc.jones_array, freq_array=uvc.freq_array,
                 time_array=uvc.time_array, history=history)
        outfile = ''.join([os.path.basename(args.calfits_file), '.x', args.extension])
        outpath = os.path.join(dirname, outfile)
        np.savez(outpath, flag_array=x_flag_array, waterfall=x_wf_t, ants=uvc.ant_array,
                 jones_array=uvc.jones_array, freq_array=uvc.freq_array,
                 time_array=uvc.time_array, history=history)

    return


def waterfall2flags(waterfall, uv):
    """
    Broadcasts a 2D waterfall of dimensions (Ntimes, Nfreqs) to a full flag array,
    defined by a UVData or UVCal object.
    Args:
        waterfall -- 2D waterfall of flags of size (Ntimes, Nfreqs).
        uv -- A UVData or UVCal object which defines the times and frequencies.
    Returns:
        flag_array -- Flag array of dimensions defined by uv which copies the values
                      of waterfall across the extra dimensions (baselines/antennas, pols)
    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError('waterfall2flags() requires a UVData or UVCal object as '
                         'the second argument.')
    if waterfall.shape != (uv.Ntimes, uv.Nfreqs):
        raise ValueError('Waterfall dimensions do not match data. Cannot broadcast.')
    if isinstance(uv, UVCal):
        flag_array = np.tile(waterfall.T[np.newaxis, np.newaxis, :, :, np.newaxis],
                             (uv.Nants_data, uv.Nspws, 1, 1, uv.Njones))
    elif isinstance(uv, UVData):
        flag_array = np.zeros_like(uv.flag_array)
        for i, t in enumerate(np.unique(uv.time_array)):
            flag_array[uv.time_array == t, :, :, :] = waterfall[i, np.newaxis, :, np.newaxis]

    return flag_array


def normalize_wf(wf, wfp):
    """ Normalize waterfall to account for data already flagged.
    Args:
        wf -- Waterfall of fractional flags.
        wfp -- Waterfall of prior fractional flags. Size must match wf.
    Returns:
        wf_norm -- Waterfall of fractional flags which were not flagged prior.
                   Note if wfp[i, j] == 1, then wf_norm[i, j] will be NaN.
    """
    if wf.shape != wfp.shape:
        raise AssertionError('waterfall and prior waterfall must be same shape.')
    ind = np.where(wfp < 1)
    wf_norm = np.nan * np.ones_like(wf)
    wf_norm[ind] = (wf[ind] - wfp[ind]) / (1. - wfp[ind])
    return wf_norm


def threshold_flags(wf, px_threshold=0.2, freq_threshold=0.5, time_threshold=0.05):
    """ Threshold flag waterfall at each pixel, as well as averages across time and frequency
    Args:
        wf -- Waterfall of fractional flags. Size (Ntimes, Nfreqs)
        px_threshold -- Fraction of flags required to threshold the pixel. Default is 0.2.
        freq_threshold -- Fraction of channels required to flag all channels at a
                          single time. Default is 0.5.
        time_threshold -- Fraction of times required to flag all times at a
                          single frequency channel. Default is 0.05.

    Return:
        wf_t -- thresholded waterfall. Boolean array, same shape as wf.
    """
    wf_t = np.zeros(wf.shape, dtype=bool)
    with warnings.catch_warnings():
        # Ignore empty slice warning which occurs for entire rows/columns of nan
        warnings.filterwarnings('ignore', message='Mean of empty slice',
                                category=RuntimeWarning)
        spec = np.nanmean(wf, axis=0)
        spec[np.isnan(spec)] = 1
        tseries = np.nanmean(wf, axis=1)
        tseries[np.isnan(tseries)] = 1
    wf_t[:, spec > time_threshold] = True
    wf_t[tseries > freq_threshold, :] = True
    with warnings.catch_warnings():
        # Explicitly handle nans in wf
        warnings.filterwarnings('ignore', message='invalid value encountered in greater',
                                category=RuntimeWarning)
        wf_t[wf > px_threshold] = True
        wf_t[np.isnan(wf)] = True  # Flag anything that was completely flagged in prior.
    return wf_t


def summarize_flags(uv, outfile, flag_array=None, prior_flags=None):
    """ Collapse several dimensions of a UVData flag array to summarize.
    Args:
        uv -- UVData object containing flag_array to be summarized
        outfile -- filename for output npz file
        flag_array -- (optional) use alternative flag_array (rather than uv.flag_array).
        prior_flags -- (optional) exclude prior flag array when calculating averages.

    Return:
        Writes an npz file to disk containing summary info. The keys are:
            waterfall - ndarray (Ntimes, Nfreqs, Npols) of rfi flags, averaged
                        over all baselines in the file.
            tmax - ndarray (Nfreqs, Npols) of max rfi fraction along time dimension.
            tmin - ndarray (Nfreqs, Npols) of min rfi fraction along time dimension.
            tmean - ndarray (Nfreqs, Npols) of average rfi fraction along time dimension.
            tstd - ndarray (Nfreqs, Npols) of standard deviation rfi fraction
                   along time dimension.
            tmedian - ndarray (Nfreqs, Npols) of median rfi fraction along time dimension.
            fmax - ndarray (Ntimes, Npols) of max rfi fraction along freq dimension.
            fmin - ndarray (Ntimes, Npols) of min rfi fraction along freq dimension.
            fmean - ndarray (Ntimes, Npols) of average rfi fraction along freq dimension.
            fstd - ndarray (Ntimes, Npols) of standard deviation rfi fraction
                   along freq dimension.
            fmedian - ndarray (Ntimes, Npols) of median rfi fraction along freq dimension.
            freqs - ndarray (Nfreqs) of frequencies in observation (Hz).
            times - ndarray (Ntimes) of times in observation (julian date).
            pols - ndarray (Npols) of polarizations in data (string format).
            version - Version string including git information.
    """
    import pyuvdata.utils as uvutils

    if flag_array is None:
        flag_array = uv.flag_array
    if prior_flags is None:
        prior_flags = np.zeros_like(flag_array)
    # Average across bls for given time
    waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
    prior_wf = np.zeros_like(waterfall)
    unit_wf = np.ones_like(prior_wf)
    waterfall_weight = np.zeros(uv.Ntimes)
    times = np.unique(uv.time_array)
    for ti, time in enumerate(times):
        ind = np.where(uv.time_array == time)[0]
        waterfall_weight[ti] = len(ind)
        waterfall[ti, :, :] = np.mean(flag_array[ind, 0, :, :], axis=0)
        prior_wf[ti, :, :] = np.mean(prior_flags[ind, 0, :, :], axis=0)
    # Normalize waterfall to account for data already flagged in file
    waterfall = (waterfall - prior_wf) / (unit_wf - prior_wf)
    # Calculate stats across time
    tmax = np.max(waterfall, axis=0)
    tmin = np.min(waterfall, axis=0)
    tmean = np.average(waterfall, axis=0, weights=waterfall_weight)
    tstd = np.sqrt(np.average((waterfall - tmean[np.newaxis, :, :])**2,
                              axis=0, weights=waterfall_weight))
    tmedian = np.median(waterfall, axis=0)
    # Calculate stats across frequency
    fmax = np.max(waterfall, axis=1)
    fmin = np.min(waterfall, axis=1)
    fmean = np.mean(waterfall, axis=1)  # no weights - all bls have all channels
    fstd = np.std(waterfall, axis=1)
    fmedian = np.median(waterfall, axis=1)
    # Some meta info
    freqs = uv.freq_array[0, :]
    pols = uvutils.polnum2str(uv.polarization_array)
    # Store data in npz
    np.savez(outfile, waterfall=waterfall, tmax=tmax, tmin=tmin, tmean=tmean,
             tstd=tstd, tmedian=tmedian, fmax=fmax, fmin=fmin, fmean=fmean,
             fstd=fstd, fmedian=fmedian, freqs=freqs, times=times, pols=pols,
             version=hera_qm_version_str)


def xrfi_apply(filename, args, history):
    """
    Read in a flag array and optionally several waterfall flags, and insert into
    a data file.

    Args:
        filename -- Data file in which update flag array.
        args -- parsed arguments via argparse.ArgumentParser.parse_args
        history -- history string to include in files
    Return:
        None
    """
    # make sure we were given files to process
    if len(filename) == 0:
        raise AssertionError('Please provide a visibility file')
    if len(filename) > 1:
        raise AssertionError('xrfi_apply currently only takes a single data file.')
    filename = filename[0]
    uvd = UVData()
    if args.infile_format == 'miriad':
        uvd.read_miriad(filename)
    elif args.infile_format == 'uvfits':
        uvd.read_uvfits(filename)
    elif args.infile_format == 'fhd':
        uvd.read_fhd(filename)
    else:
        raise ValueError('Unrecognized input file format ' + str(args.infile_format))

    # Read in flag file
    waterfalls = []
    flag_history = ''
    if args.flag_file is not None:
        d = np.load(args.flag_file)
        flag_array = d['flag_array']
        if flag_array.shape != uvd.flag_array.shape:
            raise ValueError('Flag array in ' + args.flag_file + ' does not match '
                             'shape of flag array in data file ' + filename + '.')
        flag_history += str(d['history'])
        try:
            # Flag file itself may contain a waterfall
            waterfalls.append(d['waterfall'])
        except KeyError:
            pass
    else:
        flag_array = np.zeros_like(uvd.flag_array)

    # Read in waterfalls
    if args.waterfalls is not None:
        for wfile in args.waterfalls.split(','):
            d = np.load(wfile)
            if (len(waterfalls) > 0 and d['waterfall'].shape != waterfalls[0].shape):
                raise ValueError('Not all waterfalls have the same shape, cannot combine.')
            waterfalls.append(d['waterfall'])
            if str(d['history']) not in flag_history:
                # Several files may come from same command. Cut down on repeated info.
                flag_history += str(d['history'])

    if len(waterfalls) > 0:
        wf_full = sum(waterfalls).astype(bool)  # Union all waterfalls
        flag_array += waterfall2flags(wf_full, uvd)  # Combine with flag array
    else:
        wf_full = None

    # Finally, add the flag array to the flag array in the data
    uvd.flag_array += flag_array
    # append to history
    uvd.history = uvd.history + flag_history + history

    # save output when we're done
    if args.xrfi_path == '':
        # default to the same directory
        abspath = os.path.abspath(filename)
        dirname = os.path.dirname(abspath)
    else:
        dirname = args.xrfi_path
    basename = os.path.basename(filename)
    outfile = ''.join([basename, args.extension])
    outpath = os.path.join(dirname, outfile)
    if args.outfile_format == 'miriad':
        uvd.write_miriad(outpath, clobber=args.overwrite)
    elif args.outfile_format == 'uvfits':
        if os.path.exists(outpath) and not args.overwrite:
            raise ValueError('File exists: skipping')
        uvd.write_uvfits(outpath, force_phase=True, spoof_nonessential=True)
    else:
        raise ValueError('Unrecognized output file format ' + str(args.outfile_format))
    if args.output_npz:
        # Save an npz with the final flag array and waterfall and relevant metadata
        outpath = outpath + args.out_npz_ext
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        np.savez(outpath, flag_array=uvd.flag_array, waterfall=wf_full, baseline_array=uvd.baseline_array,
                 antpairs=uvd.get_antpairs(), polarization_array=uvd.polarization_array,
                 freq_array=uvd.freq_array, time_array=uvd.time_array, lst_array=uvd.lst_array,
                 antpos=antpos, ants=ants, history=flag_history + history)
