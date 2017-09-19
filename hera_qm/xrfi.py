from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from hera_qm.version import hera_qm_version_str
import json

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
        raise ValueError('Kernel size exceeds data.')
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
         bool array: boolean array of flags
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
        bool array: boolean array of flags
    """
    # Delay import so scipy is not required for any use of hera_qm
    from scipy.signal import medfilt2d

    if Kt > d.shape[0] or Kf > d.shape[1]:
        raise ValueError('Kernel size exceeds data.')
    d = np.concatenate([d[Kt - 1::-1], d, d[:-Kt - 1:-1]], axis=0)
    d = np.concatenate([d[:, Kf - 1::-1], d, d[:, :-Kf - 1:-1]], axis=1)
    if np.iscomplexobj(d):
        d_sm_r = medfilt2d(d.real, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm_i = medfilt2d(d.imag, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
        d_sm = d_sm_r + 1j*d_sm_i
    else: d_sm = medfilt2d(d, kernel_size=(2 * Kt + 1, 2 * Kf + 1))
    d_rs = d - d_sm
    d_sq = np.abs(d_rs)**2
    # puts median on same scale as average
    sig = np.sqrt(medfilt2d(d_sq, kernel_size=(2 * Kt + 1, 2 * Kf + 1)) / .456)
    # don't divide by zero, instead turn those entries into +inf
    f = np.true_divide(d_rs, sig, where=(np.abs(sig) > 1e-7))
    f = np.where(np.abs(sig) > 1e-7, f, np.inf)
    return f[Kt:-Kt, Kf:-Kf]


#############################################################################
# RFI flagging algorithms
#############################################################################

def watershed_flag(d, f=None, sig_init=6, sig_adj=2):
    '''Generates a mask for flags using a watershed algorithm.
    Returns a watershed flagging of an array that is in units of standard
    deviation (i.e. how many sigma the datapoint is from the center).

    Args:
        d (array)[time,freq]: 2D array to perform watershed on.
            d should be in units of standard deviations.
        f (array, optional): input flags. Same size as d.
        sig_init (int): number of sigma to flag above, initially.
        sig_adj (int): number of sigma to flag above for points
            near flagged points.

    Returns:
        bool array: Array of mask values for d.
    '''
    # mask off any points above 'sig' sigma and nan's.
    f1 = np.ma.array(d, mask=np.where(d > sig_init, 1, 0))
    f1.mask |= np.isnan(f1)
    if f is not None:
        f1.mask |= np.array(f)

    # Loop over flagged points and examine adjacent points to see if they exceed sig_adj
    # Start the watershed
    prevx, prevy = 0, 0
    x, y = np.where(f1.mask)
    while x.size != prevx and y.size != prevy:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            prevx, prevy = x.size, y.size
            xp, yp = (x + dx).clip(0, f1.shape[0] - 1), (y + dy).clip(0, f1.shape[1] - 1)
            i = np.where(f1[xp, yp] > sig_adj)[0]  # if sigma > 'sigl'
            f1.mask[xp[i], yp[i]] = 1
            x, y = np.where(f1.mask)
    return f1.mask


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


def xrfi(d, f=None, Kt=8, Kf=8, sig_init=6, sig_adj=2):
    """Run best rfi excision we have. Uses detrending and watershed algorithms above.
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
    nsig = detrend_medfilt(d, Kt=Kt, Kf=Kf)
    f = watershed_flag(np.abs(nsig), f=f, sig_init=sig_init, sig_adj=sig_adj)
    return f


def xrfi_run(files, opts, history):
    """
    Run an RFI-flagging algorithm on an entire file and store results in flag array.

    Args:
       files -- a list of files to run RFI flagging on
       opts -- an optparse OptionParser instance
    Return:
       None

    This function will take in a series of files, and run an RFI-flagging algorithm to
    identify contaminated observations. Then the flags array of the object will be
    updated to reflect these identifications.
    """
    # make sure we were given files to process
    if len(files) == 0:
        raise AssertionError('Please provide a list of visibility files')

    # loop over files
    for fn in files:
        # read files in as pyuvdata object
        uvd = UVData()
        if opts.infile_format == 'miriad':
            uvd.read_miriad(fn)
        elif opts.infile_format == 'uvfits':
            uvd.read_uvfits(fn)
        elif opts.infile_format == 'fhd':
            uvd.read_fhd(fn)
        else:
            raise ValueError('Unrecognized input file format ' + str(opts.infile_format))

        # create an iterator over data contents
        for key, d in uvd.antpairpol_iter():
            ind1, ind2, ipol = uvd._key2inds(key)

            # make sure that we are selecting some number of values
            if len(ind1) > 0:
                f = uvd.flag_array[ind1, 0, :, ipol]
                if opts.algorithm == 'xrfi_simple':
                    new_f = xrfi_simple(np.abs(d), f=f, nsig_df=opts.nsig_df,
                                        nsig_dt=opts.nsig_dt, nsig_all=opts.nsig_all)
                elif opts.algorithm == 'xrfi':
                    new_f = xrfi(np.abs(d), f=f, Kt=opts.kt_size, Kf=opts.kf_size,
                                 sig_init=opts.sig_init, sig_adj=opts.sig_adj)
                else:
                    raise ValueError('Unrecognized RFI method ' + str(opts.algorithm))
                # combine old flags and new flags
                uvd.flag_array[ind1, 0, :, ipol] = np.logical_or(f, new_f)
            if len(ind2) > 0:
                f = uvd.flag_array[ind2, 0, :, ipol]
                if opts.algorithm == 'xrfi_simple':
                    new_f = xrfi_simple(np.abs(d), f=f, nsig_df=opts.nsig_df,
                                        nsig_dt=opts.nsig_dt, nsig_all=opts.nsig_all)
                elif opts.algorithm == 'xrfi':
                    new_f = xrfi(np.abs(d), f=f, Kt=opts.kt_size, Kf=opts.kf_size,
                                 sig_init=opts.sig_init, sig_adj=opts.sig_adj)
                else:
                    raise ValueError('Unrecognized RFI method ' + str(opts.algorithm))
                # combine old flags and new flags
                uvd.flag_array[ind2, 0, :, ipol] = np.logical_or(f, new_f)

        # append to history
        uvd.history = uvd.history + history

        # save output when we're done
        if opts.xrfi_path == '':
            # default to the same directory
            abspath = os.path.abspath(fn)
            dirname = os.path.dirname(abspath)
        else:
            dirname = opts.xrfi_path
        basename = os.path.basename(fn)
        filename = ''.join([basename, opts.extension])
        outpath = os.path.join(dirname, filename)
        if opts.outfile_format == 'miriad':
            uvd.write_miriad(outpath)
        elif opts.outfile_format == 'uvfits':
            uvd.write_uvfits(outpath, force_phase=True, spoof_nonessential=True)
        else:
            raise ValueError('Unrecognized output file format ' + str(opts.outfile_format))

        if opts.summary:
            sum_file = ''.join([basename, opts.summary_ext])
            sum_path = os.path.join(dirname, sum_file)
            summarize_flags(uvd, sum_path)
    return


def summarize_flags(uv, outfile):
    """ Collapse several dimensions of a UVData flag array to summarize.
    Args:
        uv -- UVData object containing flag_array to be summarized
        outfile -- filename for output npz file

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

    # Average across bls for given time
    waterfall = np.zeros((uv.Ntimes, uv.Nfreqs, uv.Npols))
    waterfall_weight = np.zeros(uv.Ntimes)
    times = np.unique(uv.time_array)
    for ti, time in enumerate(times):
        ind = np.where(uv.time_array == time)
        waterfall_weight[ti] = len(ind)
        waterfall[ti, :, :] = np.mean(uv.flag_array[ind, 0, :, :])
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
