from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
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


def flag_xants(uvd, xants):
    """Flag visibilities containing specified antennas.

    Args:
        uvd (UVData object): visibilities to be flagged
        xants (list of ints): antennas to flag
    Returns:
        uvd: UVData object, with flag_array set to True for all
             visibilities containing xants
    """
    # check that we got a UVData object
    if not isinstance(uvd, UVData):
        raise ValueError("First argument to flag_xants must be a UVData object")
    # loop over all antennas in data
    all_ants = uvd.get_ants()
    for ant in all_ants:
        # loop over list of excluded antennas to form baseline pairs
        for xant in xants:
            blts = uvd.antpair2ind(ant, xant)
            uvd.flag_array[blts, :, :, :] = True
    return uvd


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
        prevx, prevy = x.size, y.size
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            xp, yp = (x + dx).clip(0, f1.shape[0] - 1), (y + dy).clip(0, f1.shape[1] - 1)
            i = np.where(d[xp, yp] > sig_adj)[0]  # if sigma > 'sigl'
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
    if isinstance(indata, UVData):
        uvd = indata
        if len(args.filename) == 0:
            raise AssertionError('Please provide a filename to go with UVData object. '
                                 'The filename is used in conjunction with "extension" '
                                 'to determine the output filename.')
        else:
            filename = args.filename[0]
    else:
        # make sure we were given files to process
        if len(indata) == 0:
            raise AssertionError('Please provide a visibility file or UVData object')
        if len(indata) > 1:
            raise AssertionError('xrfi_run currently only takes a single data file.')
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
    d_flag_array = vis_flag(uvd, args)

    # Make a "normalized waterfall" to account for data already flagged in file
    d_wf_tot = flags2waterfall(uvd, flag_array=d_flag_array)
    d_wf_prior = flags2waterfall(uvd, flag_array=uvd.flag_array)
    unit_flags = np.ones_like(d_wf_prior)
    d_wf_norm = (d_wf_tot - d_wf_prior) / (unit_flags - d_wf_prior)
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
        if not (np.allclose(np.unique(uvd.time_array), np.unique(uvm.time_array), atol=1e-5, rtol=0) and
                np.allclose(uvd.freq_array, uvm.freq_array, atol=1., rtol=0)):
            raise ValueError('Time and frequency axes of model vis file must match'
                             'the data file.')
        m_flag_array = vis_flag(uvm, args)
        m_waterfall = flags2waterfall(uvm, flag_array=m_flag_array)
        m_wf_t = threshold_flags(m_waterfall, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)

    # Flag on gain solutions and chisquared values
    if args.calfits_file is not None:
        uvc = UVCal()
        uvc.read_calfits(args.calfits_file)
        if not (np.allclose(np.unique(uvd.time_array), np.unique(uvc.time_array), atol=1e-5, rtol=0) and
                np.allclose(uvd.freq_array, uvc.freq_array, atol=1., rtol=0)):
            raise ValueError('Time and frequency axes of calfits file must match'
                             'the data file.')
        g_flag_array, x_flag_array = cal_flag(uvc, args)
        g_waterfall = flags2waterfall(uvc, flag_array=g_flag_array)
        x_waterfall = flags2waterfall(uvc, flag_array=x_flag_array)
        g_wf_t = threshold_flags(g_waterfall, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)
        x_wf_t = threshold_flags(x_waterfall, px_threshold=args.px_threshold,
                                 freq_threshold=args.freq_threshold,
                                 time_threshold=args.time_threshold)

    # append to history
    history = 'Flagging command: "' + history + '", Using ' + hera_qm_version_str

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
    np.savez(outpath, flag_array=d_flag_array, waterfall=d_wf_t, history=history)
    if args.model_file is not None:
        outfile = ''.join([os.path.basename(args.model_file), args.extension])
        outpath = os.path.join(dirname, outfile)
        np.savez(outpath, flag_array=m_flag_array, waterfall=m_wf_t, history=history)
    if args.calfits_file is not None:
        # Save flags from gains and chisquareds in separate files
        outfile = ''.join([os.path.basename(args.calfits_file), '.g', args.extension])
        outpath = os.path.join(dirname, outfile)
        np.savez(outpath, flag_array=g_flag_array, waterfall=g_wf_t, history=history)
        outfile = ''.join([os.path.basename(args.calfits_file), '.x', args.extension])
        outpath = os.path.join(dirname, outfile)
        np.savez(outpath, flag_array=x_flag_array, waterfall=x_wf_t, history=history)

    if args.summary:
        sum_file = ''.join([basename, args.summary_ext])
        sum_path = os.path.join(dirname, sum_file)
        # Summarize using one of the raw flag arrays
        summarize_flags(uvd, sum_path, flag_array=d_flag_array)

    return


def vis_flag(uv, args):
    """
    Run an RFI-flagging algorithm on visibility data.

    Args:
        uv -- a UVData object containing visibility data to flag on.
        args -- parsed arguments via argparse.ArgumentParser.parse_args
    Return:
        flag_array -- boolean array of flags, same shape as uv.data_array
    """
    if not isinstance(uv, UVData):
        raise ValueError('First argument to vis_flags must be a UVData object.')
    flag_array = np.zeros_like(uv.flag_array)
    for key, d in uv.antpairpol_iter():
        ind1, ind2, ipol = uv._key2inds(key)
        for ind in [ind1, ind2]:
            if len(ind) == 0:
                continue
            f = uv.flag_array[ind, 0, :, ipol]
            if args.algorithm == 'xrfi_simple':
                flag_array[ind, 0, :, ipol] = xrfi_simple(np.abs(d), f=f,
                                                          nsig_df=args.nsig_df,
                                                          nsig_dt=args.nsig_dt,
                                                          nsig_all=args.nsig_all)
            elif args.algorithm == 'xrfi':
                flag_array[ind, 0, :, ipol] = xrfi(np.abs(d), f=f, Kt=args.kt_size,
                                                   Kf=args.kf_size, sig_init=args.sig_init,
                                                   sig_adj=args.sig_adj)
            else:
                raise ValueError('Unrecognized RFI method ' + str(args.algorithm))
    return flag_array


def cal_flag(uvc, args):
    """
    Run an RFI-flagging algorithm on calibration solutions and quality_array.

    Args:
        uvc -- a UVCal object containing calibration output to flag on.
                Must have cal_type=='gain'
        args -- parsed arguments via argparse.ArgumentParser.parse_args
    Return:
        flag_array -- boolean array of flags, same shape as uvc.gain_array
    """
    if not isinstance(uvc, UVCal):
        raise ValueError('First argument to cal_flags must be a UVCal object.')
    if uvc.cal_type != 'gain':
        raise ValueError('UVCal object must have cal_type=="gain".')

    g_flags = np.zeros_like(uvc.flag_array)
    x_flags = np.zeros_like(uvc.flag_array)

    for ai in range(uvc.Nants_data):
        for pi in range(uvc.Njones):
            # Note transposes are due to freq, time dimensions rather than the
            # expected time, freq
            f = uvc.flag_array[ai, 0, :, :, pi].T
            if args.algorithm == 'xrfi_simple':
                d = np.abs(uvc.gain_array[ai, 0, :, :, pi].T)
                g_flags[ai, 0, :, :, pi] = xrfi_simple(d, f=f, nsig_df=args.nsig_df,
                                                       nsig_dt=args.nsig_dt,
                                                       nsig_all=args.nsig_all).T
                d = np.abs(uvc.quality_array[ai, 0, :, :, pi].T)
                x_flags[ai, 0, :, :, pi] = xrfi_simple(d, f=f, nsig_df=args.nsig_df,
                                                       nsig_dt=args.nsig_dt,
                                                       nsig_all=args.nsig_all).T
            elif args.algorithm == 'xrfi':
                d = np.abs(uvc.gain_array[ai, 0, :, :, pi].T)
                g_flags[ai, 0, :, :, pi] = xrfi(d, f=f, Kt=args.kt_size,
                                                Kf=args.kf_size, sig_init=args.sig_init,
                                                sig_adj=args.sig_adj).T
                d = np.abs(uvc.quality_array[ai, 0, :, :, pi].T)
                x_flags[ai, 0, :, :, pi] = xrfi(d, f=f, Kt=args.kt_size,
                                                Kf=args.kf_size, sig_init=args.sig_init,
                                                sig_adj=args.sig_adj).T
            else:
                raise ValueError('Unrecognized RFI method ' + str(args.algorithm))
    return g_flags, x_flags


def flags2waterfall(uv, flag_array=None):
    """
    Convert a flag array to a 2D waterfall of dimensions (Ntimes, Nfreqs).
    Averages over baselines and polarizations (in the case of visibility data),
    or antennas and jones parameters (in case of calibrationd data).
    Args:
        uv -- A UVData or UVCal object which defines the times and frequencies,
              and supplies the flag_array to convert (if flag_array not specified)
        flag_array -- Optional flag array to convert instead of uv.flag_array.
                      Must have same dimensions as uv.flag_array.
    Returns:
        waterfall -- 2D waterfall of averaged flags, for example fraction of baselines
                     which are flagged for every time and frequency (in case of UVData input)
                     Size is (Ntimes, Nfreqs).
    """
    if not isinstance(uv, (UVData, UVCal)):
        raise ValueError('flags2waterfall() requires a UVData or UVCal object as '
                         'the first argument.')
    if flag_array is None:
        flag_array = uv.flag_array
    if uv.flag_array.shape != flag_array.shape:
        raise ValueError('Flag array must align with UVData or UVCal object.')

    if isinstance(uv, UVCal):
        waterfall = np.mean(flag_array, axis=(0, 1, 4)).T
    else:
        waterfall = np.zeros((uv.Ntimes, uv.Nfreqs))
        for i, t in enumerate(np.unique(uv.time_array)):
            waterfall[i, :] = np.mean(flag_array[uv.time_array == t, 0, :, :],
                                      axis=(0, 2))

    return waterfall


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
    spec = np.nanmean(wf, axis=0)
    wf_t[:, spec > time_threshold] = True
    tseries = np.nanmean(wf, axis=1)
    wf_t[tseries > freq_threshold, :] = True
    wf_t[wf > px_threshold] = True
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
        # Save an npz with the final flag array and waterfall
        outpath = outpath + args.out_npz_ext
        np.savez(outpath, flag_array=uvd.flag_array, waterfall=wf_full,
                 history=flag_history + history)
