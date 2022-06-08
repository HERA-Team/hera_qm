# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics using day-long autocorrelations."""
import numpy as np
from copy import deepcopy
import operator
from .xrfi import simple_flag_waterfall
from .metrics_io import write_metric_file


def nanmad(a, axis=None):
    '''Analogous to np.nanmedian, but for median absolute deviation.'''
    nanmed = np.nanmedian(a, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(a - nanmed), axis=axis)


def nanmedian_abs_diff(a, axis=0):
    '''Computes the absolute difference between neighbors along axis, then collapses 
    along that axis with the nanmedian. Useful for studying temporal variability, e.g.'''
    return np.nanmedian(np.abs(np.diff(a, axis=axis)), axis=axis)


def nanmean_abs_diff(a, axis=0):
    '''Computes the absolute difference between neighbors along axis, then collapses 
    along that axis with the nanmean. Useful for studying temporal variability, e.g.'''
    return np.nanmean(np.abs(np.diff(a, axis=axis)), axis=axis)


def _check_only_auto_keys(data):
    '''Verify that keys in data are only autocorrelation keys of the form (ant, ant, pol).'''
    from hera_cal.utils import split_pol  # delay import of hera_cal
    for bl in data:
        ap0, ap1 = split_pol(bl[2])
        if (bl[0] != bl[1]) or (ap0 != ap1):
            raise ValueError(f'{bl} is not an autocorrelation key.')


def get_auto_spectra(autos, flag_wf=None, time_avg_func=np.nanmedian, scalar_norm=True, 
                     waterfall_norm=False, norm_func=np.nanmedian, ex_ants=[]):
    '''Compute (normalized) spectrum from a set of waterfalls.
    
    Parameters
    ----------
    autos : dictionary or DataContainer
        Maps autocorrelation keys e.g. (1, 1, 'ee') to waterfalls. Imaginary parts ignored.
    flag_wf : ndarray
        Numpy array the same shape as the waterfalls in autos with flags to be treated as 
        np.nan when performing time_avg_func and freq_avg_func
    time_avg_func : function
        Function for converting a 2D numpy array to a single spectrum, collapsing over the
        0th (time) dimension. Must take axis=0 kwarg. Should be NaN-aware if flags are included. 
    scalar_norm : bool
        If True, renormalize each spectrum by norm_func of the antenna's waterfall.
    waterfall_norm : bool
        If True, renormalize each waterfall for each pol by an average waterfall created 
            using norm_func on all of all waterfalls not in ex_ants.
    norm_func : function 
        Function used for normalizing the spectra as described above. Should be NaN-aware if 
        flags are included.
    ex_ants : list of integers
        List of antenna numbers to exclude from the average waterfall if waterfall_norm is True.

    Returns
    -------
    auto_spectra : dict 
        Dictionary mapping autocorrelation key e.g. (1, 1, 'ee') to (normalized) spectrum.
    ''' 
    # get wf_shape and make empty flags if not provided
    _check_only_auto_keys(autos)
    wf_shape = next(iter(autos.values())).shape
    if flag_wf is None:
        flag_wf = np.zeros(wf_shape, dtype=bool)
        
    # pre-compute normalizing waterfall for each polarization
    if waterfall_norm:
        wf_norm = {}
        for pol in set([bl[2] for bl in autos]):
            wf_norm[pol] = []
            for i in range(wf_shape[0]):
                row_list = [np.where(flag_wf[i, :], np.nan, autos[bl][i, :].real)
                            for bl in autos if (bl[0] not in ex_ants) and (bl[2] == pol)]
                wf_norm[pol].append(norm_func(row_list, axis=0))
            wf_norm[pol] = np.vstack(wf_norm[pol])

    # compute auto_spectra
    auto_spectra = {}
    for bl in autos:
        auto = np.where(flag_wf, np.nan, autos[bl].real)
        if waterfall_norm:
            auto /= wf_norm[bl[2]]
        auto_spectra[bl] = time_avg_func(auto, axis=0)
        if scalar_norm:
            auto_spectra[bl] /= norm_func(auto)
    return auto_spectra


def spectrum_modz_scores(auto_spectra, ex_ants=[], overall_spec_func=np.nanmedian, metric_func=np.nanmedian, 
                         metric_power=1.0, metric_log=False, abs_diff=True):
    '''Computes a modified Z-score of a autocorrelation spectrum compared to all others not in ex_ants.
    
    Parameters
    ----------
    auto_spectra : dictionary
        Dictionary mapping autocorrelation key e.g. (1, 1, 'ee') to (normalized) spectrum, e.g.
        that produced by get_auto_spectra().
    ex_ants : list of integers
        list of integer antenna numbers to exclude from median and MAD when computing modified z-scores
    overall_spec_func : function
        Function for averaging together all spectra of a given polarization.
    metric_func : function
        The function used to reduce differences between each spectrum and the overall spectrum to a float
    metric_power : float
        Take each entry in the (abs) diff to this power before using metric_func to collapse to a float
    metric_log : bool
        If True, take the log of both the spectrum and the overall spectrum before taking the abs diff
    abs_diff : bool
        If True, the metric uses the abs diff with the overall spectrum. Otherwise, it's just the diff.


    Returns
    -------
    modzs : dictionary
        Dictionary mapping autocorrelation keys e.g. (1, 1, 'ee') to float modified z-scores relative 
        to all antennas not in ex_ants.
    '''
    # Check if all keys are actually autocorrelations
    _check_only_auto_keys(auto_spectra)
    
    # Get overall spectrum each polarization
    pols = set([bl[2] for bl in auto_spectra])
    overall_spectrum = {pol: overall_spec_func([spec for bl, spec in auto_spectra.items() if (bl[2] == pol) 
                                                and (bl[0] not in ex_ants)], axis=0) for pol in pols}

    # Calculate metric of distance between spectra and the mean/median spectrum
    L = lambda x : np.log(x) if metric_log else x
    A = lambda x : np.abs(x) if abs_diff else x
    diff_metrics = {bl: metric_func(A(L(auto_spectra[bl]) - L(overall_spectrum[bl[2]]))**metric_power) for bl in auto_spectra}
    
    # Calculate the modified z-score of that metric
    median_diff_metric = np.median([metric for bl, metric in diff_metrics.items() if bl[0] not in ex_ants])
    mad_diff_metric = np.median([np.abs(metric - median_diff_metric) for bl, metric in diff_metrics.items() 
                                 if bl[0] not in ex_ants])
    modzs = {bl: (diff_metrics[bl] - median_diff_metric) / mad_diff_metric / 1.4826 for bl in auto_spectra}
    return modzs


def iterative_spectrum_modz(auto_spectra, prior_ex_ants=[], modz_cut=5.0, cut_on_abs_modz=False, overall_spec_func=np.nanmedian,
                            metric_func=np.nanmedian, metric_power=1.0, metric_log=False, abs_diff=True):
    '''Iteratively re-computes modified z-scores for aucorrelation spectra by excluding antennas and recalculating.
    
    Parameters
    ----------
    auto_spectra : dictionary
        Dictionary mapping autocorrelation key e.g. (1, 1, 'ee') to (normalized) spectrum, e.g.
        that produced by get_auto_spectra().
    prior_ex_ants : list of integers
        list of integer antenna numbers to exclude from the get-go
    modz_cut : float
        Modified z-score above which to cut the worst antenna and re-run the metrics.
    cut_on_abs_modz : bool
        If True, cut ants with z-scores that are larger than modz_cut or smaller than -modz_cut.
    overall_spec_func : function
        Function for averaging together all spectra of a given polarization.
    metric_func : function
        The function used to reduce differences between each spectrum and the overall spectrum to a float
    metric_power : float
        Take each entry in the abs diff to this power before using metric_func to collapse to a float
    metric_log : bool
        If True, take the log of both the spectrum and the overall spectrum before taking the abs diff
    abs_diff : bool
        If True, the metric uses the abs diff with the overall spectrum. Otherwise, it's just the diff.

    Returns
    -------
    ex_ants : list of integers
        List of integer antenna numbers that were excluded on final iteration.
    modzs : dictionary
        Dictionary mapping autocorrelation keys e.g. (1, 1, 'ee') to float modified z-scores relative 
        to all antennas not in ex_ants. Returns results for last iteration.
    '''
    ex_ants = deepcopy(prior_ex_ants)
    # add one antenna per loop to ex_ants
    while not np.all([bl[0] in ex_ants for bl in auto_spectra]):
        # compute metric for all autos compared to the distribution of non-ex_ant antennas
        modzs = spectrum_modz_scores(auto_spectra, ex_ants=ex_ants, overall_spec_func=overall_spec_func, 
                                     metric_func=metric_func, metric_power=metric_power, 
                                     metric_log=metric_log, abs_diff=abs_diff)
        
        # figure out out worst antenna that's not already in ex_ants
        modzs_no_exants = {k: [v, np.abs(v)][cut_on_abs_modz] for k, v in modzs.items() if k[0] not in ex_ants}
        worst_ant, worst_z = max(modzs_no_exants.items(), key=operator.itemgetter(1))
        
        # cut worst antenna if it's bad enough
        if (worst_z > modz_cut):
            ex_ants.append(worst_ant[0])
        else:
            break
    
    return ex_ants, modzs


def auto_metrics_run(metric_outfile, raw_auto_files, median_round_modz_cut=8., mean_round_modz_cut=4.,
                     edge_cut=100, Kt=8, Kf=8, sig_init=5.0, sig_adj=2.0, chan_thresh_frac=.05, 
                     history='', overwrite=False):
    '''Evaluates day-long autocorrelation waterfalls for "outlierness" in shape, power, temporal 
    variability, and sharp temporal discontinuities. Each of these is assessed by collapsing each
    waterfall to a single spectrum (might be an average, might be an STD, etc.). These are then
    compared to an average spectrum and float difference is then converted into a modified Z-score
    for each metric. Antennas above some given cut are iteratively excluded from the calculation
    of the median and MAD. This process proceeds in two rounds, first by finding the worst antennas
    via median-based statistics of shape, power, etc. and then, following a relatively simple
    RFI flagging step, using more sensitive mean-based statistics.

    Parameters
    ----------
    metric_outfile : str
        Path to save metrics and associated metadata as a hdf5 file.
    raw_auto_files : str or list of str
        Path(s) to data files containing raw autocorrelations. Ideally these would be pre-selected to just
        include autocorrelations, but raw data files will work too (just more slowly)
    median_round_modz_cut : float
        Modified Z-score threshold above which to cut an antenna when either of its polarizations exceeds
        this cut. Used in Round 1 of antenna flagging, which is based on more robust median statistics. 
        Meant as only a preliminary cut of antennas to remove the worst offenders before RFI flagging.
        All statistics are still computed for cut antennas, but they are removed from the distributions that
        all other antennas are compared against.
    mean_round_modz_cut : float
        Modified Z-score threshold for excluding antennas in round 2, which uses mean-based statistics.
        Can and should genererally be more restrictive than median_round_modz_cut, but otherwise analogous.
    edge_cut : int
        Number of channels at the high and low edge of the band to flag (i.e. ignore when looking for outliers).
    Kt : int
        Number of integrations half-width of kernel for med/meanfilt in RFI flagging. 
    Kf : int
        Frequency channel half-width of kernel for med/meanfilt in RFI flagging.
    nsig_init : float
        The number of sigma in the metric above which to flag pixels. Default is 5.
    nsig_adj : float
        The number of sigma to flag above for points near flagged points. Default is 2.
    chan_thresh_frac : float
        Fraction of times flagged (excluding completely flagged integrations) above which
        to flag an entire channel. Default .05 means that channels with 5% or more of times flagged 
        (excluding completely flagged times) become completely flagged.
    history : str
        String to save in metric_outfile history field.
    overwrite : bool
        If True, overwrites file at metric_outfile.

    Returns
    -------
    ex_ants : dict
        Dict with keys r1_ex_ants and r2_ex_ants, each to a list of integer values. These represent
        antennas flagged in rounds 1 and 2 respectively. r1_ex_ants is always a subset of r2_ex_ants.
    modzs : dict
        Dict (keys of the form r[1-2]_*_modzs) of dicts mapping baseline keys e.g. (0, 0, 'ee)) to floats.
        These include the modified Z-scores for every autocorrelation for all four metrics for both round 1
        (median-based) and round 2 (mean-based).
    spectra : dict
        Dict (sting keys for each type) of dicts mapping baseline keys e.g. (0, 0, 'ee)) to 1D numpy arrays.
        For shape, we use median_spectra_normed (round 1) and mean_spectra_normed (round 2). For temporal
        variability, we use mad_spectra_normed (round 1) and std_spectra_normed (round 2). For temporal
        discontinuitites we use median_abs_diff_spectra_normed (round 1) and mean_abs_diff_spectra_normed
        (round 2). For power, we use median_spectra (round 1) and mean_spectra (round 2).
    flags : np.ndarray
        Single boolean flag waterfall computed between rounds 1 and 2. Mostly useful for visualization.
    '''

    ######################################################
    # Load Data
    ######################################################

    # Delay import of hera_cal funcitons to minimize circular dependency
    from hera_cal.io import HERAData
    from hera_cal.utils import split_pol

    # Figure out which baselines to load, if not all
    hd = HERAData(sorted(raw_auto_files))
    bls = hd.bls
    if len(hd.filepaths) > 1:  # in this caes, hd.bls will be a dictionary mapping filename to baselines
        bls = set([bl for bls in bls.values() for bl in bls])
    auto_bls = sorted([bl for bl in bls if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1])])
    pols = set([bl[2] for bl in auto_bls])
    if np.all([bl in auto_bls for bl in bls]):
        auto_bls = None  # just load the whole file, which is faster

    # Load data
    autos, _, _ = hd.read(axis='blt', bls=auto_bls)
    wf_shape = next(iter(autos.values())).shape

    # Compute initial set of flags using edge_cut
    ec_flags = np.zeros(wf_shape, dtype=bool)
    ec_flags[:, :edge_cut] = True
    ec_flags[:, -edge_cut:] = True

    ######################################################
    # Compute Statistics and Modified Z-Scores for Round 1
    ######################################################

    # median_spectra_normed are normalized time-averaged bandpasses used to assess bandpass shape
    median_spectra_normed = get_auto_spectra(autos, flag_wf=ec_flags, time_avg_func=np.nanmedian, scalar_norm=True, 
                                             waterfall_norm=False, norm_func=np.nanmedian)
    # mad_spectra_normed look at the variability of each waterfall in time, having divided out the average waterfall. 
    # These are used to assess bandpass variability over the night.
    mad_spectra_normed = get_auto_spectra(autos, flag_wf=ec_flags, time_avg_func=nanmad, scalar_norm=True, 
                                          waterfall_norm=True, norm_func=np.nanmedian, ex_ants=[])
    # median_abs_diff_spectra_normed look at the average integration-to-integration discontinuity, having divided out the
    # average waterfall. These are used to assess the relative amount of temporal discontinuities.
    median_abs_diff_spectra_normed = get_auto_spectra(autos, flag_wf=ec_flags, time_avg_func=nanmedian_abs_diff, scalar_norm=True,
                                                      waterfall_norm=True, norm_func=np.nanmedian, ex_ants=[])
    # Similar to median_spectra_normed, but without the normalization, these are used assess total power
    median_spectra = get_auto_spectra(autos, flag_wf=ec_flags, time_avg_func=np.nanmedian, scalar_norm=False, waterfall_norm=False)

    # Perform round 1 search for outliers, iterating until convergence
    r1_ex_ants = []
    while True:
        shape_ex_ants, r1_shape_modzs = iterative_spectrum_modz(median_spectra_normed, r1_ex_ants, modz_cut=median_round_modz_cut, 
                                                                abs_diff=True, overall_spec_func=np.nanmedian, metric_func=np.nanmedian,
                                                                metric_power=1.0, metric_log=False)

        temp_var_ex_ants, r1_temp_var_modzs = iterative_spectrum_modz(mad_spectra_normed, r1_ex_ants, modz_cut=median_round_modz_cut, 
                                                                      abs_diff=False, overall_spec_func=np.nanmedian, metric_func=np.nanmedian, 
                                                                      metric_power=1.0, metric_log=False)

        temp_diff_ex_ants, r1_temp_diff_modzs = iterative_spectrum_modz(median_abs_diff_spectra_normed, r1_ex_ants, modz_cut=median_round_modz_cut,
                                                                        abs_diff=False, overall_spec_func=np.nanmedian, metric_func=np.nanmedian, 
                                                                        metric_power=1.0, metric_log=False)
        
        power_ex_ants, r1_power_modzs = iterative_spectrum_modz(median_spectra, r1_ex_ants, modz_cut=median_round_modz_cut, 
                                                                abs_diff=True, overall_spec_func=np.nanmedian, metric_func=np.nanmedian, 
                                                                metric_power=1.0, metric_log=True)

        updated_ex_ants = list(set(shape_ex_ants) | set(temp_var_ex_ants) | set(temp_diff_ex_ants) | set(power_ex_ants))
        if len(updated_ex_ants) == len(r1_ex_ants):
            break
        else:
            r1_ex_ants = updated_ex_ants

    ######################################################
    # Compute Flags
    ######################################################

    # create single average waterfall of largely OK antennas, then flag on that
    avg_good_auto = np.vstack([np.mean([np.abs(autos[bl][i, :]) for bl in autos if bl[0] not in r1_ex_ants], axis=0)
                               for i in range(wf_shape[0])])
    flags = simple_flag_waterfall(avg_good_auto, Kt=Kt, Kf=Kf, sig_init=sig_init, sig_adj=sig_adj,
                                  edge_cut=edge_cut, chan_thresh_frac=chan_thresh_frac)

    ######################################################
    # Compute Statistics and Modified Z-Scores for Round 2
    ######################################################

    # Analogous statistics to the above, but this time using means, etc. instead of medians. These are more sensitive, 
    # but less robust to extreme outliers, which are now hopefully mostly gone between the ex_ants and the flagging.
    mean_spectra_normed = get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanmean, scalar_norm=True, 
                                           waterfall_norm=False, norm_func=np.nanmean)
    std_spectra_normed = get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanstd, scalar_norm=True, 
                                          waterfall_norm=True, norm_func=np.nanmean, ex_ants=r1_ex_ants)
    mean_abs_diff_spectra_normed = get_auto_spectra(autos, flag_wf=flags, time_avg_func=nanmean_abs_diff, scalar_norm=True, 
                                                    waterfall_norm=True, norm_func=np.nanmean, ex_ants=r1_ex_ants)
    mean_spectra = get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanmean, scalar_norm=False, waterfall_norm=False)

    r2_ex_ants = deepcopy(r1_ex_ants)
    while True:
        shape_ex_ants, r2_shape_modzs = iterative_spectrum_modz(mean_spectra_normed, r2_ex_ants, modz_cut=mean_round_modz_cut,
                                                                overall_spec_func=np.nanmean, metric_func=np.nanmean,
                                                                metric_power=1.0, metric_log=False)

        temp_var_ex_ants, r2_temp_var_modzs = iterative_spectrum_modz(std_spectra_normed, r2_ex_ants, modz_cut=mean_round_modz_cut,
                                                                      abs_diff=False, overall_spec_func=np.nanmean, metric_func=np.nanmean,
                                                                      metric_power=1.0, metric_log=False)

        temp_diff_ex_ants, r2_temp_diff_modzs = iterative_spectrum_modz(mean_abs_diff_spectra_normed, r2_ex_ants, modz_cut=mean_round_modz_cut,
                                                                        abs_diff=False, overall_spec_func=np.nanmean, metric_func=np.nanmean,
                                                                        metric_power=1.0, metric_log=False)
        
        power_ex_ants, r2_power_modzs = iterative_spectrum_modz(mean_spectra, r2_ex_ants, modz_cut=mean_round_modz_cut,
                                                                overall_spec_func=np.nanmean, metric_func=np.nanmean,
                                                                metric_power=1.0, metric_log=True)

        updated_ex_ants = list(set(shape_ex_ants) | set(temp_var_ex_ants) | set(temp_diff_ex_ants) | set(power_ex_ants))
        if len(updated_ex_ants) == len(r2_ex_ants):
            break
        else:
            r2_ex_ants = updated_ex_ants
            # recompute statistics that depend on the overall waterfall, and thus on the other antennas
            std_spectra_normed = get_auto_spectra(autos, flag_wf=flags, time_avg_func=np.nanstd, scalar_norm=True, 
                                                  waterfall_norm=True, norm_func=np.nanmean, ex_ants=r2_ex_ants)
            mean_abs_diff_spectra_normed = get_auto_spectra(autos, flag_wf=flags, time_avg_func=nanmean_abs_diff, scalar_norm=True, 
                                                            waterfall_norm=True, norm_func=np.nanmean, ex_ants=r2_ex_ants)

    ######################################################
    # Save results
    ######################################################
    ex_ants = {'r1_ex_ants': r1_ex_ants, 'r2_ex_ants': r2_ex_ants}
    modzs = {'r1_shape_modzs': r1_shape_modzs, 'r1_temp_var_modzs': r1_temp_var_modzs,
             'r1_temp_diff_modzs': r1_temp_diff_modzs, 'r1_power_modzs': r1_power_modzs,
             'r2_shape_modzs': r2_shape_modzs, 'r2_temp_var_modzs': r2_temp_var_modzs,
             'r2_temp_diff_modzs': r2_temp_diff_modzs, 'r2_power_modzs': r2_power_modzs}
    spectra = {'median_spectra_normed': median_spectra_normed, 'mad_spectra_normed': mad_spectra_normed,
               'median_abs_diff_spectra_normed': median_abs_diff_spectra_normed, 'median_spectra': median_spectra,
               'mean_spectra_normed': mean_spectra_normed, 'std_spectra_normed': std_spectra_normed,
               'mean_abs_diff_spectra_normed': mean_abs_diff_spectra_normed, 'mean_spectra': mean_spectra}
    parameters = {'median_round_modz_cut': median_round_modz_cut, 'mean_round_modz_cut': mean_round_modz_cut,
                  'edge_cut': edge_cut, 'Kt': Kt, 'Kf': Kf, 'sig_init': sig_init, 'sig_adj': sig_adj,'chan_thresh_frac': chan_thresh_frac}

    out_dict = {'ex_ants': ex_ants, 'modzs': modzs, 'spectra': spectra, 'flags': flags, 
                'datafile_list': raw_auto_files, 'parameters': parameters, 'history': history}
    write_metric_file(metric_outfile, out_dict, overwrite=overwrite)
    return ex_ants, modzs, spectra, flags
