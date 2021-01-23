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
                            for bl in data if (bl[0] not in ex_ants) and (bl[2] == pol)]
                wf_norm[pol].append(norm_func(row_list))
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
    '''Computes a modified z-score of a autocorrelation spectrum compared to all others not in ex_ants.
    
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
    mod_zs : dictionary
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
    mod_zs = {bl: 1.4826 * (diff_metrics[bl] - median_diff_metric) / mad_diff_metric for bl in auto_spectra}
    return mod_zs
