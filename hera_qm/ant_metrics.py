# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics."""
import numpy as np
from copy import deepcopy
import os
import re
from collections import OrderedDict
from .version import hera_qm_version_str
from . import utils, metrics_io


def get_ant_metrics_dict():
    """Return dictionary of metric names and descriptions.

    Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns
    -------
    metrics_dict : dict
        Dictionary with metric names as keys and descriptions as values.

    """
    metrics_dict = {'ant_metrics_meanVij': 'Mean of the absolute value of all '
                                           'visibilities associated with an '
                                           'antenna.',
                    'ant_metrics_meanVijXPol': 'Ratio of mean cross-pol '
                                               'visibilities to mean same-pol '
                                               'visibilities: '
                                               '(Vxy+Vyx)/(Vxx+Vyy).',
                    'ant_metrics_mod_z_scores_meanVij': 'Modified z-score of '
                                                        'the mean of the '
                                                        'absolute value of '
                                                        'all visibilities '
                                                        'associated with an '
                                                        'antenna.',
                    'ant_metrics_mod_z_scores_meanVijXPol': 'Modified z-score '
                                                            'of the ratio of '
                                                            'mean cross-pol '
                                                            'visibilities '
                                                            'to mean same-pol '
                                                            'visibilities: '
                                                            '(Vxy+Vyx)/'
                                                            '(Vxx+Vyy).',
                    'ant_metrics_crossed_ants': 'Antennas deemed to be '
                                                'cross-polarized by '
                                                'hera_qm.ant_metrics.',
                    'ant_metrics_removal_iteration': 'hera_qm.ant_metrics '
                                                     'iteration number in '
                                                     'which the antenna '
                                                     'was removed.',
                    'ant_metrics_xants': 'Antennas deemed bad by '
                                         'hera_qm.ant_metrics.',
                    'ant_metrics_dead_ants': 'Antennas deemed to be dead by '
                                             'hera_qm.ant_metrics.'}
    return metrics_dict

#######################################################################
# Low level functionality that is potentially reusable
#######################################################################


def per_antenna_modified_z_scores(metric):
    """Compute modified Z-Score over antennas for each antenna polarization.

    This function computes the per-pol modified z-score of the given metric
    dictionary for each antenna.

    The modified Z-score is defined as:
        0.6745 * (metric - median(all_metrics))/ median_absoulte_deviation

    Parameters
    ----------
    metric : dict
        Dictionary of metric data to compute z-score. Keys are expected to
        have the form: (ant, antpol)

    Returns
    -------
    zscores : dict
        Dictionary of z-scores for the given data.

    """
    zscores = {}
    antpols = set([key[1] for key in metric])
    for antpol in antpols:
        values = np.array([val for (key, val) in metric.items()
                           if key[1] == antpol])
        median = np.nanmedian(values)
        medAbsDev = np.nanmedian(np.abs(values - median))
        for (key, val) in metric.items():
            if key[1] == antpol:
                # this factor makes it comparable to a
                # standard z-score for gaussian data
                zscores[key] = 0.6745 * (val - median) / medAbsDev
    return zscores


def time_freq_abs_vis_stats(data, flags=None, time_alg=np.nanmedian, freq_alg=np.nanmedian):
    """Summarize visibility magnitudes as a single number for quick comparison to others.

    Parameters
    ----------
    data : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs)
    flags : dictionary or hera_cal DataContainer, optional
        If not None, should have the same keys and same array shapes as data
    time_alg : function, optional
        Function used to reduce a 2D or 1D numpy array to a single number.
        To handle flags properly, should be the "nan" version of the function.
    freq_alg : function, optional
        Function that reuduces a 1D array to a single number or a 2D array to a 1D
        array using the axis kwarg. If its the same as time_alg, the 2D --> float
        version will be used (no axis kwarg). To handle flags properly, should be 
        the "nan" version of the function.

    Returns
    -------
    abs_vis_stats : dictionary
        Dictionary mapping baseline keys e.g. (0, 1, 'ee') to single floats representing
        visibility amplitudes.

    """
    abs_vis_stats = {}
    for bl in data:
        data_here = deepcopy(data[bl])
        data_here[~np.isfinite(data_here)] = np.nan
        if flags is not None:
            data_here[flags[bl]] = np.nan
            
        med_abs_vis = np.nanmedian(np.abs(data_here))
        if med_abs_vis == 0:
            abs_vis_stats[bl] = 0
        else:
            if time_alg == freq_alg:  # if they are the algorithm, do it globally
                abs_vis_stats[bl] = time_alg(np.abs(data_here))
            else:
                abs_vis_stats[bl] = time_alg(freq_alg(np.abs(data_here), axis=1))
    return abs_vis_stats


def mean_Vij_metrics(abs_vis_stats, xants=[], pols=None, rawMetric=False):
    """Calculate how an antennas's average |Vij| deviates from others.

    Parameters
    ----------
    abs_vis_stats : dictionary
        Dictionary mapping baseline tuple e.g. (0, 1, 'ee') to 
        mean absolute value of visibilites over time and frequency.
    xants : list of ints or tuples, optional
        Antenna numbers or tuples e.g. (1, 'Jee') to exclude from metrics
    pols : list of str, optional
        List of visibility polarizations (e.g. ['ee','en','ne','nn']).
        Default None means all visibility polarizations are used.
    rawMetric : bool, optional
        If True, return the raw mean Vij metric instead of the modified z-score.
        Default is False.

    Returns
    -------
    meanMetrics : dict
        Dictionary indexed by (ant, antpol) of the modified z-score of the
        mean of the absolute value of all visibilities associated with an antenna.
        Very small or very large numbers are probably bad antennas.

    """
    from hera_cal.utils import split_pol, split_bl
    
    # figure out which antennas match pols and and are not in xants
    if pols is not None:
        antpols = set([ap for bl in abs_vis_stats for ap in split_pol(bl[2])])
    ants = set()
    for bl in abs_vis_stats:
        for ant in split_bl(bl):
            if (ant not in xants) and (ant[0] not in xants):
                if (pols is None) or (ant[1] in antpols):
                    ants.add(ant)
    
    # assign visibility means to each antenna in the baseline 
    per_ant_means = {ant: [] for ant in ants}
    for bl, vis_mean in abs_vis_stats.items():
        if bl[0] == bl[1]:
            continue # ignore autocorrelations
        if (pols is None) or (bl[2] in pols):
            for ant in split_bl(bl):
                if ant in ants:
                    per_ant_means[ant].append(vis_mean)
    per_ant_means = {ant: np.nanmean(per_ant_means[ant]) for ant in ants}

    if rawMetric:
        return per_ant_means
    else:
        return per_antenna_modified_z_scores(per_ant_means)


def antpol_metric_sum_ratio(cross_metrics, same_metrics):
    """Compute ratio of two metrics summed over polarizations.

    Takes the ratio of two antenna metrics, summed over both polarizations, and creates
    a new antenna metric with the same value in both polarizations for each antenna.

    Parameters
    ----------
    cross_metrics : dict
        Dict of a metrics computed with cross-polarizaed antennas. Keys are of
        the form (ant, antpol) and must match same_metrics keys.
    same_metrics : dict
        Dict of a metrics computed with non-cross-polarized antennas. Keys are of
        the form (ant, antpol) and must match cross_metrics keys.

    Returns
    -------
    cross_pol_ratio
        Dictionary of the ratio between the sum of cross_metrics and sum of same_metrics
        for each antenna provided in ants. Keys are of the form (ant, antpol) and will
        be identical for both polarizations for a given antenna by construction

    """
def mean_Vij_cross_pol_metrics(data, pols, antpols, ants, bls, xants=[],
                               rawMetric=False):
    # figure out antenna numbers and polarizations in the metrics
    antnums = set([ant[0] for metric in [cross_metrics, same_metrics] for ant in metric])
    antpols = set([ant[1] for metric in [cross_metrics, same_metrics] for ant in metric])

    # compute cross_pol_ratios
    cross_pol_ratio = {}
    for an in antnums:
        cross_sum = np.sum([cross_metrics[(an, ap)] for ap in antpols])
        same_sum = np.sum([same_metrics[(an, ap)] for ap in antpols]) 
        for ap in antpols:
            cross_pol_ratio[(an, ap)] = cross_sum / same_sum
    return cross_pol_ratio


    """Calculate the ratio of cross-pol visibilities to same-pol visibilities.

    Find which antennas are outliers based on the ratio of mean cross-pol
    visibilities to mean same-pol visibilities:
        (Vxy+Vyx)/(Vxx+Vyy).

    Parameters
    ----------
    data : array or HERAData object
        Data for all polarizations, stored in a format that supports indexing
        as data[ant1, ant2, pol].
    pols : list of str
        List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols : list of str
        List of antenna polarizations (e.g. ['x', 'y']).
    ants : list of ints
        List of all antenna indices.
    bls : list of tuples
        List of tuples of antenna pairs.
    xants : list of tuples, optional
        List of antenna-polarization tuples that should be ignored. The
        expected format is (ant, antpol). Note that if, e.g., (81, "y") is
        excluded, then (81, "x") cannot be identified as cross-polarized and
        will be exluded as well. Default is empty list.
    rawMetric : bool, optional
        If True, return the raw power ratio instead of the modified z-score.
        Default is False.

    Returns
    -------
    mean_Vij_cross_pol_metrics : dict
        Dictionary indexed by (ant, antpol) keys. Contains the modified z-scores
        of the ratio of mean visibilities, (Vxy*Vyx)/(Vxx*Vyy). Results are
        duplicated in both antpols. Very large values are likely cross-polarized.

    """
    # Compute metrics and cross pols only and and same pols only
    samePols = [pol for pol in pols if pol[0] == pol[1]]
    crossPols = [pol for pol in pols if pol[0] != pol[1]]
    full_xants = exclude_partially_excluded_ants(antpols, xants)
    meanVijMetricsSame = mean_Vij_metrics(data, samePols, antpols, ants, bls,
                                          xants=full_xants, rawMetric=True)
    meanVijMetricsCross = mean_Vij_metrics(data, crossPols, antpols, ants, bls,
                                           xants=full_xants, rawMetric=True)

    # Compute the ratio of the cross/same metrics,
    # saving the same value in each antpol
    crossPolRatio = antpol_metric_sum_ratio(ants, antpols, meanVijMetricsCross,
                                            meanVijMetricsSame,
                                            xants=full_xants)
    if rawMetric:
        return crossPolRatio
    else:
        return per_antenna_modified_z_scores(crossPolRatio)


def load_antenna_metrics(filename):
    """Load cut decisions and metrics from an HDF5 into python dictionary.

    Loading is handled via hera_qm.metrics_io.load_metric_file

    Parameters
    ----------
    filename : str
        Full path to the filename of the metric to load. Must be either
        HDF5 (recommended) or JSON (Depreciated in Future) file type.

    Returns
    -------
    metrics : dict
        Dictionary of metrics stored in the input file.

    """
    return metrics_io.load_metric_file(filename)


#######################################################################
# High level functionality for HERA
#######################################################################


class AntennaMetrics():
    """Container for holding data and meta-data for ant metrics calculations.

    This class creates an object for holding relevant visibility data and metadata,
    and provides interfaces to four antenna metrics: two identify dead antennas,
    and two identify cross-polarized antennas. These metrics can be used iteratively
    to identify bad antennas. The object handles all stroage of metrics, and supports
    writing metrics to an HDF5 filetype. The analysis functions are designed to work
    on raw data from a single observation with all four polarizations.

    """

    def __init__(self, dataFileList, fileformat='miriad'):
        """Initilize an AntennaMetrics object.

        Parameters
        ----------
        dataFileList : list of str
            List of data filenames of the four different visibility polarizations
            for the same observation.
        format : str, optional
            File type of data. Must be one of: 'miriad', 'uvh5', 'uvfits', 'fhd',
            'ms' (see pyuvdata docs). Default is 'miriad'.

        Attributes
        ----------
        hd : HERAData
            HERAData object generated from dataFileList.
        data : array
            Data contained in HERAData object.
        flags : array
            Flags contained in HERAData object.
        nsamples : array
            Nsamples contained in HERAData object.
        ants : list of ints
            List of antennas in HERAData object.
        pols : list of str
            List of polarizations in HERAData object.
        bls : list of ints
            List of baselines in HERAData object.
        dataFileList : list of str
            List of data filenames of the four different visibility polarizations
            for the same observation.
        version_str : str
            The version of the hera_qm module used to generate these metrics.
        history : str
            History to append to the metrics files when writing out files.

        """
        from hera_cal.io import HERAData

        self.hd = HERAData(dataFileList, filetype=fileformat)

        self.data, self.flags, self.nsamples = self.hd.read()
        self.ants = self.hd.get_ants()
        self.pols = [pol.lower() for pol in self.hd.get_pols()]
        self.antpols = [antpol.lower() for antpol in self.hd.get_feedpols()]
        self.bls = self.hd.get_antpairs()
        self.dataFileList = dataFileList
        self.version_str = hera_qm_version_str
        self.history = ''

        if len(self.antpols) != 2 or len(self.pols) != 4:
            raise ValueError('Missing polarization information. pols ='
                             + str(self.pols) + ' and antpols = '
                             + str(self.antpols))

    def mean_Vij_metrics(self, pols=None, xants=[], rawMetric=False):
        """Calculate how an antennas's average |Vij| deviates from others.

        Local wrapper for mean_Vij_metrics in hera_qm.ant_metrics module

        Parameters
        ----------
        pols : list of str, optional
            List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
            Default is self.pols.
        xants : list of tuples, optional
            List of antenna-polarization tuples that should be ignored. The
            expected format is (ant, antpol). Default is empty list.
        rawMetric : bool, optional
            If True, return the raw mean Vij metric instead of the modified z-score.
            Default is False.

        Returns
        -------
        meanMetrics : dict
            Dictionary indexed by (ant, antpol) keys. Contains the modified z-score
            of the mean of the absolute value of all visibilities associated with
            an antenna. Very small or very large numbers are probably bad antennas.

        """
        if pols is None:
            pols = self.pols
        return mean_Vij_metrics(self.data, pols, self.antpols,
                                self.ants, self.bls, xants=xants,
                                rawMetric=rawMetric)

    def mean_Vij_cross_pol_metrics(self, xants=[], rawMetric=False):
        """Calculate the ratio of cross-pol visibilities to same-pol visibilities.

        This method is a local wrapper for mean_Vij_cross_pol_metrics. It finds
        which antennas are outliers based on the ratio of mean cross-pol visibilities
        to mean same-pol visibilities:
            (Vxy+Vyx)/(Vxx+Vyy).

        Parameters
        ----------
        xants : list of tuples, optional
            List of antenna-polarization tuples that should be ignored. The
            expected format is (ant, antpol). Default is empty list.
        rawMetric : bool, optional
            If True, return the raw power correlations instead of the modified z-score.
            Default is False.

        Returns
        -------
        mean_Vij_cross_pol_metrics : dict
            Dictionary indexed by (ant,antpol) keys. Contains the modified z-scores of the
            ratio of mean visibilities, (Vxy+Vyx)/(Vxx+Vyy). Results are duplicated in
            both antpols. Very large values are likely cross-polarized.

        """
        return mean_Vij_cross_pol_metrics(self.data, self.pols,
                                          self.antpols, self.ants,
                                          self.bls, xants=xants,
                                          rawMetric=rawMetric)

    def reset_summary_stats(self):
        """Reset all the internal summary statistics back to empty."""
        self.xants, self.crossedAntsRemoved, self.deadAntsRemoved = [], [], []
        self.iter = 0
        self.removalIter = {}
        self.allMetrics, self.allModzScores = OrderedDict(), OrderedDict()
        self.finalMetrics, self.finalModzScores = {}, {}

    def find_totally_dead_ants(self):
        """Flag antennas whose median autoPower is 0.0.

        These antennas are marked as dead. They do not appear in recorded antenna
        metrics or zscores. Their removal iteration is -1 (i.e. before iterative
        flagging).
        """
        autoPowers = {bl: np.median(np.mean(np.abs(self.data[bl])**2, axis=0))
                      for bl in self.data.keys()}
        power_list_by_ant = {(ant, antpol): []
                             for ant in self.ants
                             for antpol in self.antpols
                             if (ant, antpol) not in self.xants}
        for ((ant0, ant1, pol), power) in autoPowers.items():
            if ((ant0, pol[0]) not in self.xants
                    and (ant1, pol[1]) not in self.xants):
                power_list_by_ant[(ant0, pol[0])].append(power)
                power_list_by_ant[(ant1, pol[1])].append(power)
        for (key, val) in power_list_by_ant.items():
            if np.median(val) == 0:
                self.xants.append(key)
                self.deadAntsRemoved.append(key)
                self.removalIter[key] = -1

    def _run_all_metrics(self, run_cross_pols=True, run_cross_pols_only=False):
        """Local call for all metrics as part of iterative flagging method.

        Parameters
        ----------
        run_cross_pols : bool, optional
            Define if mean_Vij_cross_pol_metrics is executed. Default is True.
        run_cross_pols_only : bool, optional
            Define if mean_Vij_cross_pol_metrics is the *only* metric to be run.
            Default is False.

        """
        # Compute all raw metrics
        metNames = []
        metVals = []

        if not run_cross_pols_only:
            metNames.append('meanVij')
            meanVij = self.mean_Vij_metrics(pols=self.pols,
                                            xants=self.xants,
                                            rawMetric=True)
            metVals.append(meanVij)

        if run_cross_pols:
            metNames.append('meanVijXPol')
            meanVijXPol = self.mean_Vij_cross_pol_metrics(xants=self.xants,
                                                          rawMetric=True)
            metVals.append(meanVijXPol)

        # Save all metrics and zscores
        metrics, modzScores = {}, {}
        for metric, metName in zip(metVals, metNames):
            metrics[metName] = metric
            modz = per_antenna_modified_z_scores(metric)
            modzScores[metName] = modz
            for key in metric:
                if metName in self.finalMetrics:
                    self.finalMetrics[metName][key] = metric[key]
                    self.finalModzScores[metName][key] = modz[key]
                else:
                    self.finalMetrics[metName] = {key: metric[key]}
                    self.finalModzScores[metName] = {key: modz[key]}
        self.allMetrics.update({self.iter: metrics})
        self.allModzScores.update({self.iter: modzScores})

    def iterative_antenna_metrics_and_flagging(self, crossCut=5, deadCut=5,
                                               alwaysDeadCut=10,
                                               verbose=False,
                                               run_cross_pols=True,
                                               run_cross_pols_only=False):
        """Run both Mean Vij and Mean Vij crosspol metrics and stores results in self.

        Parameters
        ----------
        crossCut : float, optional
            Modified z-score cut for most cross-polarized antennas. Default is 5 "sigmas".
        deadCut : float, optional
            Modified z-score cut for most likely dead antennas. Default is 5 "sigmas".
        alwaysDeadCut : float, optional
            Modified z-score cut for definitely dead antennas. Default is 10 "sigmas".
            These are all thrown away at once without waiting to iteratively throw away
            only the worst offender.
        run_cross_pols : bool, optional
            Define if mean_Vij_cross_pol_metrics is executed. Default is True.
        run_cross_pols_only : bool, optional
            Define if mean_Vij_cross_pol_metrics is the *only* metric to be run.
            Default is False.

        """
        self.reset_summary_stats()
        self.find_totally_dead_ants()
        self.crossCut, self.deadCut = crossCut, deadCut
        self.alwaysDeadCut = alwaysDeadCut

        # Loop over
        for iter in range(len(self.antpols) * len(self.ants)):
            self.iter = iter
            self._run_all_metrics(run_cross_pols=run_cross_pols,
                                  run_cross_pols_only=run_cross_pols_only)

            # Mostly likely dead antenna
            last_iter = list(self.allModzScores)[-1]
            worstDeadCutRatio = -1
            worstCrossCutRatio = -1

            if not run_cross_pols_only:
                deadMetrics = {ant: np.abs(metric) for ant, metric
                               in self.allModzScores[last_iter]['meanVij'].items()}
            try:
                worstDeadAnt = max(deadMetrics, key=deadMetrics.get)
                worstDeadCutRatio = np.abs(deadMetrics[worstDeadAnt]) / deadCut
            except NameError:
                # Dead metrics weren't run, but that's fine.
                pass

            if run_cross_pols:
                # Most likely cross-polarized antenna
                crossMetrics = {ant: np.abs(metric) for ant, metric
                                in self.allModzScores[last_iter]['meanVijXPol'].items()}
                worstCrossAnt = max(crossMetrics, key=crossMetrics.get)
                worstCrossCutRatio = (np.abs(crossMetrics[worstCrossAnt]) / crossCut)

            # Find the single worst antenna, remove it, log it, and run again
            if (worstCrossCutRatio >= worstDeadCutRatio
                    and worstCrossCutRatio >= 1.0):
                for antpol in self.antpols:
                    self.xants.append((worstCrossAnt[0], antpol))
                    self.crossedAntsRemoved.append((worstCrossAnt[0], antpol))
                    self.removalIter[(worstCrossAnt[0], antpol)] = iter
                    if verbose:
                        print('On iteration', iter, 'we flag\t', end='')
                        print((worstCrossAnt[0], antpol))
            elif (worstDeadCutRatio > worstCrossCutRatio
                    and worstDeadCutRatio > 1.0):
                dead_ants = set([worstDeadAnt])
                for (ant, metric) in deadMetrics.items():
                    if metric > alwaysDeadCut:
                        dead_ants.add(ant)
                for dead_ant in dead_ants:
                    self.xants.append(dead_ant)
                    self.deadAntsRemoved.append(dead_ant)
                    self.removalIter[dead_ant] = iter
                    if verbose:
                        print('On iteration', iter, 'we flag', dead_ant)
            else:
                break

    def save_antenna_metrics(self, filename, overwrite=False):
        """Output all meta-metrics and cut decisions to HDF5 file.

        Saves all cut decisions and meta-metrics in an HDF5 that can be loaded
        back into a dictionary using hera_qm.ant_metrics.load_antenna_metrics()

        Parameters
        ----------
        filename : str
            The file into which metrics will be written.
        overwrite: bool, optional
            Whether to overwrite an existing file. Default is False.

        """
        if not hasattr(self, 'xants'):
            raise KeyError(('Must run AntennaMetrics.'
                            'iterative_antenna_metrics_and_flagging() first.'))

        out_dict = {'xants': self.xants}
        out_dict['crossed_ants'] = self.crossedAntsRemoved
        out_dict['dead_ants'] = self.deadAntsRemoved
        out_dict['final_metrics'] = self.finalMetrics
        out_dict['all_metrics'] = self.allMetrics
        out_dict['final_mod_z_scores'] = self.finalModzScores
        out_dict['all_mod_z_scores'] = self.allModzScores
        out_dict['removal_iteration'] = self.removalIter
        out_dict['cross_pol_z_cut'] = self.crossCut
        out_dict['dead_ant_z_cut'] = self.deadCut
        out_dict['always_dead_ant_z_cut'] = self.alwaysDeadCut
        out_dict['datafile_list'] = self.dataFileList

        metrics_io.write_metric_file(filename, out_dict, overwrite=overwrite)


def ant_metrics_run(files, crossCut=5.0, deadCut=5.0, alwaysDeadCut=10.0,
                    metrics_path='', extension='.ant_metrics.hdf5',
                    vis_format='uvh5', verbose=True, history='',
                    run_cross_pols=True, run_cross_pols_only=False):
    """
    Run a series of ant_metrics tests on a given set of input files.

    Note
    ----
    The funciton will take in a list of files and options. It will run the
    series of ant metrics tests, and produce an HDF5 file containing the
    relevant information.

    Parameters
    ----------
    files : list of str
        List of files to run ant metrics on, one at a time. Each must include
        all both polarizations (or all 4 if run_cross_pols is True).
    crossCut : float, optional
        Modified Z-Score limit to cut cross-polarized antennas. Default is 5.0.
    deadCut : float, optional
        Modifized Z-Score limit to cut dead antennas. Default is 5.0.
    alwaysDeadCut : float, optional
        Modified Z-Score limit for antennas that are definitely dead. Antennas with
        z-scores above this limit are thrown away before iterative flagging.
        Default is 10.0.
    metrics_path : str, optional
        Full path to directory to story output metric. Default is the same directory
        as input data files.
    extension : str, optional
        File extension to add to output files. Default is ant_metrics.hdf5.
    vis_format : str, optional
        File format of input visibility data. Must be one of: 'miriad', 'uvh5',
        'uvfits', 'fhd', 'ms' (see pyuvdata docs). Default is 'uvh5'.
    verbose : bool, optional
        If True, print out statements during iterative flagging. Default is True.
    history : str, optional
        The history the add to metrics. Default is nothing (empty string).
    run_cross_pols : bool, optional
        Define if mean_Vij_cross_pol_metrics is executed. Default is True.
    run_cross_pols_only : bool, optional
        Define if mean_Vij_cross_pol_metrics is the *only* metric to be run.
        Default is False.

    Returns
    -------
    None

    """
    for file in files:
        am = AntennaMetrics(file, fileformat=vis_format)
        am.iterative_antenna_metrics_and_flagging(crossCut=crossCut,
                                                  deadCut=deadCut,
                                                  alwaysDeadCut=alwaysDeadCut,
                                                  verbose=verbose,
                                                  run_cross_pols=run_cross_pols,
                                                  run_cross_pols_only=run_cross_pols_only)
        am.history = am.history + history

        metrics_basename = utils.strip_extension(os.path.basename(file)) + extension
        if metrics_path == '':
            # default path is same directory as file
            metrics_path = os.path.dirname(os.path.abspath(file))
        am.save_antenna_metrics(os.path.join(metrics_path, metrics_basename))
