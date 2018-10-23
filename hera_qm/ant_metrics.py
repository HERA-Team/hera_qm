# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics."""
from __future__ import print_function, division, absolute_import

from six.moves import zip, range
import numpy as np
from copy import deepcopy
import json
import os
import re
from collections import OrderedDict
from .version import hera_qm_version_str
from . import utils, metrics_io
import h5py
import warnings


def get_ant_metrics_dict():
    """Return dictionary of metric names and descriptions.

    Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns:
    metrics_dict -- Dictionary with metric names as keys
                    and descriptions as values.
    """
    metrics_dict = {'ant_metrics_meanVij': 'Mean of the absolute value of all '
                                           'visibilities associated with an '
                                           'antenna.',
                    'ant_metrics_meanVijXPol': 'Ratio of mean cross-pol '
                                               'visibilities to mean same-pol '
                                               'visibilities: '
                                               '(Vxy+Vyx)/(Vxx+Vyy).',
                    'ant_metrics_redCorr': 'Extent to which baselines '
                                           'involving an antenna do not '
                                           'correlate with others they are '
                                           'nominmally redundant with.',
                    'ant_metrics_redCorrXPol': 'Mean correlation ratio between'
                                               'redundant visibilities and '
                                               'singlely-polarization '
                                               'flipped ones.',
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
                    'ant_metrics_mod_z_scores_redCorr': 'Modified z-score of '
                                                        'the extent to which '
                                                        'baselines involving '
                                                        'an antenna do not '
                                                        'correlate with others'
                                                        ' they are nominally '
                                                        'redundant with.',
                    'ant_metrics_mod_z_scores_redCorrXPol': 'Modified z-score '
                                                            'of the mean '
                                                            'correlation ratio'
                                                            ' between '
                                                            'redundant '
                                                            'visibilities and '
                                                            'singlely-'
                                                            'polarization '
                                                            'flipped ones.',
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


    Computes the per-pol modified z-score of the given metric dictionary for each antenna.

    modified Z-score = 0.6745 (metric - median(all_metrics))/ median_absoulte_deviation
        pass
    Argument:
        metric: Dictionary of metric data to compute z-score
                keys are expected to have form: (ant, antpol)
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


def mean_Vij_metrics(data, pols, antpols, ants, bls,
                     xants=[], rawMetric=False):
    """Calculate how an antennas's average |Vij| deviates from others.

    Arguments:
        data: data for all polarizations
                format that must support data[i,j,pol]
        pols : List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
        antpols: List of antenna polarizations (e.g. ['x', 'y'])
        ants: List of all antenna indices.
        bls: List of tuples of antenna pairs.
        xants: List of antennas ithat should be ignored.
               format: (ant,antpol)
        rawMetric:return the raw mean Vij metric instead of the modified z-score

    Returns:
        meanMetrics: Dictionary indexed by (ant,antpol) of the modified z-score
                     of the mean of the absolute value of all visibilities associated with an antenna.
                     Very small or very large numbers are probably bad antennas.
    """
    absVijMean = {(ant, antpol): 0.0 for ant in ants for antpol in antpols if
                  (ant, antpol) not in xants}
    visCounts = deepcopy(absVijMean)
    for (i, j) in bls:
        if i == j:
            continue
        for pol in pols:
            ants = list(zip((i, j), pol))
            if all([ant in xants for ant in ants]):
                continue
            d = data[i, j, pol]
            s = np.nansum(np.abs(d))
            for ant, antpol in ants:
                if (ant, antpol) in xants:
                    continue
                absVijMean[(ant, antpol)] += s
                visCounts[(ant, antpol)] += np.isfinite(d).sum()
    timeFreqMeans = {key: absVijMean[key] / visCounts[key]
                     for key in absVijMean}

    if rawMetric:
        return timeFreqMeans
    else:
        return per_antenna_modified_z_scores(timeFreqMeans)


def compute_median_auto_power_dict(data, pols, reds):
    """Compute the frequency median of the time averaged visibility squared.

    Arguments:
        data: Dictionary of visibility data.
              keys are in the form (ant1, ant2, pol)
        pols: List of polarizations to compute the median auto power.
              allowed values ['xx', 'yy', 'xy', 'yx']
        reds: List of lists of tuples of antenna numbers that make up redundant baseline groups.

    Returns:
        autoPower: Dictionary of the meidan of time average visibility squared.
                   keys are in the form (ant1, ant2, pol)
    """
    autoPower = {}
    for pol in pols:
        for bls in reds:
            for (i, j) in bls:
                tmp_power = np.abs(data[i, j, pol])**2
                autoPower[i, j, pol] = np.median(np.mean(tmp_power, axis=0))
    return autoPower


def red_corr_metrics(data, pols, antpols, ants, reds, xants=[],
                     rawMetric=False, crossPol=False):
    """Calculate modified Z-Score over all redundant groups for each antenna.

    Calculates the extent to which baselines involving an antenna
    do not correlate with others they are nominmally redundant with.

    Arguments:
    data: data for all polarizations
          format must support data[i,j,pol]
    pols: List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols: List of antenna polarizations (e.g. ['x', 'y'])
    ants: List of all antenna indices.
    reds: List of lists of tuples of antenna numbers that make up redundant baseline groups.
    xants: List of antennas that should be ignored.
            format: (ant, antpol)
    rawMetric: return the raw power correlations instead of the modified z-score
    crossPol: return results only when the two visibility polarizations differ by a single flip

    Returns:
    powerRedMetric: Dictionary indexed by (ant,antpol)
                    of the modified z-scores of the mean power correlations
                    inside redundant baseline groups
                    associated with each antenna.
                    Very small numbers are probably bad antennas.
    """
    # Compute power correlations and assign them to each antenna
    autoPower = compute_median_auto_power_dict(data, pols, reds)
    antCorrs = {(ant, antpol): 0.0 for ant in ants for antpol in antpols if
                (ant, antpol) not in xants}
    antCounts = deepcopy(antCorrs)
    for pol0 in pols:
        for pol1 in pols:
            iscrossed_i = (pol0[0] != pol1[0])
            iscrossed_j = (pol0[1] != pol1[1])
            onlyOnePolCrossed = (iscrossed_i ^ iscrossed_j)
            # This function can instead record correlations
            # for antennas whose counterpart are pol-swapped
            if ((not crossPol and (pol0 is pol1))
                    or (crossPol and onlyOnePolCrossed)):
                for bls in reds:
                    for n, (ant0_i, ant0_j) in enumerate(bls):
                        data0 = data[ant0_i, ant0_j, pol0]
                        for (ant1_i, ant1_j) in bls[n + 1:]:
                            data1 = data[ant1_i, ant1_j, pol1]
                            corr = np.nanmedian(np.abs(np.nanmean(data0 * data1.conj(), axis=0)))
                            corr /= np.sqrt(autoPower[ant0_i, ant0_j, pol0]
                                            * autoPower[ant1_i, ant1_j, pol1])
                            antsInvolved = [(ant0_i, pol0[0]),
                                            (ant0_j, pol0[1]),
                                            (ant1_i, pol1[0]),
                                            (ant1_j, pol1[1])]
                            if not np.any([(ant, antpol) in xants
                                           for ant, antpol in antsInvolved]):
                                # Only record the crossed antenna
                                # if i or j is crossed
                                if crossPol and iscrossed_i:
                                    antsInvolved = [(ant0_i, pol0[0]),
                                                    (ant1_i, pol1[0])]
                                elif crossPol and iscrossed_j:
                                    antsInvolved = [(ant0_j, pol0[1]),
                                                    (ant1_j, pol1[1])]
                                for ant, antpol in antsInvolved:
                                    antCorrs[(ant, antpol)] += corr
                                    antCounts[(ant, antpol)] += 1

    # Compute average and return
    for key, count in antCounts.items():
        if count > 0:
            antCorrs[key] /= count
        else:
            # Was not found in reds, should not have a valid metric.
            antCorrs[key] = np.NaN
    if rawMetric:
        return antCorrs
    else:
        return per_antenna_modified_z_scores(antCorrs)


def exclude_partially_excluded_ants(antpols, xants):
    """Create list of excluded antenna polarizations.

    Arguments:
        antpols: Single antenna polarizations to add to excluded Antennas
                 list of ['x','y'], ['x'], or ['y']
        xants: List of antennas ithat should be ignored.
               format: (ant,antpol)
    Retruns:
        xantSet: List of all antenna-polarization combinations to exclude.
    """
    xantSet = set(xants)
    for xant in xants:
        for antpol in antpols:
            xantSet.add((xant[0], antpol))
    return list(xantSet)


def antpol_metric_sum_ratio(ants, antpols, crossMetrics, sameMetrics,
                            xants=[]):
    """Compute ratio of two metrics summed over polarizations.

    Takes the ratio of two antenna metrics, summed over both polarizations, and creates a new
    antenna metric with the same value in both polarizations for each antenna.

    Arguments:
        ants: List of all antenna indices.
        antpols: List of antenna polarizations (e.g. ['x', 'y'])
        crossMetrics: Dict of a metrics computed with cross-polarizaed antennas
                      keys: (ant, antpol)
        sameMetrics: Dict of a metrics computed with non-cross-polarized antennas
                     keys: (ant, antpol)
        xants: List of antennas ithat should be ignored.
               format: (ant,antpol)

    Returns:
        crossPolRatio: Dictionary of the ratio between the sum ocrossMetrics and  sum of sameMetrics
                       for each antenna provided in ants
                       key: (ant, antpol)

    """
    crossPolRatio = {}
    for ant in ants:
        if np.all([(ant, antpol) not in xants for antpol in antpols]):
            crossSum = np.sum([crossMetrics[(ant, antpol)]
                               for antpol in antpols])
            sameSum = np.sum([sameMetrics[(ant, antpol)]
                              for antpol in antpols])
            for antpol in antpols:
                crossPolRatio[(ant, antpol)] = crossSum / sameSum
    return crossPolRatio


def mean_Vij_cross_pol_metrics(data, pols, antpols, ants, bls, xants=[],
                               rawMetric=False):
    """Calculate the ratio of cross-pol visibilities to same-pol visibilities.

    Find which antennas are outliers based on the
    ratio of mean cross-pol visibilities to mean same-pol visibilities:
    (Vxy+Vyx)/(Vxx+Vyy).

    Arguments:
    data: data for all polarizations
          format must support data[i,j,pol]
    pols: List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols: List of antenna polarizations (e.g. ['x', 'y'])
    ants: List of all antenna indices.
    bls: List of tuples of antenna pairs.
    xants: List of antennas  that should be ignored.
           format (ant,antpol) format
           e.g.,  if (81,'y') is excluded, (81,'x') cannot be identified
                  as cross-polarized and will be excluded.
    rawMetric: return the raw power ratio instead of the modified z-score

    Returns:
    mean_Vij_cross_pol_metrics: Dictionary indexed by (ant,antpol)
                                The modified z-scores of the
                                ratio of mean visibilities,
                                (Vxy+Vyx)/(Vxx+Vyy).
                                Results duplicated in both antpols.
                                Very large values are likely cross-polarized.
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


def red_corr_cross_pol_metrics(data, pols, antpols, ants, reds, xants=[],
                               rawMetric=False):
    """Calculate modified Z-Score over redundant groups; assume cross-polarized.

    Find which antennas are part of visibilities that are significantly better
    correlated with polarization-flipped visibilities in a redundant groupself.
    Returns the modified z-score.

    Arguments:
    data: data for all polarizations format must support data[i,j,pol]
    pols: List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols: List of antenna polarizations (e.g. ['x', 'y'])
    ants: List of all antenna indices.
    reds: List of lists of tuples of antenna numbers
          that make up redundant baseline groups.
    xants: List of antennas  that should be ignored.
           format (ant,antpol) format
           e.g.,  if (81,'y') is excluded, (81,'x') cannot be identified
                  as cross-polarized and will be excluded.
    rawMetric: return the raw power ratio instead of the modified z-score
                type: Boolean
                Default: False

    Returns:
    redCorrCrossPolMetrics: Dictionary indexed by (ant,antpol)
                            The modified z-scores of the mean correlation
                            ratio between redundant visibilities
                            and singlely-polarization flipped ones.
                            Very large values are probably cross-polarized.
    """
    # Compute metrics for singly flipped pols and just same pols
    full_xants = exclude_partially_excluded_ants(antpols, xants)
    samePols = [pol for pol in pols if pol[0] == pol[1]]
    redCorrMetricsSame = red_corr_metrics(data, samePols, antpols,
                                          ants, reds,
                                          xants=full_xants,
                                          rawMetric=True)
    redCorrMetricsCross = red_corr_metrics(data, pols, antpols,
                                           ants, reds,
                                           xants=full_xants,
                                           rawMetric=True,
                                           crossPol=True)

    # Compute the ratio of the cross/same metrics
    # saving the same value in each antpol
    crossPolRatio = antpol_metric_sum_ratio(ants, antpols,
                                            redCorrMetricsCross,
                                            redCorrMetricsSame,
                                            xants=full_xants)
    if rawMetric:
        return crossPolRatio
    else:
        return per_antenna_modified_z_scores(crossPolRatio)


def average_abs_metrics(metrics1, metrics2):
    """Average the absolute value of two metrics together.

    Input dictionairies are averaged for each key. All keys must match exactly.
    Arguments:
        metrics1: Dictionairy of metric data to average
        metrics2: Dictionairy of metric data to average

    Returns:
        mean_metrics: Dictionary with the same keys as inputs.
                      values are mean of both input dictionaries for each key.
    """
    if set(list(metrics1)) != set(list(metrics2)):
        raise KeyError(('Metrics being averaged have differnt '
                        '(ant,antpol) keys.'))
    return {key: np.nanmean([np.abs(metrics1[key]),
                             np.abs(metrics2[key])])
            for key in metrics1}


def load_antenna_metrics(filename):
    """Load cut decisions and metrics from an HDF5 into python dictionary.

    Loading is handled via hera_qm.metrics_io.load_metric_file
    Arguments:
        filename: Filename of the metric to load.
                  Must be either HDF5 (recommended) or JSON (Depreciated in Future) types.

    Returns:
        Dictionary of metrics stored in the input file.
    """
    return metrics_io.load_metric_file(filename)


#######################################################################
# High level functionality for HERA
#######################################################################


class AntennaMetrics():
    """Container for holding data and meta-data for ant metrics calculations.

    Object for holding relevant visibility data and metadata
    interfaces to four antenna metrics:
        Two each for identifying dead or cross-polarized antennas.
    Can iteratively identifying bad antennas
    Handles all metrics stroage, and supports writing metrics to HDF5.
    Works on raw data from a single observation with all four polarizations.
    """

    def __init__(self, dataFileList, reds, fileformat='miriad'):
        """Initilize an AntennaMetrics object.

        Arguments:
            dataFileList: List of data filenames of the four different visibility polarizations for the same observation
            reds: List of lists of tuples of antenna numbers that make up redundant baseline groups.
            format: File type of data
                    Supports: 'miriad', 'uvh5', 'uvfits', 'fhd', 'ms ' (see pyuvdata docs)
                    Default: 'miriad'.
        """
        from hera_cal.io import HERAData

        self.hd = HERAData(dataFileList, filetype=fileformat)

        self.data, self.flags, self.nsamples = self.hd.read()
        self.ants = self.hd.get_ants()
        self.pols = [pol.lower() for pol in self.hd.get_pols()]
        self.antpols = [antpol.lower() for antpol in self.hd.get_feedpols()]
        self.bls = self.hd.get_antpairs()
        self.dataFileList = dataFileList
        self.reds = reds
        self.version_str = hera_qm_version_str
        self.history = ''

        if len(self.antpols) is not 2 or len(self.pols) is not 4:
            raise ValueError('Missing polarization information. pols ='
                             + str(self.pols) + ' and antpols = '
                             + str(self.antpols))

    def mean_Vij_metrics(self, pols=None, xants=[], rawMetric=False):
        """Calculate how an antennas's average |Vij| deviates from others.

        Local wrapper for mean_Vij_metrics in hera_qm.ant_metrics module
        Arguments:

            pols : List of visibility polarizations (e.g. ['xx','xy','yx','yy'])
                   Default: self.pols
            xants: List of antennas ithat should be ignored.
                   format: (ant,antpol)
            rawMetric:return the raw mean Vij metric instead of the modified z-score

        Returns:
            meanMetrics: Dictionary indexed by (ant,antpol) of the modified z-score
                         of the mean of the absolute value of all visibilities associated with an antenna.
                         Very small or very large numbers are probably bad antennas.
        """
        if pols is None:
            pols = self.pols
        return mean_Vij_metrics(self.data, pols, self.antpols,
                                self.ants, self.bls, xants=xants,
                                rawMetric=rawMetric)

    def red_corr_metrics(self, pols=None, xants=[], rawMetric=False,
                         crossPol=False):
        """Calculate modified Z-Score over all redundant groups for each antenna.

        Local wrapper for red_corr_metrics in hera_qm.ant_metrics module.
        Calculates the extent to which baselines involving an antenna
        do not correlate with others they are nominmally redundant with.

        Arguments:
            data: data for all polarizations
                  format must support data[i,j,pol]
            pols: List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
                  Default: self.pols
            xants: List of antennas that should be ignored.
                   format: (ant, antpol)
            rawMetric: return the raw power correlations instead of the modified z-score
            crossPol: return results only when the two visibility polarizations differ by a single flip

        Returns:
            powerRedMetric: Dictionary indexed by (ant,antpol)
                            of the modified z-scores of the mean power correlations
                            inside redundant baseline groups
                            associated with each antenna.
                            Very small numbers are probably bad antennas.
        """
        if pols is None:
            pols = self.pols
        return red_corr_metrics(self.data, pols, self.antpols,
                                self.ants, self.reds, xants=xants,
                                rawMetric=rawMetric, crossPol=crossPol)

    def mean_Vij_cross_pol_metrics(self, xants=[], rawMetric=False):
        """Calculate the ratio of cross-pol visibilities to same-pol visibilities.

        Local wrapper for mean_Vij_cross_pol_metrics.
        Find which antennas are outliers based on the
        ratio of mean cross-pol visibilities to mean same-pol visibilities:
        (Vxy+Vyx)/(Vxx+Vyy).

        Arguments:
            xants: List of antennas  that should be ignored.
                   format (ant,antpol) format
                   e.g.,  if (81,'y') is excluded, (81,'x') cannot be identified
                          as cross-polarized and will be excluded.
            rawMetric: return the raw power ratio instead of the modified z-score

        Returns:
            mean_Vij_cross_pol_metrics: Dictionary indexed by (ant,antpol)
                                        The modified z-scores of the
                                        ratio of mean visibilities,
                                        (Vxy+Vyx)/(Vxx+Vyy).
                                        Results duplicated in both antpols.
                                        Very large values are likely cross-polarized.
        """
        return mean_Vij_cross_pol_metrics(self.data, self.pols,
                                          self.antpols, self.ants,
                                          self.bls, xants=xants,
                                          rawMetric=rawMetric)

    def red_corr_cross_pol_metrics(self, xants=[], rawMetric=False):
        """Calculate modified Z-Score over redundant groups; assume cross-polarized.

        Local wrapper for red_corr_cross_pol_metrics.
        Find which antennas are part of visibilities that are significantly better
        correlated with polarization-flipped visibilities in a redundant groupself.
        Returns the modified z-score.

        Arguments:
            xants: List of antennas  that should be ignored.
                   format (ant,antpol) format
                   e.g.,  if (81,'y') is excluded, (81,'x') cannot be identified
                          as cross-polarized and will be excluded.
            rawMetric: return the raw power ratio instead of the modified z-score
                        type: Boolean
                        Default: False

        Returns:
            redCorrCrossPolMetrics: Dictionary indexed by (ant,antpol)
                                    The modified z-scores of the mean correlation
                                    ratio between redundant visibilities
                                    and singlely-polarization flipped ones.
                                    Very large values are probably cross-polarized.
        """
        return red_corr_cross_pol_metrics(self.data, self.pols,
                                          self.antpols, self.ants,
                                          self.reds, xants=xants,
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

        These antennas are marked as dead
        They do not appear in recorded antenna metrics or zscores.
        Their removal iteration is -1 (i.e. before iterative flagging).
        """
        autoPowers = compute_median_auto_power_dict(self.data,
                                                    self.pols,
                                                    self.reds)
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

    def _run_all_metrics(self, run_mean_vij=True, run_red_corr=True,
                         run_cross_pols=True):
        """Local call for all metrics as part of iterative flagging method.

        Arguments:
            run_mean_vij: Boolean flag which determines if mean_Vij_metrics is executed.
                          Default is True
            run_red_corr: Boolean flag which determines if red_corr_metrics is executed.
                          Default is True
            run_cross_pols: Boolean flag which determines if mean_Vij_cross_pol_metrics and red_corr_cross_pol_metrics are executed.
                          Default is True
        """
        # Compute all raw metrics
        metNames = []
        metVals = []

        if run_mean_vij:
            metNames.append('meanVij')
            meanVij = self.mean_Vij_metrics(pols=self.pols,
                                            xants=self.xants,
                                            rawMetric=True)
            metVals.append(meanVij)

        if run_red_corr:
            metNames.append('redCorr')

            redCorr = self.red_corr_metrics(pols=['xx', 'yy'],
                                            xants=self.xants,
                                            rawMetric=True)
            metVals.append(redCorr)

        if run_cross_pols:
            metNames.append('meanVijXPol')
            metNames.append('redCorrXPol')

            meanVijXPol = self.mean_Vij_cross_pol_metrics(xants=self.xants,
                                                          rawMetric=True)
            redCorrXPol = self.red_corr_cross_pol_metrics(xants=self.xants,
                                                          rawMetric=True)
            metVals.append(meanVijXPol)
            metVals.append(redCorrXPol)

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
                                               run_mean_vij=True,
                                               run_red_corr=True,
                                               run_cross_pols=True):
        """Run all four antenna metrics and stores results in self.

        Runs all four metrics:
            Two for dead antennas
            Two for cross-polarized antennas
        Saves the results internally to this this antenna metrics object.

        Arguments:
            crossCut: Modified z-score cut for most cross-polarized antennas.
                      Default 5 "sigmas".
            deadCut: Modified z-score cut for most likely dead antennas.
                      Default 5 "sigmas".
            alwaysDeadCut: Modified z-score cut for definitely dead antennas.
                           Default 10 "sigmas".
                           These are all thrown away at once without waiting
                           to iteratively throw away only the worst offender.
            run_mean_vij: Boolean flag which determines if mean_Vij_metrics is executed.
                          Default is True
            run_red_corr: Boolean flag which determines if red_corr_metrics is executed.
                          Default is True
            run_cross_pols: Boolean flag which determines if mean_Vij_cross_pol_metrics and red_corr_cross_pol_metrics are executed.
                          Default is True
        """
        self.reset_summary_stats()
        self.find_totally_dead_ants()
        self.crossCut, self.deadCut = crossCut, deadCut
        self.alwaysDeadCut = alwaysDeadCut

        # Loop over
        for n in range(len(self.antpols) * len(self.ants)):
            self.iter = n
            self._run_all_metrics(run_mean_vij=run_mean_vij,
                                  run_red_corr=run_red_corr,
                                  run_cross_pols=run_cross_pols)

            # Mostly likely dead antenna
            last_iter = list(self.allModzScores)[-1]
            worstDeadCutRatio = None
            worstCrossCutRatio = None
            if run_mean_vij and run_red_corr:
                deadMetrics = average_abs_metrics(self.allModzScores[last_iter]['meanVij'],
                                                  self.allModzScores[last_iter]['redCorr'])
                worstDeadAnt = max(deadMetrics, key=deadMetrics.get)
                worstDeadCutRatio = np.abs(deadMetrics[worstDeadAnt]) / deadCut
            else:
                if run_mean_vij:
                    deadMetrics = self.allModzScores[last_iter]['meanVij'].copy()
                    worstDeadAnt = max(deadMetrics, key=deadMetrics.get)
                    worstDeadCutRatio = (np.abs(deadMetrics[worstDeadAnt])
                                         / deadCut)
                elif run_red_corr:
                    deadMetrics = self.allModzScores[last_iter]['redCorr'].copy()
                    worstDeadAnt = max(deadMetrics, key=deadMetrics.get)
                    worstDeadCutRatio = (np.abs(deadMetrics[worstDeadAnt])
                                         / deadCut)
            if run_cross_pols:
                # Most likely cross-polarized antenna
                crossMetrics = average_abs_metrics(self.allModzScores[last_iter]['meanVijXPol'],
                                                   self.allModzScores[last_iter]['redCorrXPol'])
                worstCrossAnt = max(crossMetrics, key=crossMetrics.get)
                worstCrossCutRatio = (np.abs(crossMetrics[worstCrossAnt])
                                      / crossCut)

            # Find the single worst antenna, remove it, log it, and run again
            if (worstCrossCutRatio >= worstDeadCutRatio
                    and worstCrossCutRatio >= 1.0):
                for antpol in self.antpols:
                    self.xants.append((worstCrossAnt[0], antpol))
                    self.crossedAntsRemoved.append((worstCrossAnt[0], antpol))
                    self.removalIter[(worstCrossAnt[0], antpol)] = n
                    if verbose:
                        print('On iteration', n, 'we flag\t', end='')
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
                    self.removalIter[dead_ant] = n
                    if verbose:
                        print('On iteration', n, 'we flag', dead_ant)
            else:
                break

    def save_antenna_metrics(self, filename):
        """Output all meta-metrics and cut decisions to HDF5 file.

        Saves all cut decisions and meta-metrics in an HDF5 that can be loaded
        back into a dictionary using hera_qm.ant_metrics.load_antenna_metrics()

        Arguments:
            filename: The file into which metrics will be written.
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
        out_dict['reds'] = self.reds

        metrics_io.write_metric_file(filename, out_dict)


def ant_metrics_run(files, pols=['xx', 'yy', 'xy', 'yx'], crossCut=5.0,
                    deadCut=5.0, alwaysDeadCut=10.0, metrics_path='',
                    extension='.ant_metrics.hdf5', vis_format='miriad',
                    verbose=True, history='',
                    run_mean_vij=True, run_red_corr=True,
                    run_cross_pols=True):
    """
    Run a series of ant_metrics tests on a given set of input files.

    Args:
        files: List of files to run ant metrics on.
               Can be any of the 4 polarizations
        pols: List of polarizations to perform metrics over.
             Allowed polarizations: 'xx', 'yy', 'xy', 'yx'
             Default: ['xx', 'yy', 'xy', 'yx']
        crossCut: Modified Z-Score limit to cut cross-polarized antennas.
                  Default: 5.0
        deadCut: Modifized Z-Score limit to cut dead antennas.
                 Default: 5.0
        alwaysDeadCut: Modified Z-Score limit for antennas that are definitely dead.
                       Antennas with above this limit are thrown away before iterative flagging.
                       Default: 10.0
        metrics_path: String path to directory to story output metric.
                      Default: same directy as input data files.
        extension: File extension to add to output files.
                   Default: ant_metrics.hdf5
        vis_format: File format of input visibility data.
                    Supports: 'miriad', 'uvh5', 'uvfits', 'fhd', 'ms' (see pyuvdata docs)
                    Default: 'miriad'
        verbose: If True print out statements during iterative flagging
        history: The history the add to metrics.
                 Default
        run_mean_vij: Boolean flag which determines if mean_Vij_metrics is executed.
                      Default is True
        run_red_corr: Boolean flag which determines if red_corr_metrics is executed.
                      Default is True
        run_cross_pols: Boolean flag which determines if mean_Vij_cross_pol_metrics and red_corr_cross_pol_metrics are executed.
                      Default is True
    Return:
       None

    The funciton will take in a list of files and options.
    It will run the series of ant metrics tests,
    and produce an HDF5 file containing the relevant information.
    The file list need only contain one polarization type for a given JD,
    the function will look for the other polarizations in the same folder.
    If not all four polarizations are found, a warning is
    generated, since the code assumes all four polarizations are present.
    """
    from hera_cal.io import HERAData
    from hera_cal.redcal import get_pos_reds

    # check the user asked to run anything
    if not any([run_mean_vij, run_red_corr, run_cross_pols]):
        raise AssertionError(("No Ant Metrics have been selected to run."
                              "Please set the correct keywords to run "
                              "the desired metrics."))

    # check that we were given some files to process
    if len(files) == 0:
        raise AssertionError('Please provide a list of visibility files')

    # generate a list of all files to be read in
    fullpol_file_list = utils.generate_fullpol_file_list(files, pols)
    if len(fullpol_file_list) == 0:
        raise AssertionError('Could not find all 4 polarizations '
                             'for any files provided')

    # get list of lists of redundant baselines, assuming redunancy information is the same for all files
    hd = HERAData(fullpol_file_list[0][0], filetype=vis_format)
    reds = get_pos_reds(hd.read()[0].antpos)
    del hd

    # do the work
    for jd_list in fullpol_file_list:
        am = AntennaMetrics(jd_list, reds, fileformat=vis_format)
        am.iterative_antenna_metrics_and_flagging(crossCut=crossCut,
                                                  deadCut=deadCut,
                                                  alwaysDeadCut=alwaysDeadCut,
                                                  verbose=verbose,
                                                  run_mean_vij=run_mean_vij,
                                                  run_red_corr=run_red_corr,
                                                  run_cross_pols=run_cross_pols)

        # add history
        am.history = am.history + history

        base_filename = jd_list[0]
        abspath = os.path.abspath(base_filename)
        dirname = os.path.dirname(abspath)
        basename = os.path.basename(base_filename)
        nopol_filename = re.sub(r'\.{}\.'.format(pols[0]), '.', basename)
        if metrics_path == '':
            # default path is same directory as file
            metrics_path = dirname
        else:
            metrics_path = metrics_path
        metrics_basename = utils.strip_extension(nopol_filename) + extension
        metrics_filename = os.path.join(metrics_path, metrics_basename)
        am.save_antenna_metrics(metrics_filename)

    return
