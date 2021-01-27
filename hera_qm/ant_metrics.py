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
    metrics_dict = {'ant_metrics_corr': 'Median value of the correlation matrix '
                                           'across all values including an '
                                           'antenna.',
                    'ant_metrics_corrXPol': 'Max difference between same-pol '
                                               'and cross-pol corr values ',
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
        visibility amplitudes. If the median value is 0, the stat will always be 0 to catch
        help catch completely dead antennas (this was observed in H1C).
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

def calc_corr_stats(data_sum, data_diff=None, flags_sum=None, flags_diff=None, time_alg=np.nanmedian, freq_alg=np.nanmedian):
    """Summarize visibility magnitudes as a single number for quick comparison to others.

    Parameters
    ----------
    data_sum : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs)
    data_diff : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs)
    flags_sum : dictionary or hera_cal DataContainer, optional
        If not None, should have the same keys and same array shapes as data
    flags_diff : dictionary or hera_cal DataContainer, optional
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
    corr_stats : dictionary
        Dictionary mapping baseline keys e.g. (0, 1, 'ee') to single floats representing
        correlation amplitudes. A value of 1 indicates a strong correlation, and a value
        of 0 indicates no correlation.

    """
    corr_metric = {}
    for bl in data_sum:
        data_sum_here = deepcopy(data_sum[bl])
        data_sum_here[~np.isfinite(data_sum_here)] = np.nan
        if data_diff is not None:
            data_diff_here = deepcopy(data_diff[bl])
            data_diff_here[~np.isfinite(data_diff_here)] = np.nan
        if flags_sum is not None:
            data_sum_here[flags_sum[bl]] = np.nan
        if flags_diff is not None:
            data_diff_here[flags_diff[bl]] = np.nan

        if data_diff is not None:
            # Standard calculation
            even = (data_sum_here + data_diff_here)/2
            odd = (data_sum_here - data_diff_here)/2
        else:
            # If diff files are not provided, use interleaving sum files
            even = data_sum_here[::2,:]
            odd = data_sum_here[1::2,:]
        even = np.divide(even,np.abs(even))
        odd = np.divide(odd,np.abs(odd))
        metric = np.abs(np.nanmean(np.multiply(even,np.conj(odd))))
        corr_metric[bl] = metric
    return corr_metric


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
        antpols = set([ap for bl in abs_vis_stats for ap in split_pol(bl[2])
                       if ((pols is None) or (bl[2] in pols))])
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
            continue  # ignore autocorrelations
        if (pols is None) or (bl[2] in pols):
            for ant in split_bl(bl):
                if ant in ants:
                    per_ant_means[ant].append(vis_mean)
    per_ant_means = {ant: np.nanmean(per_ant_means[ant]) for ant in ants}

    if rawMetric:
        return per_ant_means
    else:
        return per_antenna_modified_z_scores(per_ant_means)



def corr_cross_pol_metrics(corr_stats, xants=[]):
    """Calculate the differences in the correlation metric between polarizations.

    The four polarization combinations are xx-xy, yy-xy, xx-yx, and yy-yx. An
    antenna is considered cross-polarized if all four of these metrics are less
    than zero.

    Parameters
    ----------
    corr_stats : dictionary
        Dictionary mapping baseline tuple e.g. (0, 1, 'ee') to
        correlation metric value.
    xants : list of integers or tuples of antennas to exlcude, optional

    Returns
    -------
    cross_pol_metrics : dict
        Dictionary indexed by ant number keys. Contains the values of the
        four polarization combinations.

    """
    pols = set([bl[2] for bl in corr_stats])
    cross_pols = [pol for pol in pols if pol[0] != pol[1]]
    same_pols = [pol for pol in pols if pol[0] == pol[1]]
    print(same_pols)
    print(cross_pols)
    if (len(pols) != 4) or (len(same_pols) != 2):
        raise ValueError('There must be precisely two "cross" visbility polarizations '
                         'and two "same" polarizations but we have instead '
                         f'{cross_pols} and {same_pols}')

    cross_pol_metrics = {}
    #Iterate through all antennas
    for a1 in set([key[0] for key in corr_stats.keys()]):
        xx_xy = []
        xx_yx = []
        yy_xy = []
        yy_yx = []
        #Calculate all crosses, excluding ants in xants
        for a2 in set([key[1] for key in corr_stats.keys() if key[1] not in xants])
            xx_xy.append(np.subtract(corr_stats[(a1,a2,same_pols[0])],corr_stats[(a1,a2,cross_pols[0])]))
            xx_yx.append(np.subtract(corr_stats[(a1,a2,same_pols[0])],corr_stats[(a1,a2,cross_pols[1])]))
            yy_xy.append(np.subtract(corr_stats[(a1,a2,same_pols[1])],corr_stats[(a1,a2,cross_pols[0])]))
            yy_yx.append(np.subtract(corr_stats[(a1,a2,same_pols[1])],corr_stats[(a1,a2,cross_pols[1])]))
        #If the max value is negative (indicating crossed), then all 4 values must be negative
        cross_pol_metrics[a1] = np.max([np.nanmean(xx_xy),np.nanmean(xx_yx),np.nanmean(yy_xy),np.nanmean(yy_yx)])

    return cross_pol_metrics


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

    def __init__(self, sum_files, diff_files=None, apriori_xants=[], Nbls_per_load=None):
        """Initilize an AntennaMetrics object and load mean visibility amplitudes.

        Parameters
        ----------
        sum_files : str or list of str
            Path to file or files of raw sum data to calculate antenna metrics on
        diff_files : str or list of str
            Path to file or files of raw diff data to calculate antenna metrics on
        apriori_xants : list of integers or tuples, optional
            List of integer antenna numbers or antpol tuples e.g. (0, 'Jee') to mark
            as excluded apriori. These are included in self.xants, but not
            self.dead_ants or self.crossed_ants when writing results to disk.
        Nbls_per_load : integer, optional
            Number of baselines to load simultaneously. Trades speed for memory
            efficiency. Default None means load all baselines.

        Attributes
        ----------
        hd_sum : HERAData
            HERAData object generated from datafile_list_sum.
        hd_diff : HERAData
            HERAData object generated from datafile_list_diff.
        ants : list of tuples
            List of antenna-polarization tuples to assess
        antnums : list of ints
            List of antenna numbers
        antpols : List of str
            List of antenna polarization strings. Typically ['Jee', 'Jnn']
        bls : list of ints
            List of baselines in HERAData object.
        datafile_list_sum : list of str
            List of sum data filenames that went into this calculation.
        datafile_list_diff : list of str
            List of diff data filenames that went into this calculation.
        abs_vis_stats : dictionary
            Dictionary mapping baseline keys e.g. (0, 1, 'ee') to single floats
            representing visibility amplitudes.

        version_str : str
            The version of the hera_qm module used to generate these metrics.
        history : str
            History to append to the metrics files when writing out files.

        """
        # Instantiate HERAData object and figure out baselines
        from hera_cal.io import HERAData
        if isinstance(sum_files, str):
            sum_files = [sum_files]
        if isinstance(diff_files, str):
            diff_files = [diff_files]
        self.datafile_list_sum = sum_files
        self.hd_sum = HERAData(sum_files)
        if diff_files == None:
            self.datafile_list_diff = None
            self.hd_diff = None
        else:
            self.datafile_list_diff = diff_files
            self.hd_diff = HERAData(diff_files)
        if len(self.hd_sum.filepaths) > 1:
            # only load baselines in all files
            self.bls = sorted(set.intersection(*[set(bls) for bls in self.hd_sum.bls.values()]))
        else:
            self.bls = self.hd_sum.bls

        # Figure out which antennas are in the data
        from hera_cal.utils import split_bl, comply_pol
        self.split_bl = split_bl  # prevents the need for importing again later
        self.ants = set([ant for bl in self.bls for ant in split_bl(bl)])
        self.antnums = set([ant[0] for ant in self.ants])
        self.antpols = set([ant[1] for ant in self.ants])

        # Parse apriori_xants
        if not (isinstance(apriori_xants, list) or isinstance(apriori_xants, np.ndarray)):
            raise ValueError('apriori_xants must be a list or numpy array.')
        self.apriori_xants = set([])
        for ant in apriori_xants:
            if isinstance(ant, int):
                for ap in self.antpols:
                    self.apriori_xants.add((ant, ap))
            elif isinstance(ant, tuple):
                if (len(ant) != 2) or (comply_pol(ant[1]) not in self.antpols):
                    raise ValueError(f'{ant} is not a valid entry in apriori_xants.')
                self.apriori_xants.add((ant[0], comply_pol(ant[1])))
            else:
                raise ValueError(f'{ant} is not a valid entry in apriori_xants.')

        # Set up metadata and summary stats
        self.version_str = hera_qm_version_str
        self.history = ''
        self._reset_summary_stats()

        # Load and summarize data
        self._load_time_freq_abs_vis_stats(Nbls_per_load=Nbls_per_load)

    def _reset_summary_stats(self):
        """Reset all the internal summary statistics back to empty."""
        self.xants, self.crossed_ants, self.dead_ants = [], [], []
        self.iter = 0
        self.removal_iteration = {}
        self.all_metrics, self.all_mod_z_scores = {}, {}
        self.final_metrics, self.final_mod_z_scores = {}, {}
        for ant in self.apriori_xants:
            self.xants.append(ant)
            self.removal_iteration[ant] = -1

    def _load_time_freq_abs_vis_stats(self, Nbls_per_load=None):
        """Loop through groups of baselines to calculate self.abs_vis_stats
        using time_freq_abs_vis_stats()
        """
        if Nbls_per_load is None:
            bl_load_groups = [self.bls]
        else:
            bl_load_groups = [self.bls[i:i + Nbls_per_load]
                              for i in range(0, len(self.bls), Nbls_per_load)]

        self.abs_vis_stats = {}
        self.corr_stats = {}
        for blg in bl_load_groups:
            data_sum, flags_sum, _ = self.hd_sum.read(bls=blg, axis='blt')
            if self.hd_diff == None:
                data_diff = None
                flags_diff = None
            else:
                data_diff, flags_diff, _ = self.hd_diff.read(bls=blg, axis='blt')
            self.abs_vis_stats.update(time_freq_abs_vis_stats(data_sum, flags_sum))
            self.corr_stats.update(calc_corr_stats(data_sum, data_diff, flags_sum, flags_diff))

    def _find_totally_dead_ants(self, verbose=False):
        """Flag antennas whose median autoPower is 0.0.

        These antennas are marked as dead. They do not appear in recorded antenna
        metrics or zscores. Their removal iteration is -1 (i.e. before iterative
        flagging).
        """
        # assign abs_vis_stats to antennas
        abs_vis_stats_by_ant = {ant: [] for ant in self.ants}
        for bl in self.abs_vis_stats:
            for ant in self.split_bl(bl):
                abs_vis_stats_by_ant[ant].append(self.abs_vis_stats[bl])

        # remove antennas that are totally dead and all nans
        for ant, vis_stats in abs_vis_stats_by_ant.items():
            med = np.nanmedian(vis_stats)
            if ~np.isfinite(med) or (med == 0):
                self.xants.append(ant)
                self.dead_ants.append(ant)
                self.removal_iteration[ant] = -1
                if verbose:
                    print(f'Antenna {ant} appears totally dead and is removed.')

    def _run_all_metrics(self, run_cross_pols=True, run_cross_pols_only=False):
        """Local call for all metrics as part of iterative flagging method.

        Parameters
        ----------
        run_cross_pols : bool, optional
            Define if corr_cross_pol_metrics is executed. Default is True.
        run_cross_pols_only : bool, optional
            Define if corr_cross_pol_metrics is the *only* metric to be run.
            Default is False.

        """
        # Compute all raw metrics
        metNames = []
        metVals = []

        if run_cross_pols_only and not run_cross_pols:
            raise ValueError('Must run at least 1 metric, but run_cross_pols is False '
                             'while run_cross_pols_only is True')

        if not run_cross_pols_only:
            metNames.append('meanVij')
            meanVij = mean_Vij_metrics(self.abs_vis_stats, xants=self.xants, rawMetric=True)
            metVals.append(meanVij)

        if run_cross_pols:
            metNames.append('corrXPol')
            corrXPol = corr_cross_pol_metrics(self.abs_vis_stats,
                                                     xants=self.xants, rawMetric=True)
            metVals.append(corrXPol)

        # Save all metrics and zscores
        metrics, modzScores = {}, {}
        for metric, metName in zip(metVals, metNames):
            metrics[metName] = metric
            modz = per_antenna_modified_z_scores(metric)
            modzScores[metName] = modz
            for key in metric:
                if metName in self.final_metrics:
                    self.final_metrics[metName][key] = metric[key]
                    self.final_mod_z_scores[metName][key] = modz[key]
                else:
                    self.final_metrics[metName] = {key: metric[key]}
                    self.final_mod_z_scores[metName] = {key: modz[key]}
        self.all_metrics.update({self.iter: metrics})
        self.all_mod_z_scores.update({self.iter: modzScores})

    def iterative_antenna_metrics_and_flagging(self, crossCut=0, deadCut=0.25,
                                               verbose=False, run_cross_pols=True,
                                               run_cross_pols_only=False):
        """Run corr metric and crosspol metrics and stores results in self.

        Parameters
        ----------
        crossCut : float, optional
            Cut for most cross-polarized antennas. Default is 0.
        deadCut : float, optional
            Cut for most likely dead antennas. Default is 0.25.
        run_cross_pols : bool, optional
            Define if corr_cross_pol_metrics is executed. Default is True.
        run_cross_pols_only : bool, optional
            Define if corr_cross_pol_metrics is the *only* metric to be run.
            Default is False.

        """
        self._reset_summary_stats()
        self._find_totally_dead_ants(verbose=verbose)
        self.crossCut, self.deadCut = crossCut, deadCut

        # iteratively remove antennas, removing only the worst antenna
        for iteration in range(len(self.antpols) * len(self.ants)):
            self.iter = iteration
            self._run_all_metrics(run_cross_pols=run_cross_pols,
                                  run_cross_pols_only=run_cross_pols_only)
            worstDeadCutDiff = 1
            worstCrossCutDiff = 1

            # Find most likely dead antenna
            if not run_cross_pols_only:
                deadMetrics = {ant: np.abs(metric) for ant, metric
                               in self.all_mod_z_scores[iteration]['corr'].items()}
                worstDeadAnt = min(deadMetrics, key=deadMetrics.get)
                worstDeadCutDiff = np.abs(deadMetrics[worstDeadAnt]) - deadCut

            # Find most likely cross-polarized antenna
            if run_cross_pols:
                crossMetrics = {ant: np.max(metric) for ant, metric
                                in self.all_mod_z_scores[iteration]['corrXPol'].items()}
                worstCrossAnt = min(crossMetrics, key=crossMetrics.get)
                worstCrossCutDiff = crossMetrics[worstCrossAnt]) - crossCut

            # Find the single worst antenna, remove it, log it, and run again
            if (worstCrossCutDiff <= worstDeadCutDiff) and (worstCrossCutDiff < 0):
                for antpol in self.antpols:  # if crossed remove both polarizations
                    crossed_ant = (worstCrossAnt[0], antpol)
                    self.xants.append(crossed_ant)
                    self.crossed_ants.append(crossed_ant)
                    self.removal_iteration[crossed_ant] = iteration
                    if verbose:
                        print(f'On iteration {iteration} we flag {crossed_ant} with modified z of {crossMetrics[worstCrossAnt]}.')
            elif (worstDeadCutDiff < worstCrossCutDiff) and (worstDeadCutDiff <= 0):
                dead_ants = set([worstDeadAnt])
                for dead_ant in dead_ants:
                    self.xants.append(dead_ant)
                    self.dead_ants.append(dead_ant)
                    self.removal_iteration[dead_ant] = iteration
                    if verbose:
                        print(f'On iteration {iteration} we flag {dead_ant} with modified z of {deadMetrics[worstDeadAnt]}.')
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
        out_dict = {'xants': self.xants}
        out_dict['crossed_ants'] = self.crossed_ants
        out_dict['dead_ants'] = self.dead_ants
        out_dict['final_metrics'] = self.final_metrics
        out_dict['all_metrics'] = self.all_metrics
        out_dict['final_mod_z_scores'] = self.final_mod_z_scores
        out_dict['all_mod_z_scores'] = self.all_mod_z_scores
        out_dict['removal_iteration'] = self.removal_iteration
        out_dict['cross_pol_cut'] = self.crossCut
        out_dict['dead_ant_cut'] = self.deadCut
        out_dict['datafile_list_sum'] = self.datafile_list_sum
        out_dict['datafile_list_diff'] = self.datafile_list_diff
        out_dict['history'] = self.history

        metrics_io.write_metric_file(filename, out_dict, overwrite=overwrite)


def ant_metrics_run(sum_files, diff_files=None, apriori_xants=[], a_priori_xants_yaml=None,
                    crossCut=5.0, deadCut=0.25, run_cross_pols=True, run_cross_pols_only=False,
                    metrics_path='', extension='.ant_metrics.hdf5',
                    overwrite=False, Nbls_per_load=None, history='', verbose=True):
    """
    Run a series of ant_metrics tests on a given set of input files.

    Note
    ----
    The function will take a file or list of files and options. It will run
    ant metrics once on all files together but then save the results to an
    identical HDF5 file for each input file.

    Parameters
    ----------
    sum_files : str or list of str
        Path to file or files of raw sum data to calculate antenna metrics on.
    diff_files : str or list of str
        Path to file or files of raw diff data to calculate antenna metrics on.
    apriori_xants : list of integers or tuples, optional
        List of integer antenna numbers or antpol tuples e.g. (0, 'Jee') to mark
        as excluded apriori. These are included in self.xants, but not
        self.dead_ants or self.crossed_ants when writing results to disk.
    a_priori_xants_yaml : string, optional
        Path to a priori flagging YAML with antenna flagging information.
        See hera_qm.metrics_io.read_a_priori_ant_flags() for details.
        Frequency and time flags in the YAML are ignored.
    crossCut : float, optional
        Limit below which to cut cross-polarized antennas. Default is 0.
    deadCut : float, optional
        Limit below which to cut dead antennas. Default is 0.25.
    run_cross_pols : bool, optional
        Define if corr_cross_pol_metrics is executed. Default is True.
    run_cross_pols_only : bool, optional
        Define if corr_cross_pol_metrics is the *only* metric to be run.
        Default is False.
    metrics_path : str, optional
        Full path to directory to story output metric. Default is the same directory
        as input data files.
    extension : str, optional
        File extension to add to output files. Default is ant_metrics.hdf5.
    overwrite: bool, optional
        Whether to overwrite existing ant_metrics files. Default is False.
    Nbls_per_load : integer, optional
        Number of baselines to load simultaneously. Trades speed for memory
        efficiency. Default None means load all baselines.
    history : str, optional
        The history the add to metrics. Default is nothing (empty string).
    verbose : bool, optional
        If True, print out statements during iterative flagging. Default is True.
    """

    # load a priori exants from YAML and append to apriori_xants
    if a_priori_xants_yaml is not None:
        apaf = metrics_io.read_a_priori_ant_flags(a_priori_xants_yaml)
        apriori_xants = list(set(list(apriori_xants) + apaf))

    # run ant metrics
    am = AntennaMetrics(sum_files, diff_files,
                        apriori_xants=apriori_xants,
                        Nbls_per_load=Nbls_per_load)
    am.iterative_antenna_metrics_and_flagging(crossCut=crossCut,
                                              deadCut=deadCut,
                                              verbose=verbose,
                                              run_cross_pols=run_cross_pols,
                                              run_cross_pols_only=run_cross_pols_only)
    am.history = am.history + history

    for file in am.datafile_list_sum:
        metrics_basename = utils.strip_extension(os.path.basename(file)) + extension
        if metrics_path == '':
            # default path is same directory as file
            metrics_path = os.path.dirname(os.path.abspath(file))
        outfile = os.path.join(metrics_path, metrics_basename)
        if verbose:
            print(f'Now saving results to {outfile}')
        am.save_antenna_metrics(outfile, overwrite=overwrite)
