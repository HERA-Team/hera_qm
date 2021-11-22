# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics."""
import numpy as np
from copy import deepcopy
import os
import shutil
import re
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
    metrics_dict = {'ant_metrics_corr': 'Median value of the corr_metric '
                                           'across all values including an '
                                           'antenna.',
                    'ant_metrics_corrXPol': 'Max difference between same-pol '
                                               'and cross-pol corr values ',
                    'ant_metrics_meanVij': 'Mean of the absolute value of all '
                                           'visibilities associated with an '
                                           'antenna. LEGACY METRIC.',
                    'ant_metrics_meanVijXPol': 'Ratio of mean cross-pol '
                                               'visibilities to mean same-pol '
                                               'visibilities: '
                                               '(Vxy+Vyx)/(Vxx+Vyy). LEGACY METRIC.',
                    'ant_metrics_mod_z_scores_meanVij': 'Modified z-score of '
                                                        'the mean of the '
                                                        'absolute value of '
                                                        'all visibilities '
                                                        'associated with an '
                                                        'antenna. LEGACY METRIC.',
                    'ant_metrics_mod_z_scores_meanVijXPol': 'Modified z-score '
                                                            'of the ratio of '
                                                            'mean cross-pol '
                                                            'visibilities '
                                                            'to mean same-pol '
                                                            'visibilities: '
                                                            '(Vxy+Vyx)/'
                                                            '(Vxx+Vyy). LEGACY METRIC.',
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


def calc_corr_stats(data_sum, data_diff=None, flags=None, time_alg=np.nanmean, freq_alg=np.nanmean):
    """Calculate correlation values for all baselines, the average cross-correlation between
    even and

    Parameters
    ----------
    data_sum : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs).
        Corresponds to the even+odd output from the correlator.
    data_diff : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs)
        If not provided, data_sum will be broken into interleaving timesteps.
        Corresponds to the even-odd output from the correlator.
    flags : dictionary or hera_cal DataContainer, optional
        Times or frequencies to exclude from the calculation of the correlation metrics.
        If not None, should have the same keys and same array shapes as data_sum
    time_alg : function, optional
        Function used to reduce a 2D or 1D numpy array to a single number.
        To handle flags properly, should be the "nan" version of the function.
    freq_alg : function, optional
        Function that reduces a 1D array to a single number or a 2D array to a 1D
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
    corr_stats = {}
    for bl in data_sum:
        # turn flags and other non-finite data into nans
        data_sum_here = np.where(np.isfinite(data_sum[bl]), data_sum[bl], np.nan)
        if flags is not None:
            data_sum_here[flags[bl]] = np.nan

        # check to see if the sum file is mostly zeros, in which case the antenna is totally dead
        med_abs_sum = np.nanmedian(np.abs(data_sum_here))
        if med_abs_sum == 0:
            corr_stats[bl] = 0
            continue

        # split into even and odd
        if data_diff is not None:
            data_diff_here = np.where(np.isfinite(data_diff[bl]), data_diff[bl], np.nan)
            even = (data_sum_here + data_diff_here) / 2
            odd = (data_sum_here - data_diff_here) / 2
        if data_diff is None:
            # interleave, dropping last integraiton if there are an odd number
            last_int = (data_sum_here.shape[0] // 2) * 2
            even = data_sum_here[0:last_int:2, :]
            odd = data_sum_here[1:last_int:2, :]

        # normalize (reduces the impact of RFI by making every channel equally weighted)
        even /= np.abs(even)
        odd /= np.abs(odd)

        # reduce to a scalar statistic
        if time_alg == freq_alg:  # if they are the same algorithm, do it globally
            corr_stats[bl] = np.abs(time_alg(even * np.conj(odd)))
        else:
            corr_stats[bl] = np.abs(time_alg(freq_alg(even * np.conj(odd), axis=1)))

    return corr_stats


def corr_metrics(corr_stats, xants=[], pols=None):
    """Calculate all antennas' mean correlation values.

    Parameters
    ----------
    corr_stats : dictionary
        Dictionary mapping baseline tuple e.g. (0, 1, 'ee') to
        correlation metric averaged over time and frequency.
    xants : list of ints or tuples, optional
        Antenna numbers or tuples e.g. (1, 'Jee') to exclude from metrics
    pols : list of str, optional
        List of visibility polarizations (e.g. ['ee','en','ne','nn']).
        Defaults None means all visibility polarizations are used.

    Returns
    -------
    per_ant_mean_corr_metrics : dict
        Dictionary indexed by (ant, antpol) of the
        mean of correlation value associated with an antenna.
        Very small or very large numbers are probably bad antennas.

    """

    from hera_cal.utils import split_pol, split_bl

    # figure out which antennas match pols and and are not in xants
    if pols is not None:
        antpols = set([ap for bl in corr_stats for ap in split_pol(bl[2])
                       if ((pols is None) or (bl[2] in pols))])
    ants = set()
    for bl in corr_stats:
        for ant in split_bl(bl):
            if (ant not in xants) and (ant[0] not in xants):
                if (pols is None) or (ant[1] in antpols):
                    ants.add(ant)

    # assign correlation metrics to each antenna in the baseline
    per_ant_corrs = {ant: [] for ant in ants}
    for bl, corr_mean in corr_stats.items():
        if bl[0] == bl[1]:
            continue  # ignore autocorrelations
        if (pols is None) or (bl[2] in pols):
            if split_bl(bl)[0] in ants and split_bl(bl)[1] in ants:
                for ant in split_bl(bl):
                    per_ant_corrs[ant].append(corr_mean)
    per_ant_mean_corr_metrics = {ant: np.nanmean(per_ant_corrs[ant]) for ant in ants}

    return per_ant_mean_corr_metrics


def corr_cross_pol_metrics(corr_stats, xants=[]):
    """Calculate the differences in corr_stats between polarizations. For
    typical usage corr_stats is a measure of per-baseline average correlation
    as calculated by the calc_corr_stats method.

    The four polarization combinations are xx-xy, yy-xy, xx-yx, and yy-yx. An
    antenna is considered cross-polarized if all four of these metrics are less
    than zero.

    Parameters
    ----------
    corr_stats : dictionary
        Dictionary mapping baseline tuple e.g. (0, 1, 'ee') to
        its average corr_metric value.
    xants : list of integers or tuples of antennas to exlcude, optional

    Returns
    -------
    per_ant_corr_cross_pol_metrics : dict
        Dictionary indexed by keys (ant,antpol). Contains the max value over the
        four polarization combinations of the average (over baselines) difference
        in correlation metrics (xx-xy, xx-yx, yy-xy, yy-yx).
    """

    from hera_cal.utils import split_pol, split_bl
    from hera_cal.datacontainer import DataContainer

    # cast corr_stats as DataContainer to abstract away polarization/conjugation
    corr_stats_dc = DataContainer(corr_stats)

    # figure out pols om corr_stats and make sure they are sensible
    cross_pols = [pol for pol in corr_stats_dc.pols() if split_pol(pol)[0] != split_pol(pol)[1]]
    same_pols = [pol for pol in corr_stats_dc.pols() if split_pol(pol)[0] == split_pol(pol)[1]]
    if (len(corr_stats_dc.pols()) != 4) or (len(same_pols) != 2):
        raise ValueError('There must be precisely two "cross" visbility polarizations '
                         'and two "same" polarizations but we have instead '
                         f'{cross_pols} and {same_pols}')

    # get ants, antnums, and antpols
    ants = set()
    for bl in corr_stats:
        for ant in split_bl(bl):
            if (ant not in xants) and (ant[0] not in xants):
                ants.add(ant)
    antnums = set([ant[0] for ant in ants])
    antpols = set([ant[1] for ant in ants])

    # If an antenna is not touched, data is missing and hence set this metric to nan.
    per_ant_corr_cross_pol_metrics = {ant: np.nan for ant in ants}
    #Iterate through all antennas
    for a1 in antnums:
        # check if any pols of this ant are flagged
        if (a1 in xants) or np.any([(a1, ap) in xants for ap in antpols]):
            continue

        diffs = [[], [], [], []]
        for a2 in antnums:
            # check if any pols of this ant are flagged
            if (a2 in xants) or np.any([(a2, ap) in xants for ap in antpols]):
                continue

            # this loops over all the combinations of same and cross-pols
            # technically, this is a double-count, but the average takes that out
            for i, (sp, cp) in enumerate([(sp, cp) for sp in same_pols for cp in cross_pols]):
                if ((a1, a2, sp) in corr_stats_dc) and ((a1, a2, cp) in corr_stats_dc):
                    diffs[i].append(corr_stats_dc[(a1, a2, sp)] - corr_stats_dc[(a1, a2, cp)])

        # assign same metric to both antpols
        for ap in antpols:
            per_ant_corr_cross_pol_metrics[(a1, ap)] = np.nanmax([np.nanmean(d) for d in diffs])

    return per_ant_corr_cross_pol_metrics


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
    and provides interfaces to two antenna metrics: one for identifying dead / not
    correlating atennas and the other for identifying cross-polarized antennas. These
    metrics can be used iteratively to identify bad antennas. The object handles
    all stroage of metrics, and supports writing metrics to an HDF5 filetype.
    The analysis functions are designed to work on raw data from one or more observations
    with all four polarizations.
    """

    def __init__(self, sum_files, diff_files=None, apriori_xants=[], Nbls_per_load=None,
                 Nfiles_per_load=None, time_alg=np.nanmean, freq_alg=np.nanmean):
        """Initilize an AntennaMetrics object and load mean visibility amplitudes.

        Parameters
        ----------
        sum_files : str or list of str
            Path to file or files of raw sum data to calculate antenna metrics on
        diff_files : str or list of str
            Path to file or files of raw diff data to calculate antenna metrics on
            If not provided, even/odd correlations will be inferred with interleaving.
            Assumed to match sum_files in metadata. Flags will be ORed with sum_files.
        apriori_xants : list of integers or tuples, optional
            List of integer antenna numbers or antpol tuples e.g. (0, 'Jee') to mark
            as excluded apriori. These are included in self.xants, but not
            self.dead_ants or self.crossed_ants when writing results to disk.
        Nbls_per_load : integer, optional
            Number of baselines to load simultaneously. Trades speed for memory
            efficiency. Default None means load all baselines.
        Nfiles_per_load : integer, optional
            Number of files to load simultaneously and reduce to corr_stats.
            If None, all files are loaded simultaneously. If not None, then
            corr_stats are averaged in time with np.nanmean.
        time_alg : function, optional
            Averaging function along the time axis for producing correlation stats.
            See calc_corr_stats() for more details.
        freq_alg : function, optional
            Averaging function along the frequency axis for producing correlation stats.
            See calc_corr_stats() for more details.

        Attributes
        ----------
        hd_sum : HERAData
            HERAData object generated from sum_files.
        hd_diff : HERAData
            HERAData object generated from diff_files.
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
        
        from hera_cal.io import HERAData
        from hera_cal.utils import split_bl, comply_pol, split_pol, join_pol
        # prevents the need for importing again later
        self.HERAData = HERAData
        self.split_bl = split_bl
        self.join_pol = join_pol
        self.split_pol = split_pol

        # Instantiate HERAData object and figure out baselines
        if isinstance(sum_files, str):
            sum_files = [sum_files]
        if isinstance(diff_files, str):
            diff_files = [diff_files]
        if (diff_files is not None) and (len(diff_files) != len(sum_files)):
            raise ValueError(f'The number of sum files ({len(sum_files)}) does not match the number of diff files ({len(diff_files)}).')
        self.datafile_list_sum = sum_files
        self.hd_sum = HERAData(sum_files)
        if diff_files is None or len(diff_files) == 0:
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

        # Figure out polarizations in the data:
        from hera_cal.utils import split_bl, comply_pol, split_pol
        self.pols = set([bl[2] for bl in self.bls])
        self.cross_pols = [pol for pol in self.pols if split_pol(pol)[0] != split_pol(pol)[1]]
        self.same_pols = [pol for pol in self.pols if split_pol(pol)[0] == split_pol(pol)[1]]

        # Figure out which antennas are in the data
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
        self._load_corr_stats(Nbls_per_load=Nbls_per_load, Nfiles_per_load=Nfiles_per_load,
                              time_alg=time_alg, freq_alg=freq_alg)

    def _reset_summary_stats(self):
        """Reset all the internal summary statistics back to empty."""
        self.xants, self.crossed_ants, self.dead_ants = [], [], []
        self.iter = 0
        self.removal_iteration = {}
        self.all_metrics = {}
        self.final_metrics = {}
        for ant in self.apriori_xants:
            self.xants.append(ant)
            self.removal_iteration[ant] = -1

    def _load_corr_matrices(self, Nbls_per_load=None, Nfiles_per_load=None, time_alg=np.nanmean, freq_alg=np.nanmean):
        """Loop through groups of baselines to calculate self.corr_matrices using calc_corr_stats().
        """
        bl_load_groups = [None]  # load all baselines simultaneously
        if Nbls_per_load is not None:
            bl_load_groups = [self.bls[i:i + Nbls_per_load]
                              for i in range(0, len(self.bls), Nbls_per_load)]

        # initialize HERAData objects
        corr_stats = {bl: [] for bl in self.bls}
        hd_sums = [deepcopy(self.hd_sum)]
        hd_diffs = [deepcopy(self.hd_diff)]
        if Nfiles_per_load is not None:
            chunker = lambda paths: [paths[i:i + Nfiles_per_load] for i in range(0, len(paths), Nfiles_per_load)]
            hd_sums = [self.HERAData(filepaths) for filepaths in chunker(self.hd_sum.filepaths)]
            if self.hd_diff is not None:
                hd_diffs = [self.HERAData(filepath) for filepath in chunker(self.hd_diff.filepaths)]
        while len(hd_sums) > 0:
            # loop through baseline load groups, computing corr_stats
            for blg in bl_load_groups:
                # read data for this 
                data_sum, flags, _ = hd_sums[0].read(bls=blg, axis='blt')
                data_diff = None
                if hd_diffs[0] is not None:
                    data_diff, flags_diff, _ = hd_diffs[0].read(bls=blg, axis='blt')
                    for bl in flags:
                        flags[bl] |= flags_diff[bl]

                # compute corr_stats and append them to list, weighting by the number of files
                for bl, stat in calc_corr_stats(data_sum, data_diff=data_diff, flags=flags).items():
                    corr_stats[bl].extend([stat] * len(hd_sums[0].filepaths))

            del hd_sums[0], hd_diffs[0]  # save memory by deleting after each file load

        # reduce to a single stat per baseline (rather than per baseline, per file group)
        corr_stats = {bl: time_alg(corr_stats[bl]) for bl in self.bls}

        # convert from corr stats to corr matrices
        self.ant_to_index = {ant: i for ants in self.ants_per_antpol.values() for i, ant in enumerate(ants)}
        self.corr_matrices = {self.join_pol(ap1, ap2): np.full((len(self.ants_per_antpol[ap1]), len(self.ants_per_antpol[ap2])), np.nan)
                              for ap1 in self.antpols for ap2 in self.antpols}
        for bl in corr_stats:
            if bl[0] != bl[1]:  # ignore autocorrelations
                ant1, ant2 = self.split_bl(bl)
                self.corr_matrices[bl[2]][self.ant_to_index[ant1], self.ant_to_index[ant2]] = corr_stats[bl]
        for pol, cm in self.corr_matrices.items(): 
            self.corr_matrices[pol] = np.nanmean([cm, cm.T], axis=0)  # symmetrize

    def _find_totally_dead_ants(self, verbose=False):
        """Flag antennas whose median correlation coefficient is 0.0.

        These antennas are marked as dead. They do not appear in recorded antenna
        metrics or zscores. Their removal iteration is -1 (i.e. before iterative
        flagging).
        """
        # assign corr_stats to antennas
        corr_stats_by_ant = {ant: [] for ant in self.ants}
        for bl in self.corr_stats:
            for ant in self.split_bl(bl):
                corr_stats_by_ant[ant].append(self.corr_stats[bl])

        # remove antennas that are totally dead and all nans
        for ant, corrs in corr_stats_by_ant.items():
            med = np.nanmedian(corrs)
            if ~np.isfinite(med) or (med == 0):
                self.xants.append(ant)
                self.dead_ants.append(ant)
                self.removal_iteration[ant] = -1
                if verbose:
                    print(f'Antenna {ant} appears totally dead and is removed.')

    def _run_all_metrics(self):
        """Local call for all metrics as part of iterative flagging method.
        """
        # Compute all raw metrics
        metNames = []
        metVals = []
        metNames.append('corr')
        metVals.append(corr_metrics(self.corr_stats, xants=self.xants, pols=self.same_pols))
        metNames.append('corrXPol')
        metVals.append(corr_cross_pol_metrics(self.corr_stats, xants=self.xants))

        # Save all metrics
        metrics = {}
        for metric, metName in zip(metVals, metNames):
            metrics[metName] = metric
            for key in metric:
                if metName in self.final_metrics:
                    self.final_metrics[metName][key] = metric[key]
                else:
                    self.final_metrics[metName] = {key: metric[key]}
        self.all_metrics.update({self.iter: metrics})

    def iterative_antenna_metrics_and_flagging(self, crossCut=0, deadCut=0.4, verbose=False):
        """Run corr metric and crosspol metrics and stores results in self.

        Parameters
        ----------
        crossCut : float, optional
            Cut in cross-pol correlation metric below which to flag antennas as cross-polarized.
            Default is 0.
        deadCut : float, optional
            Cut in correlation metric below which antennas are most likely dead / not correlating.
            Default is 0.4.
        """
        self._reset_summary_stats()
        self._find_totally_dead_ants(verbose=verbose)
        self.crossCut, self.deadCut = crossCut, deadCut

        # iteratively remove antennas, removing only the worst antenna
        for iteration in range(len(self.antpols) * len(self.ants)):
            self.iter = iteration
            self._run_all_metrics()
            worstDeadCutDiff = 1
            worstCrossCutDiff = 1

            # Find most likely dead/crossed antenna
            deadMetrics = {ant: metric for ant, metric in self.all_metrics[iteration]['corr'].items() if np.isfinite(metric)}
            crossMetrics = {ant: np.max(metric) for ant, metric in self.all_metrics[iteration]['corrXPol'].items() if np.isfinite(metric)}
            if (len(deadMetrics) == 0) or (len(crossMetrics) == 0):
                break  # no unflagged antennas remain
            worstDeadAnt = min(deadMetrics, key=deadMetrics.get)
            worstDeadCutDiff = np.abs(deadMetrics[worstDeadAnt]) - deadCut
            worstCrossAnt = min(crossMetrics, key=crossMetrics.get)
            worstCrossCutDiff = crossMetrics[worstCrossAnt] - crossCut

            # Find the single worst antenna, remove it, log it, and run again
            if (worstCrossCutDiff <= worstDeadCutDiff) and (worstCrossCutDiff < 0):
                for antpol in self.antpols:  # if crossed remove both polarizations
                    crossed_ant = (worstCrossAnt[0], antpol)
                    self.xants.append(crossed_ant)
                    self.crossed_ants.append(crossed_ant)
                    self.removal_iteration[crossed_ant] = iteration
                    if verbose:
                        print(f'On iteration {iteration} we flag {crossed_ant} with cross-pol corr metric of {crossMetrics[worstCrossAnt]}.')
            elif (worstDeadCutDiff < worstCrossCutDiff) and (worstDeadCutDiff < 0):
                dead_ants = set([worstDeadAnt])
                for dead_ant in dead_ants:
                    self.xants.append(dead_ant)
                    self.dead_ants.append(dead_ant)
                    self.removal_iteration[dead_ant] = iteration
                    if verbose:
                        print(f'On iteration {iteration} we flag {dead_ant} with corr metric z of {deadMetrics[worstDeadAnt]}.')
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
        out_dict['removal_iteration'] = self.removal_iteration
        out_dict['cross_pol_cut'] = self.crossCut
        out_dict['dead_ant_cut'] = self.deadCut
        out_dict['datafile_list_sum'] = self.datafile_list_sum
        out_dict['datafile_list_diff'] = self.datafile_list_diff
        out_dict['history'] = self.history

        metrics_io.write_metric_file(filename, out_dict, overwrite=overwrite)


def ant_metrics_run(sum_files, diff_files=None, apriori_xants=[], a_priori_xants_yaml=None,
                    crossCut=0.0, deadCut=0.4, metrics_path='', extension='.ant_metrics.hdf5',
                    overwrite=False, Nbls_per_load=None, Nfiles_per_load=None, history='', verbose=True):
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
        Cut in cross-pol correlation metric below which to flag antennas as cross-polarized.
        Default is 0.
    deadCut : float, optional
        Cut in correlation metric below which antennas are most likely dead / not correlating.
        Default is 0.4.
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
    Nfiles_per_load : integer, optional
        Number of files to load simultaneously and reduce to corr_stats.
        If None, all files are loaded simultaneously. If not None, then
        corr_stats are averaged in time with np.nanmean.
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
    am = AntennaMetrics(sum_files, diff_files, apriori_xants=apriori_xants,
                        Nbls_per_load=Nbls_per_load, Nfiles_per_load=Nfiles_per_load)
    am.iterative_antenna_metrics_and_flagging(crossCut=crossCut, deadCut=deadCut, verbose=verbose)
    am.history = am.history + history

    for i, file in enumerate(am.datafile_list_sum):
        metrics_basename = utils.strip_extension(os.path.basename(file)) + extension
        if metrics_path == '':
            # default path is same directory as file
            metrics_path = os.path.dirname(os.path.abspath(file))
        outfile = os.path.join(metrics_path, metrics_basename)
        if verbose:
            print(f'Now saving results to {outfile}')
        if i == 0:
            first_outfile = outfile
            am.save_antenna_metrics(outfile, overwrite=overwrite)
        else:
            shutil.copyfile(first_outfile, outfile)
