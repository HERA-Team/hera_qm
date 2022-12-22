# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics."""
import numpy as np
from copy import deepcopy
import os
import shutil
import re
import warnings
from . import __version__
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


def calc_corr_stats(data_sum, data_diff=None):
    """For all baselines, calculate average cross-correlation between even and odd in order 
    to identify dead, cross-polarized, and non-time-locked antennas. Time and channels where
    either the even or odd data (or both) are zero are ignored, but if either even or odd 
    is entirely zero, the corr_stat will be np.nan.

    Parameters
    ----------
    data_sum : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs).
        Corresponds to the even+odd output from the correlator.
    data_diff : dictionary or hera_cal DataContainer
        Maps baseline keys e.g. (0, 1, 'ee') to numpy arrays of shape (Ntimes, Nfreqs)
        If not provided, data_sum will be broken into interleaving timesteps.
        Corresponds to the even-odd output from the correlator.

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
        data_sum_here = data_sum[bl]

        # split into even and odd
        if data_diff is not None:
            data_diff_here = data_diff[bl]
            even = data_sum_here + data_diff_here
            odd = data_sum_here - data_diff_here
        else:
            # interleave, dropping last integration if there are an odd number
            last_int = (data_sum_here.shape[0] // 2) * 2
            even = data_sum_here[0:last_int:2, :]
            odd = data_sum_here[1:last_int:2, :]

        # reduce to a scalar statistic, normalized to reduce the impact of RFI by equally weighting channels
        product = even * np.conj(odd)
        corr_stats[bl] = np.abs(np.nanmean(np.where(product == 0, np.nan, product / np.abs(product))))

    return corr_stats


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

    def __init__(self, sum_files, diff_files=None, apriori_xants=[],
                 sum_data=None, diff_data=None):
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
        sum_data : hera_cal DataContainer, optional
            Use this data instead of loading data form sum_files (which is used only for metadata).
            This option is indended for interactive use when the data are already in memory.
            Must include the full data set expected from sum_files. Nfiles_per_load and
            Nbls_per_load must both be None if this is provided.
        diff_data : hera_cal DataContainer, optional
            Same as sum_data, but for the diff files.

        Attributes
        ----------
        hd_sum : HERADataFastReader
            HERADataFastReader object generated from sum_files.
        hd_diff : HERADataFastReader
            HERADataFastReader object generated from diff_files.
        ants : list of tuples
            List of antenna-polarization tuples to assess
        antnums : list of ints
            List of antenna numbers
        antpols : List of str
            List of antenna polarization strings. Typically ['Jee', 'Jnn']
        bls : list of ints
            List of baselines in HERADataFastReader object.
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
        
        from hera_cal.io import HERADataFastReader
        from hera_cal.utils import split_bl, comply_pol, split_pol, join_pol
        # prevents the need for importing again later
        self.HERADataFastReader = HERADataFastReader
        self.split_bl = split_bl
        self.join_pol = join_pol
        self.split_pol = split_pol

        # Instantiate HERAData object and figure out baselines
        if isinstance(sum_files, str):
            sum_files = [sum_files]
        self.datafile_list_sum = sum_files  # XXX prefer not to have dummy file names
        if isinstance(diff_files, str):
            diff_files = [diff_files]
        if (diff_files is not None):
            assert len(diff_files) == len(sum_files), f'Number of sum files ({len(sum_files)}) does not match number of diff files ({len(diff_files)}).'
            self.datafile_list_diff = diff_files  # XXX prefer not to have dummy file names
        else:
            self.datafile_list_diff = None

        if sum_data is None:
            # load sum files
            hd_sum = HERADataFastReader(sum_files)
            sum_data, _, _ = hd_sum.read(read_flags=False, read_nsamples=False)
            del hd_sum

        # load diff files, if appropraite
        if diff_data is None and diff_files is not None:
            hd_diff = HERADataFastReader(diff_files)
            diff_data, _, _ = hd_diff.read(read_flags=False, read_nsamples=False)
            del hd_diff

        self.bls = list(sum_data.keys())
        assert len(self.bls) > 0, 'Make sure we have data'

        # Figure out polarizations in the data
        self.pols = set([bl[2] for bl in self.bls])
        self.cross_pols = [pol for pol in self.pols if split_pol(pol)[0] != split_pol(pol)[1]]
        self.same_pols = [pol for pol in self.pols if split_pol(pol)[0] == split_pol(pol)[1]]

        # Figure out which antennas are in the data
        self.ants = sorted(set([ant for bl in self.bls for ant in split_bl(bl)]))
        self.antnums = sorted(set([ant[0] for ant in self.ants]))
        self.antpols = sorted(set([ant[1] for ant in self.ants]))
        self.ants_per_antpol = {antpol: sorted([ant for ant in self.ants if ant[1] == antpol]) for antpol in self.antpols}

        # Parse apriori_xants
        self.apriori_xants = set()
        for ant in apriori_xants:
            if isinstance(ant, int):
                self.apriori_xants.update(set((ant, ap) for ap in self.antpols))
            elif isinstance(ant, tuple):
                assert len(ant) == 2
                ap = comply_pol(ant[1])
                assert ap in self.antpols
                self.apriori_xants.add((ant[0], ap))
            else:
                raise ValueError(f'{ant} is not a valid entry in apriori_xants.')

        # Set up metadata and summary stats
        self.version_str = __version__
        self.history = ''
        self._reset_summary_stats()

        # Load and summarize data and convert into correlation matrices
        self._update_corr_stats(sum_data=sum_data, diff_data=diff_data)

    def _reset_summary_stats(self, flag_val=np.nan):
        """Reset all the internal summary statistics back to empty."""
        # xants stores removed antennas and the iteration they were removed
        self.xants = {ant: -1 for ant in self.apriori_xants}
        self.crossed_ants = []
        self.dead_ants = []
        self.iter = 0
        self.all_metrics = {}
        self.final_metrics = {}
        self.corr_matrices = {self.join_pol(ap1, ap2): np.full((len(self.ants_per_antpol[ap1]),
                                                                len(self.ants_per_antpol[ap2])),
                                                                flag_val)
                              for ap1 in self.antpols for ap2 in self.antpols}

    def _update_corr_stats(self, sum_data, diff_data=None):
        """Extend the existing corr_stats dict of lists.
        """

        corr_stats = calc_corr_stats(sum_data, diff_data)
        # convert from corr stats to corr matrices
        self.ant_to_index = {ant: i for ants in self.ants_per_antpol.values() for i, ant in enumerate(ants)}
        for bl in corr_stats:
            if bl[0] == bl[1]:  # ignore autocorrelations
                continue
            pol = bl[2]
            ant1, ant2 = self.split_bl(bl)
            self.corr_matrices[pol][self.ant_to_index[ant1], self.ant_to_index[ant2]] = corr_stats[bl]
            self.corr_matrices[pol][self.ant_to_index[ant2], self.ant_to_index[ant1]] = corr_stats[bl]
        self.corr_matrices_for_xpol = deepcopy(self.corr_matrices)  # XXX why

    def _find_totally_dead_ants(self, verbose=False):
        """Flag antennas whose median correlation coefficient is 0.0.

        These antennas are marked as dead. They do not appear in recorded antenna
        metrics or zscores. Their removal iteration is -1 (i.e. before iterative
        flagging).
        """
        for pol in self.same_pols:
            # median over one antenna dimension
            med_corr_matrix = np.nanmedian(self.corr_matrices[pol], axis=0)
            is_dead = (med_corr_matrix == 0) | ~np.isfinite(med_corr_matrix)
            antpol = self.split_pol(pol)[0]
            dead_ants = [self.ants_per_antpol[antpol][i] for i in np.argwhere(is_dead)[:, 0]]
            for dead_ant in dead_ants:
                self._flag_corr_matrices(dead_ant)
                self.xants[dead_ant] = -1
                self.dead_ants.append(dead_ant)
                if verbose:
                    print(f'Antenna {dead_ant} appears totally dead and is removed.')

    def _flag_corr_matrices(self, ant_to_flag, flag_val=np.nan):
        """Sets all rows and columns in self.corr_matrices corresponding to the antenna to zero.
        """
        for pol in self.corr_matrices:
            ap1, ap2 = self.split_pol(pol)
            if ant_to_flag[1] == ap1:
                self.corr_matrices[pol][self.ant_to_index[ant_to_flag], :] = flag_val
            if ant_to_flag[1] == ap2:
                self.corr_matrices[pol][:, self.ant_to_index[ant_to_flag]] = flag_val

            # flag both polarizations for the xpol calculation to match previous versions of this algorithm
            self.corr_matrices_for_xpol[pol][self.ant_to_index[(ant_to_flag[0], ap1)], :] = flag_val
            self.corr_matrices_for_xpol[pol][:, self.ant_to_index[(ant_to_flag[0], ap2)]] = flag_val

    def _corr_metrics_per_ant(self):
        """Computes dictionary indexed by (ant, antpol) of the averaged unflagged correlation statistic.
        """
        per_ant_mean_corr_metrics = {}
        for pol in self.same_pols:
            # average over one antenna dimension
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                mean_corr_matrix = np.nanmean(self.corr_matrices[pol], axis=0)
            antpol = self.split_pol(pol)[0]
            per_ant_mean_corr_metrics.update(dict(zip(self.ants_per_antpol[antpol], mean_corr_matrix)))
        return per_ant_mean_corr_metrics

    def _corr_cross_pol_metrics_per_ant(self):
        """Computes dictionary indexed by (ant, antpol) of the cross-polarization statistic.
        """
        # construct all four combintions of same pols and cross pols
        matrix_pol_diffs = []
        for sp in self.same_pols:
            for cp in self.cross_pols:
                matrix_pol_diffs.append(self.corr_matrices_for_xpol[sp] - self.corr_matrices_for_xpol[cp])

        # average over one antenna dimension and then take the maximum of the four combinations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            cross_pol_metrics = np.nanmax(np.nanmean(matrix_pol_diffs, axis=1), axis=0)

        per_ant_corr_cross_pol_metrics = {}
        for _, ants in self.ants_per_antpol.items():
            per_ant_corr_cross_pol_metrics.update(dict(zip(ants, cross_pol_metrics)))
        return per_ant_corr_cross_pol_metrics

    def _run_all_metrics(self):
        """Local call for all metrics as part of iterative flagging method.
        """
        # Save all metrics
        metrics = {
            'corr': self._corr_metrics_per_ant(),
            'corrXPol': self._corr_cross_pol_metrics_per_ant(),
        }
        for name, metric in metrics.items():
            noninf_metric = {k:v for k, v in metric.items() if np.isfinite(v)}
            if not name in self.final_metrics:
                self.final_metrics[name] = {}
            self.final_metrics[name].update(noninf_metric)
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
                    self.xants[crossed_ant] = iteration
                    self.crossed_ants.append(crossed_ant)
                    self._flag_corr_matrices(crossed_ant)
                    if verbose:
                        print(f'On iteration {iteration} we flag {crossed_ant} with cross-pol corr metric of {crossMetrics[worstCrossAnt]}.')
            elif (worstDeadCutDiff < worstCrossCutDiff) and (worstDeadCutDiff < 0):
                dead_ants = set([worstDeadAnt])
                for dead_ant in dead_ants:
                    self.xants[dead_ant] = iteration
                    self.dead_ants.append(dead_ant)
                    self._flag_corr_matrices(dead_ant)
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
        out_dict = {'xants': list(self.xants.keys())}
        out_dict['crossed_ants'] = self.crossed_ants
        out_dict['dead_ants'] = self.dead_ants
        out_dict['final_metrics'] = self.final_metrics
        out_dict['all_metrics'] = self.all_metrics
        out_dict['removal_iteration'] = self.xants
        out_dict['cross_pol_cut'] = self.crossCut
        out_dict['dead_ant_cut'] = self.deadCut
        out_dict['datafile_list_sum'] = self.datafile_list_sum
        out_dict['datafile_list_diff'] = self.datafile_list_diff
        out_dict['history'] = self.history

        metrics_io.write_metric_file(filename, out_dict, overwrite=overwrite)


def ant_metrics_run(sum_files, diff_files=None, apriori_xants=[], a_priori_xants_yaml=None,
                    crossCut=0.0, deadCut=0.4, metrics_path='', extension='.ant_metrics.hdf5',
                    overwrite=False, history='', verbose=True):
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
    am = AntennaMetrics(sum_files, diff_files, apriori_xants=apriori_xants)
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
