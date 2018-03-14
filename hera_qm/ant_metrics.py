from __future__ import print_function, division, absolute_import
import numpy as np
import aipy
from copy import deepcopy
from pyuvdata import UVData
import json
import os
import re
from hera_qm.version import hera_qm_version_str
from hera_qm import utils


def get_ant_metrics_dict():
    """ Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns:
    metrics_dict -- Dictionary with metric names as keys and descriptions as values.
    """
    metrics_dict = {'ant_metrics_meanVij': 'Mean of the absolute value of all '
                    'visibilities associated with an antenna.',
                    'ant_metrics_meanVijXPol': 'Ratio of mean cross-pol visibilities '
                    'to mean same-pol visibilities: (Vxy+Vyx)/(Vxx+Vyy).',
                    'ant_metrics_redCorr': 'Extent to which baselines involving an '
                    'antenna do not correlate with others they are nominmally redundant with.',
                    'ant_metrics_redCorrXPol': 'Mean correlation ratio between '
                    'redundant visibilities and singlely-polarization flipped ones.',
                    'ant_metrics_mod_z_scores_meanVij': 'Modified z-score of the mean of the '
                    'absolute value of all visibilities associated with an antenna.',
                    'ant_metrics_mod_z_scores_meanVijXPol': 'Modified z-score of the ratio '
                    'of mean cross-pol visibilities to mean same-pol visibilities: '
                    '(Vxy+Vyx)/(Vxx+Vyy).',
                    'ant_metrics_mod_z_scores_redCorr': 'Modified z-score of the extent to '
                    'which baselines involving an antenna do not correlate with others '
                    'they are nominally redundant with.',
                    'ant_metrics_mod_z_scores_redCorrXPol': 'Modified z-score of the mean '
                    'correlation ratio between redundant visibilities and singlely-'
                    'polarization flipped ones.',
                    'ant_metrics_crossed_ants': 'Antennas deemed to be cross-polarized by '
                    'hera_qm.ant_metrics.',
                    'ant_metrics_removal_iteration': 'hera_qm.ant_metrics iteration number in '
                    'which the antenna was removed.',
                    'ant_metrics_xants': 'Antennas deemed bad by hera_qm.ant_metrics.',
                    'ant_metrics_dead_ants': 'Antennas deemed to be dead by hera_qm.ant_metrics.'}
    return metrics_dict

#######################################################################
# Low level functionality that is potentially reusable
#######################################################################


def per_antenna_modified_z_scores(metric):
    '''For a given metric, stored as a (ant,antpol) dictonary, computes the per-pol modified z-score
    for each antenna, which is the metrics, minus the median, divided by the median absolute deviation.'''
    zscores = {}
    antpols = set([key[1] for key in metric.keys()])
    for antpol in antpols:
        values = np.array([val for key, val in metric.items() if key[1] == antpol])
        median = np.nanmedian(values)
        medAbsDev = np.nanmedian(np.abs(values - median))
        for key, val in metric.items():
            if key[1] == antpol:
                zscores[key] = 0.6745 * (val - median) / medAbsDev
                # this factor makes it comparable to a standard z-score for gaussian data
    return zscores


def mean_Vij_metrics(data, pols, antpols, ants, bls, xants=[], rawMetric=False):
    '''Calculates how an antennas's average |Vij| deviates from others.

    Arguments:
    data -- data for all polarizations in a format that can support data.get_data(i,j,pol)
    pols -- List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols -- List of antenna polarizations (e.g. ['x', 'y'])
    ants -- List of all antenna indices.
    bls -- List of tuples of antenna pairs.
    xants -- list of antennas in the (ant,antpol) format that should be ignored.
    rawMetric -- return the raw mean Vij metric instead of the modified z-score

    Returns:
    meanMetrics -- a dictionary indexed by (ant,antpol) of the modified z-score of the mean of the
    absolute value of all visibilities associated with an antenna. Very small or very large numbers
    are probably bad antennas.
    '''

    absVijMean = {(ant, antpol): 0.0 for ant in ants for antpol in antpols if
                  (ant, antpol) not in xants}
    visCounts = deepcopy(absVijMean)
    for (i, j) in bls:
        if i != j:
            for pol in pols:
                for ant, antpol in zip((i, j), pol):
                    if (ant, antpol) not in xants:
                        d = data.get_data(i, j, pol)
                        absVijMean[(ant, antpol)] += np.nansum(np.abs(d))
                        visCounts[(ant, antpol)] += d.size
    timeFreqMeans = {key: absVijMean[key] / visCounts[key] for key in absVijMean.keys()}

    if rawMetric:
        return timeFreqMeans
    else:
        return per_antenna_modified_z_scores(timeFreqMeans)


def compute_median_auto_power_dict(data, pols, reds):
    '''Computes the median over frequency of the visibility squared, averaged over time.'''
    autoPower = {}
    for pol in pols:
        for bls in reds:
            for (i, j) in bls:
                autoPower[i, j, pol] = np.median(np.mean(np.abs(data.get_data(i, j, pol))**2, axis=0))
    return autoPower


def red_corr_metrics(data, pols, antpols, ants, reds, xants=[], rawMetric=False, crossPol=False):
    '''Calculates the extent to which baselines involving an antenna do not correlate
    with others they are nominmally redundant with.

    Arguments:
    data -- data for all polarizations in a format that can support data.get_data(i,j,pol)
    pols -- List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols -- List of antenna polarizations (e.g. ['x', 'y'])
    ants -- List of all antenna indices.
    reds -- List of lists of tuples of antenna numbers that make up redundant baseline groups.
    xants -- list of antennas in the (ant,antpol) format that should be ignored.
    rawMetric -- return the raw power correlations instead of the modified z-score
    crossPol -- return results only when the two visibility polarizations differ by a single flip

    Returns:
    powerRedMetric -- a dictionary indexed by (ant,antpol) of the modified z-scores of the mean
    power correlations inside redundant baseline groups that the antenna participates in.
    Very small numbers are probably bad antennas.
    '''

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
            # This function can instead record correlations for antennas whose counterpart are pol-swapped
            if (not crossPol and (pol0 is pol1)) or (crossPol and onlyOnePolCrossed):
                for bls in reds:
                    for n, (ant0_i, ant0_j) in enumerate(bls):
                        data0 = data.get_data(ant0_i, ant0_j, pol0)
                        for (ant1_i, ant1_j) in bls[n + 1:]:
                            data1 = data.get_data(ant1_i, ant1_j, pol1)
                            corr = np.median(np.abs(np.mean(data0 * data1.conj(),
                                                            axis=0)))
                            corr /= np.sqrt(autoPower[ant0_i, ant0_j, pol0] *
                                            autoPower[ant1_i, ant1_j, pol1])
                            antsInvolved = [(ant0_i, pol0[0]), (ant0_j, pol0[1]),
                                            (ant1_i, pol1[0]), (ant1_j, pol1[1])]
                            if not np.any([(ant, antpol) in xants for ant, antpol
                                           in antsInvolved]):
                                # Only record the crossed antenna if i or j is crossed
                                if crossPol and iscrossed_i:
                                    antsInvolved = [(ant0_i, pol0[0]),
                                                    (ant1_i, pol1[0])]
                                elif crossPol and iscrossed_j:
                                    antsInvolved = [(ant0_j, pol0[1]), (ant1_j, pol1[1])]
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
    '''Takes a list of excluded antennas and adds on all polarizations of those antennas.'''
    xantSet = set(xants)
    for xant in xants:
        for antpol in antpols:
            xantSet.add((xant[0], antpol))
    return list(xantSet)


def antpol_metric_sum_ratio(ants, antpols, crossMetrics, sameMetrics, xants=[]):
    '''Takes the ratio of two antenna metrics, summed over both polarizations, and creates a new
    antenna metric with the same value in both polarizations for each antenna.'''
    crossPolRatio = {}
    for ant in ants:
        if np.all([(ant, antpol) not in xants for antpol in antpols]):
            crossSum = np.sum([crossMetrics[(ant, antpol)] for antpol in antpols])
            sameSum = np.sum([sameMetrics[(ant, antpol)] for antpol in antpols])
            for antpol in antpols:
                crossPolRatio[(ant, antpol)] = crossSum / sameSum
    return crossPolRatio


def mean_Vij_cross_pol_metrics(data, pols, antpols, ants, bls, xants=[], rawMetric=False):
    '''Find which antennas are outliers based on the ratio of mean cross-pol visibilities to
    mean same-pol visibilities: (Vxy+Vyx)/(Vxx+Vyy).

    Arguments:
    data -- data for all polarizations in a format that can support data.get_data(i,j,pol)
    pols -- List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols -- List of antenna polarizations (e.g. ['x', 'y'])
    ants -- List of all antenna indices.
    bls -- List of tuples of antenna pairs.
    xants -- list of antennas in the (ant,antpol) format that should be ignored. If, e.g., (81,'y')
            is excluded, (81,'x') cannot be identified as cross-polarized and will be excluded.
    rawMetric -- return the raw power ratio instead of the modified z-score

    Returns:
    mean_Vij_cross_pol_metrics -- a dictionary indexed by (ant,antpol) of the modified z-scores of the
            ratio of mean visibilities, (Vxy+Vyx)/(Vxx+Vyy). Results duplicated in both antpols.
            Very large values are probably cross-polarized.
    '''

    # Compute metrics and cross pols only and and same pols only
    samePols = [pol for pol in pols if pol[0] == pol[1]]
    crossPols = [pol for pol in pols if pol[0] != pol[1]]
    full_xants = exclude_partially_excluded_ants(antpols, xants)
    meanVijMetricsSame = mean_Vij_metrics(data, samePols, antpols, ants, bls,
                                          xants=full_xants, rawMetric=True)
    meanVijMetricsCross = mean_Vij_metrics(data, crossPols, antpols, ants, bls,
                                           xants=full_xants, rawMetric=True)

    # Compute the ratio of the cross/same metrics, saving the same value in each antpol
    crossPolRatio = antpol_metric_sum_ratio(ants, antpols, meanVijMetricsCross,
                                            meanVijMetricsSame, xants=full_xants)
    if rawMetric:
        return crossPolRatio
    else:
        return per_antenna_modified_z_scores(crossPolRatio)


def red_corr_cross_pol_metrics(data, pols, antpols, ants, reds, xants=[], rawMetric=False):
    '''Find which antennas are part of visibilities that are significantly better correlated with
    polarization-flipped visibilities in a redundant group. Returns the modified z-score.

    Arguments:
    data -- data for all polarizations in a format that can support data.get_data(i,j,pol)
    pols -- List of visibility polarizations (e.g. ['xx','xy','yx','yy']).
    antpols -- List of antenna polarizations (e.g. ['x', 'y'])
    ants -- List of all antenna indices.
    reds -- List of lists of tuples of antenna numbers that make up redundant baseline groups.
    xants -- list of antennas in the (ant,antpol) format that should be ignored. If, e.g., (81,'y')
            is excluded, (81,'x') cannot be identified as cross-polarized and will be excluded.
    rawMetric -- return the raw correlation ratio instead of the modified z-score

    Returns:
    redCorrCrossPolMetrics -- a dictionary indexed by (ant,antpol) of the modified z-scores of the
            mean correlation ratio between redundant visibilities and singlely-polarization flipped
            ones. Very large values are probably cross-polarized.
    '''

    # Compute metrics for singly flipped pols and just same pols
    full_xants = exclude_partially_excluded_ants(antpols, xants)
    samePols = [pol for pol in pols if pol[0] == pol[1]]
    redCorrMetricsSame = red_corr_metrics(data, samePols, antpols, ants, reds,
                                          xants=full_xants, rawMetric=True)
    redCorrMetricsCross = red_corr_metrics(data, pols, antpols, ants, reds,
                                           xants=full_xants, rawMetric=True, crossPol=True)

    # Compute the ratio of the cross/same metrics, saving the same value in each antpol
    crossPolRatio = antpol_metric_sum_ratio(ants, antpols, redCorrMetricsCross,
                                            redCorrMetricsSame, xants=full_xants)
    if rawMetric:
        return crossPolRatio
    else:
        return per_antenna_modified_z_scores(crossPolRatio)


def average_abs_metrics(metrics1, metrics2):
    '''Averages the absolute value of two metrics together.'''

    if set(metrics1.keys()) != set(metrics2.keys()):
        raise KeyError('Metrics being averaged have differnt (ant,antpol) keys.')
    return {key: np.nanmean([np.abs(metrics1[key]), np.abs(metrics2[key])]) for
            key in metrics1.keys()}


def load_antenna_metrics(metricsJSONFile):
    '''Loads all cut decisions and meta-metrics from a JSON into python dictionary.'''

    with open(metricsJSONFile, 'r') as infile:
        jsonMetrics = json.load(infile)
    gvars = {'nan': np.nan, 'inf': np.inf, '-inf': -np.inf}
    return {key: (eval(str(val), gvars) if (key != 'version' and key != 'history') else str(val)) for
            key, val in jsonMetrics.items()}


#######################################################################
# High level functionality for HERA
#######################################################################


class Antenna_Metrics():
    '''Object for holding relevant visibility data and metadata with interfaces to four
    antenna metrics (two for identifying dead antennas, two for identifying cross-polarized ones),
    an iterative method for identifying one bad antenna at a time while keeping track of all
    metrics, and for writing metrics to a JSON. Works on raw data from a single observation
    with all four visibility polarizations.'''

    def __init__(self, dataFileList, reds, fileformat='miriad'):
        '''Arguments:
        dataFileList -- List of data filenames of the four different visibility
                        polarizations for the same observation
        reds -- List of lists of tuples of antenna numbers that make up redundant baseline groups
        format -- default 'miriad'. Other options: 'uvfits', 'fhd', 'ms ' (see pyuvdata docs)
        '''

        self.data = UVData()
        if fileformat == 'miriad':
            self.data.read_miriad(dataFileList)
        elif fileformat == 'uvfits':
            self.data.read_uvfits(dataFileList)
        elif fileformat == 'fhd':
            self.data.read_fhd(dataFileList)
        else:
            raise ValueError('Unrecognized file format ' + str(fileformat))
        self.ants = self.data.get_ants()
        self.pols = [pol.lower() for pol in self.data.get_pols()]
        self.antpols = [antpol.lower() for antpol in self.data.get_feedpols()]
        self.bls = self.data.get_antpairs()
        self.dataFileList = dataFileList
        self.reds = reds
        self.version_str = hera_qm_version_str
        self.history = ''

        if len(self.antpols) is not 2 or len(self.pols) is not 4:
            raise ValueError('Missing polarization information. pols =' +
                             str(self.pols) + ' and antpols = ' + str(self.antpols))

    def mean_Vij_metrics(self, pols=None, xants=[], rawMetric=False):
        '''Local wrapper for mean_Vij_metrics in hera_qm.ant_metrics module.'''

        if pols is None:
            pols = self.pols
        return mean_Vij_metrics(self.data, pols, self.antpols, self.ants, self.bls,
                                xants=xants, rawMetric=rawMetric)

    def red_corr_metrics(self, pols=None, xants=[], rawMetric=False, crossPol=False):
        '''Local wrapper for red_corr_metrics in hera_qm.ant_metrics module.'''

        if pols is None:
            pols = self.pols
        return red_corr_metrics(self.data, pols, self.antpols, self.ants, self.reds,
                                xants=xants, rawMetric=rawMetric, crossPol=crossPol)

    def mean_Vij_cross_pol_metrics(self, xants=[], rawMetric=False):
        '''Local wrapper for mean_Vij_cross_pol_metrics in hera_qm.ant_metrics module.'''

        return mean_Vij_cross_pol_metrics(self.data, self.pols, self.antpols, self.ants,
                                          self.bls, xants=xants, rawMetric=rawMetric)

    def red_corr_cross_pol_metrics(self, xants=[], rawMetric=False):
        '''Local wrapper for red_corr_cross_pol_metrics in hera_qm.ant_metrics module.'''

        return red_corr_cross_pol_metrics(self.data, self.pols, self.antpols, self.ants,
                                          self.reds, xants=xants, rawMetric=False)

    def reset_summary_stats(self):
        '''Resets all the internal summary statistics back to empty.'''

        self.xants, self.crossedAntsRemoved, self.deadAntsRemoved = [], [], []
        self.removalIter = {}
        self.allMetrics, self.allModzScores = [], []
        self.finalMetrics, self.finalModzScores = {}, {}

    def find_totally_dead_ants(self):
        '''Flags antennas whose median autoPower that they are involved in is 0.0.
        These antennas are marked as dead, but they do not appear in recorded antenna
        metrics or zscores. Their removal iteration is -1 (i.e. before iterative flagging).'''

        autoPowers = compute_median_auto_power_dict(self.data, self.pols, self.reds)
        power_list_by_ant = {(ant, antpol): [] for ant in self.ants for antpol
                             in self.antpols if (ant, antpol) not in self.xants}
        for (ant0, ant1, pol), power in autoPowers.items():
            if (ant0, pol[0]) not in self.xants and (ant1, pol[1]) not in self.xants:
                power_list_by_ant[(ant0, pol[0])].append(power)
                power_list_by_ant[(ant1, pol[1])].append(power)
        for key, val in power_list_by_ant.items():
            if np.median(val) == 0:
                self.xants.append(key)
                self.deadAntsRemoved.append(key)
                self.removalIter[key] = -1

    def _run_all_metrics(self):
        '''Designed to be run as part of AntennaMetrics.iterative_antenna_metrics_and_flagging().'''

        # Compute all raw metrics
        meanVij = self.mean_Vij_metrics(xants=self.xants, rawMetric=True)
        redCorr = self.red_corr_metrics(pols=['xx', 'yy'], xants=self.xants, rawMetric=True)
        meanVijXPol = self.mean_Vij_cross_pol_metrics(xants=self.xants, rawMetric=True)
        redCorrXPol = self.red_corr_cross_pol_metrics(xants=self.xants, rawMetric=True)

        # Save all metrics and zscores
        metrics, modzScores = {}, {}
        for metName in ['meanVij', 'redCorr', 'meanVijXPol', 'redCorrXPol']:
            metric = eval(metName)
            metrics[metName] = metric
            modz = per_antenna_modified_z_scores(metric)
            modzScores[metName] = modz
            for key in metric.keys():
                if metName in self.finalMetrics:
                    self.finalMetrics[metName][key] = metric[key]
                    self.finalModzScores[metName][key] = modz[key]
                else:
                    self.finalMetrics[metName] = {key: metric[key]}
                    self.finalModzScores[metName] = {key: modz[key]}
        self.allMetrics.append(metrics)
        self.allModzScores.append(modzScores)

    def iterative_antenna_metrics_and_flagging(self, crossCut=5, deadCut=5, alwaysDeadCut=10, verbose=False):
        '''Runs all four metrics (two for dead antennas two for cross-polarized antennas) and saves
        the results internally to this this antenna metrics object.

        Arguments:
        crossCut -- Modified z-score cut for most cross-polarized antenna. Default 5 "sigmas".
        deadCut -- Modified z-score cut for most likely dead antenna. Default 5 "sigmas".
        alwaysDeadCut -- Modified z-score cut for antennas that are definitely dead. Default 10 "sigmas".
            These are all thrown away at once without waiting to iteratively throw away only the worst offender.
        '''

        self.reset_summary_stats()
        self.find_totally_dead_ants()
        self.crossCut, self.deadCut, self.alwaysDeadCut = crossCut, deadCut, alwaysDeadCut

        # Loop over
        for n in range(len(self.antpols) * len(self.ants)):
            self._run_all_metrics()

            # Mostly likely dead antenna
            deadMetrics = average_abs_metrics(self.allModzScores[-1]['meanVij'],
                                              self.allModzScores[-1]['redCorr'])
            worstDeadAnt = max(deadMetrics, key=deadMetrics.get)
            worstDeadCutRatio = np.abs(deadMetrics[worstDeadAnt]) / deadCut

            # Most likely cross-polarized antenna
            crossMetrics = average_abs_metrics(self.allModzScores[-1]['meanVijXPol'],
                                               self.allModzScores[-1]['redCorrXPol'])
            worstCrossAnt = max(crossMetrics, key=crossMetrics.get)
            worstCrossCutRatio = np.abs(crossMetrics[worstCrossAnt]) / crossCut

            # Find the single worst antenna, remove it, log it, and run again
            if worstCrossCutRatio >= worstDeadCutRatio and worstCrossCutRatio >= 1.0:
                for antpol in self.antpols:
                    self.xants.append((worstCrossAnt[0], antpol))
                    self.crossedAntsRemoved.append((worstCrossAnt[0], antpol))
                    self.removalIter[(worstCrossAnt[0], antpol)] = n
                    if verbose:
                        print('On iteration', n, 'we flag', (worstCrossAnt[0], antpol))
            elif worstDeadCutRatio > worstCrossCutRatio and worstDeadCutRatio > 1.0:
                dead_ants = set([worstDeadAnt])
                for ant,metric in deadMetrics.items():
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

    def save_antenna_metrics(self, metricsJSONFilename):
        '''Saves all cut decisions and meta-metrics in a human-readable JSON that can be loaded
        back into a dictionary using hera_qm.ant_metrics.load_antenna_metrics().'''

        if not hasattr(self, 'xants'):
            raise KeyError('Must run AntennaMetrics.iterative_antenna_metrics_and_flagging() first.')

        allMetricsData = {'xants': str(self.xants)}
        allMetricsData['crossed_ants'] = str(self.crossedAntsRemoved)
        allMetricsData['dead_ants'] = str(self.deadAntsRemoved)
        allMetricsData['final_metrics'] = str(self.finalMetrics)
        allMetricsData['all_metrics'] = str(self.allMetrics)
        allMetricsData['final_mod_z_scores'] = str(self.finalModzScores)
        allMetricsData['all_mod_z_scores'] = str(self.allModzScores)
        allMetricsData['removal_iteration'] = str(self.removalIter)
        allMetricsData['cross_pol_z_cut'] = str(self.crossCut)
        allMetricsData['dead_ant_z_cut'] = str(self.deadCut)
        allMetricsData['always_dead_ant_z_cut'] = str(self.alwaysDeadCut)
        allMetricsData['datafile_list'] = str(self.dataFileList)
        allMetricsData['reds'] = str(self.reds)
        allMetricsData['version'] = self.version_str
        # make sure we have something in the history string to write it out
        if self.history != '':
            allMetricsData['history'] = self.history

        with open(metricsJSONFilename, 'w') as outfile:
            json.dump(allMetricsData, outfile, indent=4)


# code for running ant_metrics on a file
def ant_metrics_run(files, args, history):
    """
    Run a series of ant_metrics tests on a given set of input files.

    Args:
       files -- a list of files to run ant metrics on. Can be any of the 4 polarizations
       args -- parsed arguments via argparse.ArgumentParser.parse_args
    Return:
       None

    The funciton will take in a list of files and options. It will run the
    series of ant metrics tests, and produce a JSON file containing the relevant
    information. The file list passed in need only contain one of the polarization
    files for a given JD, and the function will look for the other polarizations
    in the same folder. If not all four polarizations are found, a warning is
    generated, since the code assumes all four polarizations are present.
    """
    try:
        from hera_cal.omni import aa_to_info
        from hera_cal.utils import get_aa_from_uv
    except(ImportError):
        from nose.plugins.skip import SkipTest
        raise SkipTest('hera_cal.omni not detected. It must be installed to calculate array info')

    # check that we were given some files to process
    if len(files) == 0:
        raise AssertionError('Please provide a list of visibility files')

    # define polarizations to look for
    if args.pol == '':
        # default polarization list
        pol_list = ['xx', 'yy', 'xy', 'yx']
    else:
        # assumes polarizations are passed in as comma-separated list, e.g. 'xx,xy,yx,yy'
        pol_list = args.pol.split(',')

    # generate a list of all files to be read in
    fullpol_file_list = utils.generate_fullpol_file_list(files, pol_list)
    if len(fullpol_file_list) == 0:
        raise AssertionError('Could not find all 4 polarizations for any files provided')

    if args.cal is not None:
        # define freqs
        # note that redundancy calculation does not depend on this, so this is just a dummy range
        freqs = np.linspace(0.1, 0.2, num=1024, endpoint=False)
        # process calfile
        aa = aipy.cal.get_aa(args.cal, freqs)
    else:
        # generate aa object from file
        # N.B.: assumes redunancy information is the same for all files passed in
        first_file = fullpol_file_list[0][0]
        uvd = UVData()
        uvd.read_miriad(first_file)
        aa = get_aa_from_uv(uvd)
        del uvd
    info = aa_to_info(aa, pols=[pol_list[-1][0]])
    reds = info.get_reds()

    # do the work
    for jd_list in fullpol_file_list:
        am = Antenna_Metrics(jd_list, reds, fileformat=args.vis_format)
        am.iterative_antenna_metrics_and_flagging(crossCut=args.crossCut, deadCut=args.deadCut,
                                                  alwaysDeadCut=args.alwaysDeadCut, verbose=args.verbose)

        # add history
        am.history = am.history + history

        base_filename = jd_list[0]
        abspath = os.path.abspath(base_filename)
        dirname = os.path.dirname(abspath)
        basename = os.path.basename(base_filename)
        nopol_filename = re.sub('\.{}\.'.format(pol_list[0]), '.', basename)
        if args.metrics_path == '':
            # default path is same directory as file
            metrics_path = dirname
        else:
            metrics_path = args.metrics_path
        metrics_basename = nopol_filename + args.extension
        metrics_filename = os.path.join(metrics_path, metrics_basename)
        am.save_antenna_metrics(metrics_filename)

    return
