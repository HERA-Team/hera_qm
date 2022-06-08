# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for computing metrics on omnical calibration solutions."""

import numpy as np
from pyuvdata import UVCal
import astropy.stats as astats
from collections import OrderedDict as odict
from . import __version__
from . import utils
import pickle as pkl
import json
import copy
import os


def get_omnical_metrics_dict():
    """Get a dictionary of metric names and their descriptions.

    A simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns
    -------
    metrics_dict : dict
        A dicationary with metric names as keys and descriptions as values.

    """
    metrics_dict = {'chisq_tot_avg_XX': 'median of chi-square across entire file for XX pol',
                    'chisq_tot_avg_YY': 'median of chi-square across entire file for YY pol',
                    'chisq_ant_avg': 'median of chi-square per-antenna across file',
                    'chisq_ant_std': 'standard dev. of chi-square per-antenna across file',
                    'chisq_good_sol_XX': 'determination of a good solution based on whether all antennas'
                    'have roughly the same chi-square standard deviation for XX pol',
                    'chisq_good_sol_YY': 'determination of a good solution based on whether all antennas'
                    'have roughly the same chi-square standard deviation for YY pol',
                    'ant_phs_std': 'gain phase standard deviation per-antenna across file',
                    'ant_phs_std_max_XX': 'maximum of ant_phs_std for XX pol',
                    'ant_phs_std_max_YY': 'maximum of ant_phs_std for YY pol',
                    'ant_phs_std_good_sol_XX': 'determination of a good solution based on whether'
                    'ant_phs_std_max is below some threshold for XX pol',
                    'ant_phs_std_good_sol_YY': 'determination of a good solution based on whether'
                    'ant_phs_std_max is below some threshold for YY pol'}
    return metrics_dict


def load_omnical_metrics(filename):
    """Load an omnical metrics file.

    Parameters
    ----------
    filename : str
        Path to an omnical metrics file.

    Returns
    -------
    metrics : dict
        A dictionary containing omnical metrics.

    Raises
    ------
    IOError:
        If the filetype inferred from the filename is not "json" or "pkl",
        an IOError is raised.

    """
    # get filetype
    filetype = filename.split('.')[-1]

    # load json
    if filetype == 'json':
        with open(filename, 'r') as f:
            metrics = json.load(f, object_pairs_hook=odict)

        # ensure keys of ant_dicts are not strings
        # loop over pols
        for key, metric in metrics.items():
            # loop over items in each pol metric dict
            for key2 in metric.keys():
                if isinstance(metric[key2], (dict, odict)):
                    if isinstance(list(metric[key2].values())[0], list):
                        metric[key2] = odict([(int(i), np.array(metric[key2][i])) for i in metric[key2]])
                    elif isinstance(list(metric[key2].values())[0], (str, np.unicode_)):
                        metric[key2] = odict([(int(i), metric[key2][i].astype(np.complex128)) for i in metric[key2]])

                elif isinstance(metric[key2], list):
                    metric[key2] = np.array(metric[key2])

    # load pickle
    elif filetype == 'pkl':
        with open(filename, 'rb') as f:
            inp = pkl.Unpickler(f)
            metrics = inp.load()
    else:
        raise IOError("Filetype not recognized, try a json or pkl file")

    return metrics


def write_metrics(metrics, filename=None, filetype='json'):
    """Write metrics to file after running self.run_metrics().

    Parameters
    ----------
    metrics : dict
        A dictionary containing the output of Omnical_Metrics.run_metrics().
    filename : str
        The base filename to write out. If not specified, the default is
        the filename saved in the metrics dictionary.
    filetype : {"json", "pkl"}, optional
        The file format of output metrics file. Default is "json".
    """
    # get pols
    pols = list(metrics.keys())

    if filename is None:
        filename = os.path.join(metrics[pols[0]]['filedir'],
                                metrics[pols[0]]['filestem'] + '.omni_metrics')

    # write to file
    if filetype == 'json':
        if filename.split('.')[-1] != 'json':
            filename += '.json'

        # change ndarrays to lists
        metrics_out = copy.deepcopy(metrics)
        # loop over pols
        for pol in metrics_out.keys():
            # loop over keys
            for key in metrics_out[pol].keys():
                if isinstance(metrics_out[pol][key], np.ndarray):
                    metrics_out[pol][key] = metrics[pol][key].tolist()
                elif isinstance(metrics_out[pol][key], (dict, odict)):
                    if list(metrics_out[pol][key].values())[0].dtype == complex:
                        metrics_out[pol][key] = odict([(j, metrics_out[pol][key][j].astype(str)) for j in metrics_out[pol][key]])
                    metrics_out[pol][key] = odict([(str(j), metrics_out[pol][key][j].tolist()) for j in metrics_out[pol][key]])
                elif isinstance(metrics_out[pol][key], (bool, np.bool_)):
                    metrics_out[pol][key] = bool(metrics_out[pol][key])
                elif isinstance(metrics_out[pol][key], (float, np.float32, np.float64)):
                    metrics_out[pol][key] = float(metrics_out[pol][key])
                elif isinstance(metrics_out[pol][key], np.integer):
                    metrics_out[pol][key] = int(metrics_out[pol][key])

        with open(filename, 'w') as outfile:
            json.dump(metrics_out, outfile, indent=4)

    elif filetype == 'pkl':
        if filename.split('.')[-1] != 'pkl':
            filename += '.pkl'
        with open(filename, 'wb') as outfile:
            outp = pkl.Pickler(outfile)
            outp.dump(metrics)


def load_firstcal_gains(fc_file):
    """Load firstcal delays and turn into phase gains.

    Parameters
    ----------
    fc_file : str
        Path to firstcal .calfits file. If a file with
        multiple polarizations is passed in, only the
        first polarization is used.

    Returns
    -------
    fc_delays : array
        The firstcal delays as saved in the file.
    fc_gains : array
        The firstcal delays in gains format.
    fc_pol : int
        The polarization of the .calfits file.
    """
    uvf = UVCal()
    uvf.read_calfits(fc_file)
    freqs = uvf.freq_array.squeeze()
    fc_gains = np.moveaxis(uvf.gain_array, 2, 3)[:, 0, :, :, 0]
    d_nu = np.mean(freqs[1:] - freqs[:-1])
    d_phi = np.abs(np.mean(np.angle(fc_gains)[:, :, 1:] - np.angle(fc_gains)[:, :, :-1], axis=2))
    fc_delays = (d_phi / d_nu) / (2 * np.pi)
    fc_pol = uvf.jones_array[0]
    return fc_delays, fc_gains, fc_pol


def plot_phs_metric(metrics, plot_type='std', ax=None, save=False,
                    fname=None, outpath=None,
                    plot_kwargs={'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}):
    """Plot omnical phase metric.

    Parameters
    ----------
    metrics : dict
        A dictionary of metrics from OmniCal_Metrics.run_metrics().
    plot_type : {"std", "ft", "hist"}, optional
        The plot type to make. "std" plots the standard deviations of
        omni-firstcal phase differences. Large standard deviations
        are not ideal. "hist" plots the histogram of gain phase
        solutions. "ft" plots the FFT of the antenna gain differences.
        Default is "std".
    ax : matplotlib axis object, optional
        The axis on which to generate plots. If None, axes will be generated
        as necessary. Default is None.
    save : bool, optional
        If True, save image as a png. Default is False.
    fname : str, optional
        Filename to save image as. If not specified, a filename is generated
        based on the input filename saved in the metrics dictionary.
    outpath : str, optional
        Path to place file in. If not specified, will default to location
        of the input *omni.calfits file.
    plot_kwargs : dict, optional
        A dictionary of keyword arguments to be passed to the plot command.
        Default is {'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}

    Returns
    -------
    fig : matplotlib figure object
        If ax is None, the figure created to generate the plot is returned.

    """
    import matplotlib.pyplot as plt
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    if plot_type == 'std':
        # get y data
        ant_phs_std = np.array(list(metrics['ant_phs_std'].values()))
        phs_std_cut = metrics['phs_std_cut']
        ymax = np.max([ant_phs_std.max() * 1.1, phs_std_cut * 1.2])

        # make grid and plot points
        ax.grid(True)
        ax.plot(ant_phs_std, **plot_kwargs)
        p1 = ax.axhline(phs_std_cut, color='darkorange')
        ax.axhspan(phs_std_cut, ymax, color='darkorange', alpha=0.2)
        ax.legend([p1], ["phase std cut"], fontsize=12)

        # adjust axes
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('phase stand dev across file [radians]', fontsize=14)
        ax.tick_params(size=8)
        ax.set_xticks(range(metrics['Nants']))
        ax.set_xticklabels(metrics['ant_array'])
        for label in ax.get_xticklabels():
            label.set_rotation(20)
        ax.set_ylim(0, ymax)
        ax.set_title("{0}".format(metrics['filename']))

    if plot_type == 'hist':
        ylines = np.array(list(metrics['ant_phs_hists'].values()))
        ax.grid(True)
        plist = [ax.plot(metrics['ant_phs_hist_bins'], ylines[i], alpha=0.75)
                 for i in range(len(ylines))]
        ax.set_xlabel('gain phase [radians]', fontsize=14)
        ax1 = fig.add_axes([0.99, 0.1, 0.03, 0.8])
        ax1.axis('off')
        ax1.legend(np.concatenate(plist), metrics['ant_array'])
        ax.set_title("gain phase histogram for {0}".format(metrics['filename']))

    if plot_type == 'ft':
        ylines = np.abs(np.array(list(metrics['ant_gain_fft'].values())))
        ax.grid(True)
        plist = [ax.plot(metrics['ant_gain_dly'] * 1e9, ylines[i], alpha=0.75)
                 for i in range(len(ylines))]
        ax.set_xlabel('delay [ns]', fontsize=14)
        ax.set_ylabel(r'radians sec$^{-1}$', fontsize=14)
        ax.set_yscale('log')
        ax1 = fig.add_axes([0.99, 0.1, 0.03, 0.8])
        ax1.axis('off')
        ax1.legend(np.concatenate(plist), metrics['ant_array'])
        ax.set_title("abs val of FT(complex gains) for {0}".format(metrics['filename']))

    if save is True:
        if fname is None:
            fname = (utils.strip_extension(metrics['filename'])
                     + '.phs_{0}.png'.format(plot_type))
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        if custom_ax is False:
            fig.savefig(fname, bbox_inches='tight')
        else:
            ax.figure.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig


def plot_chisq_metric(metrics, ax=None, save=False, fname=None, outpath=None,
                      plot_kwargs={'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}):
    """Plot Chi-Square output from omnical for each antenna.

    Parameters
    ----------
    metrics : dict
        A metrics dictionary from OmniCal_Metrics.run_metrics().
    ax : matplotlib axis object, optional
        The axis on which to generate plots. If None, axes will be generated
        as necessary. Default is None.
    save : bool, optional
        If True, save image as a png. Default is False.
    fname : str, optional
        Filename to save image as. If not specified, a filename is generated
        based on the input filename saved in the metrics dictionary.
    outpath : str, optional
        Path to place file in. If not specified, will default to location
        of the input *omni.calfits file.
    plot_kwargs : dict, optional
        A dictionary of keyword arguments to be passed to the plot command.
        Default is {'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}.

    Returns
    -------
    fig : matplotlib figure object
        If ax is None, the figure created to generate the plot is returned.

    """
    import matplotlib.pyplot as plt
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=(8, 6))

        if ax is None:
            ax = fig.add_subplot(111)

    # get y data
    yloc = metrics['chisq_ant_std_loc']
    ysig = metrics['chisq_ant_std_scale']
    ycut = yloc + metrics['chisq_std_zscore_cut'] * ysig
    chisq_ant_std = np.array(list(metrics['chisq_ant_std'].values()))
    ymax = np.max([chisq_ant_std.max() * 1.2, ycut + ysig])

    # make grid and plots
    ax.grid(True)
    p1 = ax.axhspan(yloc - ysig, yloc + ysig,
                    color='green', alpha=0.1)
    p2 = ax.axhline(yloc, color='steelblue')
    p3 = ax.axhline(ycut, color='darkred')
    ax.axhspan(ycut, ymax, color='darkred', alpha=0.2)
    ax.plot(chisq_ant_std, **plot_kwargs)
    ax.legend([p2, p1, p3], ["avg std", "1-sigma", "std cut"], fontsize=12)

    # adjust axes
    ax.set_xlabel('antenna number', fontsize=14)
    ax.set_ylabel('chi-square stand dev across file', fontsize=14)
    ax.tick_params(size=8)
    ax.set_xticks(range(metrics['Nants']))
    ax.set_xticklabels(metrics['ant_array'])
    [t.set_rotation(20) for t in ax.get_xticklabels()]
    ax.set_ylim(0, ymax)
    ax.set_title("{0}".format(metrics['filename']))

    if save is True:
        if fname is None:
            fname = utils.strip_extension(metrics['filename']) + '.chisq_std.png'
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        if custom_ax is False:
            fig.savefig(fname, bbox_inches='tight')
        else:
            ax.figure.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig


class OmniCal_Metrics(object):
    """Class for computing and storing omnical metrics.

    This class provides methods for storing omnical gain solutions
    and running metrics on them.
    """

    jones2pol = {-5: 'XX', -6: 'YY', -7: 'XY', -8: 'YX'}

    def __init__(self, omni_calfits, history=''):
        """Initialize an Omnical Metrics object.

        Parameters
        ----------
        omni_calfits : str
            Path to a calfits file output from omnical, typically
            ending in *.omni.calfits.
        history : str, optional
            History string to be added to output files. Default is empty string.

        Attributes
        ----------
        filedir : str
            The path to the directory of the input omni_calfits file.
        filestem : str:
            The basename of the omni_calfits file.
        filename : str
            The filename of the input omni_calfits file.
        version_str : str
            The current version of the hera_qm repo.
        history : str
            A history string to be added to output files.
        firstcal_file : str
            Path to the firstcal file associated with the same observation.
            Only used to remove firstcal delays from omnical gain solutions.
        uv : UVCal object
            A UVCal object corresponding to the data saved in the calfits file.
        Nants : int
            The number of antennas in the data file.
        freqs : array
            The frequencies in the calibration file, in Hz.
        Nfreqs : int
            The number of frequency channels in the file.
        jones : array
            The array of Jones parameters in the file, as integers.
        pols : array
            An array of polarization strings corresponding to the Jones numbers.
        Npols : int
            The number of polarizations in the file.
        times : array
            The time values in the file, in JD.
        Ntimes : int
            The number of times in the file.
        ant_array : array
            An array containing the antenna numbers in the data files.
        omni_gains : array
            An array of omnical gain solutions saved in the file. The time
            and frequency axes are transposed relative to the file.
        chisq : array
            An array of the per-antenna chi-squared of the gain solutions saved
            in the file. The time and frequency axes are transposed relative
            to the file.
        chisq_tavg : array
            An array of the chi-squared values with the median taken along
            the time axis.

        """
        # Get file info and other relevant metadata
        self.filedir = os.path.dirname(omni_calfits)
        self.filestem = '.'.join(os.path.basename(omni_calfits).split('.')[:-1])
        self.filename = os.path.basename(omni_calfits)
        self.version_str = __version__
        self.history = history
        self.firstcal_file = None

        # Instantiate Data Object
        self.uv = UVCal()
        self.uv.read_calfits(omni_calfits)

        # Get relevant metadata
        self.Nants = self.uv.Nants_data
        self.freqs = self.uv.freq_array.squeeze()
        self.Nfreqs = len(self.freqs)
        self.jones = self.uv.jones_array
        self.pols = np.array([self.jones2pol[x] for x in self.jones])
        self.Npols = len(self.pols)
        self.times = self.uv.time_array
        self.Ntimes = self.uv.Ntimes
        self.ant_array = self.uv.ant_array

        # Get omnical gains, move time axis in front of freq axis
        self.omni_gains = np.moveaxis(self.uv.gain_array, 2, 3)[:, 0, :, :, :]

        # Assign total chisq array
        self.chisq = np.moveaxis(self.uv.quality_array, 2, 3)[:, 0, :, :, :]
        self.chisq_tavg = np.median(self.chisq, axis=1)

    def run_metrics(self, fcfiles=None, cut_edges=True, Ncut=100,
                    phs_std_cut=0.3, chisq_std_zscore_cut=4.0):
        """Compute the metric values for the data in the object.

        This function runs the metrics functions on each polarization
        (e.g. XX, YY) individually and then packs them into a single metrics
        dictionary.

        Parameters
        ----------
        fcfiles : list
            A list of single-pol firstcal delay solution files matching ordering
            of polarization in .omni.calfits file.
        cut_edges : bool
            If True, removes Ncut channels from top and bottom of band. Default is True.
        Ncut : int
            Number of channels to remove from top and bottom of band (if cut_edges is True).
            Default is 100.
        phs_std_cut : float
            The cut for standard deviation of phase solutions in radians. Default is 0.3.
        chisq_std_zscore : float
            The sigma tolerance (or z-score tolerance) for standard deviation of
            the chi-squared fluctuations. Default is 4.0.

        Returns
        -------
        full_metrics : OrderedDictionary
            An OrderedDictionary containing polarizations as primary keys,
            which have additional OrderedDictionaries as values. These
            secondary dictionaries actually contain the metrics for the
            given polarization.

        """
        # build firstcal_gains if fed fc files
        run_fc_metrics = False
        if fcfiles is not None:
            self.load_firstcal_gains(fcfiles)
            run_fc_metrics = True

        full_metrics = odict()

        # loop over polarization
        for poli, pol in enumerate(self.pols):
            # select freq channels
            if cut_edges is True:
                self.band = np.arange(Ncut, self.Nfreqs - Ncut)
            else:
                self.band = np.arange(self.Nfreqs)

            # get chisq metrics
            (chisq_avg, chisq_tot_avg, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
             chisq_ant_std_scale, chisq_ant_std_zscore, chisq_ant_std_zscore_max,
             chisq_good_sol) = self.chisq_metric(self.chisq[:, :, :, poli], chisq_std_zscore_cut=chisq_std_zscore_cut)

            if run_fc_metrics is True:
                # run phs FT metric
                ant_gain_fft, ant_gain_dly = self.phs_FT_metric(self.gain_diff[:, :, :, poli])

                # run phs std metric
                (ant_phs_std, ant_phs_std_max, ant_phs_std_good_sol, ant_phs_std_per_time, ant_phs_hists,
                 ant_phs_hist_bins) = self.phs_std_metric(self.gain_diff[:, :, :, poli], phs_std_cut=phs_std_cut)

            # initialize metrics
            metrics = odict()

            metrics['chisq_avg'] = chisq_avg
            metrics['chisq_tot_avg'] = chisq_tot_avg
            metrics['chisq_ant_avg'] = chisq_ant_avg
            metrics['chisq_ant_std'] = chisq_ant_std
            metrics['chisq_ant_std_loc'] = chisq_ant_std_loc
            metrics['chisq_ant_std_scale'] = chisq_ant_std_scale
            metrics['chisq_ant_std_zscore'] = chisq_ant_std_zscore
            metrics['chisq_ant_std_zscore_max'] = chisq_ant_std_zscore_max
            metrics['chisq_std_zscore_cut'] = chisq_std_zscore_cut
            metrics['chisq_good_sol'] = chisq_good_sol

            metrics['freqs'] = self.freqs
            metrics['Nfreqs'] = self.Nfreqs
            metrics['cut_edges'] = cut_edges
            metrics['Ncut'] = Ncut
            metrics['band'] = self.band
            metrics['ant_array'] = self.ant_array
            metrics['jones'] = self.jones[poli]
            metrics['pol'] = self.pols[poli]
            metrics['ant_pol'] = self.pols[poli][0]
            metrics['times'] = self.times
            metrics['Ntimes'] = self.Ntimes
            metrics['Nants'] = self.Nants

            metrics['version'] = self.version_str
            metrics['history'] = self.history
            metrics['filename'] = self.filename
            metrics['filestem'] = self.filestem
            metrics['filedir'] = self.filedir

            metrics['ant_gain_fft'] = None
            metrics['ant_gain_dly'] = None
            metrics['ant_phs_std'] = None
            metrics['ant_phs_std_max'] = None
            metrics['ant_phs_std_good_sol'] = None
            metrics['phs_std_cut'] = None
            metrics['ant_phs_std_per_time'] = None
            metrics['ant_phs_hists'] = None
            metrics['ant_phs_hist_bins'] = None

            if run_fc_metrics is True:
                metrics['ant_gain_fft'] = ant_gain_fft
                metrics['ant_gain_dly'] = ant_gain_dly
                metrics['ant_phs_std'] = ant_phs_std
                metrics['ant_phs_std_max'] = ant_phs_std_max
                metrics['ant_phs_std_good_sol'] = ant_phs_std_good_sol
                metrics['phs_std_cut'] = phs_std_cut
                metrics['ant_phs_std_per_time'] = ant_phs_std_per_time
                metrics['ant_phs_hists'] = ant_phs_hists
                metrics['ant_phs_hist_bins'] = ant_phs_hist_bins

            full_metrics[pol] = metrics

        return full_metrics

    def load_firstcal_gains(self, fc_files):
        """Load firstcal delay solutions as gains.

        This is a wrapper for omnical_metrics.load_firstcal_gains.
        It attaches firstcal_delays and firstcal_gains to the object
        and calculates gain_diff.

        Parameters
        ----------
        fc_files : list
            A list of strings representing paths to firstcal files.

        Returns
        -------
        None

        Raises
        ------
        ValueError:
            A ValueError is raised if the firstcal file contains a polarization
            not present in the omnical solutions.

        """
        # check if fcfiles is a str, if so change to list
        if isinstance(fc_files, str):
            fc_files = [fc_files]

        firstcal_delays = []
        firstcal_gains = []
        pol_sort = []
        for fcfile in fc_files:
            fc_delays, fc_gains, fc_pol = load_firstcal_gains(fcfile)
            # convert to 'xy' pol convention and then select omni_pol from fc_pols
            fc_pol = self.jones2pol[fc_pol]
            if fc_pol not in self.pols:
                raise ValueError("firstcal_pol={} not in list of omnical pols={}".format(fc_pol, self.pols))
            pol_sort.append(list(self.pols).index(fc_pol))
            firstcal_delays.append(fc_delays)
            firstcal_gains.append(fc_gains)
        self.firstcal_delays = np.moveaxis(np.array(firstcal_delays), 0, 2)[:, :, pol_sort]
        self.firstcal_gains = np.moveaxis(np.array(firstcal_gains), 0, 3)[:, :, :, pol_sort]
        self.gain_diff = self.omni_gains / self.firstcal_gains

    def chisq_metric(self, chisq, chisq_std_zscore_cut=4.0, return_dict=True):
        """Compute chi-squared metrics.

        Parameters
        ----------
        chisq : ndarray, shape=(Nants, Ntimes, Nfreqs)
            An ndarray containing chi-squared value for each antenna (single pol).
        chisq_std_zscore_cut : float, optional
            The sigma tolerance (or z-score tolerance) for standard deviation of
            the chi-squared fluctuations. Default is 4.0.
        return_dict : bool, optional
            If True, return per-antenna metrics as a dictionary with antenna number as key
            rather than an ndarray. Default is True.

        Returns
        -------
        ret : tuple
            A length-9 tuple containing the various chi-squared metrics.

        """
        # Get chisq statistics
        chisq_avg = np.median(np.median(chisq[:, :, self.band], axis=0), axis=1).astype(np.float64)
        chisq_tot_avg = astats.biweight_location(chisq[:, :, self.band]).astype(np.float64)
        chisq_ant_avg = np.array(list(map(astats.biweight_location, chisq[:, :, self.band]))).astype(np.float64)
        chisq_ant_std = np.sqrt(np.array(list(map(astats.biweight_midvariance, chisq[:, :, self.band]))))
        chisq_ant_std_loc = astats.biweight_location(chisq_ant_std).astype(np.float64)
        chisq_ant_std_scale = np.sqrt(astats.biweight_midvariance(chisq_ant_std))
        chisq_ant_std_zscore = (chisq_ant_std - chisq_ant_std_loc) / chisq_ant_std_scale
        chisq_ant_std_zscore_max = np.max(np.abs(chisq_ant_std_zscore))
        chisq_good_sol = chisq_ant_std_zscore_max < chisq_std_zscore_cut

        # convert to dictionaries
        if return_dict is True:
            chisq_ant_std = odict(zip(self.ant_array, chisq_ant_std))
            chisq_ant_avg = odict(zip(self.ant_array, chisq_ant_avg))

        return (chisq_avg, chisq_tot_avg, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
                chisq_ant_std_scale, chisq_ant_std_zscore, chisq_ant_std_zscore_max, chisq_good_sol)

    def phs_FT_metric(self, gain_diff, return_dict=True):
        """Compute the Fourier-transform-of-phase metric.

        Takes the square of the real-valued FT of the phase
        difference between omnical and firstcal solutions and uses it
        to assess noise level across freq.

        Parameters
        ----------
        gain_diff : ndarray, dtype=complex, shape=(Nants, Ntimes, Nfreqs)
            A complex ndarray containing omnical gain phases divided by
            firstcal gain phases for a single pol.
        return_dict : bool, optional
            If True, return antenna output as dictionary with antenna name as key.
            Default is True

        Returns
        -------
        ant_gain_fft : ndarray
            The FFT of complex gain array with the median computed over time.
        ant_gain_delay : ndarray
            The delay values corresponding to the FFT frequencies.

        """
        # take fft of complex gains over frequency per-antenna
        ant_gain_fft = np.fft.fftshift(np.fft.fft(gain_diff[:, :, self.band], axis=2))
        Nfreq = len(self.band)
        ant_gain_dly = (np.arange(-Nfreq // 2, Nfreq // 2)
                        / (self.freqs[self.band][-1] - self.freqs[self.band][0]))

        # median over time
        ant_gain_fft = np.median(ant_gain_fft, axis=1)

        if return_dict is True:
            ant_gain_fft = odict(zip(self.ant_array, ant_gain_fft))

        return ant_gain_fft, ant_gain_dly

    def phs_std_metric(self, gain_diff, phs_std_cut=0.3, return_dict=True):
        """Compute the standard deviation of gain phase.

        This metric takes the variance of the phase difference between omnical
        and firstcal solutions (in radians) to assess its frequency variability.

        Parameters
        ----------
        gain_diff : ndarray, dtype=complex float, shape=(Nants, Ntimes, Nfreqs)
            A complex ndarray containing omnical gains divided by firstcal gains.
        phs_std_cut : float, optional
            The cut for standard deviation of phase solutions in radians. Default
            is 0.3.
        return_dict : bool, optional
            If True, return per-antenna output as dictionary with antenna name as key.
            Default is True.

        Returns
        -------
        ant_phs_std : ndarray, shape=(Nants,)
            The phase stand dev for each antenna.
        ant_phs_std_max : float
            The maximum of ant_phs_std.
        phs_std_band_ants : ndarray
            An array containing antennas whose phs_std didn't meet phs_std_cut.
        phs_std_good_sol : bool
            A boolean with metric's good / bad determination of entire solution.
        """
        # get real-valued phase
        phs_diff = np.angle(gain_diff)

        # subtract antenna mean
        phs_diff_mean = np.median(phs_diff, axis=2)
        phs_diff -= phs_diff_mean[:, :, np.newaxis]

        # take standard deviation across time & freq
        ant_phs_std = np.sqrt(np.array(list(map(astats.biweight_midvariance,
                                                phs_diff[:, :, self.band]))))
        ant_phs_std_max = np.max(ant_phs_std)
        ant_phs_std_good_sol = ant_phs_std_max < phs_std_cut

        # get std across freq per antenna per time integration
        ant_phs_std_per_time = np.sqrt(astats.biweight_midvariance(phs_diff[:, :, self.band], axis=2))

        # make histogram
        ant_phs_hists = []
        for phs in phs_diff:
            hist, bins = np.histogram(phs, bins=128, range=(-np.pi, np.pi), density=True)
            bins = 0.5 * (bins[1:] + bins[:-1])
            ant_phs_hists.append(hist)
        ant_phs_hists = np.array(ant_phs_hists)

        ant_phs_hist_bins = bins

        if return_dict is True:
            ant_phs_std = odict(zip(self.ant_array, ant_phs_std))
            ant_phs_std_per_time = odict(zip(self.ant_array, ant_phs_std_per_time))
            ant_phs_hists = odict(zip(self.ant_array, ant_phs_hists))

        return ant_phs_std, ant_phs_std_max, ant_phs_std_good_sol, ant_phs_std_per_time, ant_phs_hists, ant_phs_hist_bins

    def plot_gains(self, ants=None, time_index=0, pol_index=0, divide_fc=False,
                   plot_type='phs', ax=None, save=False, fname=None, outpath=None,
                   plot_kwargs={'marker': 'o', 'markersize': 2, 'alpha': 0.75, 'linestyle': ''}):
        """Plot omnical gain solutions for each antenna.

        Parameters
        ----------
        ants : list, optional
            A list of ant numbers to plot. If not specified all antennas
            are plotted.
        time_index : int, optional
            The index of time axis to plot. Default is 0.
        pol_index : int, optional
            The index of cross polarization axis to plot. Default is 0.
        divide_fc : bool, optional
            If True, divide out firstcal gains from omnical gains.
            Note that this only works if self.firstcal_file is not None.
            Default is False.
        plot_type : {"phs", "amp"}, optional
            The type of plot to make. "phs" is phase, "amp" is amplitude of the
            complex gain solutions. Default is "phs"
        ax : matplotlib axis object, optional
            The axis on which to generate plots. If None, axes will be generated
            as necessary. Default is None.
        save : bool, optional
            If True, save image as a png. Default is False.
        fname : str, optional
            Filename to save image as. If not specified, a filename is generated
            based on the input filename saved in the metrics dictionary.
        outpath : str, optional
            Path to place file in. If not specified, will default to location
            of the input *omni.calfits file.
        plot_kwargs : dict, optional
            A dictionary of keyword arguments to be passed to the plotting function.
            Default is {'marker': 'o', 'markersize': 2, 'alpha': 0.75, 'linestyle': ''}.

        Returns
        -------
        fig : matplotlib figure object
            If ax is None, the figure created to generate the plot is returned.

        """
        import matplotlib.pyplot as plt

        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

        if ants is None:
            ants = np.arange(self.Nants)
        else:
            ants = np.array(list(map(lambda x: np.where(self.ant_array == x)[0][0], ants)))

        if plot_type == 'phs':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, pol_index].T
            if divide_fc is True:
                gains = self.gain_diff[ants, time_index, :, pol_index].T
            plist = np.array(ax.plot(self.freqs / 1e6, np.angle(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain phase [radians]', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(plist, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} pol".format(self.filename, self.times[time_index], self.pols[pol_index]))

        elif plot_type == 'amp':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, pol_index].T.copy()
            if divide_fc is True:
                gains /= self.firstcal_gains[ants, time_index, :, pol_index].T
            plist = np.array(ax.plot(self.freqs / 1e6, np.abs(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain amplitude', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(plist, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} pol".format(self.filename, self.times[time_index], self.pols[pol_index]))

        if save is True:
            if fname is None:
                fname = utils.strip_extension(self.filename) + '.gain_{0}.png'.format(plot_type)
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
            if custom_ax is False:
                fig.savefig(fname, bbox_inches='tight')
            else:
                ax.figure.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig

    def plot_chisq_tavg(self, pol_index=0, ants=None, ax=None, save=False, fname=None, outpath=None):
        """Plot Omnical chi-squared averaged over time.

        Parameters
        ----------
        pol_index : int, optional
            The polarization index to plot. Default is 0.
        ants : list, optional
            A list of ant numbers to plot. If not specified all antennas
            are plotted.
        ax : matplotlib axis object, optional
            The axis on which to generate plots. If None, axes will be generated
            as necessary. Default is None.
        save : bool, optional
            If True, save image as a png. Default is False.
        fname : str, optional
            Filename to save image as. If not specified, a filename is generated
            based on the input filename saved in the metrics dictionary.
        outpath : str, optional
            Path to place file in. If not specified, will default to location
            of the input *omni.calfits file.
        plot_kwargs : dict, optional
            A dictionary of keyword arguments to be passed to the plotting function.

        Returns
        -------
        fig : matplotlib figure object
            If ax is None, the figure created to generate the plot is returned.

        """
        import matplotlib.pyplot as plt

        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

        # get antennas
        if ants is None:
            ants = np.arange(self.Nants)
        else:
            ants = np.array(list(map(lambda x: np.where(self.ant_array == x)[0][0], ants)))

        # make grid and plots
        ax.grid(True)
        ydata = self.chisq_tavg[ants, :, pol_index].T
        ylim = np.percentile(ydata.ravel(), 75) * 3
        plist = ax.plot(self.freqs / 1e6, ydata)
        ax.set_xlabel('frequency [MHz]', fontsize=14)
        ax.set_ylabel('chi-square avg over time', fontsize=14)
        ax.set_ylim(0, ylim * 2)
        ax.set_title("{0} : {1} pol".format(self.filename, self.pols[pol_index]))

        ax = ax.figure.add_axes([0.99, 0.1, 0.02, 0.8])
        ax.axis('off')
        ax.legend(plist, self.ant_array)

        if save is True:
            if fname is None:
                fname = utils.strip_extension(self.filename) + '.chisq_tavg.png'
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
        if (custom_ax is False) and save:
            fig.savefig(fname, bbox_inches='tight')
        elif save:
            ax.figure.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig

    def plot_metrics(self, metrics):
        """Plot all metrics.

        metrics : dict
            A nested polarization dictionary from within a
            Omnical_Metrics.run_metrics() dictionary output.
            For example:
                full_metrics = Omnical_Metrics.run_metrics()
                plot_metrics(full_metrics['XX'])

        """
        # plot chisq metric
        plot_chisq_metric(metrics, save=True)
        # plot phs metrics
        plot_phs_metric(metrics, plot_type='std', save=True)
        plot_phs_metric(metrics, plot_type='hist', save=True)
        plot_phs_metric(metrics, plot_type='ft', save=True)


def omnical_metrics_run(files, args, history):
    """Run OmniCal Metrics on a set of input files.

    This function will produce a JSON file containing the series of metrics.

    Parameters
    ----------
    files : list of str
        A list of *omni.calfits file to run metrics on.
    args : argparse.Namespace
        Parsed command-line arguments generated via argparse.ArgumentParser.parse_args.
    history : str
        History string to append to omnical metrics object.

    Returns
    -------
    None

    Raises
    ------
    AssertionError:
        If the length of files to process is 0, an AssertionError is raised.

    """
    if len(files) == 0:
        raise AssertionError('Please provide a list of calfits files')

    for i, filename in enumerate(files):
        om = OmniCal_Metrics(filename, history=history)
        if args.fc_files is not None:
            fc_files = list(map(lambda x: x.split(','), args.fc_files.split('|')))
            full_metrics = om.run_metrics(fcfiles=fc_files[i],
                                          cut_edges=args.no_bandcut is False,
                                          phs_std_cut=args.phs_std_cut,
                                          chisq_std_zscore_cut=args.chisq_std_zscore_cut)
        else:
            full_metrics = om.run_metrics(cut_edges=args.no_bandcut is False,
                                          phs_std_cut=args.phs_std_cut,
                                          chisq_std_zscore_cut=args.chisq_std_zscore_cut)

        # iterate over pols
        for p, pol in enumerate(full_metrics.keys()):
            if args.make_plots is True:
                om.plot_metrics(full_metrics[pol])

        abspath = os.path.abspath(filename)
        dirname = os.path.dirname(abspath)
        if args.metrics_path == '':
            # default path is same directory as file
            metrics_path = dirname
        else:
            metrics_path = args.metrics_path
            print(metrics_path)
        metrics_basename = utils.strip_extension(os.path.basename(filename)) + args.extension
        metrics_filename = os.path.join(metrics_path, metrics_basename)
        write_metrics(full_metrics, filename=metrics_filename)
