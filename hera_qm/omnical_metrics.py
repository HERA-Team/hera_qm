import matplotlib.pyplot as plt
import numpy as np
from pyuvdata import UVCal
import pkg_resources
pkg_resources.require('astropy>=2.0')
import astropy.stats as astats
from collections import OrderedDict
from hera_qm.version import hera_qm_version_str
import json
import cPickle as pkl
import copy
import os
from scipy.signal import medfilt


def get_omnical_metrics_dict():
    """ Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns:
    metrics_dict : dictionary
        metric names as keys and descriptions as values.
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
    """
    load omnical metrics file

    Input:
    ------
    filename : str
        path to omnical metrics file

    Output:
    metrics : dictionary
        dictionary containing omnical metrics
    """
    # get filetype
    filetype = filename.split('.')[-1]

    # load json
    if filetype == 'json':
        with open(filename, 'r') as f:
            metrics = json.load(f, object_pairs_hook=OrderedDict)

        # ensure keys of ant_dicts are not strings
        # loop over pols
        for h, p in enumerate(metrics.keys()):
            # loop over items in each pol metric dict
            for k in metrics[p].keys():
                if type(metrics[p][k]) is OrderedDict:
                    for i in metrics[p][k].keys():
                        if type(metrics[p][k][i]) is list:
                            metrics[p][k][i] = np.array(metrics[p][k].pop(i))
                        metrics[p][k][int(i)] = metrics[p][k].pop(i)
                        # check for complex array
                        try:
                            if metrics[p][k][int(i)].dtype.type == np.unicode_:
                                metrics[p][k][int(i)] = metrics[p][k][int(i)].astype(np.complex128)
                        except:
                            pass

                elif type(metrics[p][k]) is list:
                    metrics[p][k] = np.array(metrics[p][k])

    # load pickle
    elif filetype == 'pkl':
        with open(filename, 'rb') as f:
            inp = pkl.Unpickler(f)
            metrics = inp.load()
    else:
        raise IOError("Filetype not recognized, try a json or pkl file")

    return metrics

def write_metrics(metrics, filename=None, filetype='json'):
    """
    Write metrics to file after running self.run_metrics()

    Input:
    ------
    metrics : dictionary
        Omnical_Metrics.run_metrics() output

    filename : str, default=None
        filename to write out, will use filename by default

    filetype : str, default='json', option=['json', 'pkl']
        specify file format of output metrics file
    """
    # get pols
    pols = metrics.keys()

    if filename is None:
        filename = os.path.join(metrics[pols[0]]['filedir'], metrics[pols[0]]['filestem'] + '.omni_metrics')

    # write to file
    if filetype == 'json':
        if filename.split('.')[-1] != 'json':
            filename += '.json'

        # change ndarrays to lists
        metrics_out = copy.deepcopy(metrics)
        # loop over pols
        for h, pol in enumerate(metrics_out.keys()):
            # loop over keys
            for i, k in enumerate(metrics_out[pol].keys()):
                if type(metrics_out[pol][k]) is np.ndarray:
                    metrics_out[pol][k] = metrics_out[pol][k].tolist()
                elif type(metrics_out[pol][k]) is OrderedDict:
                    # loop over keys
                    for j in metrics_out[pol][k].keys():
                        if type(metrics_out[pol][k][j]) is np.ndarray:
                            # check for complex
                            if metrics_out[pol][k][j].dtype.type == np.complex128:
                                metrics_out[pol][k][j] = metrics_out[pol][k][j].astype(np.str)
                            metrics_out[pol][k][j] = metrics_out[pol][k][j].tolist()
                elif type(metrics_out[pol][k]) is np.bool_:
                    metrics_out[pol][k] = bool(metrics_out[pol][k])

        with open(filename, 'w') as f:
            json.dump(metrics_out, f, indent=4)

    elif filetype == 'pkl':
        if filename.split('.')[-1] != 'pkl':
            filename += '.pkl'
        with open(filename, 'wb') as f:
            outp = pkl.Pickler(f)
            outp.dump(metrics)


def load_firstcal_gains(fc_file):
    """
    load firstcal delays and turn into phase gains

    fc_file : str
        path to firstcal .calfits file (single polarization)

    jones2pol : dict
        dictionary containing jones integers as keys and 
        X-Y pols as values

    """
    uvf = UVCal()
    uvf.read_calfits(fc_file)
    freqs = uvf.freq_array.squeeze()
    fc_gains = np.moveaxis(uvf.gain_array, 2, 3)[:, 0, :, :, 0]
    d_nu = np.mean(freqs[1:]-freqs[:-1])
    d_phi = np.abs(np.mean(np.angle(fc_gains)[:, :, 1:] - np.angle(fc_gains)[:, :, :-1], axis=2))
    fc_delays = (d_phi / d_nu)/(2*np.pi)
    fc_pol = uvf.jones_array[0]
    return fc_delays, fc_gains, fc_pol


def plot_phs_metric(metrics, plot_type='std', ax=None, save=False,
                    fname=None, outpath=None,
                    plot_kwargs={'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}):
    """
    Plot omnical phase metric

    Input:
    ------
    metrics : dictionary
        a metrics dictionary from OmniCal_Metrics.run_metrics()

    plot_type : str, default='std', options=['std', 'ft', 'hist']
        specify plot type

        'std' plots stand deviations of omni-firstcal phase difference.
        large stand deviations are not ideal

        'hist' plots the histogram of gain phase solutions

    ax : matplotlib axis object

    save : bool, default=False
        if True, save image as a png

    fname : str, default=None
        filename to save image as

    outpath : str, default=None
        path to place file in
        will default to location of *omni.calfits file
    """
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    if plot_type == 'std':
        # get y data
        ant_phs_std = np.array(metrics['ant_phs_std'].values())
        phs_std_cut = metrics['phs_std_cut']
        ant_phs_std_max = metrics['ant_phs_std_max']
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
        [t.set_rotation(20) for t in ax.get_xticklabels()]
        ax.set_ylim(0, ymax)
        ax.set_title("{0}".format(metrics['filename']))

    if plot_type == 'hist':
        ylines = np.array(metrics['ant_phs_hists'].values())
        ax.grid(True)
        p = [ax.plot(metrics['ant_phs_hist_bins'], ylines[i], alpha=0.75) for i in range(len(ylines))]
        ax.set_xlabel('gain phase [radians]', fontsize=14)
        ax1 = fig.add_axes([0.99, 0.1, 0.03, 0.8])
        ax1.axis('off')
        ax1.legend(np.concatenate(p), metrics['ant_array'])
        ax.set_title("gain phase histogram for {0}".format(metrics['filename']))

    if plot_type == 'ft':
        ylines = np.abs(np.array(metrics['ant_gain_fft'].values()))
        ax.grid(True)
        p = [ax.plot(metrics['ant_gain_dly']*1e9, ylines[i], alpha=0.75) for i in range(len(ylines))]
        ax.set_xlabel('delay [ns]', fontsize=14)
        ax.set_ylabel(r'radians sec$^{-1}$', fontsize=14)
        ax.set_yscale('log')
        ax1 = fig.add_axes([0.99, 0.1, 0.03, 0.8])
        ax1.axis('off')
        ax1.legend(np.concatenate(p), metrics['ant_array'])
        ax.set_title("abs val of FT(complex gains) for {0}".format(metrics['filename']))

    if save is True:
        if fname is None:
            fname = metrics['filename'] + '.phs_{0}.png'.format(plot_type)
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        if custom_ax == False:
            fig.savefig(fname, bbox_inches='tight')
        else:
            ax.figure.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig

def plot_chisq_metric(metrics, ax=None, save=False, fname=None, outpath=None,
                      plot_kwargs={'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}):
    """
    Plot Chi-Square output from omnical for each antenna

    Input:
    ------
    metrics : dictionary
        metrics dictionary from OmniCal_Metrics.run_metrics()

    ax : matplotlib axis object, default=False

    save : bool, default=False
        if True, save image as a png

    fname : str, default=None
        filename to save image as

    outpath : str, default=None
        path to place file in
        will default to location of *omni.calfits file
    """
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
    chisq_ant_std = np.array(metrics['chisq_ant_std'].values())
    ymax = np.max([chisq_ant_std.max() * 1.2, ycut + ysig])

    # make grid and plots
    ax.grid(True)
    p1 = ax.axhspan(yloc - ysig, yloc + ysig,
                    color='green', alpha=0.1)
    p2 = ax.axhline(yloc, color='steelblue')
    p3 = ax.axhline(ycut, color='darkred')
    p4 = ax.axhspan(ycut, ymax, color='darkred', alpha=0.2)
    p5 = ax.plot(chisq_ant_std, **plot_kwargs)
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
            fname = metrics['filename'] + '.chisq_std.png'
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        if custom_ax == False:
            fig.savefig(fname, bbox_inches='tight')
        else:
            ax.figure.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig

class OmniCal_Metrics(object):
    """
    OmniCal_Metrics class for holding omnical data
    and running metrics on them.
    """

    jones2pol = {-5: 'XX', -6: 'YY', -7: 'XY', -8: 'YX'}

    def __init__(self, omni_calfits, history=''):
        """
        Omnical Metrics initialization

        Input:
        ------
        omni_calfits : str
            calfits file output from omnical, typically
            ending in *.omni.calfits

        history : str
            history string
        """
        # Get file info and other relevant metadata
        self.filedir = os.path.dirname(omni_calfits)
        self.filestem = '.'.join(os.path.basename(omni_calfits).split('.')[:-1])
        self.filename = os.path.basename(omni_calfits)
        self.version_str = hera_qm_version_str
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
        self.pols = np.array(map(lambda x: self.jones2pol[x], self.jones))
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
        """
        run metrics function on each polarization (e.g. XX, YY) individually
        and then pack into a single metrics dictionary

        Input:
        ------
        fcfiles : list, default=None
            list of single-pol firstcal delay solution files matching ordering of
            pol in .omni.calfits file

        see self.compile_metrics() for details on remaining keyword arguments
        """
        # build firstcal_gains if fed fc files
        run_fc_metrics = False
        if fcfiles is not None:
            self.load_firstcal_gains(fcfiles)
            run_fc_metrics = True

        full_metrics = OrderedDict()

        # loop over polarization
        for i, pol in enumerate(self.pols):
            # select freq channels
            if cut_edges is True:
                self.band = np.arange(Ncut, self.Nfreqs - Ncut)
            else:
                self.band = np.arange(self.Nfreqs)

            # get chisq metrics
            (chisq_avg, chisq_tot_avg, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
             chisq_ant_std_scale, chisq_ant_std_zscore, chisq_ant_std_zscore_max,
             chisq_good_sol) = self.chisq_metric(self.chisq[:, :, :, i], chisq_std_zscore_cut=chisq_std_zscore_cut)

            if run_fc_metrics is True:
                # run phs FT metric
                ant_gain_fft, ant_gain_dly = self.phs_FT_metric(self.gain_diff[:, :, :, i])

                # run phs std metric
                (ant_phs_std, ant_phs_std_max, ant_phs_std_good_sol, ant_phs_std_per_time, ant_phs_hists,
                 ant_phs_hist_bins) = self.phs_std_metric(self.gain_diff[:, :, :, i], phs_std_cut=phs_std_cut)

            # initialize metrics
            metrics = OrderedDict()

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
            metrics['jones'] = self.jones[i]
            metrics['pol'] = self.pols[i]
            metrics['ant_pol'] = self.pols[i][0]
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
        """
        wrapper for omnical_metrics.load_firstcal_gains.
        attach firstcal_delays and firstcal_gains to class
        and calculate gain_diff
        fc_files : list, dtype=str
            list of paths to firstcal files
        """
        # check if fcfiles is a str, if so change to list
        if type(fc_files) is str:
            fc_files = [fc_files]

        firstcal_delays = []
        firstcal_gains = []
        pol_sort = []
        for i, fcfile in enumerate(fc_files):
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
        """
        chi square metrics

        chisq : ndarray, shape=(Nants, Ntimes, Nfreqs)
            ndarray containing chisq for each antenna (single pol)

        chisq_std_cut : float, default=5.0
            sigma tolerance (or z-score tolerance) for std of chisq fluctuations

        return_dict : bool, default=True
            return per-antenna metrics as a dictionary, with antenna number as key
            rather than an ndarray
        """
        # Get chisq statistics
        chisq_avg = np.median(np.median(chisq[:, :, self.band], axis=0), axis=1).astype(np.float64)
        chisq_tot_avg = astats.biweight_location(chisq[:, :, self.band]).astype(np.float64)
        chisq_ant_avg = np.array(map(astats.biweight_location, chisq[:, :, self.band])).astype(np.float64)
        chisq_ant_std = np.sqrt(np.array(map(astats.biweight_midvariance, chisq[:, :, self.band])))
        chisq_ant_std_loc = astats.biweight_location(chisq_ant_std).astype(np.float64)
        chisq_ant_std_scale = np.sqrt(astats.biweight_midvariance(chisq_ant_std))
        chisq_ant_std_zscore = (chisq_ant_std - chisq_ant_std_loc) / chisq_ant_std_scale
        chisq_ant_std_zscore_max = np.max(np.abs(chisq_ant_std_zscore))
        chisq_good_sol = chisq_ant_std_zscore_max < chisq_std_zscore_cut

        # convert to dictionaries
        if return_dict is True:
            chisq_ant_std = OrderedDict(zip(self.ant_array, chisq_ant_std))
            chisq_ant_avg = OrderedDict(zip(self.ant_array, chisq_ant_avg))

        return (chisq_avg, chisq_tot_avg, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
                chisq_ant_std_scale, chisq_ant_std_zscore, chisq_ant_std_zscore_max, chisq_good_sol)

    def phs_FT_metric(self, gain_diff, return_dict=True):
        """
        Takes the square of the real-valued FT of the phase
        difference between omnical and firstcal solutions and uses it
        to assess noise level across freq

        Input:
        ------
        gain_diff : ndarray, dtype=complex, shape=(Nants, Ntimes, Nfreqs)
            complex ndarray containing omnical gain phases divided by firstcal gain phases (single pol)

        return_dict : bool
            return antenna output as dictionary with antenna name as key

        Output:
        -------
        ant_gain_fft : ndarray
            fft of complex gain array averaged over time

        """
        # take fft of complex gains over frequency per-antenna
        ant_gain_fft = np.fft.fftshift(np.fft.fft(gain_diff[:, :, self.band], axis=2))
        Nfreq = len(self.band)
        ant_gain_dly = np.arange(-Nfreq//2, Nfreq//2)/(self.freqs[self.band][-1]-self.freqs[self.band][0])

        # median over time
        ant_gain_fft = np.median(ant_gain_fft, axis=1)

        if return_dict is True:
            ant_gain_fft = OrderedDict(zip(self.ant_array, ant_gain_fft))

        return ant_gain_fft, ant_gain_dly

    def phs_std_metric(self, gain_diff, phs_std_cut=0.3, return_dict=True):
        """
        Takes the variance of the phase difference between omnical
        and firstcal solutions (in radians) to assess its frequency variability

        Input:
        ------
        gain_diff : ndarray, dtype=complex float, shape=(Nants, Ntimes, Nfreqs)
            complex ndarray containing omnical gains divided by firstcal gains

        phs_std_cut : float, default=0.5
            cut for standard deviation of phase solutions in radians

        return_dict : bool
            return per-antenna output as dictionary with antenna name as key

        Output:
        -------
        ant_phs_std : ndarray, shape=(Nants,)
            phase stand dev for each antenna

        ant_phs_std_max : float
            maximum of ant_phs_std

        phs_std_band_ants : ndarray
            array containing antennas whose phs_std didn't meet phs_std_cut

        phs_std_good_sol : bool
            boolean with metric's good / bad determination of entire solution
        """
        # get real-valued phase
        phs_diff = np.angle(gain_diff)

        # subtract antenna mean
        phs_diff_mean = np.median(phs_diff, axis=2)
        phs_diff -= phs_diff_mean[:, :, np.newaxis]

        # take standard deviation across time & freq
        ant_phs_std = np.sqrt(np.array(map(astats.biweight_midvariance, phs_diff[:, :, self.band])))
        ant_phs_std_max = np.max(ant_phs_std)
        ant_phs_std_good_sol = ant_phs_std_max < phs_std_cut

        # get std across freq per antenna per time integration
        ant_phs_std_per_time = np.sqrt(astats.biweight_midvariance(phs_diff[:, :, self.band], axis=2))

        # make histogram
        ant_phs_hists = []
        for i, phs in enumerate(phs_diff):
            h, bins = np.histogram(phs, bins=128, range=(-np.pi, np.pi), normed=True)
            bins = 0.5*(bins[1:]+bins[:-1])
            ant_phs_hists.append(h)
        ant_phs_hists = np.array(ant_phs_hists)

        ant_phs_hist_bins = bins

        if return_dict is True:
            ant_phs_std = OrderedDict(zip(self.ant_array, ant_phs_std))
            ant_phs_std_per_time = OrderedDict(zip(self.ant_array, ant_phs_std_per_time))
            ant_phs_hists = OrderedDict(zip(self.ant_array, ant_phs_hists))

        return ant_phs_std, ant_phs_std_max, ant_phs_std_good_sol, ant_phs_std_per_time, ant_phs_hists, ant_phs_hist_bins

    def plot_gains(self, ants=None, time_index=0, pol_index=0, divide_fc=False,
                   plot_type='phs', ax=None, save=False, fname=None, outpath=None,
                   plot_kwargs={'marker': 'o', 'markersize': 2, 'alpha': 0.75, 'linestyle': ''}):
        """
        Plot omnical gain solutions for each antenna

        Input:
        ------
        ants : list
            list of ant numbers to plot

        time_index : int, default=0
            index of time axis

        pol_index : int, default=0
            index of cross polarization axis

        plot_type : str, default='phs', options=['phs', 'amp']

        divide_fc : bool, defaul=False
            divide out firstcal gains from omnical gains.
            only works if self.firstcal_file is not None

        ax : matplotlib axis object

        save : bool, default=False
            if True, save image as a png

        fname : str, default=None
            filename to save image as

        outpath : str, default=None
            path to place file in
            will default to location of *omni.calfits file
        """
        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

        if ants is None:
            ants = np.arange(self.Nants)
        else:
            ants = np.array(map(lambda x: np.where(self.ant_array == x)[0][0], ants))

        if plot_type == 'phs':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, pol_index].T
            if divide_fc is True:
                gains = self.gain_diff[ants, time_index, :, pol_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.angle(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain phase [radians]', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} pol".format(self.filename, self.times[time_index], self.pols[pol_index]))

        elif plot_type == 'amp':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, pol_index].T.copy()
            if divide_fc is True:
                gains /= self.firstcal_gains[ants, time_index, :, pol_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.abs(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain amplitude', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} pol".format(self.filename, self.times[time_index], self.pols[pol_index]))

        if save is True:
            if fname is None:
                fname = self.filename + '.gain_{0}.png'.format(plot_type)
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
            if custom_ax == False:
                fig.savefig(fname, bbox_inches='tight')
            else:
                ax.figure.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig


    def plot_chisq_tavg(self, pol_index=0, ants=None, ax=None, save=False, fname=None, outpath=None):
        """
        Plot Omnical chi-square averaged over time

        Input:
        ------
        pol_index : int, default=0
            polarization index across polarization axis

        ants : list
            list of ant numbers to plot

        pol_index : int, default=0
            index of cross polarization axis

        ax : matplotlib axis object

        save : bool, default=False
            if True, save image as a png

        fname : str, default=None
            filename to save image as

        outpath : str, default=None
            path to place file in
            will default to location of *omni.calfits file
        """
        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)

        # get antennas
        if ants is None:
            ants = np.arange(self.Nants)
        else:
            ants = np.array(map(lambda x: np.where(self.ant_array == x)[0][0], ants))

        # make grid and plots
        ax.grid(True)
        ydata = self.chisq_tavg[ants, :, pol_index].T
        ylim = np.percentile(ydata.ravel(), 75)*3
        p = ax.plot(self.freqs / 1e6, ydata)
        ax.set_xlabel('frequency [MHz]', fontsize=14)
        ax.set_ylabel('chi-square avg over time', fontsize=14)
        ax.set_ylim(0, ylim*2)
        ax.set_title("{0} : {1} pol".format(self.filename, self.pols[pol_index]))

        ax = ax.figure.add_axes([0.99, 0.1, 0.02, 0.8])
        ax.axis('off')
        ax.legend(p, self.ant_array)

        if save is True:
            if fname is None:
                fname = self.filename + '.chisq_tavg.png'
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
        if custom_ax == False:
            fig.savefig(fname, bbox_inches='tight')
        else:
            ax.figure.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig

    def plot_metrics(self, metrics):
        """
        plot all metrics

        metrics : dictionary
            a nested polarization dictionary from within a
            Omnical_Metrics.run_metrics() dictionary output
            ex:
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
    """
    Run OmniCal Metrics on a set of input files. It will produce
    a JSON file containing result of metrics

    Input:
    ------
    files : a list of *omni.calfits file to run metrics on

    args : parsed arguments via argparse.ArgumentParser.parse_args

    Output:
    -------
    None
    """
    if len(files) == 0:
        raise AssertionError('Please provide a list of calfits files')

    for i, filename in enumerate(files):
        om = OmniCal_Metrics(filename, history=history)
        if args.fc_files is not None:
            fc_files = map(lambda x: x.split(','), args.fc_files.split('|'))
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
        basename = os.path.basename(abspath)
        if args.metrics_path == '':
            # default path is same directory as file
            metrics_path = dirname
        else:
            metrics_path = args.metrics_path
            print(metrics_path)
        metrics_basename = os.path.basename(filename) + args.extension
        metrics_filename = os.path.join(metrics_path, metrics_basename)
        write_metrics(full_metrics, filename=metrics_filename)
