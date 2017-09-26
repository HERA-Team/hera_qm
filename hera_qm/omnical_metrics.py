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
    metrics_dict = {'avg_chisq': 'average of chi-square for all antennas',
                    'tot_phs_noise': 'average of phase noise power across all antennas',
                    'tot_phs_std': 'average of phase standard deviation across all antennas',
                    'phs_noise_good_sol': 'determination of good solution for phase noise metric',
                    'phs_std_good_sol': 'determination of good solution for phase std metric',
                    'chisq_ant_avg': 'average chisquare value for each antenna averaged '
                    'over frequency and time',
                    'ant_phs_noise': 'phase noise power for each antenna',
                    'ant_phs_std': 'phase standard deviation for each antenna',
                    'chisq_bad_ants': 'list of bad antennas from chisq metric',
                    'phs_noise_bad_ants': 'list of bad antennas from phs noise metric',
                    'phs_std_bad_ants': 'list of bad antennas from phs std metric'}
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

        # ensure keys of dicts are not strings
        for k in metrics.keys():
            if type(metrics[k]) is OrderedDict:
                for i in metrics[k].keys():
                    if type(metrics[k][i]) is list:
                        metrics[k][i] = np.array(metrics[k].pop(i))
                    metrics[k][int(i)] = metrics[k].pop(i)

            elif type(metrics[k]) is list:
                metrics[k] = np.array(metrics[k])

    # load pickle
    elif filetype == 'pkl':
        with open(filename, 'rb') as f:
            inp = pkl.Unpickler(f)
            metrics = inp.load()
    else:
        raise IOError("Filetype not recognized, try a json or pkl file")

    return metrics


def load_firstcal_gains(fc_file):
    """
    load firstcal delays and turn into phase gains

    fc_file : str
        path to firstcal .calfits file

    jones2pol : dict
        dictionary containing jones integers as keys and 
        X-Y xpols as values

    getpol : str, options=['XX', 'XY', 'YX', 'YY']
        string containing the pol of data you want to extract
        needs a jones2pol dictionary to work

    """
    uvf = UVCal()
    uvf.read_calfits(fc_file)
    freqs = uvf.freq_array.squeeze()
    firstcal_delays = np.moveaxis(uvf.delay_array, 2, 3)[:, 0, :, :, :]
    firstcal_gains = np.array(map(lambda x: np.exp(-2j * np.pi * freqs.reshape(1, -1, 1) * x), firstcal_delays))
    fc_pols = uvf.jones_array
    return firstcal_delays, firstcal_gains, fc_pols

def plot_phs_metric(metrics, plot_type='std', ax=None, save=False,
                    fname=None, outpath=None, xpol_index=0,
                    plot_kwargs={'marker': 'o', 'color': 'k', 'linestyle': '', 'markersize': 6}):
    """
    Plot omnical phase metric

    Input:
    ------
    metrics : dictionary
        a metrics dictionary from OmniCal_Metrics.run_metrics()

    plot_type : str, default='std', options=['std', 'ft']
        specify plot type

        'std' plots stand deviations of omni-firstcal phase difference.
        large stand deviations are not ideal

        'ft' plots the phase noise level
        see self.phs_FT_metric for details

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
        tot_phs_std = metrics['tot_phs_std']
        ymax = np.max([ant_phs_std.max() * 1.1, phs_std_cut * 1.2])

        # make grid and plot points
        ax.grid(True)
        ax.plot(ant_phs_std, **plot_kwargs)
        p1 = ax.axhline(phs_std_cut, color='darkorange')
        ax.axhspan(phs_std_cut, ymax, color='darkorange', alpha=0.2)
        p2 = ax.axhline(tot_phs_std, color='steelblue')
        ax.legend([p1, p2], ["std cut", "avg phs std"], fontsize=12)

        # adjust axes
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('phase stand dev across file [radians]', fontsize=14)
        ax.tick_params(size=8)
        ax.set_xticks(range(metrics['Nants']))
        ax.set_xticklabels(metrics['ant_array'])
        [t.set_rotation(20) for t in ax.get_xticklabels()]
        ax.set_ylim(0, ymax)
        ax.set_title("{0}".format(metrics['filename']))

    elif plot_type == 'ft':
        # get y data
        ant_phs_noise = np.array(metrics['ant_phs_noise'].values())
        phs_noise_cut = metrics['phs_noise_cut']
        tot_phs_noise = metrics['tot_phs_noise']
        ymax = np.max([ant_phs_noise.max()*1.1, phs_noise_cut*1.2])

        # make grid and plot points
        ax.grid(True)
        ax.plot(ant_phs_noise, **plot_kwargs)
        p1 = ax.axhline(phs_noise_cut, color='darkorange')
        ax.axhspan(phs_noise_cut, ymax, color='darkorange', alpha=0.2)
        p2 = ax.axhline(tot_phs_noise, color='steelblue')
        ax.legend([p1, p2], ["phase noise cut", "avg phase noise"], fontsize=12)

        # adjust axes
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('phase noise level', fontsize=14)
        ax.tick_params(size=8)
        ax.set_xticks(range(metrics['Nants']))
        ax.set_xticklabels(metrics['ant_array'])
        [t.set_rotation(20) for t in ax.get_xticklabels()]
        ax.set_ylim(0, ymax)
        ax.set_title("{0}".format(metrics['filename']))

    if save is True and custom_ax is False:
        if fname is None:
            fname = metrics['filename'] + '.phs_{0}.png'.format(plot_type)
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        fig.savefig(fname, bbox_inches='tight')


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
    ycut = yloc + metrics['chisq_std_cut'] * ysig
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

    if save is True and custom_ax is False:
        if fname is None:
            fname = metrics['filename'] + '.chisq_std.png'
        if outpath is None:
            fname = os.path.join(metrics['filedir'], fname)
        else:
            fname = os.path.join(outpath, fname)
        fig.savefig(fname, bbox_inches='tight')


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
        self.xpols = np.array(map(lambda x: self.jones2pol[x], self.jones))
        self.Nxpols = len(self.pols)
        self.times = self.uv.time_array
        self.Ntimes = self.uv.Ntimes
        self.ant_array = self.uv.ant_array

        # Get omnical gains, move time axis in front of freq axis
        self.omni_gains = np.moveaxis(self.uv.gain_array, 2, 3)[:, 0, :, :, :]

        # Assign total chisq array
        self.chisq = np.moveaxis(self.uv.quality_array, 2, 3)[:, 0, :, :, :]
        self.chisq_tavg = np.median(self.chisq, axis=1)

    def run_metrics(self, fcfiles=None, cut_edges=True, Ncut=100,
                    phs_noise_cut=1.0, phs_std_cut=0.3, chisq_std_cut=5.0):
        """
        run metrics function on each polarization (e.g. XX, YY) individually
        and then pack into a single metrics dictionary

        Input:
        ------
        fcfiles : list, default=None
            list of single-pol firstcal delay solution files matching ordering of
            xpol in .omni.calfits file

        see self.compile_metrics() for details on remaining keyword arguments
        """
        full_metrics = OrderedDict()
        for i, xpol in enumerate(self.xpols):
            metrics = self.run_metrics(firstcal_file=fcfiles[i], pol_index=i, cut_edges=cut_edges, Ncut=Ncut,
                                       phs_noise_cut=1.0, phs_std_cut=0.3, chisq_std_cut=5.0)
            full_metrics[xpol] = metrics
        return full_metrics


    def compile_metrics(self, firstcal_file=None, pol_index=0, cut_edges=True, Ncut=100,
                        phs_noise_cut=1.0, phs_std_cut=0.3, chisq_std_cut=5.0):
        """
        run omnical metrics on a single polarization (e.g. XX, XY, YX or YY)

        Input:
        ------
        firstcal_file : str, default=None
            path to a FirstCal *.calfits file
            if fed, it will perform firstcal comparison metrics

        pol_index : int, default=0
            polarization index across polarization axis

        cut_edges : bool, defaul=True
            cut bandpass edges before metrics.
            if True, cuts Ncut number of channels from each band edge

        Ncut : int, default=100
            number of frequency channels to cut from bandpass edges before
            calculating metrics

        phs_noise_cut : float, default=1.0
            cut in phase noise level w.r.t. frequency for
            good/bad determination of phase solution.
            see self.phs_FT_metric for details

        phs_std_cut : float, default=0.3
            cut in phase variability for
            good/bad determination.
            see self.phs_std_metric for details

        chisq_std_cut : float, default=5.0
            sets the cut in chisq variablity for good/bad determination.
            the cut is the standard deviation of the
            chisq per-antenna times chisq_std_cut, which means
            chisq_std_cut is like a "sigma" threshold for each antenna's
            chisq variability.
        """
        # assign fc filename
        self.firstcal_file = firstcal_file

        # get polarization of omni data
        omni_pol = self.xpols[pol_index]

        # select freq channels
        if cut_edges is True:
            self.band = np.arange(Ncut, self.Nfreqs - Ncut)
        else:
            self.band = np.arange(self.Nfreqs)

        # check pol_index is within bounds
        if pol_index > self.Npols-1:
            raise KeyError("pol_index={0} is larger than number of polarizations present in omni.calfits file".format(pol_index))

        # get chisq metrics
        (avg_chisq, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
         chisq_ant_std_scale, chisq_bad_ants) = self.chisq_metric(self.chisq[:, :, :, pol_index], chisq_std_cut=chisq_std_cut)

        if firstcal_file is not None:
            # load fc gain solutions
            firstcal_delays, firstcal_gains, fc_pols = load_firstcal_gains(firstcal_file)

            # convert to 'xy' pol convention and then select omni_pol from fc_pols
            fc_pols = map(lambda x: self.jones2pol[x], fc_pols)
            if omni_pol not in fc_pols:
                raise ValueError("omni_pol={0} not in list of pols from firstcal_file={1]".format(omni_pol, firstcal_file))
            fc_pol_index = fc_pols.index(omni_pol)
            self.firstcal_delays = firstcal_delays[:, :, :, fc_pol_index]
            self.firstcal_gains = firstcal_gains[:, :, :, fc_pol_index]

            # get gain difference between omnical and firstcal gain solutions
            self.gain_diff = self.omni_gains[:, :, :, pol_index] * self.firstcal_gains.conj()

            # run phs FT metric
            (ant_phs_noise, tot_phs_noise, phs_noise_bad_ants,
            phs_noise_good_sol) = self.phs_FT_metric(np.angle(self.gain_diff),
                                                     phs_noise_cut=phs_noise_cut)

            # run phs std metric
            (ant_phs_std, tot_phs_std, phs_std_bad_ants,
            phs_std_good_sol) = self.phs_std_metric(np.angle(self.gain_diff),
                                                    phs_std_cut=phs_std_cut)

        # initialize metrics
        metrics = OrderedDict()

        metrics['avg_chisq'] = avg_chisq
        metrics['chisq_ant_avg'] = chisq_ant_avg
        metrics['chisq_ant_std'] = chisq_ant_std
        metrics['chisq_ant_std_loc'] = chisq_ant_std_loc
        metrics['chisq_ant_std_scale'] = chisq_ant_std_scale
        metrics['chisq_bad_ants'] = chisq_bad_ants
        metrics['chisq_std_cut'] = chisq_std_cut

        metrics['freqs'] = self.freqs
        metrics['Nfreqs'] = self.Nfreqs
        metrics['cut_edges'] = cut_edges
        metrics['Ncut'] = Ncut
        metrics['band'] = self.band
        metrics['ant_array'] = self.ant_array
        metrics['jones'] = self.jones[pol_index]
        metrics['xpol'] = self.xpols[pol_index]
        metrics['ant_pol'] = self.xpols[pol_index][0]
        metrics['times'] = self.times
        metrics['Ntimes'] = self.Ntimes
        metrics['Nants'] = self.Nants

        metrics['version'] = self.version_str
        metrics['history'] = self.history
        metrics['filename'] = self.filename
        metrics['filestem'] = self.filestem
        metrics['filedir'] = self.filedir

        metrics['ant_phs_noise'] = None
        metrics['tot_phs_noise'] = None
        metrics['phs_noise_bad_ants'] = None
        metrics['phs_noise_good_sol'] = None
        metrics['phs_noise_cut'] = None
        metrics['ant_phs_std'] = None
        metrics['tot_phs_std'] = None
        metrics['phs_std_bad_ants'] = None
        metrics['phs_std_good_sol'] = None
        metrics['phs_std_cut'] = None

        if firstcal_file is not None:
            metrics['ant_phs_noise'] = ant_phs_noise
            metrics['tot_phs_noise'] = tot_phs_noise
            metrics['phs_noise_bad_ants'] = phs_noise_bad_ants
            metrics['phs_noise_good_sol'] = phs_noise_good_sol
            metrics['phs_noise_cut'] = phs_noise_cut
            metrics['ant_phs_std'] = ant_phs_std
            metrics['tot_phs_std'] = tot_phs_std
            metrics['phs_std_bad_ants'] = phs_std_bad_ants
            metrics['phs_std_good_sol'] = phs_std_good_sol
            metrics['phs_std_cut'] = phs_std_cut

        return metrics

    def write_metrics(self, metrics, filename=None, filetype='json'):
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
        if filename is None:
            filename = os.path.join(self.filedir, self.filestem + '.omni_metrics')

        # write to file
        if filetype == 'json':
            if filename.split('.')[-1] != 'json':
                filename += '.json'

            # change ndarrays to lists
            metrics_out = copy.deepcopy(metrics)
            for i, k in enumerate(metrics_out.keys()):
                if type(metrics_out[k]) is np.ndarray:
                    metrics_out[k] = metrics_out[k].tolist()
                elif type(metrics_out[k]) is OrderedDict:
                    for j in metrics_out[k].keys():
                        if type(metrics_out[k][j]) is np.ndarray:
                            metrics_out[k][j] = metrics_out[k][j].tolist()
                elif type(metrics_out[k]) is np.bool_:
                    metrics_out[k] = bool(metrics_out[k])

            with open(filename, 'w') as f:
                json.dump(metrics_out, f, indent=4)

        elif filetype == 'pkl':
            if filename.split('.')[-1] != 'pkl':
                filename += '.pkl'
            with open(filename, 'wb') as f:
                outp = pkl.Pickler(f)
                outp.dump(metrics)

    def chisq_metric(self, chisq, chisq_std_cut=5.0, return_dict=True):
        """
        chi square metrics

        chisq : ndarray, shape=(Nants, Ntimes, Nfreqs)
            ndarray containing chisq for each antenna (single xpol)

        chisq_std_cut : float, default=5.0
            sigma tolerance (or z-score tolerance) for std of chisq fluctuations

        return_dict : bool, default=True
            return per-antenna metrics as a dictionary, with antenna number as key
            rather than an ndarray
        """
        # Get chisq statistics
        avg_chisq = astats.biweight_location(chisq[:, :, self.band]).astype(np.float64)
        chisq_ant_avg = np.array(map(astats.biweight_location, chisq[:, :, self.band])).astype(np.float64)
        chisq_ant_std = np.sqrt(np.array(map(astats.biweight_midvariance, chisq[:, :, self.band])))
        chisq_ant_std_loc = astats.biweight_location(chisq_ant_std).astype(np.float64)
        chisq_ant_std_scale = np.sqrt(astats.biweight_midvariance(chisq_ant_std))

        # Pick out "bad" antennas from chisq_std_cut, which is a cut in
        # the standard deviation of the chisq fluctuations across time & frequency
        chisq_bad_ants = self.ant_array[np.where((chisq_ant_std - chisq_ant_std_loc) > chisq_ant_std_scale * chisq_std_cut)]

        # convert to dictionaries
        if return_dict is True:
            chisq_ant_std = OrderedDict(zip(self.ant_array, chisq_ant_std))
            chisq_ant_avg = OrderedDict(zip(self.ant_array, chisq_ant_avg))

        return (avg_chisq, chisq_ant_avg, chisq_ant_std, chisq_ant_std_loc,
                chisq_ant_std_scale, chisq_bad_ants)

    def phs_FT_metric(self, phs_diff, phs_noise_cut=1.0, return_dict=True):
        """
        Takes the square of the real-valued FT of the phase
        difference between omnical and fistcal solutions and uses it
        to assess noise level across freq

        Input:
        ------
        phs_diff : ndarray, shape=(Nants, Ntimes, Nfreqs)
            real ndarray containing difference between omnical gain phases
            and firstcal gain phases (single xpol)

        phs_noise_cut : float, default=1.0
            phase noise level cut.
            the noise level is estimated by taking the absolute value of
            the real-valued fourier transform of the omnical gain
            phase - firstcal gain phase, taking a median filter across
            the modes, and then taking the median of the last 100 modes.

        return_dict : bool
            return antenna output as dictionary with antenna name as key

        Output:
        -------
        ant_phs_noise : ndarray, shape=(Nants,)
            phs noise for each antenna

        tot_phs_noise : float
            average of ant_phs_noise

        phs_noise_bad_ants : ndarray
            array of antennas whose phs_noise is above the cut

        phs_noise_good_sol : bool
            boolean with the metric's good / bad determination for
            the entire solution.
        """
        # take rfft
        freq_smooth = int(self.Nfreqs / 70)
        if freq_smooth % 2 == 0:
            freq_smooth += 1
        rfft = medfilt(np.abs(np.fft.rfft(phs_diff[:, :, self.band], axis=2)), kernel_size=(1, 1, freq_smooth))

        # Get phase noise
        freq_width = int(self.Nfreqs/100)
        phs_noise = np.median(rfft[:, :, -freq_width:], axis=2)

        # Calculate metrics
        ant_phs_noise = np.array(map(np.median, phs_noise))
        tot_phs_noise = np.max(ant_phs_noise)
        phs_noise_bad_ants = self.ant_array[np.where(ant_phs_noise > phs_noise_cut)]
        phs_noise_good_sol = tot_phs_noise < phs_noise_cut

        if return_dict is True:
            ant_phs_noise = OrderedDict(zip(self.ant_array, ant_phs_noise))

        return (ant_phs_noise, tot_phs_noise,
                phs_noise_bad_ants, phs_noise_good_sol)

    def phs_std_metric(self, phs_diff, phs_std_cut=0.3, frq_xchan_cut=0.3, return_dict=True):
        """
        Takes the variance of the phase difference between omnical
        and firstcal solutions (in radians) to assess its frequency variability

        Input:
        ------
        phs_diff : ndarray, dtype=float, shape=(Nants, Ntimes, Nfreqs)
            real-valued ndarray containing angle of (omnical gains divided by firstcal gains)

        phs_std_cut : float, default=0.5
            cut for standard deviation of phase solutions in radians

        return_dict : bool
            return per-antenna output as dictionary with antenna name as key

        Output:
        -------
        ant_phs_std : ndarray, shape=(Nants,)
            phase stand dev for each antenna

        tot_phs_std : float
            average of ant_phs_std

        phs_std_band_ants : ndarray
            array containing antennas whose phs_std didn't meet phs_std_cut

        phs_std_good_sol : bool
            boolean with metric's good / bad determination of entire solution
        """
        # take robust standard deviation across time & freq
        ant_phs_std = np.sqrt(np.array(map(astats.biweight_midvariance, phs_diff[:, :, self.band])))
        max_ant_phs_std = np.max(ant_phs_std)

        # get std across freq per antenna per time integration
        ant_phs_std_per_time = np.sqrt(astats.biweight_midvariance(phs_diff[:, :, self.band], axis=2))

        # take robust standard deviation across antenna & time
        frq_phs_std = np.sqrt(np.array(map(astats.biweight_midvariance, phs_diff[:, :, self.band].T)))
        frq_xchan = np.sum(freq_phs_std > phs_std_cut) / len(self.band)

        # get goodness of variability
        ant_phs_std_good_sol = max_ant_phs_std < phs_std_cut
        frq_xchan_good_sol = frq_xchan < frq_xchan_cut

        if return_dict is True:
            ant_phs_std = OrderedDict(zip(self.ant_array, ant_phs_std))
            ant_phs_std_per_time = OrderedDict(zip(self.ant_array, ant_phs_std_per_time))

        return ant_phs_std, tot_phs_std, phs_std_bad_ants, phs_std_good_sol, ant_phs_std_per_time

    def plot_gains(self, ants=None, time_index=0, xpol_index=0, divide_fc=False,
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

        xpol_index : int, default=0
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
            gains = self.omni_gains[ants, time_index, :, xpol_index].T.copy()
            if divide_fc is True:
                gains /= self.firstcal_gains[ants, time_index, :, xpol_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.angle(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain phase [radians]', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} xpol".format(self.filename, self.times[time_index], self.xpols[xpol_index]))

        elif plot_type == 'amp':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, xpol_index].T.copy()
            if divide_fc is True:
                gains /= self.firstcal_gains[ants, time_index, :, xpol_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.abs(gains), **plot_kwargs)).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain amplitude', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1} : {2} pol".format(self.filename, self.times[time_index], self.xpols[xpol_index]))

        if save is True and custom_ax is False:
            if fname is None:
                fname = self.filename + '.gain_{0}.png'.format(plot_type)
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
            fig.savefig(fname, bbox_inches='tight')

    def plot_chisq_tavg(self, pol_index=0, ants=None, xpol_index=0, ax=None, save=False, fname=None, outpath=None):
        """
        Plot Omnical chi-square averaged over time

        Input:
        ------
        pol_index : int, default=0
            polarization index across polarization axis

        ants : list
            list of ant numbers to plot

        xpol_index : int, default=0
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
        ylim = np.percentile(ydata.ravel(), 75)
        p = ax.plot(self.freqs / 1e6, ydata)
        ax.set_xlabel('frequency [MHz]', fontsize=14)
        ax.set_ylabel('chi-square avg over time', fontsize=14)
        ax.set_ylim(0, ylim*2)
        ax.set_title("{0} : {1} xpol".format(self.filename, self.xpols[pol_index]))

        ax = ax.figure.add_axes([0.99, 0.1, 0.02, 0.8])
        ax.axis('off')
        ax.legend(p, self.ant_array)

        if save is True and custom_ax is False:
            if fname is None:
                fname = self.filename + '.chisq_tavg.png'
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
            fig.savefig(fname, bbox_inches='tight')

    def plot_metrics(self, metrics):
        """
        plot all metrics

        metrics : dictionary
            Omnical_Metrics.run_metrics() dictionary output

        """
        # plot chisq metric
        plot_chisq_metric(metrics, save=True)
        # plot phs metrics
        plot_phs_metric(metrics, plot_type='std', save=True)
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
        if len(args.fc_files) > 0:
            om.run_metrics(firstcal_file=args.fc_files[i],
                           cut_edges=args.no_bandcut is False,
                           phs_noise_cut=args.phs_noise_cut,
                           phs_std_cut=args.phs_std_cut,
                           chisq_std_cut=args.chisq_std_cut)
        else:
            om.run_metrics(cut_edges=args.no_bandcut is False,
                           phs_noise_cut=args.phs_noise_cut,
                           phs_std_cut=args.phs_std_cut,
                           chisq_std_cut=args.chisq_std_cut)

        if args.make_plots is True:
            om.plot_metrics()

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
        om.write_metrics(filename=metrics_filename)
