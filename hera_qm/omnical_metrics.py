"""
OmniCal Metrics
"""
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
    metrics_dict -- Dictionary with metric names as keys and descriptions as values.
    """
    metrics_dict = {'omnical_quality': 'Quality of cal solution (chi-squared) '
                    'for each antenna.',
                    'omnical_total_quality': 'Quality of overall cal solution '
                    '(chi-squared) across entire array.'}
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

def plot_phs_metric(metrics, plot_type='std', ax=None, save=False, fname=None, outpath=None, **kwargs):
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

    ax : matplotli axis object

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
        ymax = np.max([ant_phs_std.max()*1.1, phs_std_cut*1.2])

        # make grid and plot points
        ax.grid(True)
        ax.plot(ant_phs_std, marker='o', color='k', linestyle='', markersize=5)
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
        ax.plot(ant_phs_noise, marker='o', color='k', linestyle='', markersize=5)
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


def plot_chisq_metric(metrics, ax=None, save=False, fname=None, outpath=None, **kwargs):
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
    ycut = yloc + metrics['chisq_std_cut']*ysig
    chisq_ant_std = np.array(metrics['chisq_ant_std'].values())
    ymax = np.max([chisq_ant_std.max()*1.2, ycut+ysig])

    # make grid and plots
    ax.grid(True)
    p1 = ax.axhspan(yloc - ysig, yloc + ysig, color='green', alpha=0.1)
    p2 = ax.axhline(yloc, color='steelblue')
    p3 = ax.axhline(ycut, color='darkred')
    p4 = ax.axhspan(ycut, ymax, color='darkred', alpha=0.2)
    p5 = ax.plot(chisq_ant_std, marker='o', color='k', linestyle='', markersize=6)
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

    jones2pol = {-5:'XX',-6:'YY',-7:'XY',-8:'YX'}

    def __init__(self, omni_calfits):
        """
        Omnical Metrics initialization

        Input:
        ------
        omni_calfits : str
            calfits file output from omnical, typically
            ending in *.omni.calfits
        """
        # Get file info and other relevant metadata
        self.filedir     = os.path.dirname(omni_calfits)
        self.filestem    = '.'.join(os.path.basename(omni_calfits).split('.')[:-1])
        self.filename    = os.path.basename(omni_calfits)
        self.version_str = hera_qm_version_str
        self.history     = ''

        # Instantiate Data Object
        self.uv = UVCal()
        self.uv.read_calfits(omni_calfits)

        # Get relevant metadata
        self.Nants  = self.uv.Nants_data
        self.freqs  = self.uv.freq_array.squeeze()
        self.Nfreqs = len(self.freqs)
        self.jones  = self.uv.jones_array
        self.pols   = np.array(map(lambda x: self.jones2pol[x], self.jones))
        self.Npols  = self.uv.Njones
        self.times  = self.uv.time_array
        self.Ntimes = self.uv.Ntimes
        self.ant_array = self.uv.ant_array

        # Get omnical gains, move time axis in front of freq axis
        self.omni_gains = np.moveaxis(self.uv.gain_array, 2, 3)[:, 0, :, :, :]

        # Assign total chisq array
        self.chisq = np.moveaxis(self.uv.quality_array, 2, 3)[:, 0, :, :, :]

    def run_metrics(self, firstcal_file=None, cut_band=True, phs_noise_cut=1.0,
                          phs_std_cut=0.3, chisq_std_cut=5.0):
        """
        run omnical metrics

        Input:
        ------
        firstcal_file : str, default=None
            path to a FirstCal *.calfits file
            if fed, it will perform firstcal comparison metrics

        cut_band : bool, defaul=True
            cut bandpass edges before metrics.
            if True, cuts 0:102 and 922:1024 channels from band

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
            the cut is the standard deviation of the chisq per-antenna times
            chisq_std_cut, which means chisq_std_cut is like a "sigma" threshold
            for each antenna's chisq variability.
        """
        # assign fc filename
        self.firstcal_file = firstcal_file

        # select freq channels
        if cut_band == True:
            self.band = np.arange(self.Nfreqs/10, self.Nfreqs - self.Nfreqs/10)
        else:
            self.band = np.arange(self.Nfreqs)

        # Get time averaged chisq for each antenna
        self.chisq_tavg = np.median(self.chisq, axis=1)

        # Get robust standard deviation of chisq for each antenna
        chisq_ant_std       = np.sqrt(np.array(map(astats.biweight_midvariance, self.chisq[:, :, self.band, :])))
        chisq_ant_std_loc   = astats.biweight_location(chisq_ant_std)
        chisq_ant_std_scale = np.sqrt(astats.biweight_midvariance(chisq_ant_std))

        # Pick out "bad" antennas from chisq_std_cut, which is a cut in the standard deviation
        # of the chisq fluctuations across time, frequency and polarizations
        chisq_bad_ants = self.ant_array[np.where((chisq_ant_std-chisq_ant_std_loc) > chisq_ant_std_scale * chisq_std_cut)]

        # convert to dictionaries
        chisq_ant_std = OrderedDict(zip(self.ant_array, chisq_ant_std))

        if firstcal_file is not None:
            # load fc gain solutions
            uvfc = UVCal()
            uvfc.read_calfits(firstcal_file)

            # check Nants_data is same as omnical
            if uvfc.Nants_data != self.Nants:
                raise Exception("Nants_data for firstcal file is not the same as omnical file.")

            # turn firstcal delay into phase
            self.firstcal_delays = np.moveaxis(uvfc.delay_array, 2, 3)[:, 0, :, :, :]
            self.firstcal_gains = np.array(map(lambda x: np.exp(-2j*np.pi*self.freqs.reshape(1,-1,1)*x), \
                                                    self.firstcal_delays))

            # get gain difference between omnical and firstcal gain solutions
            self.gain_diff = self.omni_gains * self.firstcal_gains.conj()

            # run phs FT metric
            (ant_phs_noise, tot_phs_noise, phs_noise_bad_ants,
            phs_noise_good_sol) = self.phs_FT_metric(np.angle(self.gain_diff), phs_noise_cut=phs_noise_cut)

            # run phs std metric
            (ant_phs_std, tot_phs_std, phs_std_bad_ants, 
            phs_std_good_sol) = self.phs_std_metric(np.angle(self.gain_diff), phs_std_cut=phs_std_cut,)

        # initialize metrics
        metrics                        = OrderedDict()

        metrics['chisq_ant_std']       = chisq_ant_std
        metrics['chisq_ant_std_loc']   = chisq_ant_std_loc
        metrics['chisq_ant_std_scale'] = chisq_ant_std_scale
        metrics['chisq_bad_ants']      = chisq_bad_ants
        metrics['chisq_std_cut']       = chisq_std_cut

        metrics['freqs']               = self.freqs
        metrics['Nfreqs']              = self.Nfreqs
        metrics['cut_band']            = cut_band
        metrics['band']                = self.band
        metrics['ant_array']           = self.ant_array
        metrics['jones']               = self.jones
        metrics['pols']                = self.pols
        metrics['Npols']               = self.Npols
        metrics['times']               = self.times
        metrics['Ntimes']              = self.Ntimes
        metrics['Nants']               = self.Nants

        metrics['version']             = self.version_str
        metrics['history']             = self.history
        metrics['filename']            = self.filename
        metrics['filestem']            = self.filestem
        metrics['filedir']             = self.filedir

        metrics['ant_phs_noise']       = None
        metrics['tot_phs_noise']       = None
        metrics['phs_noise_bad_ants']  = None
        metrics['phs_noise_good_sol']  = None
        metrics['phs_noise_cut']       = None
        metrics['ant_phs_std']         = None
        metrics['tot_phs_std']         = None
        metrics['phs_std_bad_ants']    = None
        metrics['phs_std_good_sol']    = None
        metrics['phs_std_cut']         = None

        if firstcal_file is not None:
            metrics['ant_phs_noise']       = ant_phs_noise
            metrics['tot_phs_noise']       = tot_phs_noise
            metrics['phs_noise_bad_ants']  = phs_noise_bad_ants
            metrics['phs_noise_good_sol']  = phs_noise_good_sol
            metrics['phs_noise_cut']       = phs_noise_cut
            metrics['ant_phs_std']         = ant_phs_std
            metrics['tot_phs_std']         = tot_phs_std
            metrics['phs_std_bad_ants']    = phs_std_bad_ants
            metrics['phs_std_good_sol']    = phs_std_good_sol
            metrics['phs_std_cut']         = phs_std_cut

        self.metrics = metrics

    def write_metrics(self, filename=None, filetype='json'):
        """
        Write metrics to file after running self.run_metrics()

        Input:
        ------
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
            metrics_out = copy.deepcopy(self.metrics)
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
                outp.dump(self.metrics)

    def load_metrics(self, filename):
        """
        load omnical metrics

        Input:
        ------
        filename : str
            path to omnical metrics file

        Result:
        --------
        self.metrics dictionary
        """
        self.metrics = load_omnical_metrics(filename)

    def phs_FT_metric(self, phs_diff, phs_noise_cut=1.0, return_dict=True):
        """
        Takes the square of the real-valued FT of the phase difference between
        omnical and fistcal solutions and uses it to assess noise level across freq

        Input:
        ------
        phs_diff : ndarray, dtype=float, shape=(Nants, Ntimes, Nfreqs, Npols)
            real ndarray containing difference between omnical gain phases
            and firstcal gain phases

        phs_noise_cut : float, default=1.0
            phase noise level cut.
            the noise level is estimated by taking the absolute value of the 
            real-valued fourier transform of the omnical gain phase - firstcal
            gain phase, taking a median filter across the modes, and then taking 
            the median of the last 100 modes. 

        return_dict : bool
            return per-antenna output as dictionary with antenna name as key

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
        rfft = medfilt(np.abs(np.fft.rfft(phs_diff[:, :, self.band, :], axis=2)), kernel_size=(1, 1, 15, 1))

        # Get phase noise
        phs_noise = np.median(rfft[:, :, -100:, :], axis=2)

        # Calculate metrics
        ant_phs_noise   = np.array(map(np.median, phs_noise))
        tot_phs_noise   = np.median(ant_phs_noise)
        phs_noise_bad_ants  = self.ant_array[np.where(ant_phs_noise > phs_noise_cut)]
        phs_noise_good_sol  = tot_phs_noise < phs_noise_cut

        if return_dict == True:
            ant_phs_noise = OrderedDict(zip(self.ant_array, ant_phs_noise))

        return ant_phs_noise, tot_phs_noise, phs_noise_bad_ants, phs_noise_good_sol

    def phs_std_metric(self, phs_diff, phs_std_cut=0.3, return_dict=True):
        """
        Takes the variance of the phase difference between omnical
        and firstcal solutions (in radians) to assess its frequency variability

        Input:
        ------
        phs_diff : ndarray, dtype=complex, shape=(Nants, Ntimes, Nfreqs, Npols)
            complex ndarray containing omnical gains divided by firstcal gains

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
        # take robust standard deviation
        ant_phs_std = np.sqrt(np.array(map(astats.biweight_midvariance, phs_diff[:, :, self.band, :])))
        tot_phs_std = np.median(ant_phs_std)

        # get goodness of variability
        phs_std_bad_ants = self.ant_array[np.where(ant_phs_std > phs_std_cut)]
        phs_std_good_sol = tot_phs_std < phs_std_cut

        if return_dict == True:
            ant_phs_std = OrderedDict(zip(self.ant_array, ant_phs_std))

        return ant_phs_std, tot_phs_std, phs_std_bad_ants, phs_std_good_sol

    def plot_chisq_metric(self, ax=None, save=False, fname=None, **kwargs):
        """
        Plot Chi-Square output from omnical for each antenna

        Input:
        ------
        ax : matplotlib axis object, default=None

        save : bool, default=False
            if True, save image as a png

        fname : str, default=None
            filename to save image as

        outpath : str, default=None
            path to place file in
            will default to location of *omni.calfits file
        """
        if hasattr(self, 'metrics') == False:
            raise Exception("Must run self.run_metrics() before plotting routines...")

        plot_chisq_metric(self.metrics, ax=ax, save=save, fname=fname, **kwargs)

    def plot_phs_metric(self, plot_type='std', ax=None, save=False, fname=None, outpath=None, **kwargs):
        """
        Plot omnical phase metric

        Input:
        ------
        plot_type : str, default='std', options=['std', 'ft']
            specify plot type

            'std' plots stand deviations of omni-firstcal phase difference.
            large stand deviations are not ideal

            'ft' plots the "noise_level"
            see self.phs_FT_metric for details.

        ax : matplotlib axis object

        save : bool, default=False
            if True, save image as a png

        fname : str, default=None
            filename to save image as

        outpath : str, default=None
            path to place file in
            will default to location of *omni.calfits file
        """
        if hasattr(self, 'metrics') == False:
            raise Exception("Must run self.run_metrics() before plotting routines...")

        if self.firstcal_file is None:
            raise Exception("Must supply firstcal_file in order to plot phase metrics...")

        plot_phs_metric(self.metrics, plot_type=plot_type, ax=ax, save=save, fname=fname, outpath=outpath, **kwargs)

    def plot_gains(self, ants=None, time_index=0, jones_index=0, divide_fc=False, plot_type='phs', ax=None,
                         save=False, fname=None, outpath=None):
        """
        Plot omnical gain solutions for each antenna

        Input:
        ------
        ants : list
            list of ant numbers to plot
        time_index : int, default=0
            index of time array

        jones_index : int, default=0
            index of jones (polarization) array

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
            ants = np.array(map(lambda x: np.where(self.ant_array==x)[0][0], ants))

        if plot_type == 'phs':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, jones_index].T.copy()
            if divide_fc == True:
                gains /= self.firstcal_gains[ants, time_index, :, jones_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.angle(gains), marker='o', markersize=3, alpha=0.75, linestyle='')).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain phase [radians]', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1}".format(self.filename, self.times[time_index]))

        elif plot_type == 'amp':
            # make grid and plot
            ax.grid(True)
            gains = self.omni_gains[ants, time_index, :, jones_index].T.copy()
            if divide_fc == True:
                gains /= self.firstcal_gains[ants, time_index, :, jones_index].T
            p = np.array(ax.plot(self.freqs/1e6, np.abs(gains), marker='o', markersize=3, alpha=0.75, linestyle='')).ravel()

            # axes
            ax.set_xlabel('frequency [MHz]', fontsize=14)
            ax.set_ylabel('gain amplitude', fontsize=14)
            ax1 = ax.figure.add_axes([0.99, 0.1, 0.03, 0.8])
            ax1.axis('off')
            ax1.legend(p, self.ant_array[ants])
            ax.set_title("{0} : JD={1}".format(self.filename, self.times[time_index]))

        if save is True and custom_ax is False:
            if fname is None:
                fname = self.filename + '.gain_{0}.png'.format(plot_type)
            if outpath is None:
                fname = os.path.join(self.filedir, fname)
            else:
                fname = os.path.join(outpath, fname)
            fig.savefig(fname, bbox_inches='tight')

    def plot_chisq_tavg(self, ants=None, jones_index=0, ax=None, save=False, fname=None, outpath=None):
        """
        Plot Omnical chi-square averaged over time

        Input:
        ------
        ants : list
            list of ant numbers to plot

        jones_index : int, default=0
            index of jones (polarization) array

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
            ants = np.array(map(lambda x: np.where(self.ant_array==x)[0][0], ants))

        # make grid and plots
        ax.grid(True)
        p = ax.plot(self.freqs/1e6, self.chisq_tavg[ants, :, jones_index].T)
        ax.set_xlabel('frequency [MHz]', fontsize=14)
        ax.set_ylabel('chi-square avg over time', fontsize=14)
        ax.set_ylim(0, self.metrics['chisq_ant_std_loc']+self.metrics['chisq_ant_std_scale']*30)
        ax.set_title("{0} : {1} pol".format(self.filename, self.pols[jones_index]))

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

    def plot_metrics(self):
        """
        plot all metrics

        """
        # plot chisq metric
        self.plot_chisq_metric(save=True)
        # plot phs metrics
        self.plot_phs_metric(plot_type='std', save=True)
        self.plot_phs_metric(plot_type='ft', save=True)

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
        om = OmniCal_Metrics(filename)
        if len(args.fc_files) > 0:
            om.run_metrics(firstcal_file=args.fc_files[i], cut_band=args.no_bandcut==False,
                phs_noise_cut=args.phs_noise_cut, phs_std_cut=args.phs_std_cut, chisq_std_cut=args.chisq_std_cut)
        else:
            om.run_metrics(cut_band=args.no_bandcut==False,
                phs_noise_cut=args.phs_noise_cut, phs_std_cut=args.phs_std_cut, chisq_std_cut=args.chisq_std_cut)

        om.history = om.history + history
        if args.make_plots == True:
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

