# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""FirstCal metrics."""
import numpy as np
from pyuvdata import UVCal, utils as uvutils
import astropy.stats as astats
from collections import OrderedDict
from . import __version__
from . import utils, metrics_io
import copy
import os

try:
    from sklearn import gaussian_process as gp
    sklearn_import = True
except ImportError:
    print("could not import sklearn")
    sklearn_import = False


def get_firstcal_metrics_dict():
    """Return dictionary with metric names as keys.

    Finds all metrics computed on firstcal solutions and returns all dictionaries and keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns
    -------
    metrics_dict : dict
        Dictionary with metric names as keys and descriptions as values.

    """
    metrics_dict = {'firstcal_metrics_good_sol': 'Whether full firstcal solution'
                    'is good (1) or bad(0).',
                    'firstcal_metrics_good_sol_x': 'Whether full x firstcal solution'
                    'is good (1) or bad(0).',
                    'firstcal_metrics_good_sol_y': 'Whether full y firstcal solution'
                    'is good (1) or bad(0).',
                    'firstcal_metrics_agg_std': 'Aggregate standard deviation '
                    'of delay solutions',
                    'firstcal_metrics_max_std_x': 'Maximum antenna standard deviation '
                    'of xx delay solutions',
                    'firstcal_metrics_max_std_y': 'Maximum antenna standard deviation '
                    'of yy delay solutions',
                    'firstcal_metrics_agg_std_x': 'Aggregate standard deviation '
                    'of xx delay solutions',
                    'firstcal_metrics_agg_std_y': 'Aggregate standard deviation '
                    'of yy delay solutions',
                    'firstcal_metrics_ant_z_scores': 'Z-scores for each antenna '
                    'delay solution w.r.t. agg_std',
                    'firstcal_metrics_ant_avg': 'Average delay solution for '
                    'each antenna.',
                    'firstcal_metrics_ant_std': 'Standard deviation of each '
                    'antennas delay solution across time.',
                    'firstcal_metrics_bad_ants': 'Antennas flagged as bad due '
                    'to large variation in delay solution.',
                    'firstcal_metrics_rot_ants': 'Antennas flagged as being '
                    'rotated by 180 degrees.'}
    return metrics_dict


def load_firstcal_metrics(filename):
    """
    Read-in a firstcal_metrics file and return dictionary.

    Loading is handled via hera_qm.metrics_io.load_metric_file

    Parameters
    ----------
    filename : str
        Full path to filename of the metric to load. Must be either HDF5 (recommended)
        or JSON or PKL (Depreciated in Future) types.

    Returns
    -------
    metrics : dict
        Dictionary of metrics stored in the input file.

    """
    return metrics_io.load_metric_file(filename)


def plot_stds(metrics, fname=None, ax=None, xaxis='ant', kwargs={}, save=False):
    """
    Plot standard deviation of delay solutions per-ant or per-time.

    Parameters
    ----------
    metrics : dict
        A "metrics" dictionary from FirstCalMetrics.run_metrics().
    fname : str, optional
        Full path to filename. Default is None.
    xaxis : {"ant", "time"}, optional
        What to plot on the x-axis. Must be one of: "ant", "time". These are
        antenna number or time-stamp respectively. Default is "ant".
    ax : matplotlib axis object, optional
        Where to plot the data. Default is None (meaning a new axis will be created).
    kwargs : dict, optional
        Plotting kwargs. Potential keys (descriptions) are: "cmap" (colormap).
    save : bool, optional
        If True, save the image to the specified filename. Default is False.

    Returns
    -------
    fig : matplotlib figure object
        If ax is not specified, the matplotlib figure object is returned.

    """
    import matplotlib.pyplot as plt
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    # choose xaxis
    if xaxis == 'ant':
        Nants = len(metrics['ants'])
        xax = range(Nants)
        yax = list(metrics['ant_std'].values())
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'Spectral'
        cmap_func = plt.get_cmap(kwargs['cmap'])
        cmap = cmap_func(np.linspace(0, 0.95, len(xax)))
        ax.grid(True, zorder=0)
        ax.tick_params(size=8)
        ax.scatter(xax, yax, c=cmap, alpha=0.85, marker='o', edgecolor='k',
                   s=70, zorder=3)
        ax.set_xlim(-1, Nants)
        ax.set_xticks(range(Nants))
        ax.set_xticklabels(metrics['ants'])
        for label in ax.get_xticklabels():
            label.set_rotation(20)
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

    elif xaxis == 'time':
        xax = metrics['frac_JD']
        yax = list(metrics['time_std'].values())
        ax.grid(True, zorder=0)
        ax.tick_params(size=8)
        ax.plot(xax, yax, c='k', marker='.', linestyle='-', alpha=0.85, zorder=1)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
        ax.set_xlabel('fractional JD of {}'.format(metrics['start_JD']), fontsize=14)
        ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

    else:
        raise NameError('xaxis kwarg not recognized, try "ant" or "time"')

    if save is True and custom_ax is False:
        if fname is None:
            fname = metrics['fc_filestem'] + '.stds.png'
        fig.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig


def plot_zscores(metrics, fname=None, plot_type='full', ax=None, figsize=(10, 6),
                 save=False, kwargs={'cmap': 'Spectral'}, plot_abs=False):
    """
    Plot z_scores for antenna delay solution.

    Parameters
    ----------
    metrics : dict
        A FirstCalMetrics "metrics" dictionary.
    fname : str, optional
        Full path to the output filename. Default is None.
    plot_type : {"full", "time_avg"}, optional
        Type of plot to make. "full" plots the z-score for each entry for
        antennas and times. "time_avg" plots the z-score for each antenna
        averaged over time. Default is "full".
    ax : matplotlib axis object, optional
        Where to plot the data. Default is None (meaning a new axis will be created).
    figsize : tuple, optional
        Figure size if creating a figure. Default is (10, 6).
    save : bool, optional
        If True, save the image to the specified filename. Default is False.
    kwargs : dict, optional
        Plotting kwargs. Potential keys (descriptions) are: "vmin" (colormap minimum
        value), "vmax" (colormap maximum value), "cmap" (colormap). If plot type is
        "full", the kwargs dictionary is unpacked and passed to `matshow`.
        Default is {'cmap': 'Spectral'}
    plot_abs : bool, optional
        If True, plot the absolute value of z-scores instead of actual values.
        Default is False

    Returns
    -------
    fig : matplotlib figure object
        If ax is not specified, the matplotlib figure object is returned.

    """
    import matplotlib.pyplot as plt

    # Get ax if not provided
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # unpack some variables
    z_scores = np.array(list(metrics['z_scores'].values()))
    ant_z_scores = np.array(list(metrics['ant_z_scores'].values()))
    Nants = len(metrics['ants'])
    if plot_abs is True:
        z_scores = np.abs(z_scores)
        if 'vmin' not in kwargs:
            kwargs['vmin'] = 0
        if 'vmax' not in kwargs:
            kwargs['vmax'] = 5
    else:
        if 'vmin' not in kwargs:
            kwargs['vmin'] = -5
        if 'vmax' not in kwargs:
            kwargs['vmax'] = 5

    # Plot zscores
    if plot_type == 'full':
        # plot
        xlen = [np.min(metrics['frac_JD']), np.max(metrics['frac_JD'])]
        xlen_round = [np.ceil(np.min(metrics['frac_JD']) * 1000) / 1000,
                      np.floor(np.max(metrics['frac_JD']) * 1000) / 1000]
        ylen = [0, Nants]
        cax = ax.matshow(z_scores, origin='lower', aspect='auto',
                         extent=[xlen[0], xlen[1], ylen[0], ylen[1]], **kwargs)

        # define ticks
        ax.xaxis.set_ticks_position('bottom')
        xticks = np.arange(xlen_round[0], xlen_round[1] + 1e-5, 0.001)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(Nants) + 0.5)
        ax.set_yticklabels(metrics['ants'])
        [t.set_rotation(20) for t in ax.get_yticklabels()]
        ax.tick_params(size=8)

        # set labels
        ax.set_xlabel('fraction of JD %d' % metrics['start_JD'], fontsize=14)
        ax.set_ylabel('antenna number', fontsize=14)

        # set colorbar
        fig.colorbar(cax, label='z-score')

    elif plot_type == 'time_avg':
        # plot
        ax.grid(True, zorder=0)
        ax.bar(range(len(ant_z_scores)), ant_z_scores, align='center', color='steelblue', alpha=0.75,
               zorder=3)

        # define ticks
        ax.set_xlim(-1, Nants)
        ax.set_ylim(0, kwargs['vmax'])

        ax.set_xticks(range(Nants))
        ax.set_xticklabels(metrics['ants'])
        ax.tick_params(size=8)

        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('time-averaged z-score', fontsize=14)

    else:
        raise NameError("plot_type not understood, try 'full' or 'time_avg'")

    if save is True and custom_ax is False:
        if fname is None:
            fname = metrics['fc_filestem'] + '.zscrs.png'
        fig.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig


class FirstCalMetrics(object):
    """Object to store and compute FirstCal metric data.

    The FirstCalMetrics class is used for holding firstcal data, running metrics,
    and plotting delay solutions. Currently it only supports single polarization
    solutions.

    """

    # sklearn import statement
    sklearn_import = sklearn_import

    def __init__(self, calfits_files, use_gp=True):
        """Initilize the object.

        Parameters
        ----------
        calfits_files : str or list
            Filename for a *.first.calfits file or a list of (time-ordered)
            .first.calfits files of the same polarization.
        use_gp : bool, optional
            If True, use a Gaussian process model to subtract underlying smooth
            delay solution behavior over time from fluctuations. Default is True.

        Attributes
        ----------
        UVC : pyuvdata.UVCal() object
            The resulting UVCal object from reading in the calfits files.
        Nants : int
            The number of antennas in the UVCal object.
        Ntimes : int
            The number of times in the UVCal object
        delays : ndarray, shape=(Nants, Ntimes)
            The firstcal delay solutions in nanoseconds.
        delay_avgs : ndarray, shape=(Nants,)
            The median delay solutions across time in nanoseconds.
        delay_fluctuations : ndarray, shape=(Nants, Ntimes)
            The firstcal delay solution fluctuations from the time average in
            nanoseconds.
        start_JD : float
            The integer JD of the start of the UVCal object.
        frac_JD : ndarray, shape=(Ntimes,)
            The time-stamps of each integration in units of the fraction of start_JD.
            E.g., 2457966.53433 becomes 0.53433.
        ants : ndarray, shape=(Nants,)
            The antenna numbers contained in the calfits files.
        pol : {"x", "y"}
            Polarization of the files examined.
        fc_basename : str
            Basename of the calfits file, or first calfits file in the list.
        fc_filename : str
            Filename of the calfits file, or first calfits file in the list.
        fc_filestem : str
            Filename minus extension of the calfits file, or first calfits file in the list.
        times : ndarray, shape=(Ntimes,)
            The times contained in the UVCal object.
        minutes : ndarray, shape=(Ntimes,)
            The number of minutes of the fractional JD.
        ants
            The list of antennas in the UVCal object.
        version_str : str
            The version of the hera_qm module used to generate these metrics.
        history : str
            History to append to the metrics files when writing out files.

        """
        # Instantiate UVCal and read calfits
        self.UVC = UVCal()
        self.UVC.read_calfits(calfits_files)

        self.pols = np.array([uvutils.polnum2str(jones, x_orientation=self.UVC.x_orientation)
                              for jones in self.UVC.jones_array])
        self.Npols = self.pols.size

        # get file prefix
        if isinstance(calfits_files, list):
            calfits_file = calfits_files[0]
        else:
            calfits_file = calfits_files
        self.fc_basename = os.path.basename(calfits_file)
        self.fc_filename = calfits_file
        self.fc_filestem = utils.strip_extension(self.fc_filename)

        # get other relevant arrays
        self.times = self.UVC.time_array
        self.Ntimes = len(list(set(self.times)))
        self.start_JD = np.floor(self.times).min()
        self.frac_JD = self.times - self.start_JD
        self.minutes = 24 * 60 * (self.frac_JD - self.frac_JD.min())
        self.Nants = self.UVC.Nants_data
        self.ants = self.UVC.ant_array
        self.version_str = __version__
        self.history = ''

        # Get the firstcal delays and/or gains and/or rotated antennas
        if self.UVC.cal_type == 'gain':
            # get delays
            freqs = self.UVC.freq_array.squeeze()
            # the unwrap is dove over the frequency axis
            fc_gains = self.UVC.gain_array[:, 0, :, :, :]
            fc_phi = np.unwrap(np.angle(fc_gains), axis=1)
            d_nu = np.median(np.diff(freqs))
            d_phi = np.median(fc_phi[:, 1:, :, :] - fc_phi[:, :-1, :, :], axis=1)
            gain_slope = (d_phi / d_nu)
            self.delays = gain_slope / (-2 * np.pi)
            self.gains = fc_gains

            # get delay offsets at nu = 0 Hz, and then get rotated antennas
            self.offsets = fc_phi[:, 0, :, :] - gain_slope * freqs[0]
            # find where the offest have a difference of pi from 0
            rot_offset_bool = np.isclose(np.pi, np.mod(np.abs(self.offsets), 2 * np.pi), atol=0.1).T
            rot_offset_bool = np.any(rot_offset_bool, axis=(0, 1))
            self.rot_ants = np.unique(self.ants[rot_offset_bool])

        elif self.UVC.cal_type == 'delay':
            self.delays = self.UVC.delay_array.squeeze()
            self.gains = None
            self.offsets = None
            self.rot_ants = []

        # Calculate avg delay solution and subtract to get delay_fluctuations
        delay_flags = np.all(self.UVC.flag_array, axis=(1, 2))
        self.delays = self.delays * 1e9
        self.delays[delay_flags] = np.nan
        self.delay_avgs = np.nanmedian(self.delays, axis=1, keepdims=True)
        self.delay_avgs[~np.isfinite(self.delay_avgs)] = 0
        self.delays[delay_flags] = 0
        self.delay_fluctuations = (self.delays - self.delay_avgs)

        # use gaussian process model to subtract underlying mean function
        if use_gp is True and self.sklearn_import is True:
            # initialize GP kernel.
            # RBF is a squared exponential kernel with a minimum length_scale_bound of 0.01 JD, meaning
            # the GP solution won't have time fluctuations quicker than ~0.01 JD, which will preserve
            # short time fluctuations. WhiteKernel is a Gaussian white noise component with a fiducial
            # noise level of 0.01 nanoseconds. Both of these are hyperparameters that are fit for via
            # a gradient descent algorithm in the GP.fit() routine, so length_scale=0.2 and
            # noise_level=0.01 are just initial conditions and are not the final hyperparameter solution
            kernel = (gp.kernels.RBF(length_scale=0.2, length_scale_bounds=(0.01, 1.0))
                      + gp.kernels.WhiteKernel(noise_level=0.01))
            xdata = self.frac_JD.reshape(-1, 1)
            self.delay_smooths = copy.copy(self.delay_fluctuations)
            # iterate over each antenna
            for anti in range(self.Nants):
                # get ydata
                ydata = copy.copy(self.delay_fluctuations[anti, :, :])
                # scale by std
                ystd = np.sqrt([astats.biweight_midvariance(ydata[~delay_flags[anti, :, ip], ip])
                                for ip in range(self.Npols)])
                ydata /= ystd
                GP = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                for pol_cnt in range(self.Npols):
                    if np.all(np.isfinite(ydata[..., pol_cnt])):
                        # fit GP and remove from delay fluctuations but only one polarization at a time
                        GP.fit(xdata, ydata[..., pol_cnt])
                        ymodel = (GP.predict(xdata) * ystd[pol_cnt])
                        self.delay_fluctuations[anti, :, pol_cnt] -= ymodel
                        self.delay_smooths[anti, :, pol_cnt] = ymodel

    def run_metrics(self, std_cut=0.5):
        """Compute all metrics and save to dictionary.

        Run all metrics, put them in "metrics" dictionary, and attach metrics to
        the object instance.

        Parameters
        ----------
        std_cut : float, optional
            Delay standard deviation for determining good or bad. Default is 0.5.

        Results
        -------
        metrics : dict
            A dictionary with keys: "good_sol", "bad_ants", "z_scores",
            "ant_z_scores".

            "good_sol" is a boolean, and is a statement of the goodness
            of a full FirstCal solution. It is True if the aggregate
            standard deviation is less than std_cut. Otherwise it is False.

            "bad_ants" is a list of bad antennas that don't meet the tolerance
            set by std_cut.

            "z_scores" is a dictionary that contains the z-score for each
            antenna and time-stamp with respect to the standard deviation of
            all antennas and all times.

            "ant_z_scores" is a dictionary that contains the z-score for each
            antenna with respect to the standard deviation of all antennas,
            which is then averaged over time.

        """
        full_metrics = OrderedDict()
        for pol_ind, pol in enumerate(self.pols):
            # Calculate std and zscores
            (ant_avg, ant_std, time_std, agg_std, max_std,
             z_scores, ant_z_scores) = self.delay_std(pol_ind=pol_ind, return_dict=True)

            # Given delay standard dev. cut, find "bad" ants
            # also determine if full solultion is bad
            if max_std > std_cut:
                good_sol = False
            else:
                good_sol = True

            bad_ants = []
            for ant in ant_std:
                if ant_std[ant] > std_cut:
                    bad_ants.append(ant)

            # put into dictionary
            metrics = OrderedDict()
            metrics['good_sol'] = bool(good_sol)
            metrics['ants'] = self.ants
            metrics['bad_ants'] = bad_ants
            metrics['z_scores'] = z_scores
            metrics['ant_z_scores'] = ant_z_scores
            metrics['ant_avg'] = ant_avg
            metrics['ant_std'] = ant_std
            metrics['time_std'] = time_std
            metrics['agg_std'] = agg_std
            metrics['max_std'] = max_std
            metrics['times'] = self.times
            metrics['version'] = self.version_str
            metrics['fc_filename'] = self.fc_filename
            metrics['fc_filestem'] = self.fc_filestem
            metrics['start_JD'] = self.start_JD
            metrics['frac_JD'] = self.frac_JD
            metrics['std_cut'] = std_cut
            metrics['pol'] = self.pols[pol_ind]
            metrics['rot_ants'] = self.rot_ants

            if self.history != '':
                metrics['history'] = self.history

            full_metrics[pol] = metrics

        self.metrics = full_metrics

    def write_metrics(self, filename=None, filetype='h5', overwrite=False):
        """
        Write metrics to file after running run_metrics().

        Parameters
        ----------
        filename : str, optional
            The full filename to write without filetype suffix. If not provided,
            self.fc_filestem is used.
        filetype : {"h5", "hdf5", "json", "pkl"}, optional
            The filetype to save the metrics as. "h5" and "hdf5" save an HDF5
            filetype, "json" saves JSON, and "pkl" saves a python pickle. "json"
            and "pkl" are deprecated, and should not be used. Default is "h5".
        overwrite : bool, optional
            If True, overwrite the file if it exists.

        """
        if filename is None:
            filename = self.fc_filestem + ".first_metrics"
        if filetype == 'pkl':
            if filename.split('.')[-1] != 'pkl':
                filename += '.pkl'
        elif filetype == 'json':
            if filename.split('.')[-1] != 'json':
                filename += '.json'
        elif filetype in ['h5', 'hdf5']:
            if filename.split('.')[-1] not in ['hdf5', 'h5']:
                filename += '.hdf5'
        else:
            raise ValueError("Output filetype is not an accepted value. "
                             "Allowed values are: ['json', 'pkl', 'h5', 'hdf5'] "
                             "Received value: {0}".format(filetype))
        metrics_io.write_metric_file(filename=filename,
                                     input_dict=self.metrics,
                                     overwrite=overwrite)

    def load_metrics(self, filename):
        """
        Read-in a firstcal_metrics file and append to class.

        Parameters
        ----------
        filename : str
            Full path to the filename to read in.

        Results
        -------
        metrics : dict
            Dictionary containing the metrics saved in the file.

        """
        self.metrics = load_firstcal_metrics(filename)

    def delay_std(self, pol_ind, return_dict=False):
        """
        Calculate standard deviations of per-antenna delay solutions.

        Calculate standard deviations of per-antenna delay solutions and aggregate
        delay solutions. Assign a z-score for each (antenna, time) delay solution
        with respect to the aggregate mean and standard deviation.


        Note
        ----
        This method Uses the square root of the result of astropy.stats.biweight_midvariance
        for a robust measure of the standard deviation (i.e., it is not greatly influenced
        by outliers).

        Parameters
        ----------
        return_dict : bool, optional
            If True, return time_std, ant_std and z_scores as a dictionary with
            "ant" as keys and "times" as keys. If False, return these values
            as ndarrays. Default is False

        Returns
        -------
        ant_avg : ndarray, shape=(Nants,)
            The average delay solution across time for each antenna.
        ant_std : ndarray, shape=(Nants,)
            The standard deviation of delay solutions across time for each antenna.
        time_std : ndarray, shape=(Ntimes,)
            The standard deviation of delay solutions across ants for each time-stamp.
        agg_std : float
            The aggregate standard deviation of delay solutions across all antennas and
            all times.
        max_std : float
            The maximum antenna standard deviation of delay solutions.
        z_scores : ndarray, shape=(Nants, Ntimes)
            The z-score (standard score) for each (antenna, time) delay solution with
            respect to the aggregate standard deviation.
        ant_z_scores : ndarray, shape=(Nants,)
            The absolute value of z-scores for each antenna & time, which is then
            averaged over time.

        """
        # calculate standard deviations, ignoring antenna-times flagged for all freqs
        delay_flags = np.all(self.UVC.flag_array, axis=(1, 2))[:, :, pol_ind]
        ant_avg = self.delay_avgs[:, :, pol_ind]
        ant_avg = self.delay_avgs[:, :, pol_ind]
        ant_std = np.sqrt([astats.biweight_midvariance((self.delay_fluctuations[i, :, pol_ind])[~delay_flags[i, :]])
                           for i in range(self.Nants)])
        ant_std[~np.isfinite(ant_std)] = 0.0
        time_std = np.sqrt([astats.biweight_midvariance((self.delay_fluctuations[:, i, pol_ind])[~delay_flags[:, i]])
                           for i in range(self.Ntimes)])
        time_std[~np.isfinite(time_std)] = 0.0
        agg_std = np.sqrt(astats.biweight_midvariance((self.delay_fluctuations[:, :, pol_ind])[~delay_flags]))
        if not np.isfinite(agg_std):
            agg_std = 0.0
        max_std = np.max(ant_std)

        # calculate z-scores
        z_scores = self.delay_fluctuations[:, :, pol_ind] / agg_std
        ant_z_scores = np.median(np.abs(z_scores), axis=1)

        # convert to ordered dict if desired
        if return_dict is True:
            ant_avg_d = OrderedDict()
            time_std_d = OrderedDict()
            ant_std_d = OrderedDict()
            z_scores_d = OrderedDict()
            ant_z_scores_d = OrderedDict()
            for i, ant in enumerate(self.ants):
                ant_avg_d[ant] = ant_avg[i]
                ant_std_d[ant] = ant_std[i]
                z_scores_d[ant] = z_scores[i]
                ant_z_scores_d[ant] = ant_z_scores[i]
            for i, t in enumerate(self.times):
                # casting the time as a string here so that there is
                # a lower chance to lose JD information from truncation
                # or rounding.
                # We choose a 7 decimal precision to give ~10ms precision
                # in the JDs
                time_str = "{0:.7f}".format(t)
                time_std_d[time_str] = time_std[i]

            ant_avg = ant_avg_d
            time_std = time_std_d
            ant_std = ant_std_d
            z_scores = z_scores_d
            ant_z_scores = ant_z_scores_d

        return ant_avg, ant_std, time_std, agg_std, max_std, z_scores, ant_z_scores

    def plot_delays(self, ants=None, plot_type='both', cmap='nipy_spectral', ax=None, save=False, fname=None,
                    plt_kwargs={'markersize': 5, 'alpha': 0.75}):
        """Plot delay solutions from a calfits file.

        This method plots either:
            1. per-antenna delay solution in nanosec
            2. per-antenna delay solution subtracting per-ant median
            3. both


        Parameters
        ----------
        ants : list, optional
            List specifying which antennas to plot. If None, all antennas are plotted.
        plot_type : {"solution", "fluctuation", "both"}, optional
            Specify which type of plot to make. "solution" means plotting the full delay
            solution. "fluctuation" plots just the deviations from the average. "both"
            will make a plot of both types. Default is "both"
        ax : list, optional
            A list containing matplotlib axis objects on which to make plots.
            If None, the method will create new figures and axes as needed. If this
            is not None, the list must contain enough subplots given the type specified
            in plot_type, plus one additional axis for the legend at the end.
        cmap : str, optional
            The colormap for different antennas. Default is "nipy_spectral".
        save : bool, optional
            If True, save the plot as a png file. Note that this only works if the figure
            is generated inside of the function (i.e., if ax==None). Default is False.
        fname : str, optional
            Full path to the filename to save the plot as. Default is self.fc_filestem.
        plt_kwargs : dict, optional
            Keyword arguments for ax.plot() calls. Default is {"markersize": 5, "alpha": 0.75}.

        Returns
        -------
        fig : matplotlib figure object
            If ax is not specified, the figure generated inside the method is returned.

        """
        import matplotlib.pyplot as plt
        # Init figure and ax if needed
        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8, 8))
            self.fig = fig
            fig.subplots_adjust(hspace=0.3)
            if plot_type == 'both':
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                ax = [ax1, ax2]
            else:
                ax = fig.add_subplot(111)

        # match ants fed to true ants
        if ants is not None:
            plot_ants = [np.where(self.ants == ant)[0][0] for ant in ants if ant in self.ants]
        else:
            plot_ants = range(self.Nants)

        # Get a colormap
        try:
            cm_func = plt.get_cmap(cmap)
            cm = cm_func(np.linspace(0, 0.95, len(plot_ants)))
        except ValueError:
            print("cmap not recognized, using spectral")
            cm_func = plt.get_cmap('nipy_spectral')
            cm = cm_func(np.linspace(0, 0.95, len(plot_ants)))

        # plot delay solutions
        if (plot_type == 'both') or (plot_type == 'solution'):
            if plot_type == 'both':
                axes = ax
                ax = axes[0]
            plabel = []
            ax.grid(True, zorder=0)
            for color, delay in zip(cm, self.delays[plot_ants]):
                pl, = ax.plot(self.frac_JD, delay, marker='.',
                              c=color, **plt_kwargs)
                plabel.append(pl)
            ax.set_xlabel('fraction of JD %d' % self.start_JD, fontsize=14)
            ax.set_ylabel('delay solution [ns]', fontsize=14)
            if plot_type == 'both':
                ax = axes

        # plot delay fluctuation
        if (plot_type == 'both') or (plot_type == 'fluctuation'):
            if plot_type == 'both':
                axes = ax
                ax = axes[1]
            plabel = []
            ax.grid(True, zorder=0)
            for color, delay_f in zip(cm, self.delay_fluctuations[plot_ants]):
                pl, = ax.plot(self.frac_JD, delay_f,
                              marker='.', c=color, **plt_kwargs)
                plabel.append(pl)
            ax.set_xlabel('fraction of JD %d' % self.start_JD, fontsize=14)
            ax.set_ylabel('delay fluctuation [ns]', fontsize=14)
            if plot_type == 'both':
                ax = axes

        # add legend
        if custom_ax is False:
            ax = fig.add_axes([1.0, 0.1, 0.05, 0.8])
            ax.axis('off')
            ax.legend(plabel, self.ants[plot_ants])
        else:
            ax[-1].legend(plabel, self.ants[plot_ants])

        if save is True and custom_ax is False:
            if fname is None:
                fname = self.fc_filestem + '.dlys.png'
            fig.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig

    def plot_zscores(self, fname=None, plot_type='full', pol=None, ax=None, figsize=(10, 6),
                     save=False, kwargs={'cmap': 'Spectral'}, plot_abs=False):
        """Plot z_scores for antenna delay solutions.

        Parameters
        ----------
        fname : str, optional
            Full path to the output filename.
        plot_type : {"full", "time_avg"}, optional
            Type of plot to make. "full" plots the z-score for each entry for
            antennas and times. "time_avg" plots the z-score for each antenna
            averaged over time. Default is "full".
        pol : str, optional
            Polarization to plot. Default is the first one saved in the metrics object.
        ax : matplotlib axis object, optional
            Matplotlib axis object to add plot to. If not specified, a new axis object
            is created.
        figsize : tuple, optional
            The figsize to use when creating a new figure (i.e., if ax is None). Default
            is (10, 6).
        save : bool, optional
            If True, save figure to fname. Default is False.
        kwargs : dict, optional
            Keyword arguments to the plot_zscores function. Default is {"cmap": "Spectral"}.
        plot_abs : bool, optional
            If True, plot the absolue value of the z-scored. Default is False.

        """
        # make sure metrics has been run
        if hasattr(self, 'metrics') is False:
            raise NameError("You need to run FirstCalMetrics.run_metrics() "
                            + "in order to plot delay z_scores")
        if pol is None:
            pol = list(self.metrics.keys())[0]
        fig = plot_zscores(self.metrics[pol], fname=fname, plot_type=plot_type, ax=ax, figsize=figsize,
                           save=save, kwargs=kwargs, plot_abs=plot_abs)
        return fig

    def plot_stds(self, fname=None, pol=None, ax=None, xaxis='ant', kwargs={}, save=False):
        """Plot standard deviation of delay solutions per-ant or per-time.

        Parameters
        ----------
        fname : str, optional
            Full path to the output filename.
        pol : str, optional
            Polarization to plot. Default is the first one saved in the metrics object.
        xaxis : {"ant", "time"}, optional
            What to plot on the x-axis. Must be one of: "ant", "time". These are
            antenna number or time-stamp respectively. Default is "ant".
        ax : matplotlib axis object, optional
            Matplotlib axis object to add plot to. If not specified, a new axis object
            is created.
        kwargs : dict, optional
            Plotting keyword arguments. Default is empty dict.
        save : bool, optional
            If True, save the image to fname. Default is False.

        """
        # make sure metrics has been run
        if hasattr(self, 'metrics') is False:
            raise NameError("You need to run FirstCalMetrics.run_metrics() "
                            + "in order to plot delay stds")
        if pol is None:
            pol = list(self.metrics.keys())[0]
        fig = plot_stds(self.metrics[pol], fname=fname, ax=ax, xaxis=xaxis, kwargs=kwargs, save=save)
        return fig


# code for running firstcal_metrics on a file
def firstcal_metrics_run(files, args, history):
    """
    Run firstcal_metrics tests on a given set of input files.

    This function will take in a list of files and options. It will run the firstcal
    metrics and produce a JSON file containing the relevant information.

    Parameters
    ----------
    files : list of str
        A list of files to run firstcal metrics on.
    args : argparse.Namespace
        The parsed command-line arguments generated via argparse.ArgumentParser.parse_args.

    """
    if len(files) == 0:
        raise AssertionError('Please provide a list of calfits files')

    for filename in files:
        fm = FirstCalMetrics(filename)
        fm.run_metrics(std_cut=args.std_cut)

        # add history
        fm.history = fm.history + history

        abspath = os.path.abspath(filename)
        dirname = os.path.dirname(abspath)
        if args.metrics_path == '':
            # default path is same directory as file
            metrics_path = dirname
        else:
            metrics_path = args.metrics_path
        print(metrics_path)
        metrics_basename = utils.strip_extension(os.path.basename(filename)) + args.extension
        # Sometimes the firstcal output has an extra '.first.' in the name
        # the replace attempts to remove this but does nothing if it does not find `.first.` in the string
        metrics_filename = os.path.join(metrics_path, metrics_basename).replace('.first.', '.')
        fm.write_metrics(filename=metrics_filename, filetype=args.filetype,
                         overwrite=args.clobber)
