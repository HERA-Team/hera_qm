"""
FirstCal metrics
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
try:
    from sklearn import gaussian_process as gp
    sklearn_import = True
except ImportError:
    print "could not import sklearn"
    sklearn_import = False


def get_firstcal_metrics_dict():
    """ Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns:
    metrics_dict -- Dictionary with metric names as keys and descriptions as values.
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
    Read-in a firstcal_metrics file and return dictionary

    Input:
    ------
    filename : str
        filename to read in

    Returns:
    --------
    metrics : dict
        dictionary holding metrics
    """
    # get filetype
    filetype = filename.split('.')[-1]

    # load json
    if filetype == 'json':
        with open(filename, 'r') as f:
            metrics = json.load(f, object_pairs_hook=OrderedDict)

        # ensure keys of dicts are not strings
        for i in metrics['ant_avg'].keys():
            metrics['ant_avg'][int(i)] = metrics['ant_avg'].pop(i)
        for i in metrics['ant_std'].keys():
            metrics['ant_std'][int(i)] = metrics['ant_std'].pop(i)
        for i in metrics['time_std'].keys():
            metrics['time_std'][float(i)] = metrics['time_std'].pop(i)
        for i in metrics['z_scores'].keys():
            metrics['z_scores'][int(i)] = metrics['z_scores'].pop(i)
        for i in metrics['ant_z_scores'].keys():
            metrics['ant_z_scores'][int(i)] = metrics['ant_z_scores'].pop(i)

    # load pickle
    elif filetype == 'pkl':
        with open(filename, 'rb') as f:
            inp = pkl.Unpickler(f)
            metrics = inp.load()
    else:
        raise IOError("Filetype not recognized, try a json or pkl file")

    return metrics


def plot_stds(metrics, fname=None, ax=None, xaxis='ant', kwargs={}, save=False):
    """
    Plot standard deviation of delay solutions per-ant or per-time

    Input:
    ------

    metrics : dictionary
        a "metrics" dictionary from FirstCal_Metrics.run_metrics()

    fname : str, default=None
        filename

    xaxis : str, default='ant', option=['ant', 'time']
        what to plot on the xaxis, antennas or time stamp

    ax : axis object, default=None
        matplotlib axis object

    kwargs : dict
        plotting kwargs

    save : bool, default=False
        save image to file

    """
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

    # choose xaxis
    if xaxis == 'ant':
        Nants = len(metrics['ants'])
        xax = range(Nants)
        yax = metrics['ant_std'].values()
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
        [t.set_rotation(20) for t in ax.get_xticklabels()]
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

    elif xaxis == 'time':
        xax = metrics['frac_JD']
        yax = metrics['time_std'].values()
        ax.grid(True, zorder=0)
        ax.tick_params(size=8)
        ax.plot(xax, yax, c='k', marker='.', linestyle='-', alpha=0.85, zorder=1)
        [t.set_rotation(20) for t in ax.get_xticklabels()]
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
    Plot z_scores for antenna delay solution

    Input:
    ------

    metrics : dict
        a FirstCal_Metrics "metrics" dictionary

    fname : str, default=None
        filename

    plot_type : str, default='full'
        Type of plot to make
        'full' : plot zscore for each (N_ant, N_times)
        'time_avg' : plot zscore for each (N_ant,) avg over time

    ax : axis object, default=None
        matplotlib axis object

    figsize : tuple, default=(10,6)
        figsize if creating figure

    save : bool, default=False
        save figure to file

    kwargs : dict
        plotting kwargs
    """
    # Get ax if not provided
    custom_ax = True
    if ax is None:
        custom_ax = False
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    # unpack some variables
    z_scores = np.array(metrics['z_scores'].values())
    ant_z_scores = np.array(metrics['ant_z_scores'].values())
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
        cmap_func = plt.get_cmap(kwargs['cmap'])
        cmap = cmap_func(np.linspace(0, 0.95, Nants))
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

class FirstCal_Metrics(object):
    """
    FirstCal_Metrics class for holding firstcal data,
    running metrics, and plotting delay solutions.
    Currently only supports single polarization solutions.
    """

    # sklearn import statement
    sklearn_import = sklearn_import

    def __init__(self, calfits_files, use_gp=True):
        """
        Input:
        ------
        calfits_files : str or list
            filename for a *.first.calfits file
            or a list of .first.calfits files (time-ordered)
            of the same polarization

        use_gp : bool, default=True
            use gaussian process model to 
            subtract underlying smooth delay solution
            behavior over time from fluctuations

        Result:
        -------
        self.UVC : pyuvdata.UVCal() instance

        self.delays : ndarray, shape=(N_ant, N_times)
            firstcal delay solutions in nanoseconds

        self.delay_avgs : ndarray, shape=(N_ant,)
            median delay solutions across time [nanosec]

        self.delay_fluctuations : ndarray, shape=(N_ant, N_times)
            firstcal delay solution fluctuations from time average [nanosec]

        self.frac_JD : ndarray, shape=(N_times,)
            ndarray containing time-stamps of each integration
            in units of the fraction of current JD
            i.e. 2457966.53433 -> 0.53433

        self.ants : ndarray, shape=(N_ants,)
            ndarray containing antenna numbers

        self.pol : str
            Polarization, 'y' or 'x' currently supported
        """
        # Instantiate UVCal and read calfits
        self.UVC = UVCal()
        self.UVC.read_calfits(calfits_files)

        if len(self.UVC.jones_array) > 1:
            raise ValueError('Sorry, only single pol firstcal solutions are '
                             'currently supported.')
        pol_dict = {-5: 'x', -6: 'y'}
        try:
            self.pol = pol_dict[self.UVC.jones_array[0]]
        except KeyError:
            raise ValueError('Sorry, only calibration polarizations "x" and '
                             '"y" are currently supported.')

        # get file prefix
        if type(calfits_files) is list:
            calfits_file = calfits_files[0]
        else:
            calfits_file = calfits_files
        self.fc_basename = os.path.basename(calfits_file)
        self.fc_filename = calfits_file
        self.fc_filestem = '.'.join(self.fc_filename.split('.')[:-1])

        # get other relevant arrays
        self.times = self.UVC.time_array
        self.start_JD = np.floor(self.times).min()
        self.frac_JD = self.times - self.start_JD
        self.minutes = 24 * 60 * (self.frac_JD - self.frac_JD.min())
        self.Nants = self.UVC.Nants_data
        self.ants = self.UVC.ant_array
        self.version_str = hera_qm_version_str
        self.history = ''

        # Get the firstcal delays and/or gains and/or rotated antennas
        if self.UVC.cal_type == 'gain':
            # get delays
            freqs = self.UVC.freq_array.squeeze()
            fc_gains = np.moveaxis(self.UVC.gain_array, 2, 3)[:, 0, :, :, 0]
            fc_phi = np.unwrap(np.angle(fc_gains))
            d_nu = np.median(np.diff(freqs))
            d_phi = np.median(fc_phi[:, :, 1:] - fc_phi[:, :, :-1], axis=2)
            gain_slope = (d_phi / d_nu)
            self.delays = gain_slope / (-2*np.pi)
            self.gains = fc_gains

            # get delay offsets at nu = 0 Hz, and then get rotated antennas
            self.offsets = fc_phi[:, :, 0] - gain_slope * freqs[0]
            self.rot_ants = np.unique(map(lambda x: self.ants[x], (np.isclose(np.pi, np.abs(self.offsets) % (2 * np.pi), atol=1.0)).T)).tolist()

        elif self.UVC.cal_type == 'delay':
            self.delays = self.UVC.delay_array.squeeze()
            self.gains = None
            self.offsets = None
            self.rot_ants = []

        # Calculate avg delay solution and subtract to get delay_fluctuations
        self.delays = self.delays * 1e9
        self.delay_avgs = np.median(self.delays, axis=1)
        self.delay_fluctuations = (self.delays.T - self.delay_avgs).T

        # use gaussian process model to subtract underlying mean function
        if use_gp is True and self.sklearn_import is True:
	    # initialize GP kernel.		
	    # RBF is a squared exponential kernel with a minimum length_scale_bound of 0.01 JD, meaning		
	    # the GP solution won't have time fluctuations quicker than ~0.01 JD, which will preserve 		
	    # short time fluctuations. WhiteKernel is a Gaussian white noise component with a fiducial		
	    # noise level of 0.01 nanoseconds. Both of these are hyperparameters that are fit for via		
	    # a gradient descent algorithm in the GP.fit() routine, so length_scale=0.2 and		
	    # noise_level=0.01 are just initial conditions and are not the final hyperparameter solution
            kernel = gp.kernels.RBF(length_scale=0.2, length_scale_bounds=(0.01, 1.0)) + gp.kernels.WhiteKernel(noise_level=0.01)
            x = self.frac_JD.reshape(-1, 1)
            self.delay_smooths = []
            # iterate over each antenna
            for i in range(self.Nants):
                # get ydata
                y = copy.copy(self.delay_fluctuations[i])
                # scale by std
                ystd = np.sqrt(astats.biweight_midvariance(y))
                y /= ystd
                # fit GP and remove from delay fluctuations
                GP = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
                GP.fit(x, y)
                ymodel = GP.predict(x).ravel() * ystd
                self.delay_fluctuations[i] -= ymodel
                self.delay_smooths.append(ymodel)
            self.delay_smooths = np.array(self.delay_smooths)


    def run_metrics(self, std_cut=0.5):
        """
        Run all metrics, put them in "metrics" dictionary
        and attach metrics to class

        Input:
        ------
        std_cut : float, default=0.5
            delay stand. dev cut for good / bad determination

        Result:
        -------
        Create a self.metrics dictionary containing:

        good_sol : bool
            Statement on goodness of full FirstCal solution,
            determined True if aggregate stand. dev is < std_cut
            otherwise False

        bad_ants : list
            list of bad antennas that don't meet std_cut tolerance

        z_scores : dictionary
            contains z_score for each antenna and each time-stamp
            w.r.t. standard deviation of all antennas and all times

        ant_z_scores : dictionary
            contains z_score for each antenna w.r.t stand dev. of
            all antennas, then averaged over time
        """
        # Calculate std and zscores
        (ant_avg, ant_std, time_std, agg_std, max_std,
         z_scores, ant_z_scores) = self.delay_std(return_dict=True)

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
        metrics['good_sol'] = good_sol
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
        metrics['pol'] = self.pol
        metrics['rot_ants'] = self.rot_ants

        if self.history != '':
            metrics['history'] = self.history

        self.metrics = metrics

    def write_metrics(self, filename=None, filetype='json'):
        """
        Write metrics to file after running run_metrics()

        filename : str, default=None
            filename without filetype suffix
            if None, use default filename stem

        filetype : str, default='json'
            filetype to write out to
            options = ['json', 'pkl']

        """
        # get filename prefix
        if filename is None:
            filename = self.fc_filestem + ".first_metrics"

        # write to file
        if filetype == 'json':
            if filename.split('.')[-1] != 'json':
                filename += '.json'
            # change ndarrays to lists
            metrics_out = copy.deepcopy(self.metrics)
            metrics_out['frac_JD'] = list(metrics_out['frac_JD'])
            metrics_out['times'] = list(metrics_out['times'])
            metrics_out['ants'] = list(metrics_out['ants'])
            metrics_out['rot_ants'] = list(metrics_out['rot_ants'])
            for k in metrics_out['z_scores'].keys():
                metrics_out['z_scores'][k] = list(metrics_out['z_scores'][k])

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
        Read-in a firstcal_metrics file and append to class

        Input:
        ------
        filename : str
            filename to read in

        Result:
        -------
        self.metrics dictionary
        """
        self.metrics = load_firstcal_metrics(filename)

    def delay_std(self, return_dict=False):
        """
        Calculate standard deviations of per-antenna delay solutions
        and aggregate delay solutions. Assign a z-score for
        each (antenna, time) delay solution w.r.t. aggregate
        mean and standard deviation.

        Uses sqrt( astropy.stats.biweight_midvariance ) for a robust measure
        of the std

        Input:
        ------

        return_dict : bool, [default=False]
            if True, return time_std, ant_std and z_scores
            as a dictionary with "ant" as keys
            and "times" as keys
            else, return as ndarrays

        Output:
        --------
        return ant_avg, ant_std, time_std, agg_std, max_std, z_scores

        ant_avg : ndarray, shape=(N_ants,)
            average delay solution across time for each antenna

        ant_std : ndarray, shape=(N_ants,)
            standard deviation of delay solutions across time
            for each antenna

        time_std : ndarray, shape=(N_times,)
            standard deviation of delay solutions across ants
            for each time-stamp

        agg_std : float
            aggregate standard deviation of delay solutions
            across all antennas and all times

	max_std : float
            maximum antenna standard deviation of delay solutions

        z_scores : ndarray, shape=(N_ant, N_times)
            z_scores (standard scores) for each (antenna, time)
            delay solution w.r.t. agg_std

        ant_z_scores : ndarray, shape=(N_ant,)
            absolute_value(z_scores) for each antenna & time
            then averaged over time

        """
        # calculate standard deviations
        ant_avg = self.delay_avgs
        ant_std = np.sqrt(astats.biweight_midvariance(self.delay_fluctuations, axis=1))
        time_std = np.sqrt(astats.biweight_midvariance(self.delay_fluctuations, axis=0))
        agg_std = np.sqrt(astats.biweight_midvariance(self.delay_fluctuations))
        max_std = np.max(ant_std)

        # calculate z-scores
        z_scores = self.delay_fluctuations / agg_std
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
                time_std_d[t] = time_std[i]

            ant_avg = ant_avg_d
            time_std = time_std_d
            ant_std = ant_std_d
            z_scores = z_scores_d
            ant_z_scores = ant_z_scores_d

        return ant_avg, ant_std, time_std, agg_std, max_std, z_scores, ant_z_scores

    def plot_delays(self, ants=None, plot_type='both', cmap='nipy_spectral', ax=None, save=False, fname=None,
                    plt_kwargs={'markersize': 5, 'alpha': 0.75}):
        """
        plot delay solutions from a calfits file
        plot either
            1. per-antenna delay solution in nanosec
            2. per-antenna delay solution subtracting per-ant median
            3. both

        Input:
        ------

        ants : list, [default=None]
            specify which antennas to plot.
            will plot all by default

        plot_type : str, [default='both']
            specify which type of plot to make
            'solution' for full delay solution
            'fluctuation' for just flucutations from avg
            'both' for both

        ax : list, [default=None]
            list containing matplotlib axis objects
            to make plots in.
            if None, will create a figure and axes by default.
            if not None, ax must contain enough subplots
            given specification of plot_type plus one
            more axis for a legend at the end

        cmap : str, [default='spectral']
            colormap for different antennas

        save : bool, [default=False]
            if True save plot as png
            only works if fig is defined in function
            i.e. if ax == None

        fname : str, [default=self.fc_filestem+'.png']
            filename to save plot as
            default is self.fc_filestem

        plt_kwargs : dict, [default={'markersize':8,'alpha':0.75}]
            keyword arguments for ax.plot() calls
            other than "c" and "marker" which are
            already defined
        """
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
            for i, index in enumerate(plot_ants):
                p, = ax.plot(self.frac_JD, self.delays[index], marker='.',
                             c=cm[i], **plt_kwargs)
                plabel.append(p)
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
            for i, index in enumerate(plot_ants):
                p, = ax.plot(self.frac_JD, self.delay_fluctuations[index],
                             marker='.', c=cm[i], **plt_kwargs)
                plabel.append(p)
            ax.set_xlabel('fraction of JD %d' % self.start_JD, fontsize=14)
            ax.set_ylabel('delay fluctuation [ns]', fontsize=14)
            if plot_type == 'both':
                ax = axes

        # add legend
        if custom_ax is False:
            ax = fig.add_axes([1.0, 0.1, 0.05, 0.8])
            ax.axis('off')
            ax.legend(plabel, [self.ants[i] for i in plot_ants])
        else:
            ax[-1].legend(plabel, [self.ants[i] for i in plot_ants])

        if save is True and custom_ax is False:
            if fname is None:
                fname = self.fc_filestem + '.dlys.png'
            fig.savefig(fname, bbox_inches='tight')

        if custom_ax is False:
            return fig

    def plot_zscores(self, fname=None, plot_type='full', ax=None, figsize=(10, 6),
                     save=False, kwargs={'cmap': 'Spectral'}, plot_abs=False):
        """
        Plot z_scores for antenna delay solution

        Input:
        ------

        fname : str, default=None
            filename

        plot_type : str, default='full'
            Type of plot to make
            'full' : plot zscore for each (N_ant, N_times)
            'time_avg' : plot zscore for each (N_ant,) avg over time

        ax : axis object, default=None
            matplotlib axis object

        figsize : tuple, default=(10,6)
            figsize if creating figure

        save : bool, default=False
            save figure to file

        kwargs : dict
            plotting kwargs
        """
        # make sure metrics has been run
        if hasattr(self, 'metrics') == False:
            raise NameError("You need to run FirstCal_Metrics.run_metrics() " +
                            "in order to plot delay z_scores")
        fig = plot_zscores(self.metrics, fname=fname, plot_type=plot_type, ax=ax, figsize=figsize,
                     save=save, kwargs=kwargs, plot_abs=plot_abs)
        return fig

    def plot_stds(self, fname=None, ax=None, xaxis='ant', kwargs={}, save=False):
        """
        Plot standard deviation of delay solutions per-ant or per-time

        Input:
        ------

        fname : str, default=None
            filename

        xaxis : str, default='ant', option=['ant', 'time']
            what to plot on the xaxis, antennas or time stamp

        ax : axis object, default=None
            matplotlib axis object

        kwargs : dict
            plotting kwargs

        save : bool, default=False
            save image to file

        """
        # make sure metrics has been run
        if hasattr(self, 'metrics') == False:
            raise NameError("You need to run FirstCal_Metrics.run_metrics() " +
                            "in order to plot delay stds")
        fig = plot_stds(self.metrics, fname=fname, ax=ax, xaxis=xaxis, kwargs=kwargs, save=save)
        return fig


# code for running firstcal_metrics on a file
def firstcal_metrics_run(files, args, history):
    """
    Run firstcal_metrics tests on a given set of input files.

    Args:
        files -- a list of files to run firstcal metrics on.
        args -- parsed arguments via argparse.ArgumentParser.parse_args
    Return:
        None

    The function will take in a list of files and options. It will run the firstcal
    metrics and produce a JSON file containing the relevant information.
    """
    if len(files) == 0:
        raise AssertionError('Please provide a list of calfits files')

    for i, filename in enumerate(files):
        fm = FirstCal_Metrics(filename)
        fm.run_metrics(std_cut=args.std_cut)
            
        # add history
        fm.history = fm.history + history

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
        fm.write_metrics(filename=metrics_filename)
