"""
FirstCal metrics

"""
import numpy as np
import os
from pyuvdata import UVData, UVCal
import hera_cal as hc
import matplotlib.pyplot as plt
import matplotlib
import astropy.stats as astats
from collections import OrderedDict
import json
import cPickle as pkl


class FirstCal_Metrics(object):
    """
    FirstCal_Metrics class for holding firstcal data,
    running metrics, and plotting delay solutions

    FirstCal_Metrics currently only works on a 
    per-file basis, as pyuvadata.UVCal also only
    works on a per-file basis. It also only works
    with files that have more than a few time integrations,
    otherwise the standard deviation metric is not well-defined.
    """

    def __init__(self, calfits_file):
        """
        Input:
        ------
        calfits_file : str
            filename for a *.first.calfits file

        Result:
        -------
        self.UVC : pyuvdata.UVCal() instance

        self.delsys : ndarray, shape=(N_ant, N_times)
            firstcal delay solutions

        self.delay_offsets : ndarray, shape=(N_ant, N_times)
            firstcal delay solution offsets from time average

        self.frac_JD : ndarray, shape=(N_times,)
            ndarray containing time-stamps of each integration
            in fraction of current JD

        self.ants : ndarray, shape=(N_ants,)
            ndarray containing antenna numbers
        """

        # Instantiate UVCal and read calfits
        self.UVC = UVCal()
        self.UVC.read_calfits(calfits_file)

        # get file prefix
        self.file_stem = '.'.join(calfits_file.split('.')[:-1])

        # Calculate median delay
        self.delays = self.UVC.delay_array.squeeze() * 1e9
        self.delay_avgs = np.median(self.delays, axis=1)
        self.delay_offsets = (self.delays.T - self.delay_avgs).T

        # get other relevant arrays
        self.times = self.UVC.time_array
        self.start_JD = np.floor(self.times).min()
        self.frac_JD = self.times - self.start_JD
        self.minutes = 24 * 60 * (self.frac_JD - self.frac_JD.min())
        self.Nants = self.UVC.Nants_data
        self.ants = self.UVC.ant_array

    def run_metrics(self, std_cut=0.5, output=False):
        """
        Run all metrics and write results to file

        filename : str, default=None
            filename for output w/o filetype suffix
            if None, uses calfile stem

        std_cut : float, default=0.5
            delay stand. dev cut for good / bad determination

        output : bool, default=False
            return with function

        filetype : str, default='pkl'
            output filetype
            options = ['pkl', 'json']

        write : bool, default=True
            write output file or not

        Output:
        -------
        Create a self.result dictionary
        if output == True:
            return self.result

        """
        # Calculate std and zscores
        self.ant_std, self.time_std, self.agg_std, self.z_scores = self.delay_std(return_dict=True)

        # Given delay std cut find "bad" ant
        # determine if full sol is bad
        if self.agg_std > std_cut:
            self.full_sol = 'bad'
        else:
            self.full_sol = 'good'

        self.bad_ants = []
        for ant in self.ant_std:
            if self.ant_std[ant] > std_cut:
                self.bad_ants.append(ant)

        # put into dictionary
        result = OrderedDict()
        result['full_sol'] = self.full_sol
        result['bad_ant'] = self.bad_ants
        result['z_scores'] = self.z_scores
        result['ant_std'] = self.ant_std
        result['time_std'] = self.time_std
        result['agg_std'] = self.agg_std
        result['times'] = self.times
        self.result = result

        if output == True:
            return self.result
       
    def write_metrics(self, filename=None, filetype='pkl'):
        """
        Write metrics to file after running run_metrics()

        filename : str, default=None
            filename without filetype suffix
            if None, use default filename stem

        filetype : str, default='pkl'
            filetype to write out to
            options = ['json', 'pkl']

        """
        # get filename prefix
        if filename is None:
            filename = self.file_stem + ".metrics"

        # write to file
        if filetype == 'json':
            filename +='.json'
            # change ndarrays to lists
            self.result['times'] = list(self.result['times'])
            for k in self.result['z_scores'].keys():
                self.result['z_scores'][k] = list(self.result['z_scores'][k])
                   
            result_str = json.dumps(self.result)
            with open(filename, 'w') as f:
                json.dump(result_str, f, indent=4)

        elif filetype == 'pkl':
            filename += '.pkl'
            with open(filename, 'wb') as f:
                outp = pkl.Pickler(f)
                outp.dump(self.result)

    def load_metrics(self, filename):
        """
        Read-in a firstcal_metrics file and append to class

        Input:
        ------
        filename : str
            filename to read in

        """
        # get filetype
        filetype = filename.split('.')[-1]
        if filetype == 'json':
            with open(filename, 'r') as f:
                self.result = json.load(f)

        elif filetype == 'pkl':
            with open(filename, 'rb') as f:
                inp = pkl.Unpickler(f)
                self.result = inp.load() 
        else:
            raise IOError("Filetype not recognized, try a json or pkl file")


    def delay_std(self, return_dict=False):
        """
        Calculate standard deviations of per-antenna delay solutions
        and aggregate delay solutions. Assign a z-score for
        each (antenna, time) delay solution w.r.t. aggregate
        mean and standard deviation.

        Uses astropy.stats.biweight_midvariance for a robust measure
        of the std

        Input:
        ------

        return_dict : bool, [default=False]
            if True, return time_std, ant_std and z_scores
            as a dictionary with "ant" as keys in str
            and "times" as keys in str
            else, return as ndarrays

        Output:
        --------
        return ant_std, time_std, agg_std, z_scores

        ant_std : ndarray, shape=(N_ants,)
            standard deviation of delay solutions across time
            for each antenna

        time_std : ndarray, shape=(N_times,)
            standard deviation of delay solutions across ants
            for each time-stamp

        agg_std : float
            aggregate standard deviation of delay solutions
            across all antennas

        z_scores : ndarray, shape=(N_ant, N_times)
            absolute value of z_scores (standard scores)
            for each (antenna, time) delay solution w.r.t. agg_std

        """
        # calculate standard deviations
        ant_std = astats.biweight_midvariance(self.delay_offsets, axis=1)
        time_std = astats.biweight_midvariance(self.delay_offsets, axis=0)
        agg_std = astats.biweight_midvariance(self.delay_offsets)

        # calculate z-scores
        z_scores = np.abs(self.delay_offsets / agg_std)
       
        # convert to ordered dict if desired 
        if return_dict == True:
            time_std_d = OrderedDict()
            ant_std_d = OrderedDict()
            z_scores_d = OrderedDict()
            for i, ant in enumerate(self.ants):
                ant_std_d[str(ant)] = ant_std[i]
                z_scores_d[str(ant)] = z_scores[i]
            for i, t in enumerate(self.times):
                time_std_d[str(t)] = time_std[i]

            time_std = time_std_d
            ant_std = ant_std_d
            z_scores = z_scores_d

        return ant_std, time_std, agg_std, z_scores


    def plot_delays(self, ants=None, plot_type='both', ax=None,
                    cm='spectral', save=False, fname=None,
                    plt_kwargs={'markersize':8, 'alpha':0.75}):
        """
        plot delay solutions from a calfits file
        plot either
            1. per-antenna delay solution in nanosec
            2. per-antenna delay solution offset from average
            3. both

        Input:
        ------

        ants : list, [default=None]
            specify which antennas to plot.
            will plot all by default

        plot_type : str, [default='both']
            specify which type of plot to make
            'solution' for full delay solution
            'offset' for just offset from avg
            'both' for both

        ax : list, [default=None]
            list containing matplotlib axis objects 
            to make plots in.
            if None, will create a figure and axes by default.
            if not None, ax must contain enough subplots
            given specification of plot_type plus one
            more axis for a legend at the end
         
        cm : str, [default='spectral']
            select matplotlib colormap to use

        save : bool, [default=False]
            if True save plot as png
            only works if fig is defined in function
            i.e. if ax == None

        fname : str, [default=self.file_stem+'.png']
            filename to save plot as
            default is self.file_stem
 
        plt_kwargs : dict, [default={'markersize':8,'alpha':0.75}]
            keyword arguments for ax.plot() calls
            other than "c" and "marker" which are 
            already defined 
        """
        # Init figure and ax if needed
        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8,8))
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
            plot_ants = [np.where(self.ants==ant)[0][0] for ant in ants if ant in self.ants]
        else:
            plot_ants = range(self.Nants)

        # Get a colormap
        try:
            cmap = getattr(matplotlib.cm, cm)
            cmap = cmap(np.linspace(0, 0.95, len(plot_ants)))
        except AttributeError:
            print("matplotlib colormap not recognized, using spectral")
            cmap = matplotlib.cm.spectral(np.linspace(0, 0.95, len(plot_ants)))

        # plot delay solutions
        if (plot_type == 'both') or (plot_type == 'solution'):
            if plot_type == 'both':
                axes = ax
                ax = axes[0]
            plabel = []
            ax.grid(True)
            for i, index in enumerate(plot_ants):
                p, = ax.plot(self.frac_JD, self.delays[index], marker='.',
                                c=cmap[i], **plt_kwargs)
                plabel.append(p)
            ax.set_xlabel('fraction of JD %d'%self.start_JD, fontsize=14)
            ax.set_ylabel('delay solution [ns]', fontsize=14)
            if plot_type == 'both':
                ax = axes

        # plot delay offset
        if (plot_type == 'both') or (plot_type == 'offset'):
            if plot_type == 'both':
                axes = ax
                ax = axes[1]
            plabel = []
            ax.grid(True)
            for i, index in enumerate(plot_ants):
                p, = ax.plot(self.frac_JD, self.delay_offsets[index],
                                marker='.', c=cmap[i], **plt_kwargs)
                plabel.append(p)
            ax.set_xlabel('fraction of JD %d'%self.start_JD, fontsize=14)
            ax.set_ylabel('delay offset [ns]', fontsize=14)
            if plot_type == 'both':
                ax = axes

        # add legend
        if custom_ax == False:
            ax = fig.add_axes([1.0,0.1,0.05,0.8])
            ax.axis('off')
            ax.legend(plabel, [self.ants[i] for i in plot_ants])
        else:
            ax[-1].legend(plabel, [self.ants[i] for i in plot_ants])

        if save == True and custom_ax == False:
            if fname is None:
                fname = self.file_stem + '.dlys.png'
            fig.savefig(fname, bbox_inches='tight')


    def plot_zscores(self, fname=None, plot_type='full', cm='viridis_r', ax=None, figsize=(10,6),
                        save=False,
                        kwargs={'cmap':'viridis_r','vmin':0,'vmax':5}):
        """
        Plot antenna delay solution z_scores

        Input:
        ------

        fname : str, default=None
            filename

        plot_type : str, default='full'
            Type of plot to make
            'full' : plot zscore for each (N_ant, N_times)
            'time_avg' : plot zscore for each (N_ant,) avg over time

        cm : str, default='viridis_r'
            colormap

        ax : axis object, default=None
            matplotlib axis object

        figsize : tuple, default=(10,6)
            figsize if creating figure

        save : bool, default=False
            save figure to file

        kwargs : dict
            plotting kwargs
        """

        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        # Get zscores
        ant_std, time_std, agg_std, z_scores = self.delay_std()

        # Plot zscores
        if plot_type == 'full':
            # plot
            xlen = [self.frac_JD.min(), self.frac_JD.max()]
            xlen_round = [np.ceil(self.frac_JD.min()*1000)/1000,
                                np.floor(self.frac_JD.max()*1000)/1000]
            ylen = [0, len(self.ants)]
            cax = ax.matshow(z_scores, origin='lower', aspect='auto',
                    extent=[xlen[0], xlen[1], ylen[0], ylen[1]], **kwargs)

            # define ticks
            ax.xaxis.set_ticks_position('bottom')
            xticks = np.arange(xlen_round[0], xlen_round[1]+1e-5, 0.001)
            ax.set_xticks(xticks)
            #[t.set_rotation(20) for t in ax.get_xticklabels()]
            ax.set_yticks(np.arange(len(self.ants))+0.5)
            ax.set_yticklabels(self.ants)
            [t.set_rotation(20) for t in ax.get_yticklabels()]

            # set labels
            ax.set_xlabel('fraction of JD %d'%self.start_JD, fontsize=14)
            ax.set_ylabel('antenna number', fontsize=14)

            # set colorbar
            fig.colorbar(cax, label='z-score')
           
        if plot_type == 'time_avg':
            # plot
            cmap = matplotlib.cm.spectral(np.linspace(0, 0.95, len(self.ants)))
            z_scores = np.mean(z_scores, axis=1)
            ax.grid(True)
#            ax.scatter(range(len(z_scores)), z_scores,  c=cmap, alpha=0.85,
#                            marker='o', s=70, edgecolor='None')
            ax.bar(range(len(z_scores)), z_scores, align='center', color='b', alpha=0.4)

            # define ticks
            ax.set_xlim(-1, len(self.ants))
            ax.set_ylim(0,5)

            ax.set_xticks(range(len(self.ants)))
            ax.set_xticklabels(self.ants)
            [t.set_rotation(20) for t in ax.get_xticklabels()]
             
            ax.set_xlabel('antenna number', fontsize=14)
            ax.set_ylabel('time-averaged z-score', fontsize=14)

        if save == True and custom_ax == False:
            if fname is None:
                fname = self.file_stem + '.zscrs.png'
            fig.savefig(fname, bbox_inches='tight')


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
        custom_ax = True
        if ax is None:
            custom_ax = False
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(111)

        # get std
        ant_std, time_std, agg_std, z_scores = self.delay_std()

        # choose xaxis
        if xaxis == 'ant':
            xax = range(len(ant_std))
            yax = ant_std
            cmap = matplotlib.cm.spectral(np.linspace(0, 0.95, len(xax)))
            ax.grid(True)
            ax.scatter(xax, yax, c=cmap, alpha=0.85,
                    marker='o', edgecolor='None', s=70)
            ax.set_xlim(-1, len(self.ants))
            ax.set_xticks(range(len(self.ants)))
            ax.set_xticklabels(self.ants)
            [t.set_rotation(20) for t in ax.get_xticklabels()]
            ax.set_xlabel('antenna number', fontsize=14)
            ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

        elif xaxis == 'time':
            xax = self.frac_JD
            yax = time_std
            cmap=None
            ax.grid(True)
            ax.plot(xax, yax, c='k', marker='.', linestyle='-', alpha=0.85)
            [t.set_rotation(20) for t in ax.get_xticklabels()]
            ax.set_xlabel('fractional JD of {}'.format(self.start_JD), fontsize=14)
            ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

        if save == True and custom_ax == False:
            if fname is None:
                fname = self.file_stem + '.stds.png'
            fig.savefig(fname, bbox_inches='tight')

