# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

"""
FirstCal metrics plotting
"""


def plot_delays(FC, ants=None, plot_type='both', cmap='nipy_spectral', ax=None, save=False, fname=None,
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

    fname : str, [default=FC.fc_filestem+'.png']
        filename to save plot as
        default is FC.fc_filestem

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
        FC.fig = fig
        fig.subplots_adjust(hspace=0.3)
        if plot_type == 'both':
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax = [ax1, ax2]
        else:
            ax = fig.add_subplot(111)

    # match ants fed to true ants
    if ants is not None:
        plot_ants = [np.where(FC.ants == ant)[0][0] for ant in ants if ant in FC.ants]
    else:
        plot_ants = range(FC.Nants)

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
            p, = ax.plot(FC.frac_JD, FC.delays[index], marker='.',
                         c=cm[i], **plt_kwargs)
            plabel.append(p)
        ax.set_xlabel('fraction of JD %d' % FC.start_JD, fontsize=14)
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
            p, = ax.plot(FC.frac_JD, FC.delay_fluctuations[index],
                         marker='.', c=cm[i], **plt_kwargs)
            plabel.append(p)
        ax.set_xlabel('fraction of JD %d' % FC.start_JD, fontsize=14)
        ax.set_ylabel('delay fluctuation [ns]', fontsize=14)
        if plot_type == 'both':
            ax = axes

    # add legend
    if custom_ax is False:
        ax = fig.add_axes([1.0, 0.1, 0.05, 0.8])
        ax.axis('off')
        ax.legend(plabel, [FC.ants[i] for i in plot_ants])
    else:
        ax[-1].legend(plabel, [FC.ants[i] for i in plot_ants])

    if save is True and custom_ax is False:
        if fname is None:
            fname = FC.fc_filestem + '.dlys.png'
        fig.savefig(fname, bbox_inches='tight')

    if custom_ax is False:
        return fig


def plot_zscores(FC, fname=None, plot_type='full', ax=None, figsize=(10, 6),
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
    if hasattr(FC, 'metrics') is False:
        raise NameError("You need to run FirstCal_Metrics.run_metrics() "
                        + "in order to plot delay z_scores")
    fig = plot_zscores(FC.metrics, fname=fname, plot_type=plot_type, ax=ax, figsize=figsize,
                       save=save, kwargs=kwargs, plot_abs=plot_abs)
    return fig


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
        [t.set_rotation(20) for t in ax.get_xticklabels()]
        ax.set_xlabel('antenna number', fontsize=14)
        ax.set_ylabel('delay solution standard deviation [ns]', fontsize=14)

    elif xaxis == 'time':
        xax = metrics['frac_JD']
        yax = list(metrics['time_std'].values())
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
    if hasattr(self, 'metrics') is False:
        raise NameError("You need to run FirstCal_Metrics.run_metrics() "
                        + "in order to plot delay stds")
    fig = plot_stds(self.metrics, fname=fname, ax=ax, xaxis=xaxis, kwargs=kwargs, save=save)
    return fig
