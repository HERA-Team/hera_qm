# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""Module for computing visibility metrics."""

import numpy as np
from pyuvdata import UVData
import copy
from . import utils
from . import __version__


def check_noise_variance(uvd):
    """Calculate the noise levels of each baseline/pol relative to the autos.

    This function calculates the noise for each baseline/pol by differencing
    along the frequency dimension and compares to the noise on the auto-spectra
    for each antenna in the baseline.

    Parameters
    ----------
    uvd : UVData object
        A UVData object with data to be analyzed.

    Returns
    -------
    Cij : dict
        A dictionary of variance measurements with keys (ant1, ant2, pol).

    Raises
    ------
    NotImplementedError:
        If the UVData object contains data where integration_time is not
        the same for all integrations, a NotImplementedError is raised. In
        other words, this is *not* applicable to BDA data.

    """
    Cij = {}
    for key, data in uvd.antpairpol_iter():
        inds = uvd.antpair2ind(key[0], key[1])
        integration_time = uvd.integration_time[inds]
        if not len(set(integration_time)) == 1:
            raise NotImplementedError(("Integration times which vary with "
                                       "time are currently not supported."))
        else:
            integration_time = integration_time[0]
        w = uvd.get_nsamples(key)
        bl = (key[0], key[1])
        ai = uvd.get_data((key[0], key[0], key[2])).real
        aj = uvd.get_data((key[1], key[1], key[2])).real
        ww = w[1:, 1:] * w[1:, :-1] * w[:-1, 1:] * w[:-1, :-1]

        dd = (((data[:-1, :-1] - data[:-1, 1:]) - (data[1:, :-1] - data[1:, 1:])) * ww
              / np.sqrt(4))
        dai = (((ai[:-1, :-1] + ai[:-1, 1:]) + (ai[1:, :-1] + ai[1:, 1:])) * ww
               / 4)
        daj = (((aj[:-1, :-1] + aj[:-1, 1:]) + (aj[1:, :-1] + aj[1:, 1:])) * ww
               / 4)
        Cij[key] = (np.sum(np.abs(dd)**2, axis=0) / np.sum(dai * daj, axis=0)
                    * (uvd.channel_width * integration_time))

    return Cij


def sequential_diff(data, t_int=None, axis=(0,), pad=True, run_check=True, history=''):
    """Take a sequential (forward) difference to estimate visibility noise.

    Parameters
    ----------
    data : ndarray or UVData object
        A 2D or 3D ndarray containing visibility data, with
        shape (Ntimes, Nfreqs, :). Or UVData object
        holding visibility data.
    t_int : ndarray, optional
        A 2D or 3D ndarray containing the integration time
        of the visibilities matching shape of data.
        If None and data is UVData, will use UVData ~flags *
        nsample * integration_time as input.
    axis : int or tuple, optional
        Axes along which to take sequential difference. Default is (0,)
    pad : bool, optional
        If True, insert an extra (flagged) column at the end
        of axis such that diff_data has same shape as input.
        Default is True.
    run_check : bool, optional
        If True, run the UVData.check() function when complete.
        Default is True.
    history : str, optional
        A string to prepend to history of UVData if provided,
        in addition to a standard history comment. Default is empty string.

    Returns
    -------
    diff_data : ndarray or UVData object
        A 2D or 3D complex differenced data array,
        or UVData object holding differenced data
        divided by sqrt(2).

    diff_t_int : ndarray
        If input data is ndarray, this is the average
        integration time of differenced pixels, unless
        one of them is 0.0 in which case output is 0.0.

    Raises
    ------
    ValueError:
        A ValueError is raised if:
        - "axis" is not 0 or 1.
        - t_int is not an array.
        - data is not an array or a UVData object.

    """
    # type check
    if isinstance(axis, (int, np.integer)):
        axis = (axis,)
    if not np.all([ax in [0, 1] for ax in axis]):
        raise ValueError("axis must be 0, 1 or both")

    # perform differencing if data is an ndarray
    if isinstance(data, np.ndarray):
        # get t_int
        if t_int is None:
            t_int = np.ones_like(data, dtype=np.float64)
        else:
            if not isinstance(t_int, np.ndarray):
                raise ValueError("t_int must be ndarray if data is ndarray")

        # pad
        if pad:
            for ax in axis:
                # get padding vector
                zero_slice = utils.dynamic_slice(np.zeros_like(data, dtype=np.float64), slice(0, 1), axis=ax)
                # pad arrays
                data = np.concatenate([data, zero_slice], axis=ax)
                t_int = np.concatenate([t_int, zero_slice], axis=ax)

        for ax in axis:
            # difference data to get noise
            data = (utils.dynamic_slice(data, slice(1, None), axis=ax)
                    - utils.dynamic_slice(data, slice(None, -1), axis=ax))

            # get average t_int
            t1 = utils.dynamic_slice(t_int, slice(1, None), axis=ax)
            t2 = utils.dynamic_slice(t_int, slice(None, -1), axis=ax)
            where = ~(np.isclose(t1, 0.0) + np.isclose(t2, 0.0))
            t_int = 0.5 * (t1 + t2)

            # take inverse sum
            inv_sum = np.true_divide(1., t1, where=where) + np.true_divide(1., t2, where=where)
            inv_t_int = np.true_divide(1., inv_sum, where=where)

            # get noise correction factor
            corr = np.true_divide(inv_t_int, t_int, where=where)
            # remove potentially invalid entries from masked values in true_divide
            corr[~where] = 0.
            corr = np.sqrt(corr)

            # set bad pixel correction to 1.0 and t_int to 0.0 (i.e. flagged pixels)
            corr[~where] = 1.0
            t_int[~where] = 0.0

            # normalize noise by 1. / sqrt(t_int / inv_t_int)
            data *= corr

        return data, t_int

    # iterate over bls if UVData
    elif isinstance(data, UVData):
        # copy object
        uvd = copy.deepcopy(data)

        if not pad:
            # get new time and freq arrays and UVData object
            # this is equivalent to assuming a forward difference
            times = None
            freqs = None
            for ax in axis:
                if ax == 0:
                    times = np.unique(uvd.time_array)[:-1]
                elif ax == 1:
                    freqs = np.unique(uvd.freq_array)[:-1]
            if times is not None or freqs is not None:
                uvd.select(times=times, frequencies=freqs)

        # iterate over baselines
        bls = uvd.get_antpairs()
        for bl in bls:
            # get blt slice
            bl_slice = uvd.antpair2ind(bl, ordered=False)

            # configure data and t_int
            bl_data = data.get_data(bl, squeeze='none')[:, 0, :, :]
            bl_t = (data.get_nsamples(bl, squeeze='none')[:, 0, :, :]
                    * (~data.get_flags(bl, squeeze='none')[:, 0, :, :]).astype(np.float64)
                    * data.integration_time[data.antpair2ind(bl, ordered=False)][:, None, None])

            # take difference
            bl_data, bl_t = sequential_diff(bl_data, t_int=bl_t, axis=axis, pad=pad)

            # configure output flags, nsample
            flags = np.isclose(bl_t, 0.0)
            nsample = bl_t / uvd.integration_time[uvd.antpair2ind(bl, ordered=False)][:, None, None]

            # assign data
            uvd.data_array[bl_slice, 0, :, :] = bl_data
            uvd.flag_array[bl_slice, 0, :, :] = flags
            uvd.nsample_array[bl_slice, 0, :, :] = nsample

        # run check
        if run_check:
            uvd.check()

        # add to history
        uvd.history = "Took sequential difference with hera_qm [{}]\n{}\n{}\n{}" \
                      .format(__version__, history, '-' * 50, uvd.history)

        return uvd

    else:
        raise ValueError("Didn't recognize input data structure")


def vis_bl_bl_cov(uvd1, uvd2, bls, iterax=None, return_corr=False):
    """Calculate visibility data covariance.

    Calculate the visibility data covariance or correlation matrix
    from uvd between specified baselines, optionally as a fuction of
    frequency _or_ time.

    If return_corr == True, the correlation matrix holds the
    covariance between baseline1 and baseline2 divided
    by the standard deviation of baseline1 times the standard
    deviation of baseline2.

    Parameters
    ----------
    uvd1 : UVData object
        A single-pol UVData object holding visibility data.
    uvd2 : UVData object
        A single-pol UVData object holding visibility data.
    bls : list
        A list of antenna-pair tuples or UVData baseline integers to
        correlate.
    iterax : {"freq", "time", None}, optional
        A data axis to iterate calculation over. If not specified,
        no iteration is done. Default is None.
    return_corr : bool, optional
        If True, calculate and return correlation matrix. Default is False.

    Returns
    -------
    bl_bl_cov : ndarray
        A covariance (or correlation) matrix across baselines.
        This ndarray is 4 dimensional, with shape (Nbls, Nbls, Ntimes, Nfreqs)
        where
            Ntimes == 1 if iterax != 'time'.
            Nfreqs == 1 if iterax != 'freq'.

    Raises
    ------
    AssertionError:
        An AssertionError is raised if:
        - uvd1 or uvd2 is not a UVData object.
        - uvd1 or uvd2 has more than 1 polarization.
        - bls is not a list of baselines.
        - iterax is not one of the above options.
        - the times or frequencies of uvd1 and uvd2 disagree.

    """
    # type checks
    assert isinstance(uvd1, UVData) and isinstance(uvd2, UVData), \
        "uvd1 and uvd2 must be UVData objects"
    assert uvd1.Npols == 1 and uvd2.Npols == 1, \
        "uvd1 and uvd2 must be single-polarization objects"
    assert isinstance(bls, (list, np.ndarray)), "bls must be a list of baselines"
    if isinstance(bls[0], (int, np.integer)):
        bls = [uvd.baseline_to_antnums(bl) for bl in bls]
    assert iterax in [None, 'time', 'freq'], \
        "iterax {} not recognized".format(iterax)
    assert uvd1.Ntimes == uvd2.Ntimes, "Ntimes must agree between uvd1 and uvd2"
    assert uvd1.Nfreqs == uvd2.Nfreqs, "Nfreqs must agree between uvd1 and uvd2"

    # get Nfreqs and Ntimes
    if iterax == 'time':
        Ntimes = uvd1.Ntimes
        Nfreqs = 1
        sumaxes = (1,)
    elif iterax == 'freq':
        Ntimes = 1
        Nfreqs = uvd1.Nfreqs
        sumaxes = (0,)
    elif iterax is None:
        Ntimes = 1
        Nfreqs = 1
        sumaxes = (0, 1)

    # construct empty data dictionaries
    d1 = {}
    w1 = {}
    m1 = {}
    d2 = {}
    w2 = {}
    m2 = {}
    for bl in bls:
        d1[bl] = uvd1.get_data(bl)
        d2[bl] = uvd2.get_data(bl)
        w1[bl] = (~uvd1.get_flags(bl)).astype(np.float64)
        w2[bl] = (~uvd2.get_flags(bl)).astype(np.float64)
        m1[bl] = (np.sum(d1[bl] * w1[bl], axis=sumaxes, keepdims=True)
                  / np.sum(w1[bl], axis=sumaxes, keepdims=True).clip(1e-10, np.inf))
        m2[bl] = (np.sum(d2[bl] * w2[bl], axis=sumaxes, keepdims=True)
                  / np.sum(w2[bl], axis=sumaxes, keepdims=True).clip(1e-10, np.inf))

    # setup empty cov array
    Nbls = len(bls)
    cov = np.empty((Nbls, Nbls, Ntimes, Nfreqs), dtype=np.complex128) * np.nan

    # iterate over bls
    for bl1i, bl1 in enumerate(bls):
        # skip if completely flagged
        if np.isclose(m1[bl1], 0.0).all():
            continue

        # iterate over bls
        for bl2i, bl2 in enumerate(bls):
            # skip if completely flagged
            if np.isclose(m2[bl2], 0.0).all():
                continue

            # get cov
            w12 = w1[bl1] * w2[bl2]
            cov[bl1i, bl2i] = (np.sum((d1[bl1] - m1[bl1]) * (d2[bl2] - m2[bl2]).conj()
                                      * w12, axis=sumaxes, keepdims=True)
                               / np.sum(w12, axis=sumaxes, keepdims=True).clip(1e-10, np.inf))

    # calculate correlation matrix
    if return_corr:
        # get stds of bls
        std = np.empty((2, Nbls, Ntimes, Nfreqs), dtype=np.complex128) * np.nan
        for bli, bl in enumerate(bls):
            d1diff = d1[bl] - m1[bl]
            std[0, bli] = np.sqrt(np.abs(np.sum(d1diff * d1diff.conj() * w1[bl],
                                                axis=sumaxes, keepdims=True)
                                         / np.sum(w1[bl], axis=sumaxes, keepdims=True).clip(1e-10, np.inf)))
            d2diff = d2[bl] - m2[bl]
            std[1, bli] = np.sqrt(np.abs(np.sum(d2diff * d2diff.conj() * w2[bl],
                                                axis=sumaxes, keepdims=True)
                                         / np.sum(w2[bl], axis=sumaxes, keepdims=True).clip(1e-10, np.inf)))

        # turn cov into corr
        for bli in range(Nbls):
            for blj in range(Nbls):
                cov[bli, blj] /= std[0, bli] * std[1, blj]
    return cov


def plot_bl_bl_cov(uvd1, uvd2, bls, plot_corr=False, ax=None, cmap='viridis',
                   vmin=None, vmax=None, component='abs', colorbar=True,
                   tlsize=10, tlrot=35, figsize=None, times=None, freqs=None):
    """Plot the visibility data covariance or correlation matrix.

    This function plots the covariance or correlation matrix of uvd across
    a set of specified baselines.

    Parameters
    ----------
    uvd1 : UVData object
        A single-pol UVData object holding visibility data.
    uvd2 : UVData object
        A single-pol UVData object holding visibility data.
    bls : list
        A list of antenna-pair tuples or UVData baseline integers to
        correlate.
    plot_corr : bool, optional
        If True, calculate and plot correlation matrix instead.
        Default is False.
    ax : matplotlib.axes.Axis object, optional
        If provided, the axis object to add plot to. If not specified,
        a new axis will be created. Default is None.
    cmap : str, optional
        Colormap to use. Default is "viridis".
    vmin : float, optional
        Colorscale min. Default is None.
    vmax : float, optional
        Colorscale max. Default is None.
    component : {"real", "imag", "abs"}, optional
        Component of matrix to plot. Default is "abs".
    colorbar : bool, optional
        If True, plot a colorbar. Default is True.
    tlsize : int, optional
        Tick-label size. Default is 10.
    tlrot : int, optional
        Tick-label rotation in degrees. Default is 35.
    figsize : tuple, optional
        Len-2 integer tuple for figure-size if ax is None. Default is None.
    times : list, optional
        List of times to select on UVData before calculating matrices.
        Cannot be fed if freqs if also fed. Default is None.
    freqs : list, optional
        List of frequencies to select on UVData before calculating
        matrices. Cannot be fed if times is also fed. Default is None.

    Returns
    -------
    fig : matplotlib.pyplot.Figure object
        If ax is None, the figure object that the plot is added to.

    Raises
    ------
    ValueError:
        A ValueError is raised if "component" not in the above list.
    AssertionError:
        An AssertionError is raised if times and freqs are both specified,
        or if all the data to be plotted is flagged.

    """
    import matplotlib.pyplot as plt
    # selections
    assert times is None or freqs is None, \
        "times and freqs cannot both be fed at the same time"
    if times is not None or freqs is not None:
        uvd1 = uvd1.select(times=times, frequencies=freqs, inplace=False, run_check=False)
        uvd2 = uvd2.select(times=times, frequencies=freqs, inplace=False, run_check=False)

    # get covariance
    bl_bl_cov = vis_bl_bl_cov(uvd1, uvd2, bls, return_corr=plot_corr).squeeze()
    assert not np.isnan(bl_bl_cov).all(), "All data are flagged!"
    if component == 'abs':
        bl_bl_cov = np.abs(bl_bl_cov)
    elif component == 'real':
        bl_bl_cov = np.real(bl_bl_cov)
    elif component == 'imag':
        bl_bl_cov = np.imag(bl_bl_cov)
    else:
        raise ValueError("Didn't recognize component {}".format(component))

    # setup figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        newfig = True
    else:
        fig = ax.get_figure()
        newfig = False

    # plot
    cax = ax.matshow(bl_bl_cov, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')

    # ticks
    Nbls = len(bls)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(labelsize=tlsize, rotation=tlrot)
    ax.set_xticks(np.arange(Nbls))
    ax.set_yticks(np.arange(Nbls))
    ax.set_xticklabels(bls)
    ax.set_yticklabels(bls)

    # colorbar
    if colorbar:
        cbar = fig.colorbar(cax, ax=ax)
        if plot_corr:
            action = "corr"
        else:
            action = "cov"
        label = r"$\rm {}\ {}(V_{{1}}, V_{{2}})\ [{}\ \cdot\ {}]$" \
                .format(component, action, uvd1.vis_units, uvd2.vis_units)
        cbar.set_label(label, fontsize=12)

    # axes labels
    if ax.get_xlabel() == "":
        ax.set_xlabel("Baseline", fontsize=14)
    if ax.get_ylabel() == "":
        ax.set_ylabel("Baseline", fontsize=14)

    if newfig:
        return fig


def plot_bl_bl_scatter(uvd1, uvd2, bls, component='real', whiten=False, colorbar=True,
                       axes=None, colorax='freq', alpha=1, msize=1, marker='.', grid=True,
                       one2one=True, loglog=False, freqs=None, times=None, figsize=None,
                       xylim=None, cbfontsize=10, axfontsize=14, force_plot=False,
                       tlsize=10, facecolor='lightgrey', cmap='viridis'):
    """Make a scatter - matrix plot.

    This plot shows the covariance of visibility data between baselines in uvd1
    with baselines in uvd2.

    Parameters
    ----------
    uvd1 : UVData object
        A single-pol UVData object holding visibility data.
    uvd2 : UVData object
        A single-pol UVData object holding visibility data.
    bls : list
        A list of antenna-pair tuples or UVData baseline integers to
        correlate.
    component : {"real", "imag", "abs", "angle"}, optional
        Component of visibility data to plot. Default is "real".
    whiten : bool, optional
        If True, divide data component by abs of data before plotting.
        Default is False.
    colorbar : bool, optional
        If True, add a colorbar. Default is True.
    axes : ndarray, optional
        ndarray of axes objects to use in plotting. If not specified, axes will
        be generated as needed to create plots. Default is None.
    colorax : {"freq", "time"}, optional
        Data axis to colorize. Default is "freq".
    alpha : float, optional
        Transparency of points. Default is 1.
    msize : int, optional
        Marker size. Default is 1.
    marker : str, optional
        Type of marker to use. Default is ".".
    grid : bool, optional
        If True, add a grid to the scatter plots. Default is True.
    one2one : bool, optional
        If True, add a 1-to-1 line. Default is True.
    loglog : bool, optional
        If True, logscale the x and y axes. Default is False.
    freqs : ndarray, optional
        Array of frequencies to select on before plotting. If not specified,
        all frequencies are used.
    times : ndarray, optional
        Array of times to select on before plotting. If not specified, all
        times are used.
    figsize : tuple, optional
        Figure size if axes is None.
    xylim : tuple, optional
        xy limits of the subplots.
    cbfontsize : int, optional
        Fontsize of colorbar label and ticks. Default is 10.
    axfontsize : int, optional
        Fontsize of axes labels. Default is 14.
    force_plot : bool, optional
        If Nbls > 10 and force_plot is False, this function errors. Default is False.
    tlsize : int, optional
        Ticklabel size of subplots. Default is 10.
    facecolor : str, optional
        Facecolor of subplots. Default is "lightgrey".
    cmap : str, optional
        Colormap to use for scatter plot. Default is "viridis".

    Returns
    -------
    fig : matplotlib.pyplot.Figure object
        If axes is None, the figure object that the plot is added to.

    Raises
    ------
    ValueError:
        A ValueError is raised if "component" not in the above list, if Nbls > 10
        and force_plot is False, if "colorax" is not in the above list, or if
        appropriate plot limits cannot be determine when xylim is not specified.
    AssertionError:
        An AssertionError is raised if times and freqs are both specified, if
        the times or frequencies arrays disagree for the input UVData objects,
        or if axes are specified incorrectly.
    """
    import matplotlib.pyplot as plt
    # selections
    assert times is None or freqs is None, \
        "times and freqs cannot both be fed at the same time"
    if times is not None or freqs is not None:
        uvd1 = uvd1.select(times=times, frequencies=freqs, inplace=False, run_check=False)
        uvd2 = uvd2.select(times=times, frequencies=freqs, inplace=False, run_check=False)
    assert uvd1.Ntimes == uvd2.Ntimes, "Ntimes must agree between uvd1 and uvd2"
    assert uvd1.Nfreqs == uvd2.Nfreqs, "Nfreqs must agree between uvd1 and uvd2"

    if component == 'abs':
        cast = np.abs
    elif component == 'real':
        cast = np.real
    elif component == 'imag':
        cast = np.imag
    elif component == 'angle':
        cast = np.angle
    else:
        raise ValueError("Didn't recognize component {}".format(component))

    # setup figure
    Nbls = len(bls)
    if Nbls >= 10 and force_plot is False:
        raise ValueError("Trying to plot >= 10 bls and force_plot = False, quitting...")
    if axes is None:
        fig, axes = plt.subplots(Nbls, Nbls, figsize=figsize)
        newfig = True
    else:
        assert isinstance(axes, np.ndarray), "axes must be a 2D ndarray"
        assert axes.ndim == 2, "axes must be a 2D ndarray"
        assert axes.size == Nbls**2, "axes.size must equal Nbls^2"
        fig = axes[0, 0].get_figure()
        newfig = False

    # get colorax
    if colorax == 'freq':
        color = np.repeat(uvd1.freq_array / 1e6, uvd1.Ntimes, axis=0).ravel()
        clabel = r"$\rm Frequency\ [MHz]$"
    elif colorax == 'time':
        jd = int(np.floor(np.median(uvd1.time_array)))
        color = np.repeat(np.unique(uvd1.time_array)[:, None] % jd, uvd1.Nfreqs, axis=1).ravel()
        clabel = r"$\rm Julian\ Date\ \%\ {}$".format(jd)
    else:
        raise ValueError("Didn't recognize colorax {}".format(colorax))

    # iterate over bl-bl pairs
    for bli, bl1 in enumerate(bls):
        for blj, bl2 in enumerate(bls):
            # get ax
            ax = axes[bli, blj]

            # turn on grid
            if grid:
                ax.grid()

            # facecolor
            ax.set_facecolor(facecolor)

            # get data
            try:
                d1 = uvd1.get_data(bl1).ravel().copy()
                d2 = uvd2.get_data(bl2).ravel().copy()
                f1 = uvd1.get_flags(bl1).ravel()
                f2 = uvd2.get_flags(bl2).ravel()
            except KeyError:
                # data key didn't exist...
                d1 = np.zeros_like(color)
                d2 = np.zeros_like(color)
                f1 = np.ones_like(color, dtype=np.bool_)
                f2 = np.ones_like(color, dtype=np.bool_)

            d1[f1] *= np.nan
            d2[f2] *= np.nan
            if whiten:
                d1 /= np.abs(d1)
                d2 /= np.abs(d2)
            d1 = cast(d1)
            d2 = cast(d2)

            # plot
            cax = ax.scatter(d1, d2, alpha=alpha, s=msize, cmap=cmap, c=color, marker=marker)
            if (bli == 0 and blj == 0) and (f1.all() or f2.all()) and (xylim is None):
                raise ValueError("xylim was not specified and is therefore determined by\n"
                                 "range of first bl-pair, but these data are completely flagged.")

            # logscale
            if loglog:
                ax.set_xscale('log')
                ax.set_yscale('log')

            # ax ticks: all subplots should have same range
            if xylim is None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
            else:
                xlim, ylim = xylim, xylim
            neg = np.min([xlim[0], ylim[0]])
            pos = np.max([xlim[1], ylim[1]])
            xylim = (neg, pos)
            ax.set_xlim([neg, pos])
            ax.set_ylim([neg, pos])
            ax.neg = neg
            ax.pos = pos

            # ax tick labels
            if bli == Nbls - 1:
                # last row, make axes labels
                if ax.get_xlabel() == "":
                    ax.set_xlabel("{} {} [{}]".format(bl2, component, uvd2.vis_units),
                                  fontsize=axfontsize)
                _ = [tl.set_size(tlsize) for tl in ax.get_xticklabels()]
            else:
                # not last row, get rid of tick labels
                ax.set_xticklabels([])
            if blj == 0:
                # first column, make axes labels
                if ax.get_ylabel() == "":
                    ax.set_ylabel("{} {} [{}]".format(bl1, component, uvd1.vis_units),
                                  fontsize=axfontsize)
                _ = [tl.set_size(tlsize) for tl in ax.get_yticklabels()]
            else:
                # not first column, get rid of tick labels
                ax.set_yticklabels([])

            # plot one2one
            if one2one:
                ax.plot([neg, pos], [neg, pos], color='k', lw=1, ls='--')

    # colorbar
    if colorbar:
        fig.subplots_adjust(right=0.90)
        cbax = fig.add_axes([0.90, 0.15, 0.05, 0.7])
        cbax.axis('off')
        cbar = fig.colorbar(cax, ax=cbax, fraction=0.75, aspect=40)
        cbar.set_label(clabel, fontsize=cbfontsize)
        _ = [tl.set_size(cbfontsize) for tl in cbar.ax.get_yticklabels()]

    if newfig:
        return fig
