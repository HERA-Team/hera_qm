# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
import numpy as np
from pyuvdata import UVData
import matplotlib.pyplot as plt


def check_noise_variance(data):
    '''Function to calculate the noise levels of each baseline/pol combination,
    relative to the noise on the autos.

    Args:
        data (UVData): UVData object with data.

    Returns:
        Cij (dict): dictionary of variance measurements with keywords of (ant1, ant2, pol)
    '''
    Cij = {}
    for key, d in data.antpairpol_iter():
        w = data.get_nsamples(key)
        bl = (key[0], key[1])
        ai = data.get_data((key[0], key[0], key[2])).real
        aj = data.get_data((key[1], key[1], key[2])).real
        ww = w[1:, 1:] * w[1:, :-1] * w[:-1, 1:] * w[:-1, :-1]
        dd = ((d[:-1, :-1] - d[:-1, 1:]) - (d[1:, :-1] - d[1:, 1:])) * ww / np.sqrt(4)
        dai = ((ai[:-1, :-1] + ai[:-1, 1:]) + (ai[1:, :-1] + ai[1:, 1:])) * ww / 4
        daj = ((aj[:-1, :-1] + aj[:-1, 1:]) + (aj[1:, :-1] + aj[1:, 1:])) * ww / 4
        Cij[key] = (np.sum(np.abs(dd)**2, axis=0) / np.sum(dai * daj, axis=0) *
                    (data.channel_width * data.integration_time))

    return Cij


def vis_bl_cov(uvd1, uvd2, bls, iterax=None):
    """
    Calculate visibility data covariance from uvd1 with uvd2
    between specified baselines, optionally as a fuction of
    frequency _or_ time.

    Parameters
    ----------
    uvd1 : UVData object
        A single-pol UVData object holding visibility data

    uvd2 : UVData object
        A single-pol UVData object holding visibility data

    bls : list
        A list of antenna-pair tuples or UVData baseline integers to
        correlate.

    iterax : str, optional
        A data axis to iterate calculation over. Options=['freq', 'time'].

    Returns
    -------
    blcov : ndarray
        A covariance matrix of the data spanning Nbls x Nbls.
        This ndarray is 4 dimensional, with shape (Nbls, Nbls, Ntimes^1, Nfreqs^2)
            1 : Ntimes == 1 if iterax != 'time'
            2 : Nfreqs == 1 if iterax != 'freq'
    """
    # type checks
    assert isinstance(uvd1, UVData) and isinstance(uvd2, UVData), "uvd1 and uvd2 must be UVData objects"
    assert uvd1.Npols == 1 and uvd2.Npols == 1, "uvd1 and uvd2 must be single-polarization objects"
    assert isinstance(bls, (list, np.ndarray)), "bls must be a list of baselines"
    if isinstance(bls[0], (int, np.integer)): bls = [uvd1.baseline_to_antnums(bl) for bl in bls]
    assert iterax in [None, 'time', 'freq'], "iterax {} not recognized".format(iterax)
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

    # construct empty cov
    Nbls = len(bls)
    cov = np.empty((Nbls, Nbls, Ntimes, Nfreqs), dtype=np.complex) * np.nan

    # iterate over bls
    for i, bl1 in enumerate(bls):
        for j, bl2 in enumerate(bls):
            # get data and weights
            d1 = uvd1.get_data(bl1)
            w1 = (~uvd1.get_flags(bl1)).astype(np.float)
            d2 = uvd2.get_data(bl2)
            w2 = (~uvd2.get_flags(bl2)).astype(np.float)

            # skip if completely flagged
            if np.isclose(w1, 0.0).all() or np.isclose(w2, 0.0).all():
                continue

            # get means
            m1 = np.sum(d1 * w1, axis=sumaxes, keepdims=True) / np.sum(w1, axis=sumaxes, keepdims=True).clip(1e-10, np.inf)
            m2 = np.sum(d2 * w2, axis=sumaxes, keepdims=True) / np.sum(w2, axis=sumaxes, keepdims=True).clip(1e-10, np.inf)

            # get cov
            c = np.sum((d1 - m1) * w1 * (d2 - m2) * w2, axis=sumaxes, keepdims=True) / np.sum(w1 * w2).clip(1e-10, np.inf)

            # assign
            cov[i, j] = c

    return cov


def plot_bl_cov(uvd1, uvd2, bls, ax=None, cmap='viridis', vmin=None, vmax=None,
                component='abs', colorbar=True, tlsize=10, tlrot=35, figsize=None):
    """
    Plot the visibility data covariance between uvd1 and uvd2
    across a set of specified baselines.

    Parameters
    ----------
    uvd1 : UVData object

    uvd2 : UVData object

    bls : list
        List of baseline antenna-pairs

    ax : matplotlib.axes.Axis object

    cmap : str
        Colormap to use

    vmin, vmax : float
        Colorscale min and max

    component : str
        Component of complex covariance to plot. Options = ['real', 'imag', 'abs'].

    colorbar : bool
        If True, plot a colorbar

    tlsize : int
        Tick-label size.

    tlrot : int
        Tick-label rotation in degrees.

    figsize : tuple
        Len-2 integer tuple for figure-size if ax is None.

    Returns
    -------
    if ax is None:
        fig : matplotlib.pyplot.Figure object
    """
    # get covariance
    blcov = vis_bl_cov(uvd1, uvd2, bls).squeeze()
    assert not np.isnan(blcov).all(), "All data are flagged!"
    if component == 'abs':
        blcov = np.abs(blcov)
    elif component == 'real':
        blcov = np.real(blcov)
    elif component == 'imag':
        blcov = np.imag(blcov)

    # setup figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        newfig = True
    else:
        fig = ax.get_figure()
        newfig = False

    # plot 
    cax = ax.matshow(blcov, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')

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
        cbar.set_label(r"$\rm {}\ cov(V_{{1}},\ V_{{2}}^{{\ast}})\ [{}\ \cdot\ {}]$".format(component, uvd1.vis_units, uvd2.vis_units), fontsize=12)

    # axes labels
    if ax.get_xlabel() == "":
        ax.set_xlabel("Baseline", fontsize=14)
    if ax.get_ylabel() == "":
        ax.set_ylabel("Baseline", fontsize=14)

    if newfig:
        return fig
