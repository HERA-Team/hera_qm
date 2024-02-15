#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import os
import argparse
import numpy as np
import re
import matplotlib as mpl
import redis as redis_lib
from hera_mc import mc, sys_handling, cm_utils
from astropy import time
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import pyuvdata.utils as uvutils
import aipy.miriad as apm


parser = argparse.ArgumentParser(description='Plot auto locations and magnitudes')
parser.add_argument('-s', '--show', dest='show', action='store_true', default=False,
                    help='Show the plot. Default: False -- i.e., just save a png')
parser.add_argument('-l', '--log', dest='log', action='store_true', default=True,
                    help='Take 10*log10() of data before plotting. Default:True')
parser.add_argument('--outpath', default='', type=str,
                    help='Path to save output plots to. Default is same directory as file.')
parser.add_argument('--outbase', default='', type=str,
                    help='Base for output file names. Default is JD of data.')
parser.add_argument('--idbaddies', dest='idbaddies', action='store_true', default=False,
                    help='Identify potential misbehaving antennas. Default: False')
parser.add_argument('files', metavar='files', type=str, nargs='*', default=[],
                    help='Files for which to plot auto views.')
args = parser.parse_args()

if not args.show:
    # use a matplotlib backend that doesn't require an X session
    mpl.use('Agg')
# this has to be called after matplotlibs use()
import matplotlib.pyplot as plt  # noqa

# Get auto data
autos = {}
amps = {}
times = {}
if args.log:
    autos_raw = {}
if len(args.files) == 0:
    # No file given, use redis db
    redis = redis_lib.Redis('redishost')
    keys = [k for k in redis.keys() if k.startswith('visdata')]
    for key in keys:
        ant = int(re.findall(r'visdata://(\d+)/', key)[0])
        pol = key[-2:]
        autos[(ant, pol)] = np.fromstring(redis.hgetall(key).get('data'), dtype=np.float32)
        amps[(ant, pol)] = np.median(autos[(ant, pol)])
        if args.log:
            autos_raw[(ant, pol)] = autos[(ant, pol)]
            autos[(ant, pol)] = 10.0 * np.log10(autos[(ant, pol)])
            amps[(ant, pol)] = 10.0 * np.log10(amps[(ant, pol)])
        times[(ant, pol)] = float(redis.hgetall(key).get('time', 0))
else:
    counts = {}
    for fname in args.files:
        uvd = apm.UV(fname)
        pol = uvutils.polnum2str(uvd['pol']).lower()
        uvd.select('auto', 1, 1)
        for (uvw, this_t, (ant1, ant2)), auto, fname in uvd.all(raw=True):
            try:
                counts[(ant1, pol)] += 1
                autos[(ant1, pol)] += auto
                times[(ant1, pol)] += this_t
            except KeyError:
                counts[(ant1, pol)] = 1
                autos[(ant1, pol)] = auto
                times[(ant1, pol)] = this_t
    for key in autos.keys():
        autos[key] /= counts[key]
        times[key] /= counts[key]
        amps[key] = np.median(autos[key])
        if args.log:
            autos_raw[key] = autos[key]
            autos[key] = 10.0 * np.log10(autos[key])
            amps[key] = 10.0 * np.log10(amps[key])
    del(uvd)
    del(counts)

ants = np.unique([ant for (ant, pol) in autos.keys()])
# Find most recent time, only keep spectra from that time
latest = np.max(times.values())
for key, this_t in times.items():
    if latest - this_t > 1. / 60. / 24.:
        # more than a minute from latest, use NaN to flag
        autos[key] = np.nan
        amps[key] = np.nan
latest = time.Time(latest, format='jd')

# Get cminfo
parser = mc.get_mc_argument_parser()
mcargs = parser.parse_args(args=[])  # args=[] to throw away command line arguments
db = mc.connect_to_mc_db(mcargs)
session = db.sessionmaker()
hsession = sys_handling.Handling(session)
stations_conn = hsession.get_all_fully_connected_at_date(at_date=latest)
antpos = np.zeros((np.max(ants) + 1, 2))
ants_connected = []
antnames = ["" for x in range(np.max(ants) + 1)]
for stn in stations_conn:
    # make sure that antenna in cminfo is in data file
    try:
        antpos[stn['antenna_number'], :] = [stn['easting'], stn['northing']]
    except IndexError:
        continue
    ants_connected.append(stn['antenna_number'])
    antnames[stn['antenna_number']] = stn['station_name']
array_center = np.mean(antpos[antpos[:, 0] != 0, :], axis=0)
antpos -= array_center

# Get receiverator and PAM info
receiverators = ["" for x in range(np.max(ants) + 1)]
rxr_nums = np.zeros(np.max(ants) + 1, dtype=int)
pams = ["" for x in range(np.max(ants) + 1)]
for ant in ants_connected:
    pinfo = hsession.get_pam_info(antnames[ant], latest)
    receiverators[ant] = pinfo['e'][0][:-1]
    pams[ant] = pinfo['e'][1]
    result = re.findall(r'RI(\d+)', pinfo['e'][0])[0]
    rxr_nums[ant] = int(result)

# Pick a colormap to highlight "good", "bad", and "really bad"
# TODO: Fine tune this
goodbad = plt.get_cmap('RdYlGn')

# Construct path stuff for output
if args.outpath == '':
    # default path is same directory as file
    try:
        outpath = os.path.dirname(os.path.abspath(args.files[0]))
    except IndexError:
        outpath = os.path.abspath(os.path.curdir)
else:
    outpath = args.outpath
if args.outbase == '':
    try:
        basename = '.'.join(os.path.basename(args.files[0]).split('.')[0:3])
    except IndexError:
        basename = '%5f' % latest.jd
else:
    basename = args.outbase

# Plot autos vs positions
pol_labels = {'xx': 'E', 'yy': 'N'}
poli = {'xx': 0, 'yy': 1}
if args.log:
    vmin = -30
    vmax = 15
else:
    vmin = 0
    vmax = 12
fig = plt.figure(figsize=(10, 8))
for ant in ants_connected:
    for pol in ['xx', 'yy']:
        try:
            if not np.isnan(amps[(ant, pol)]):
                ax = plt.scatter(antpos[ant, 0], antpos[ant, 1] + 3 * (poli[pol] - 0.5),
                                 c=amps[(ant, pol)], vmin=vmin, vmax=vmax, cmap=goodbad)
            else:
                plt.scatter(antpos[ant, 0], antpos[ant, 1] + 3 * (poli[pol] - 0.5),
                            marker='x', color='k')
        except KeyError:
            plt.scatter(antpos[ant, 0], antpos[ant, 1] + 3 * (poli[pol] - 0.5),
                        marker='x', color='k')
    text = (str(ant) + '\n' + pams[ant] + '\n' + receiverators[ant])
    plt.annotate(text, xy=antpos[ant, 0:2] + [1, 0], textcoords='data',
                 verticalalignment='center')
if args.log:
    label = '10log10(Median Autos)'
else:
    label = 'Median Autos'
plt.colorbar(ax, label=label)
xr = antpos[ants_connected, 0].max() - antpos[ants_connected, 0].min()
yr = antpos[ants_connected, 1].max() - antpos[ants_connected, 1].min()
plt.xlim([antpos[ants_connected, 0].min() - 0.05 * xr, antpos[ants_connected, 0].max() + 0.2 * xr])
plt.ylim([antpos[ants_connected, 1].min() - 0.05 * yr, antpos[ants_connected, 1].max() + 0.1 * yr])
plt.title(str(latest.datetime) + ' UTC; JD=' + str(latest.jd))
# Add polarization key
for pol in ['xx', 'yy']:
    xpos = antpos[ants_connected, 0].min()
    ypos = antpos[ants_connected, 1].min() + 3 * (poli[pol])
    plt.scatter(xpos, ypos, c=vmax, vmin=vmin, vmax=vmax, cmap=goodbad)
    plt.annotate(pol_labels[pol] + ' pol', xy=[xpos + 1, ypos], textcoords='data', verticalalignment='center')
xmin = plt.gca().get_xlim()[0]
ymin = plt.gca().get_ylim()[0]
plt.plot([xmin, xmin + xr / 6., xmin + xr / 6.], [ymin * 3. / 4., ymin * 3. / 4., ymin], 'k')
# Save file
filename = os.path.join(outpath, basename + '.auto_v_pos.png')
plt.savefig(filename)


# Autos v rxr
fig, axarr = plt.subplots(1, np.max(rxr_nums), sharex=True, sharey=True, figsize=(15, 8))
for rxr in np.unique(rxr_nums[ants_connected]):
    ind = np.where(rxr_nums == rxr)[0]
    for anti, ant in enumerate(ind):
        for pol in ['xx', 'yy']:
            try:
                if not np.isnan(amps[(ant, pol)]):
                    ax = axarr[rxr - 1].scatter(0, anti + 0.3 * poli[pol], c=amps[(ant, pol)],
                                                vmin=vmin, vmax=vmax, cmap=goodbad)
                else:
                    axarr[rxr - 1].scatter(0, anti + 0.3 * poli[pol], marker='x', color='k')
            except BaseException:
                axarr[rxr - 1].scatter(0, anti + 0.3 * poli[pol], marker='x', color='k')
        axarr[rxr - 1].annotate(str(ant) + ',' + pams[ant], xy=[0.01, i])
    axarr[rxr - 1].set_yticks([])
    axarr[rxr - 1].set_xticks([])
for rxr in range(np.max(rxr_nums)):
    axarr[rxr].set_title('Rxr ' + str(rxr + 1))
plt.xlim([-.01, .1])
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(wspace=0)
# cbar_ax = f.add_axes([.14, .05, .72, .05])
cbar_ax = fig.add_axes([.13, .05, .67, .05])
if args.log:
    label = '10log10(Median Autos)'
else:
    label = 'Median Autos'
fig.colorbar(ax, cax=cbar_ax, orientation='horizontal', label=label)
fig.suptitle(str(latest.datetime) + ' UTC')
# Add polarization key
p_ax = fig.add_axes([0.82, 0.03, 0.06, 0.08])
for pol in ['xx', 'yy']:
    yval = poli[pol]
    plt.scatter(0, yval, c=vmax, vmin=vmin, vmax=vmax, cmap=goodbad)
    plt.annotate(pol_labels[pol] + ' pol', xy=[1, yval], textcoords='data', verticalalignment='center')
plt.xlim([-1, 4])
plt.ylim([-1, 2])
p_ax.set_xticks([])
p_ax.set_yticks([])
# Save file
filename = os.path.join(outpath, basename + '.auto_v_rxr.png')
plt.savefig(filename)

# Plot spectra
fig = plt.figure(figsize=(20, 12))
# Number of subplots in X and Y directions
nants = len(ants)
nx = int(np.ceil(np.log2(nants + 1)))
ny = int(np.ceil(nants / float(nx)))
pol_colors = {'xx': 'r', 'yy': 'b'}
ants = np.sort(ants)
for ai, ant in enumerate(ants):
    ax = plt.subplot(nx, ny, ai + 1)
    for pol in ['xx', 'yy']:
        try:
            plt.plot(autos[(ant, pol)], pol_colors[pol], label=pol_labels[pol])
        except KeyError:
            continue
    if ant in ants_connected:
        lcolor = 'r'
    else:
        lcolor = 'k'
    plt.text(0.8, 0.8, str(ant), fontsize=12, transform=ax.transAxes, color=lcolor)
    # Axis stuff
    ax.axes.get_xaxis().set_ticklabels([])
    if ai % ny:
        ax.axes.get_yaxis().set_ticklabels([])
    else:
        if args.log:
            ax.set_ylabel('10log10')
        else:
            ax.set_ylabel('linear')
    ax.set_ylim([vmin, 1.3 * vmax])
    if ai == 0:
        plt.legend(loc='best')
fig.suptitle(str(latest.datetime) + ' UTC; JD=' + str(latest.jd))
filename = os.path.join(outpath, basename + '.auto_specs.png')
plt.savefig(filename)

# Plot rms values at input of ADC's


def plot_rms(data, POL='xx'):
    """Plots rms values at input to ADC's by looking at the autos.

    Args:
        data (dict): Dictionary of auto's.
        POL (str): String of polarization."""
    fig = plt.figure(figsize=(20, 12))
    CHUNK = 256
    BINS = 24
    rmscolors = 'bgrcmy'
    ants = sorted([i for (i, pp) in data if pp == POL])
    Nrow = np.ceil(np.sqrt(len(ants)))
    Mcol = np.ceil(len(ants) / float(Nrow))
    bins = np.logspace(-2, 4, BINS, base=2.)
    for cnt, ant in enumerate(ants):
        ax = plt.subplot(Nrow, Mcol, cnt + 1)
        for ch in xrange(0, 1024, CHUNK):
            dchunk = data[ant, POL][ch:ch + CHUNK].flatten()
            hist, hbins = np.histogram(np.sqrt(dchunk / 2), bins)
            hist = 10**np.log10(hist + .1)
            hbins = 0.5 * (hbins[1:] + hbins[:-1])
            ax.fill_between(np.log2(bins), hist, .1, where=hist > .1, color=rmscolors[j], alpha=.5)
        bounds = np.where(bins < 2**0, dchunk.size, np.where(bins > 2**2, dchunk.size, 0))
        ax.fill_between(np.log2(bins), bounds, .1, where=bounds > .1, color='black', alpha=.6)
        ax.set_yscale('log')
        plt.xlim(-2, 3)
        plt.ylim(d.size / 1e2, d.size)
        if ant in ants_connected:
            lcolor = 'r'
        else:
            lcolor = 'b'
        plt.text(0.5, 0.8, str(ant), fontsize=12, transform=ax.transAxes, color=lcolor)
        ax.get_yaxis().set_visible(False)
        if cnt < (Nrow - 1) * Mcol:
            ax.get_xaxis().set_ticklabels([])
        else:
            plt.xlabel(r'$V_{\rm rms}$ [bits]')
        plt.grid()
    fig.suptitle(str(latest.datetime) + ' UTC' + '    {0}'.format(POL) + '; JD=' + str(latest.jd))
    plt.tight_layout(rect=(0, 0, 1, .95))


for pol in ['xx', 'yy']:
    if args.log:
        plot_rms(autos_raw, POL=pol)
    else:
        plot_rms(autos, POL=pol)
    filename = os.path.join(outpath, basename + '.{0}.auto_rms_values.png'.format(pol))
    plt.savefig(filename)

# ID some potential baddies
if args.idbaddies:
    baddies = [str((key[0], pol_labels[key[1]])) for key, val in amps.items() if
               val < 0.75 * (vmax - vmin) + vmin and key[0] in ants_connected]
    filename = os.path.join(outpath, basename + '.baddies.txt')
    np.savetxt(filename, baddies, fmt='%s', header='You may want to check these antennas:')
