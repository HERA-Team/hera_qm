#!/usr/bin/env python
"""
Create a plot of the antenna autocorrelations stored in redis.
"""

import redis as redis_lib
import numpy as np
import matplotlib as mpl
import argparse

parser = argparse.ArgumentParser(description='Plot autocorrelations from redis',
                                 epilog='NB: plots are autoscaled for each antenna')
parser.add_argument('-a', '--ants', dest='ants', type=str, default='all',
                    help='Comma separated list of antennas to plot (0-indexed), or "all".')
parser.add_argument('-s', '--show', dest='show', action='store_true', default=False,
                    help='Show the plot. Default: False -- i.e., just save a png')
parser.add_argument('-l', '--log', dest='log', action='store_true', default=False,
                    help='Take 10*log10() of data before plotting. Default:False')

args = parser.parse_args()

if not args.show:
    # use a matplotlib backend that doesn't require an X session
    mpl.use('Agg')
# this has to be called after matplotlibs use()
import matplotlib.pyplot as plt

if args.ants == 'all':
    ants = range(128)  # TODO: make this programatic
else:
    try:
        ants = map(int, args.ants.split(','))
    except:
        print "Failed to parse antenna list. Exiting."
        exit()

redis = redis_lib.Redis('redishost')

# TODO: Make these programatic
bw = 0.1
f0 = 0.1
nchan = 1024
sdf = float(bw) / nchan

freqs = np.linspace(f0, f0 + bw - sdf / 2., nchan)

# Number of subplots in X and Y directions
nants = len(ants)
ny = int(np.ceil(np.log2(nants + 1)))
nx = int(np.ceil(nants / float(ny)))

jd = None
filename = '/dev/null'

plt.figure()
for na, ant in enumerate(ants):
    print "Getting data for antenna %d" % ant
    try:
        xx = redis.hgetall("visdata://%d/%d/xx" % (ant, ant))
    except:
        print "Couldn't get %d:xx data from redis" % ant
        xx = {}
    try:
        yy = redis.hgetall("visdata://%d/%d/yy" % (ant, ant))
    except:
        print "Couldn't get %d:yy data from redis" % ant
        yy = {}

    if jd is None:
        # Get time from xx or yy, defaulting to 0 if not present
        jd = xx.get('time', 0) or yy.get('time', 0)
        filename = 'rms.%.5f.png' % float(jd)

    # data is returned as strings
    xxdata = xx.get('data')
    yydata = yy.get('data')

    if xxdata:
        # Unpack string into NArray
        xxdata = np.fromstring(xxdata, dtype=np.float32)
        ax = plt.subplot(nx, ny, na + 1)
        if args.log:
            plt.plot(freqs, 10 * np.log10(xxdata), label='xx')
        else:
            plt.plot(freqs, xxdata, label='xx')

    if yydata:
        # Unpack string into NArray
        yydata = np.fromstring(yydata, dtype=np.float32)
        ax = plt.subplot(nx, ny, na + 1)
        if args.log:
            plt.plot(freqs, 10 * np.log10(yydata), label='yy')
        else:
            plt.plot(freqs, yydata, label='yy')

    if xxdata is not None or yydata is not None:
        plt.text(0.8, 0.8, '%d' % ant, fontsize=12, transform=ax.transAxes)

    # turn off axis text
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticklabels([])
    cur_axes.axes.get_yaxis().set_ticklabels([])

    # plt.title('Ant: %d'%ant)
    # put in a legend for the first plot
    if ant == 0:
        plt.legend(loc='best')

plt.savefig(filename)
if args.show:
    plt.show()
