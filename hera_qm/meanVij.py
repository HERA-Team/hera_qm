#! /usr/bin/env python
import sys
import numpy as np
import aipy
import optparse
import capo
import re
import os
import sys
from matplotlib import pyplot as plt


def file2jd(zenuv):
    return float(re.findall(r'\d+\.\d+', zenuv)[0])  # get JD of a file

o = optparse.OptionParser()
o.set_usage('meanVij.py [options] *.uvcRRE')
o.set_description(__doc__)
aipy.scripting.add_standard_options(o, cal=True, pol=True)
o.add_option('-c', '--chan', dest='chan', default=None,
             help='Channel range in form "lo_hi" to average over.')
o.add_option('-t', '--time', dest='time', default=None,
             help='Time range in form "lo_hi" (index) to average over.')
o.add_option('--ba', dest='badants', default='', help='comma separated list '
             'of bad antennas.')
o.add_option('--ants', dest='include_ants', default=None, help='comma separated '
             'list of antennas to include.')
o.add_option('--outpath', default=None, help='Full path of file to print list '
             'of bad antennae [1 sigma deviants] to.')
o.add_option('--autos', dest='autos', default=False, action='store_true',
             help='Plot autocorrelations instead of crosses.')
o.add_option('--antpos', dest='pos', default=False, action='store_true',
             help='Plot mean(Vij) as color on antenna positions.')
o.add_option('--plotMeanVij', dest='pmvij', default=False, action='store_true',
             help='Plot mean(Vij) as a function of antenna number.')
o.add_option('--hist', dest='hist', default=False, action='store_true',
             help='Plot histogram of mean(Vij).')
o.add_option('--list', dest='list', default=False, action='store_true',
             help='List all antennas and mean(Vij) in decending order.')
o.add_option('--skiplast', dest='skip', default=False, action='store_true',
             help='Skip final file, which can sometimes screw things up with '
             'too few time integrations.')
opts, args = o.parse_args(sys.argv[1:])
if opts.skip:
    args = args[:-1]

# parse options
if opts.time is not None:
    tlo, thi = map(int, opts.time.split('_'))
else:
    tlo, thi = 0, None
if opts.chan is not None:
    clo, chi = map(int, opts.chan.split('_'))
else:
    clo, chi = 0, None
if not len(opts.badants) == 0:
    badants = map(int, opts.badants.split(','))
else:
    badants = []
if opts.include_ants is None:
    include_ants = None
else:
    include_ants = map(int, opts.include_ants.split(','))

vis_stor = {}
flg_stor = {}

# Loop through files
for uv in args:
    print '    Reading %s...' % uv
    if opts.autos:
        times, data, flags = capo.arp.get_dict_of_uv_data([uv], 'auto', opts.pol)
    else:
        times, data, flags = capo.arp.get_dict_of_uv_data([uv], 'cross', opts.pol)
    ants_data = np.unique([[a, b] for a, b in data.keys()])
    if include_ants is not None:
        ants_data = [ant for ant in ants_data if ant in include_ants]
    for bl in data:
        if include_ants is not None:
            if (bl[0] not in include_ants) or (bl[1] not in include_ants):
                continue
        for ant in bl:
            try:
                d = data[bl][opts.pol].T
            except(KeyError):
                if bl[0] not in badants and bl[1] not in badants:
                    print 'KeyError on {bl}'.format(bl=bl)
                continue
            try:
                vis_stor[ant] += np.abs(d)
                flg_stor[ant] += np.ones_like(vis_stor[ant])
            except(KeyError):
                vis_stor[ant] = np.abs(d)
                flg_stor[ant] = np.ones_like(vis_stor[ant])

final_ants = np.sort(np.array(vis_stor.keys()))
# average all abs visibilities |Vij| over j per i
avgs = {}
for ant in final_ants:
    avgs[ant] = np.nanmean(vis_stor[ant][clo:chi, tlo:thi] /
                           flg_stor[ant][clo:chi, tlo:thi])

avgs_include = [avgs[i] for i in final_ants if i not in badants]
mean_avg = np.nanmean(avgs_include)
std_avg = np.nanstd(avgs_include)
med_avg = np.nanmedian(avgs_include)
mad_avg = np.nanmedian(np.abs(avgs_include - med_avg))
cut = med_avg - 2 * mad_avg

if opts.pos:
    # get antenna positions
    print 'reading, %s' % opts.cal
    exec("import {calfile} as cal".format(calfile=opts.cal))
    antpos = cal.prms['antpos_ideal']

    _x, _y, _avg = [], [], []
    for i in final_ants:
        x, y = antpos[i]['top_x'], antpos[i]['top_y']
        _x.append(x)
        _y.append(y)
        _avg.append(avgs[i])

    # Plot channel/time average on antenna positions
    print 'Plotting antpos'
    plt.scatter(_x, _y, s=40, c=np.log10(np.array(_avg)))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(r'$\left\langle |V_{ij}| \right\rangle_{j,t,\nu}$')

    # Cross out bad ants
    for i in badants:
        plt.plot(_x[i], _y[i], 'kx', ms=20)

    plt.xlabel('E-W')
    plt.ylabel('N-S')
    plt.show()
    plt.close()

if opts.pmvij:
    for i in final_ants:
        plt.plot(i, avgs[i], 'bo', ms=8)
    for i in badants:
        try:
            plt.plot(i, avgs[i], 'kx', ms=10)
        except(KeyError):
            plt.plot(i, 0, 'kx', ms=10)
    plt.axhline(med_avg, color='k')
    plt.axhline(cut, ls='--', color='r')
    plt.fill_between(final_ants, med_avg - mad_avg, med_avg + mad_avg,
                     color='b', alpha=0.5)
    plt.fill_between(final_ants, med_avg - 2 * mad_avg, med_avg + 2 * mad_avg,
                     color='b', alpha=0.2)
    plt.grid()
    plt.xlim(final_ants.min() - 0.5, final_ants.max() + 0.5)
    plt.ylabel(r'$\left\langle |V_{ij}| \right\rangle_{j,t,\nu}$', size=15)
    plt.xlabel('Antenna number')
    plt.show()

if opts.hist:
    temp = plt.hist([avgs[i] for i in final_ants if np.isfinite(avgs[i])], 20)
    plt.axvline(med_avg, color='k')
    plt.axvline(cut, ls='--', color='red')
    plt.xlabel(r'$\left\langle |V_{ij}| \right\rangle_{j,t,\nu}$', size=15)
    plt.ylabel('Count')
    plt.title('Mean Vij Histogram')
    plt.show()

out = []
for i in avgs:
    if avgs[i] <= cut:
        out.append(i)

JD = int(file2jd(args[0]))
if len(args) > 1:
    heading = str(JD) + ' Baddies:\n'
else:
    heading = args[0] + ' Baddies:\n'
out = str(out)[1:-1] + '\n'
if opts.list:
    data = np.array([avgs[ant] for ant in final_ants])
    inds = np.argsort(data)
    data = data[inds[-1::-1]]
    sorted_ants = final_ants[inds[-1::-1]]
    data = np.stack([sorted_ants, data], axis=1)

if opts.outpath is not None:
    if os.path.exists(opts.outpath):
        with open(opts.outpath, 'a') as outfile:
            outfile.write(heading)
            outfile.write(out)
            if opts.list:
                outfile.write('All ants and mean(Vij): \n')
                np.savetxt(outfile, data)
    else:
        with open(opts.outpath, 'w+') as outfile:
            outfile.write(heading)
            outfile.write(out)
            if opts.list:
                outfile.write('All ants and mean(Vij): \n')
                np.savetxt(outfile, data)
else:
    print(heading)
    print(out)
    if opts.list:
        print('All ants and mean V(ij):')
        print(data)
