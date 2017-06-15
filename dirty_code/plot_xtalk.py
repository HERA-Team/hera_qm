#! /usr/bin/env python

from pylab import *
import sys
import numpy as n 
import optparse

o = optparse.OptionParser()
o.set_usage('plot_xtalk.py *.uv')
o.set_description(__doc__)
o.add_option('--plotbl',default='0_16;',help='The baselines to plot, split by ";" [default=0_16;]')
o.add_option('--verbose', action='store_true',
    help="Print a lot of stuff.")
#o.add_option('--legend',action='store_true',
#    help="""enable the plot legend. Its usually in the way, so its disabled by
#default""")
o.add_option('--yscale',type='float',
    help='set the amount to offset each file')
opts, args = o.parse_args(sys.argv[1:])
bls = opts.plotbl.split(';')
if args[0].endswith('npz'):
    SIGMAS = []
    AVGS = []
    for F in sys.argv[1:]:
        A = n.load(F)
        SIGMAS.append(A['SIGMA'])
        AVGS.append(A['AVG'])
        freqs = A['freqs']*1e3
    figure(31,figsize=(6,5))
    clf()
    ch = n.argwhere(n.logical_and(freqs>120,freqs<180))
    for i,AVG in enumerate(AVGS):
        plot(freqs[ch],n.real(AVG[ch])/n.max(n.real(AVGS)),label='day %d'%i)
    SIGMAS=n.array(SIGMAS) 
    SIGMA = n.sqrt(n.mean(SIGMAS**2,axis=0))[ch].squeeze()/n.max(n.real(AVGS))
    AVG = n.mean(AVGS,axis=0)[ch].squeeze()/n.max(n.real(AVGS))
    F = freqs[ch].squeeze()
    y1 = AVG-SIGMA/2
    y2 = AVG+SIGMA/2
    fill_between(F,y1=n.real(y1),y2=n.real(y2),
        facecolor='0.3',
        edgecolor='0.3',
        zorder=20,
        alpha=0.9)
    plot(freqs[ch],n.real(n.mean(AVGS,axis=0)[ch]),'k')
    ylabel('$\mathcal{R}e(V_{8,12})$  [counts]')
    xlabel('frequency [MHz]')
    legend(loc='lower right',ncol=4,columnspacing=0.5,numpoints=4,prop={'size':10})
    subplots_adjust(left=0.15,bottom=0.15)
    show()
else:
    import pickle
    for i,filename in enumerate(args):
	print filename
        P = pickle.load(open(filename))
        for bl in bls:
            try:
                plot(P['freqs']*1e3,i*opts.yscale+n.real(P[bl]),label=bl+'-'+filename)
            except(KeyError):
               print opts.plotbl,"not found in ",filename
            #if i>10:break
            text(100,i*opts.yscale,filename,fontsize=10)
        xlabel('MHz')
        ylabel('real(<vis>)')
    grid()
    show()
    
