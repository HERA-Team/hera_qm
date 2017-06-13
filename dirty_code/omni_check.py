#! /usr/bin/env python

import omnical
import aipy
import pylab
import numpy
import capo
import pickle
import matplotlib.pyplot as plt
import optparse
import os, sys

### Options ###
o = optparse.OptionParser()
o.set_usage('omni_check.py [options] *.npz')
aipy.scripting.add_standard_options(o,pol=True)
o.add_option('--chisq',dest='chisq',default=False,action="store_true",
            help='Plot chisq.')
o.add_option('--gains',dest='gains',default=False,action="store_true",
            help='Plot gains of each antenna solved for.')
o.add_option('--chisqant',dest='chisqant',default=False,action="store_true",
            help='Plot chisqs per antenna.')
o.set_description(__doc__)
#o.add_option('-C',dest='cal',default='psa6622_v003',type='string',
#            help='Path and name of calfile.')
opts,args = o.parse_args(sys.argv[1:])


### Plot ChiSq ####
#if opts.pol == -1: pol = args[0].split('.')[3] #XXX hard-coded for *pol.npz files
pol = opts.pol
if opts.chisq == True:
    chisqs = []
    for i,file in enumerate(args):
        print 'Reading',file
        file = numpy.load(file)
        try: #reads *pol.npz files
            chisq = file['chisq '+str(pol)] #shape is (#times, #freqs)
        except: #reads .npz files
            chisq = file['chisq']
        for t in range(len(chisq)):
            chisqs.append(chisq[t])
            #chisq is sum of square of (model-data) on all visibilities per time/freq snapshot

    cs = numpy.array(chisqs)
    plt.imshow(numpy.log(cs),aspect='auto',interpolation='nearest',vmax=7,vmin=-6)
    plt.xlabel('Freq Channel',fontsize=10)
    plt.ylabel('Time',fontsize=10)
    plt.tick_params(axis='both',which='major',labelsize=8)
    plt.title('Omnical ChiSquare',fontsize=12)
    plt.colorbar()
    plt.show()


### Plot Gains or Chisqants ###
if opts.gains == True or opts.chisqant == True:
    gains = {} #or chisqant values, depending on option
    for i, f in enumerate(args): #loop over files
        print 'Reading',f
        file = numpy.load(f)
        for a in range(128):
            if opts.chisqant == True: #chisqant / gain
                try: value = file['chisq'+str(a)+pol[0]]/file[str(a)+pol[0]] #XXX only 0th element of pol
                except: continue
                try: gains[a].append(value)
                except: gains[a] = [value]
                vmax=0.05
                vmin=0.0
            if opts.gains == True:
                print "option doesn't exist yet"
                sys.exit()
                try: value = file[str(a)+pol[0]] #XXX only the 0th element of pol
                except: continue
                try: gains[a].append(value)
                except: gains[a] = [value]
                vmax=1.5
        file.close()
    for key in gains.keys():
        gains[key] = numpy.vstack(numpy.abs(gains[key])) #cool thing to stack 2D arrays that only match in 1 dimension
        mk = numpy.ma.masked_where(gains[key] == 1,gains[key]).mask #flags
        gains[key] = numpy.ma.masked_array(gains[key],mask=mk) #masked array
    #Plotting
    means = []
    ants = []
    f,axarr = plt.subplots(8,14,figsize=(14,8),sharex=True,sharey=True)
    for ant in range(max(map(int,gains.keys()))+1):
        i1 = ant/14 #row number
        i2 = ant%14 #col number
        axarr[i1,i2].set_title(ant,fontsize=10)
        axarr[i1,i2].tick_params(axis='both',which='both',labelsize=8)
        try:
            means.append(numpy.median(gains[ant][:,:])) #median, not mean #XXX freq range restriction
            ants.append(ant)
            axarr[i1,i2].imshow(gains[ant],vmax=vmax,aspect='auto',interpolation='nearest')
        except: continue
    f.subplots_adjust(hspace=0.7)
    print 'Bad Antennas (starting with highest chisq):',[ants[i] for i in numpy.argsort(means)[::-1]]
    plt.show()  
    baddies = [ants[i] for i in numpy.where(means > numpy.mean(means)+1.0*numpy.std(means))[0]]
    cut = numpy.mean(means)+1.0*numpy.std(means)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    A = numpy.linspace(0,len(means)-1,len(means))
    B = numpy.sort(means)[::-1]
    plt.plot(A,B,'ko')
    for index,(i,j) in enumerate(zip(A,B)):
        ax.annotate([ants[k] for k in numpy.argsort(means)[::-1]][index], xy=(i,j),size=10)
        ax.axhline(cut, color='purple', label='Avg+Std')
    plt.title('Median Chisq (high to low)')
    plt.show()
    print '1 sigma cut on median chisq: ',baddies
