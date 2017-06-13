#!/usr/bin/env python
"""
Performs a sigma cut on the median, overall chisquare from Omnical, per JD.
This scripts acts on the accumulated overall chisqare files formed by capo/sak/scripts/accumulate_omnichisq.py
Returns a text file of the JDs that deviate by less than sigma, and optional plots.
"""
import sys, os, numpy as np, matplotlib.pyplot as plt, optparse
o = optparse.OptionParser()
#o.add_usage('work_with_omnichisq.py [options] *npz')
o.add_option('-c','--chan',default='0_203',help='Channel range to perform median and stdev over.')
o.add_option('-n','--nsigma',default=1,help='Number of sigma to clip on.')
o.add_option('--plot_per_JD',action='store_true',help='Plot waterfall of chisquares per JD. Note that days missing data will be stretched by matplotlib to fill the maximum size imshow frame.')
o.add_option('--mx',default=5,help='Epochs are delineated by their differing flux scales. If plot_per_JD==True, the vmax of the imshow windows is given by this parameter. The minimum is zero.')
o.add_option('--outfile',default=os.getcwd()+'/goodJDs.txt',help='Full path and name of file to output "good" JDs to. The full argument of this scripts execution will always be a header to this file.')
o.add_option('-i',action='store_true',help='Launch interactive session after the rest of the stuff happens.')
opts,chisqfiles = o.parse_args(sys.argv[1:])
cmin,cmax = map(int,opts.chan.split('_'))

if opts.plot_per_JD: 
    Nmx = 5
    if len(chisqfiles)>=4*(Nmx**2):
        raise Exception('Not implemented for that many JDs')
    
    f1,axarr1 = plt.subplots(Nmx,Nmx,sharex=True,sharey=True)
    if not len(chisqfiles)>(Nmx**2): 
        f2,axarr2,f3,axarr3,f4,axarr4=None,None,None,None,None,None
    else: 
        f2,axarr2 = plt.subplots(Nmx,Nmx,sharex=True,sharey=True)
        f3,axarr3 = plt.subplots(Nmx,Nmx,sharex=True,sharey=True)
        f4,axarr4 = plt.subplots(Nmx,Nmx,sharex=True,sharey=True)

jds,meds,stds = [],[],[]

for i,f in enumerate(chisqfiles):
    jd = chisqfiles[i].split('.')[1]
    data = np.load(chisqfiles[i])['chisq']
    #analyze
    print jd,data.shape
    jds.append(int(jd))
    dd = data[:,cmin:cmax]
    meds.append(np.median(dd))
    stds.append(0.67449*np.nanstd(dd)) #median absolute deviation
    #plot
    if opts.plot_per_JD: 
        if i<Nmx**2: ax = axarr1.ravel()[i]
        elif i>=Nmx**2 and i<2*(Nmx**2): ax = axarr2.ravel()[i-Nmx**2]
        elif i>=2*(Nmx**2) and i<3*(Nmx**2): ax = axarr3.ravel()[i-2*(Nmx**2)]
        else: ax = axarr4.ravel()[i-3*(Nmx**2)]
        if not i==len(chisqfiles)-1: ax.imshow(data,aspect='auto',interpolation='None',vmax=opts.mx, vmin=0, extent=[0,data.shape[1],data.shape[0],0])
        else: ax.imshow(data,aspect='auto',interpolation='None',vmax=opts.mx, vmin=0, extent=[0,data.shape[1],1330,0])
        ax.set_title(jd)
if opts.plot_per_JD:
    plt.show()
    plt.close()

#plots of medians
plt.figure(3)
plt.errorbar(np.array(jds)-jds[0],meds,yerr=stds,fmt='o')
plt.xlabel('JD-%i'%jds[0],size=15)
plt.ylabel(r'$\langle\chi^2\rangle^{\rm median}_{t,\nu}$',size=20)
plt.grid()

f=plt.figure(4)
ax = f.add_subplot(111)
for i in range(len(jds)):
    plt.plot(i,sorted(meds)[i],'ko')
    ax.annotate(str(np.array(jds)[np.argsort(meds)][i]-2450000),xy=(i,sorted(meds)[i]))
plt.axhline(np.mean(meds)+np.std(meds),color='m',label='Avg+Std')
plt.legend(loc='best')
ax.set_ylabel(r'$\langle\chi^2\rangle^{\rm median}_{\rm t,%i:%i}$'%(cmin,cmax))
plt.grid()
plt.show()
plt.close()

#write
goodJDs = np.array(jds)[np.array(meds)<(np.mean(meds)+np.std(meds))]
np.savetxt(opts.outfile,goodJDs,header=' '.join(sys.argv[:]),fmt='%i')

if opts.i: import IPython;IPython.embed()
