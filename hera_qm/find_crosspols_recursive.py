#!/usr/bin/env python
import aipy, capo as C, optparse, numpy as np, matplotlib.pyplot as plt, sys
o = optparse.OptionParser()
o.set_description(__doc__)
o.set_usage('find_crosspols_recursive.py *xx.uvcRRE (only use XX pol, script will find the others).')
aipy.scripting.add_standard_options(o, cal=True)
o.add_option('-A','--startant',dest='startant',default='0',help='antenna to start relative to (default is all antennae, i.e. start with zero)')
o.add_option('-b','--badants',dest='ba',default=None,help='bad antennae to remove, separated by commas. e.g. "-b 1,2,3"')
o.add_option('-v','--verbose',dest='verb',action='store_true',help='Toggle verbosity')
opts, args = o.parse_args(sys.argv[1:])

#parse files: assumes both pols are in the same directory
if 'xx' in args[0]:
    xxfiles = args
    xyfiles = [w.replace('xx','xy') for w in args]
    yxfiles = [w.replace('xx','yx') for w in args]
    yyfiles = [w.replace('xx','yy') for w in args]
else:
    print 'Hand this script the xx files. Assumes all pols are in the same directory'
    raise Exception

#parse array data

if not opts.ba is None: badants = map(int,opts.ba.split(','))
else: badants = []
print 'reading, %s'%opts.cal
exec("import {calfile} as cal".format(calfile=opts.cal))
antpos = cal.prms['antpos']
nants = len(antpos.keys())

all_avgs = np.zeros((nants,nants))
offenders = []
if opts.verb: print 'Rel_ant : avg>2sigma'
for anchor_ant in range(int(opts.startant),nants):
    if anchor_ant in badants:
        all_avgs[anchor_ant,:] = np.nan
        continue
    #data read
    tpxx,dpxx,fpxx = C.arp.get_dict_of_uv_data(xxfiles,antstr=str(anchor_ant),polstr='xx')
    tpxy,dpxy,fpxy = C.arp.get_dict_of_uv_data(xyfiles,antstr=str(anchor_ant),polstr='xy')
    tpyx,dpyx,fpyx = C.arp.get_dict_of_uv_data(yxfiles,antstr=str(anchor_ant),polstr='yx')
    tpyy,dpyy,fpyy = C.arp.get_dict_of_uv_data(yyfiles,antstr=str(anchor_ant),polstr='yy')
    #data analysis
    avg_ratios = []
    for ant in range(nants):
        #parse miriad keys
        if anchor_ant == ant:
            #neglect auto
            avg_ratios.append(np.nan)
            continue
        elif anchor_ant < ant: tup = (anchor_ant,ant)
        else: tup = (ant,anchor_ant)
        try:
            avg_xx = np.nanmean(np.absolute(dpxx[tup]['xx']))
        except IndexError:
            print 'Index error on antenna %i'%ant
            continue
        avg_xy = np.nanmean(np.absolute(dpxy[tup]['xy']))
        avg_yx = np.nanmean(np.absolute(dpyx[tup]['yx']))
        avg_yy = np.nanmean(np.absolute(dpyy[tup]['yy']))
        ratio = (avg_xy + avg_yx)/(avg_xx + avg_yy)
        avg_ratios.append(ratio)
    del(tpxx);del(dpxx);del(fpxx)
    del(tpxy);del(dpxy);del(fpxy)
    del(tpyx);del(dpyx);del(fpyx)
    del(tpyy);del(dpyy);del(fpyy)
    
    #crosspol'd antennae are the ones that deviate from avg by 2 sigma
    crosspols = np.where(avg_ratios > np.nanmean(avg_ratios)+2*np.nanstd(avg_ratios))[0]
    if opts.verb: print anchor_ant,':',crosspols
    offenders.append(crosspols)
    all_avgs[anchor_ant,:] = avg_ratios
