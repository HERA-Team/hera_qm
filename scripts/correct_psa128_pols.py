#!/usr/bin/env python

import aipy
import capo
import numpy
import sys, optparse, os

o = optparse.OptionParser()
o.set_usage('correct_psa128_pols.py *xx.uvcRRE [xx pol only, it will find the other pols, assuming they are in the same directories]')
o.add_option('-a','--ant',default='', help='Comma-separated antenna numbers to swap pols.')
o.add_option('-b','--badant',default=None,help='Comma-separated antenna numbers to remove from file')
o.add_option('-v','--verbose',action='store_true',help='Toggle verbosity')
opts,args = o.parse_args(sys.argv[1:])

ANTS = map(int,opts.ant.split(',')) #antennas that need to be swapped x->y, y->x
if not opts.badant is None: 
    BADANTS = map(int,opts.badant.split(','))
else:
    BADANTS = []

def mfunc(uv,p,d,f):
    ant1,ant2 = p[2]
    if ant1 in BADANTS or ant2 in BADANTS: return p,None,None
    if ant1 not in ANTS and ant2 not in ANTS: return p,d,f
    if ant1 in ANTS:
        if pol[0] == 'x': newpol1 = 'y'
        if pol[0] == 'y': newpol1 = 'x'
    else: newpol1 = pol[0]
    if ant2 in ANTS:
        if pol[1] == 'x': newpol2 = 'y'
        if pol[1] == 'y': newpol2 = 'x'
    else: newpol2 = pol[1]
    newpol = newpol1+newpol2
    index = numpy.where(t[newpol]['times'] == p[1])[0][0] #times must match exactly
    try:
        d = data[newpol][p[2]][newpol][index] #collect information from correct pol file
        f = flags[newpol][p[2]][newpol][index]
    except: #if baseline doesn't exist for some reason  
        missing_bls.append(p[2])
        return p,None,None
    return p,d,f

pols = ['xx','xy','yx','yy']
for filename in args:
    files,t,data,flags = {},{},{},{}
    for pol in pols:
        if not os.path.exists(filename.replace('xx',pol)):
            print '%s not found'%filename.replace('xx',pol)
            break #some files do not have all pols due to incomplete restore
        files[pol] = filename.replace('xx',pol) #dictionary of files by pol
        t[pol],data[pol],flags[pol] = capo.miriad.read_files([files[pol]], antstr='all', polstr=pol, verbose=True) #read all the pol files
    
    if len(files)!=4: continue #some files do not have all pols due to incomplete restore (2)
    
    check = [numpy.all(t['xx']['times'] == t['xy']['times']),numpy.all(t['xx']['times'] == t['yx']['times']),numpy.all(t['xx']['times'] == t['yy']['times'])]
    if not numpy.all(check):
        print 'missing integrations... skipping'
        continue #some files do not have all times - incomplete restore?
    
    for pol in files: #loop through 4 pols
        missing_bls = []
        uvi = aipy.miriad.UV(files[pol])
        print files[pol], '->', files[pol]+'c'
        if os.path.exists(files[pol]+'c'):
            print '   File exists... skipping.'
            continue
        uvo = aipy.miriad.UV(files[pol]+'c',status='new')
        uvo.init_from_uv(uvi)
        uvo.pipe(uvi,raw=True,mfunc=mfunc,append2hist='CORRECT_PSA128_POLS:'+' '.join(sys.argv)+'\n')
        index = len(missing_bls) / len(t[pol]['lsts'])
        uvo['history'] += ' Missing bls: ' + str(missing_bls[:index])
        if opts.verbose and len(missing_bls)>0:
            print ' Missing bls: ' + str(missing_bls[:index])
