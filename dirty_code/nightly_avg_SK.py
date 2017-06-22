#! /usr/bin/env python
"""
Averages the outputs of uv_avg into a single nightly file
"""


import aipy as a, numpy as n, sys, os, optparse, pickle,re

o = optparse.OptionParser()
o.set_usage('nightly_avg.py [options] *.uv')
o.add_option('--average_all',action='store_true',
     help='Average everything into a single file instead of nightly')
o.set_description(__doc__)
a.scripting.add_standard_options(o, ant=False, pol=True, cal=False) #SK
opts,args = o.parse_args(sys.argv[1:])


#break the list of files into nights
def file2jd(file):
    return float(re.findall('\D+(\d+.\d+)\D+',file)[0])
nights = {}
nights_c  = {}
#nights = set([round(file2jd(file),0) for file in args])
#nights = dict(zip(nights,[[]]*len(nights)))
for file in args:
    print file
    
    assert(opts.pol in file.split('.'))#SK
    
    jd = file2jd(file)
    if opts.average_all: 
        night = n.int(n.floor(file2jd(args[0])))
    else:
        night = n.int(n.floor(jd))
    F = open(file)
    AVG = pickle.load(F)
    if not nights.has_key(night): 
        nights[night] = {}
        nights_c[night] = {}
    for bl in AVG:
        if bl=='freqs':freqs = AVG['freqs'];continue
        if bl=='counts':continue
        nights[night][bl] = nights[night].get(bl,0) + AVG[bl]*AVG['counts'][bl].astype(float)
        nights_c[night][bl] = nights_c[night].get(bl,0) + AVG['counts'][bl]
for night in nights:
    for bl in nights[night]:
        N = nights_c[night][bl]
        nights[night][bl][N>0] /= N[N>0]
    nights[night]['counts'] = nights_c[night]
    nights[night]['freqs'] = freqs
for night in nights:
    outfile = str(night)+'.%s.avg.pkl'%(opts.pol)#SK
    print "writing: ",outfile
    F = open(outfile,'w')
    pickle.dump(nights[night],F)
    F.close()
