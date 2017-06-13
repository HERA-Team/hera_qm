#!/usr/bin/env python
import numpy as np, matplotlib.pyplot as plt, glob, os, sys
import optparse
o = optparse.OptionParser()
o.add_option('-y', type='str', help='Year label (2013 or 2014)')
o.add_option('-e', type='str', help='Epoch label (1 or 2)')
o.add_option('--minJD', type='int', help='Minimum JD')
o.add_option('--maxJD', type='int', help='Maximum JD')
o.add_option('--dest', type='str', default=os.getcwd(), help='Destination of output file')
opts,args = o.parse_args(sys.argv[1:])

path2epoch='/data4/paper/%sEoR/Analysis/ProcessedData/epoch%s/omni_v3_xtalk/'%(opts.y,opts.e)

for jd in range(opts.minJD,opts.maxJD):
    for pol in ['xx','yy']:
        globname = 'zen.%i.*.%s.npz'%(jd,pol)
        files2query = []
        for f in sorted(glob.glob(path2epoch+globname)):
            files2query.append(f)
        if len(files2query) == 0:
            print 'No files of type %s'%path2epoch+globname
            continue
        chisq_of_jdpol = np.zeros((len(files2query)*19,203),dtype='float32')
        
        #begin file read
        for i,f in enumerate(files2query):
            d = np.load(f)
            try:
                if d['chisq'].shape[0] != 19:
                    pad = np.zeros((19-d['chisq'].shape[0], 203))
                    csq = np.vstack((d['chisq'],pad))
                else:
                    csq = d['chisq']
                chisq_of_jdpol[i*19:(i+1)*19,:] = csq
            except ValueError:
                print 'file %s has shape %s'%(f,str(d['chisq'].shape))
                #this should not happen
        outname = opts.dest+'/omnichisq.%i.%s.npz'%(jd,pol)
        print 'Saving %s'%outname
        np.savez(outname,chisq=chisq_of_jdpol)

