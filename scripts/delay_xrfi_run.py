#!/usr/bin/env python

import sys
import numpy as np
from hera_qm import utils as qm_utils
from hera_qm import xrfi
from hera_cal import delay_filter
from hera_cal import io
from pyuvdata import UVData

a = qm_utils.get_metrics_ArgumentParser('delay_xrfi_run')
args = a.parse_args()
filename = args.filename
history = ' '.join(sys.argv)

# Read data, apply delay filter, update UVData object
uv = UVData()
uv.read_miriad(filename)
# apply a priori waterfall flags
waterfalls = []
if args.waterfalls is not None:
    for wfile in args.waterfalls.split(','):
        d = np.load(wfile)
        waterfalls.append(d['waterfall'])
    if len(waterfalls) > 0:
        wf_full = sum(waterfalls).astype(bool)
        uv.flag_array += xrfi.waterfall2flags(wf_full, uv)

# Stuff into delay filter object, run delay filter
dfil = delay_filter.Delay_Filter()
dfil.load_data(uv)
dfil.run_filter(standoff=args.standoff, horizon=args.horizon, tol=args.tol,
                window=args.window, skip_wgt=args.skip_wgt, maxiter=args.maxiter)
io.update_uvdata(dfil.input_data, data=dfil.filtered_residuals)

# Run xrfi
xrfi.xrfi_run(dfil.input_data, args, history)
