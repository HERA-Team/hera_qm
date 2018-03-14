#!/usr/bin/env python

import sys
from hera_qm import utils as qm_utils
from hera_qm import xrfi
from hera_cal import delay_filter
from hera_cal import io
from pyuvdata import UVData

ax = qm_utils.get_metrics_ArgumentParser('xrfi_run')
ad = delay_filter.delay_filter_argparser()
args_x = ax.parse_known_args()[0]
args_d = ad.parse_known_args()[0]
filename = args_x.filename
history = ' '.join(sys.argv)

# Read data, apply delay filter, update UVData object
dfil = delay_filter.Delay_Filter()
uv = UVData()
uv.read_miriad(filename)
dfil.load_data(uv)
dfil.run_filter(standoff=args_d.standoff, horizon=args_d.horizon, tol=args_d.tol,
                window=args_d.window, skip_wgt=args_d.skip_wgt, maxiter=args_d.maxiter)
io.update_uvdata(dfil.input_data, data=dfil.filtered_residuals)

# Run xrfi
xrfi.xrfi_run(dfil.input_data, args_x, history)
