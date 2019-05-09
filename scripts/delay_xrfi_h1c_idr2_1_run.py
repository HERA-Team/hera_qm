#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
import numpy as np
from hera_qm import utils as qm_utils
from hera_qm import xrfi
from hera_cal import delay_filter
from hera_cal import io
from pyuvdata import UVData

ap = qm_utils.get_metrics_ArgumentParser('delay_xrfi_h1c_idr2_1_run')
args = ap.parse_args()
filename = args.filename
history = ' '.join(sys.argv)

# Read data, apply delay filter, update UVData object
uv = UVData()
uv.read_miriad(filename)
# apply a priori waterfall flags
if args.waterfalls is not None:
    waterfalls = args.waterfalls.split(',')
    if len(waterfalls) > 0:
        xrfi.flag_apply(waterfalls, uv, force_pol=True)

# set kwargs
kwargs = {}
if args.window == 'tukey':
    kwargs['alpha'] = args.alpha

# Stuff into delay filter object, run delay filter
dfil = delay_filter.Delay_Filter()
dfil.load_data(uv)
dfil.run_filter(standoff=args.standoff, horizon=args.horizon, tol=args.tol,
                window=args.window, skip_wgt=args.skip_wgt, maxiter=args.maxiter, **kwargs)
io.update_uvdata(dfil.input_data, data=dfil.filtered_residuals, flags=dfil.flags)

# Run xrfi
xrfi.xrfi_h1c_run(dfil.input_data, history, infile_format=args.infile_format,
                  extension=args.extension, summary=args.summary, summary_ext=args.summary_ext,
                  xrfi_path=args.xrfi_path, model_file=args.model_file,
                  model_file_format=args.model_file_format, calfits_file=args.calfits_file,
                  kt_size=args.kt_size, kf_size=args.kf_size, sig_init=args.sig_init,
                  sig_adj=args.sig_adj, px_threshold=args.px_threshold,
                  freq_threshold=args.freq_threshold, time_threshold=args.time_threshold,
                  ex_ants=args.ex_ants, metrics_file=args.metrics_file, filename=args.filename)
