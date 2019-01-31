#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

a = utils.get_metrics_ArgumentParser('delay_xrfi_run')
args = a.parse_args()
history = ' '.join(sys.argv)

# Run xrfi
xrfi.delay_xrfi_run(args.vis_file, args.cal_metrics, args.cal_flags, history,
                    input_cal=args.input_cal, standoff=args.standoff, horizon=args.horizon,
                    tol=args.tol, window=args.window, skip_wgt=args.skip_wgt,
                    maxiter=args.maxiter, alpha=args.alpha, metrics_ext=args.metrics_ext,
                    flags_ext=args.flags_ext, cal_ext=args.cal_ext, kt_size=args.kt_size,
                    kf_size=args.kf_size, sig_init=args.sig_init, sig_adj=args.sig_adj,
                    freq_threshold=args.freq_threshold, time_threshold=args.time_threshold,
                    ex_ants=args.ex_ants, metrics_file=args.metrics_file)
