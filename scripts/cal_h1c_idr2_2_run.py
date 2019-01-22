#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

a = utils.get_metrics_ArgumentParser('xrfi_cal_h1c_idr2_2_run')
args = a.parse_args()
history = ' '.join(sys.argv)

xrfi.xrfi_cal_h1c_idr2_2_run(args.omni_calfits_file, abs_calfits_file, args.model_file,
                             history, metrics_ext=args.metrics_ext,
                             flag_ext=args.flags_ext, xrfi_path=args.xrfi_path,
                             kt_size=args.kt_size, kf_size=args.kf_size,
                             sig_init=args.sig_init, sig_adj=args.sig_adj,
                             freq_threshold=args.freq_threshold,
                             time_threshold=args.time_threshold, ex_ants=args.ex_ants,
                             metrics_file=args.metrics_file.)
