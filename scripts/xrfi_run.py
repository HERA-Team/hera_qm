#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

a = utils.get_metrics_ArgumentParser('xrfi_run')
args = a.parse_args()
history = ' '.join(sys.argv)

xrfi.xrfi_run(args.ocalfits_file, args.acalfits_file, args.model_file, args.data_file,
              history, init_metrics_ext=args.init_metrics_ext,
              init_flags_ext=args.init_flags_ext, final_metrics_ext=args.final_metrics_ext,
              final_flags_ext=args.final_flags_ext, xrfi_path=args.xrfi_path,
              kt_size=args.kt_size, kf_size=args.kf_size, sig_init=args.sig_init,
              sig_adj=args.sig_adj, freq_threshold=args.freq_threshold,
              time_threshold=args.time_threshold, ex_ants=args.ex_ants,
              metrics_file=args.metrics_file, cal_ext=args.cal_ext, clobber=args.clobber)
