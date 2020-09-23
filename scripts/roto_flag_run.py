#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('roto_flag_run')
args = ap.parse_args()
history = ' '.join(sys.argv)

correlations='cross'
if args.use_autos:
    correlations='both'

if args.flag_files == 'none':
    args.flag_files = None

if not args.flag_only or args.fname == args.data_files[0]:
    xrfi.roto_flag_run(data_files=args.data_files, flag_files=args.flag_files,
                       a_priori_flag_yaml=args.a_priori_flag_yaml,
                       flag_percentile_freq=args.flag_percentile_freq,
                       flag_percentile_time=args.flag_percentile_time,
                       Nwf_per_load=args.Nwf_per_load,
                       niters=args.niters, correlations=correlations,
                       kt_size=args.kt_size, kf_size=args.kf_size,
                       output_label=args.output_label, clobber=args.clobber,
                       flag_kernel=True, cal_files=args.cal_files,
                       cal_label=args.cal_label,
                       metric_only_mode=args.metric_only, flag_only_mode=args.flag_only)
