#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_run_data_only')
args = ap.parse_args()
history = ' '.join(sys.argv)

xrfi.xrfi_run(data_files=args.data_files,
              a_priori_flag_yaml=args.a_priori_flag_yaml,
              use_cross_pol_vis=not(args.skip_cross_pol_vis),
              cross_median_filter=args.cross_median_filter,
              cross_mean_filter=not(args.skip_cross_mean_filter),
              history=history,
              xrfi_path=args.xrfi_path,
              throw_away_edges=not(args.keep_edge_times),
              kt_size=args.kt_size,
              kf_size=args.kf_size,
              sig_init_med=args.sig_init_med,
              sig_adj_med=args.sig_adj_med,
              sig_init_mean=args.sig_init_mean,
              sig_adj_mean=args.sig_adj_mean,
              ex_ants=args.ex_ants,
              Nwf_per_load=args.Nwf_per_load,
              metrics_files=args.metrics_files,
              clobber=args.clobber)
