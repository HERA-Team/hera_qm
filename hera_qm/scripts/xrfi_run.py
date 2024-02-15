#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_run')
args = ap.parse_args()
history = ' '.join(sys.argv)

xrfi.xrfi_run(args.ocalfits_files,
              args.acalfits_files,
              args.model_files,
              args.data_files, 
              a_priori_flag_yaml=args.a_priori_flag_yaml,
              a_priori_ants_only=not(args.a_apriori_times_and_freqs),
              use_cross_pol_vis=not(args.skip_cross_pol_vis),
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
              omnical_median_filter=not(args.skip_omnical_median_filter),
              omnical_mean_filter=not(args.skip_omnical_mean_filter),
              omnical_chi2_median_filter=not(args.skip_omnical_chi2_median_filter),
              omnical_chi2_mean_filter=not(args.skip_omnical_chi2_mean_filter),
              omnical_zscore_filter=not(args.skip_omnical_zscore_filter),
              abscal_median_filter=not(args.skip_abscal_median_filter),
              abscal_mean_filter=not(args.skip_abscal_mean_filter),
              abscal_chi2_median_filter=not(args.skip_abscal_chi2_median_filter),
              abscal_chi2_mean_filter=not(args.skip_abscal_chi2_mean_filter),
              abscal_zscore_filter=not(args.skip_abscal_zscore_filter),
              omnivis_median_filter=not(args.skip_omnivis_median_filter),
              omnivis_mean_filter=not(args.skip_omnivis_mean_filter),
              auto_median_filter=not(args.skip_auto_median_filter),
              auto_mean_filter=not(args.skip_auto_mean_filter),
              cross_median_filter=args.use_cross_median_filter,
              cross_mean_filter=not(args.skip_cross_mean_filter),
              metrics_files=args.metrics_files, clobber=args.clobber)
