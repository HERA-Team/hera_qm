#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License

from hera_qm import utils
from hera_qm.auto_metrics import auto_metrics_run
import sys

ap = utils.get_metrics_ArgumentParser('auto_metrics')
args = ap.parse_args()
history = ' '.join(sys.argv)

auto_metrics_run(args.metric_outfile,
                 args.raw_auto_files,
                 median_round_modz_cut=args.median_round_modz_cut,
                 mean_round_modz_cut=args.mean_round_modz_cut,
                 edge_cut=args.edge_cut,
                 Kt=args.Kt,
                 Kf=args.Kf,
                 sig_init=args.sig_init,
                 sig_adj=args.sig_adj,
                 chan_thresh_frac=args.chan_thresh_frac,
                 history=history,
                 overwrite=args.clobber)
