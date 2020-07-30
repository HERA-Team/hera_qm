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

xrfi.xrfi_run(args.ocalfits_files, args.acalfits_files, args.model_files, args.data_files,
              history, xrfi_path=args.xrfi_path, throw_away_edges=not(args.keep_edge_times),
              kt_size=args.kt_size, kf_size=args.kf_size, sig_init=args.sig_init,
              sig_adj=args.sig_adj, ex_ants=args.ex_ants,
              metrics_file=args.metrics_file, clobber=args.clobber)
