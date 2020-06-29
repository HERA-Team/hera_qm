#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_h3c_idr2_1_run')
args = ap.parse_args()
flag_command = ' '.join(sys.argv)

xrfi.xrfi_h3c_idr2_1_run(args.ocalfits_files, args.acalfits_files, args.model_files,
                         args.data_files, flag_command, xrfi_path=args.xrfi_path,
                         kt_size=args.kt_size, kf_size=args.kf_size, sig_init=args.sig_init,
                         sig_adj=args.sig_adj, ex_ants=args.ex_ants,
                         metrics_file=args.metrics_file, clobber=args.clobber)
