#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_day_threshold_run')
args = ap.parse_args()
history = ' '.join(sys.argv)

if args.run_if_first is None or sorted(args.data_files)[0] == args.run_if_first:
    xrfi.day_threshold_run(args.data_files, history, kt_size=args.kt_size,
                           kf_size=args.kf_size, nsig_f=args.nsig_f, nsig_t=args.nsig_t,
                           clobber=args.clobber)
else:
    print(sorted(args.data_files)[0], 'is not', args.run_if_first, '...skipping.')
