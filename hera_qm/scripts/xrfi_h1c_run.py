#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_h1c_run')
args = ap.parse_args()
filename = args.filename
history = ' '.join(sys.argv)

xrfi.xrfi_h1c_run(filename, history, infile_format=args.infile_format,
                  extension=args.extension, summary=args.summary, summary_ext=args.summary_ext,
                  xrfi_path=args.xrfi_path, model_file=args.model_file,
                  model_file_format=args.model_file_format, calfits_file=args.calfits_file,
                  kt_size=args.kt_size, kf_size=args.kf_size, sig_init=args.sig_init,
                  sig_adj=args.sig_adj, px_threshold=args.px_threshold,
                  freq_threshold=args.freq_threshold, time_threshold=args.time_threshold,
                  ex_ants=args.ex_ants, metrics_file=args.metrics_file, filename=filename[0])
