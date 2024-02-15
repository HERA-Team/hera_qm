#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import sys
from hera_qm import utils
from hera_qm import xrfi

ap = utils.get_metrics_ArgumentParser('xrfi_apply')
args = ap.parse_args()
filename = args.filename
history = ' '.join(sys.argv)

xrfi.xrfi_h1c_apply(filename, history, infile_format=args.infile_format, xrfi_path=args.xrfi_path,
                    outfile_format=args.outfile_format, extension=args.extension,
                    overwrite=args.overwrite, flag_file=args.flag_file, waterfalls=args.waterfalls,
                    output_uvflag=args.output_uvflag, output_uvflag_ext=args.output_uvflag_ext)
