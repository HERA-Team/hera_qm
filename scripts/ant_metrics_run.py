#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

from hera_qm import utils
from hera_qm import ant_metrics
import sys

ap = utils.get_metrics_ArgumentParser('ant_metrics')
args = ap.parse_args()
history = ' '.join(sys.argv)
ant_metrics.ant_metrics_run(args.sum_files,
                            diff_files=args.diff_files,
                            apriori_xants=args.apriori_xants,
                            a_priori_xants_yaml=args.a_priori_xants_yaml,
                            crossCut=args.crossCut,
                            deadCut=args.deadCut,
                            metrics_path=args.metrics_path,
                            extension=args.extension,
                            overwrite=args.clobber,
                            Nbls_per_load=args.Nbls_per_load,
                            verbose=args.verbose,
                            history=history)
