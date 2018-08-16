#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

from hera_qm import utils
from hera_qm import ant_metrics
import sys

a = utils.get_metrics_ArgumentParser('ant_metrics')
args = a.parse_args()
history = ' '.join(sys.argv)
if args.pol == '':
    args.pol = ['xx', 'yy', 'xy', 'xy']
else:
    args.pol = list(args.pol.split(','))
ant_metrics.ant_metrics_run(args.files, pols=args.pol, crossCut=args.crossCut,
                            deadCut=args.deadCut,
                            alwaysDeatCut=args.alwaysDeatCut,
                            metrics_path=args.metrics_path,
                            extension=args.extension,
                            vis_format=args.vis_format,
                            verbose=args.verbose, history)
