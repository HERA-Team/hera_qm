#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

from hera_qm import utils
from hera_qm import omnical_metrics
import sys

ap = utils.get_metrics_ArgumentParser('omnical_metrics')
args = ap.parse_args()
files = args.files
history = ' '.join(sys.argv)

omnical_metrics.omnical_metrics_run(files, args, history)
