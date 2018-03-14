#!/usr/bin/env python

from hera_qm import utils
from hera_qm import firstcal_metrics
import sys

a = utils.get_metrics_ArgumentParser('firstcal_metrics')
args = a.parse_args()
files = args.files
history = ' '.join(sys.argv)

firstcal_metrics.firstcal_metrics_run(files, args, history)
