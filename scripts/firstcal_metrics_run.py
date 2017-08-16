#!/usr/bin/env python

from hera_qm import utils
from hera_qm import firstcal_metrics
import sys

o = utils.get_metrics_OptionParser('firstcal_metrics')
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

firstcal_metrics.firstcal_metrics_run(files, opts, history)
