#!/usr/bin/env python

from hera_qm import utils
from hera_qm import ant_metrics
import sys

o = utils.get_metrics_OptionParser('ant_metrics')
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

ant_metrics.ant_metrics_run(files, opts, history)
