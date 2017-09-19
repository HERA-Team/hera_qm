#!/usr/bin/env python

from hera_qm import utils
from hera_qm import ant_metrics
import sys

a = utils.get_metrics_ArgumentParser('ant_metrics')
args = a.parse_args()
history = ' '.join(sys.argv)

ant_metrics.ant_metrics_run(args.files, args, history)
