#!/usr/bin/env python

from hera_qm import utils
from hera_qm import omnical_metrics
import sys

a = utils.get_metrics_ArgumentParser('omnical_metrics')
args = a.parse_args()
files = args.files
history = ' '.join(sys.argv)

omnical_metrics.omnical_metrics_run(files, args, history)
