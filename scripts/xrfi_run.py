#!/usr/bin/env python

import sys
from hera_qm import utils
from hera_qm import xrfi

a = utils.get_metrics_ArgumentParser('xrfi')
args = a.parse_args()
files = args.files
history = ' '.join(sys.argv)

xrfi.xrfi_run(files, args, history)
