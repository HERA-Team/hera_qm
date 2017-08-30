#!/usr/bin/env python

import sys
from hera_qm import utils
from hera_qm import xrfi

o = utils.get_metrics_OptionParser('xrfi')
opts, files = o.parse_args(sys.argv[1:])
history = ' '.join(sys.argv)

xrfi.xrfi_run(files, opts, history)
