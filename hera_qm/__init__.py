# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""init file for hera_qm."""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hera_qm")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

del version
del PackageNotFoundError

from . import xrfi  # noqa
from . import vis_metrics  # noqa
from . import ant_metrics  # noqa
from . import auto_metrics  # noqa
from . import firstcal_metrics  # noqa
from . import omnical_metrics  # noqa 
from . import metrics_io  # noqa