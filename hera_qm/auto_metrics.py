# -*- coding: utf-8 -*-
# Copyright (c) 2021 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to compute per Antenna metrics using day-long autocorrelations."""
import numpy as np
from copy import deepcopy
from .version import hera_qm_version_str
