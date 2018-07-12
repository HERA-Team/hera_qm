from __future__ import division
import unittest
import nose.tools as nt
import os
import numpy as np
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
from hera_qm.data import DATA_PATH
from xrfi import UVFlag


def test_uvflag_init_UVData():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))
    uvf = UVFlag(uv)
    nt.assert_true(uv.flag_array.shape == uvf.flag_arrays.shape)
