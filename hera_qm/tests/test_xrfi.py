# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

from __future__ import division
import unittest
import nose.tools as nt
import glob
import os
import shutil
import hera_qm.xrfi as xrfi
import numpy as np
import hera_qm.tests as qmtest
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH


test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_c_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'
test_outfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')


class TestFlagXants():
    def test_uvdata(self):
        uv = UVData()
        uv.read_miriad(test_d_file)
        xant = uv.get_ants()[0]
        xrfi.flag_xants(uv, xant)
        nt.assert_true(np.all(uv.flag_array[uv.ant_1_array == xant, :, :, :]))
        nt.assert_true(np.all(uv.flag_array[uv.ant_2_array == xant, :, :, :]))

    def test_uvcal(self):
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        xant = uvc.ant_array[0]
        xrfi.flag_xants(uvc, xant)
        nt.assert_true(np.all(uvc.flag_array[0, :, :, :, :]))
