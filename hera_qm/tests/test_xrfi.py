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
from hera_qm import UVFlag


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

    def test_uvflag(self):
        uvf = UVFlag(test_f_file)
        uvf.to_flag()
        xant = uvf.ant_1_array[0]
        xrfi.flag_xants(uvf, xant)
        nt.assert_true(np.all(uvf.flag_array[uvf.ant_1_array == xant, :, :, :]))
        nt.assert_true(np.all(uvf.flag_array[uvf.ant_2_array == xant, :, :, :]))

    def test_input_error(self):
        nt.assert_raises(ValueError, xrfi.flag_xants, 4, 0)

    def test_uvflag_waterfall_error(self):
        uvf = UVFlag(test_f_file)
        uvf.to_waterfall()
        uvf.to_flag()
        nt.assert_raises(ValueError, xrfi.flag_xants, uvf, 0)

    def test_uvflag_not_flag_error(self):
        uvf = UVFlag(test_f_file)
        nt.assert_raises(ValueError, xrfi.flag_xants, uvf, 0)

    def test_not_inplace_uvflag(self):
        uvf = UVFlag(test_f_file)
        xant = uvf.ant_1_array[0]
        uvf2 = xrfi.flag_xants(uvf, xant, inplace=False)
        nt.assert_true(np.all(uvf2.flag_array[uvf2.ant_1_array == xant, :, :, :]))
        nt.assert_true(np.all(uvf2.flag_array[uvf2.ant_2_array == xant, :, :, :]))

    def test_not_inplace_uvdata(self):
        uv = UVData()
        uv.read_miriad(test_d_file)
        xant = uv.get_ants()[0]
        uv2 = xrfi.flag_xants(uv, xant, inplace=False)
        nt.assert_true(np.all(uv2.flag_array[uv2.ant_1_array == xant, :, :, :]))
        nt.assert_true(np.all(uv2.flag_array[uv2.ant_2_array == xant, :, :, :]))


class TestPreProcessingFunctions():
    def __init__(self):
        self.size = 100

    def test_medmin(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_medminfilt(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_detrend_deriv(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_detrend_medminfilt(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_detrend_medfilt(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)


class TestFlaggingFunctions():

    def test_watershed_flag(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_ws_flag_waterfall(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_flag(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_unflag(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_flag_apply(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)


class TestHighLevelFunctions():

    def test_calculate_metric(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)


class TestPipelines():

    def test_xrfi_h1c_pipe(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)


class TestWrappers():

    def test_xrfi_h1c_run(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_xrfi_h1c_apply(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)
