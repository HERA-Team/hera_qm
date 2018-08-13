# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import
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
        # make fake data
        data = np.zeros((self.size, self.size))
        for i in range(data.shape[1]):
            data[:, i] = i * np.ones_like(data[:, i])
        # medmin should be self.size - 1 for these data
        medmin = xrfi.medmin(data)
        nt.assert_true(np.allclose(medmin, self.size - 1))

    def test_medminfilt(self):
        # make fake data
        data = np.zeros((self.size, self.size))
        for i in range(data.shape[1]):
            data[:, i] = i * np.ones_like(data[:, i])
        # run medmin filt
        Kt = 8
        Kf = 8
        d_filt = xrfi.medminfilt(data, Kt=Kt, Kf=Kf)

        # build up "answer" array
        ans = np.zeros((self.size, self.size))
        for i in range(data.shape[1]):
            if i < self.size - Kf:
                ans[:, i] = i + (Kf - 1)
            else:
                ans[:, i] = self.size - 1
        nt.assert_true(np.allclose(d_filt, ans))

        # test cases where filters are larger than data dimensions
        Kt = self.size + 1
        Kf = self.size + 1
        d_filt = uvtest.checkWarnings(xrfi.medminfilt, [data, Kt, Kf], nwarnings=2,
                                      category=[UserWarning, UserWarning],
                                      message=['Kt value {:d} is larger than the data'.format(Kt),
                                               'Kf value {:d} is larger than the data'.format(Kt)])
        ans = (self.size - 1) * np.ones_like(d_filt)
        nt.assert_true(np.allclose(d_filt, ans))

    def test_detrend_deriv(self):
        # make fake data
        data = np.zeros((self.size, self.size))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = j * i**2 + j**3
        # run detrend_deriv in both dimensions
        dtdf = xrfi.detrend_deriv(data, df=True, dt=True)
        ans = np.ones_like(dtdf)
        nt.assert_true(np.allclose(dtdf, ans))

        # only run along frequency
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = j**3
        df = xrfi.detrend_deriv(data, df=True, dt=False)
        ans = np.ones_like(df)
        nt.assert_true(np.allclose(df, ans))

        # only run along time
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data[i, j] = i**3
        dt = xrfi.detrend_deriv(data, df=False, dt=True)
        ans = np.ones_like(dt)
        nt.assert_true(np.allclose(dt, ans))

        # catch error of df and dt both being False
        nt.assert_raises(ValueError, xrfi.detrend_deriv, data, False, False)

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
