# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
from __future__ import print_function, division, absolute_import

import unittest
import nose.tools as nt
import glob
import os
import shutil
import hera_qm.xrfi as xrfi
import numpy as np
import warnings
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
from pyuvdata import utils as uvutils
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH
from hera_qm import UVFlag


test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_uvfits_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.uvfits')
test_uvh5_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvh5')
test_c_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'
test_f_file_flags = test_d_file + '.testuvflag.flags.h5'  # version in 'flag' mode
test_outfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')
xrfi_path = os.path.join(DATA_PATH, 'test_output')


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


class TestResolveXrfiPath():
    def test_resolve_xrfi_path_given(self):
        dirname = xrfi.resolve_xrfi_path(xrfi_path, test_d_file)
        nt.assert_equal(xrfi_path, dirname)

    def test_resolve_xrfi_path_empty(self):
        dirname = xrfi.resolve_xrfi_path('', test_d_file)
        nt.assert_equal(os.path.dirname(os.path.abspath(test_d_file)), dirname)


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

        # Test error when wrong dimensions are passed
        nt.assert_raises(ValueError, xrfi.medmin, np.ones((5, 4, 3)))

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
        d_filt = uvtest.checkWarnings(xrfi.medminfilt, [data], {'Kt': Kt, 'Kf': Kf}, nwarnings=2,
                                      category=[UserWarning, UserWarning],
                                      message=['Kt value {:d} is larger than the data'.format(Kt),
                                               'Kf value {:d} is larger than the data'.format(Kf)])
        ans = (self.size - 1) * np.ones_like(d_filt)
        nt.assert_true(np.allclose(d_filt, ans))

        # Test error when wrong dimensions are passed
        nt.assert_raises(ValueError, xrfi.medminfilt, np.ones((5, 4, 3)))

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
        nt.assert_raises(ValueError, xrfi.detrend_deriv, data, dt=False, df=False)

        # Test error when wrong dimensions are passed
        nt.assert_raises(ValueError, xrfi.detrend_deriv, np.ones((5, 4, 3)))

    def test_detrend_medminfilt(self):
        # make fake data
        data = np.zeros((self.size, self.size))
        for i in range(data.shape[1]):
            data[:, i] = i * np.ones_like(data[:, i])
        # run detrend_medminfilt
        Kt = 8
        Kf = 8
        dm = xrfi.detrend_medminfilt(data, Kt=Kt, Kf=Kf)

        # read in "answer" array
        # this is output that corresponds to self.size==100, Kt==8, Kf==8
        ans_fn = os.path.join(DATA_PATH, 'test_detrend_medminfilt_ans.txt')
        ans = np.loadtxt(ans_fn)
        nt.assert_true(np.allclose(ans, dm))

        # Test error when wrong dimensions are passed
        nt.assert_raises(ValueError, xrfi.detrend_medminfilt, np.ones((5, 4, 3)))

    def test_detrend_medfilt(self):
        # make fake data
        data = np.zeros((self.size, self.size))
        for i in range(data.shape[1]):
            data[:, i] = i * np.ones_like(data[:, i])
        # run detrend medfilt
        Kt = 101
        Kf = 101
        dm = uvtest.checkWarnings(xrfi.detrend_medfilt, [data, None, Kt, Kf], nwarnings=2,
                                  category=[UserWarning, UserWarning],
                                  message=['Kt value {:d} is larger than the data'.format(Kt),
                                           'Kf value {:d} is larger than the data'.format(Kf)])

        # read in "answer" array
        # this is output that corresponds to self.size==100, Kt==101, Kf==101
        ans_fn = os.path.join(DATA_PATH, 'test_detrend_medfilt_ans.txt')
        ans = np.loadtxt(ans_fn)
        nt.assert_true(np.allclose(ans, dm))

        # use complex data
        data = np.zeros((self.size, self.size), dtype=np.complex)
        for i in range(data.shape[1]):
            data[:, i] = (i * np.ones_like(data[:, i], dtype=np.float)
                          + 1j * i * np.ones_like(data[:, i], dtype=np.float))
        # run detrend_medfilt
        Kt = 58
        Kf = 58
        dm = xrfi.detrend_medfilt(data, Kt=Kt, Kf=Kf)

        # read in "answer" array
        # this is output that corresponds to self.size=100, Kt=58, Kf=58
        ans_fn = os.path.join(DATA_PATH, 'test_detrend_medfilt_complex_ans.txt')
        ans = np.genfromtxt(ans_fn, dtype=np.complex)
        nt.assert_true(np.allclose(ans, dm))

        # Test error when wrong dimensions are passed
        nt.assert_raises(ValueError, xrfi.detrend_medfilt, np.ones((5, 4, 3)))


class TestFlaggingFunctions():

    def test_watershed_flag(self):
        # generate a metrics and flag UVFlag object
        uv = UVData()
        uv.read_miriad(test_d_file)
        uvm = UVFlag(uv, history='I made this')
        uvf = UVFlag(uv, mode='flag')

        # set metric and flag arrays to specific values
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[0, 0, 1, 0] = 7.
        uvf.flag_array[0, 0, 0, 0] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[0, 0, :2, 0] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test flagging channels adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[:, :, 1, :] = 1.
        uvf.flag_array[:, :, 0, :] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test flagging times adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        times = np.unique(uv.time_array)
        inds1 = np.where(uv.time_array == times[0])[0]
        inds2 = np.where(uv.time_array == times[1])[0]
        uvm.metric_array[inds2, 0, :, 0] = 1.
        uvf.flag_array[inds1, 0, :, 0] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[inds1, 0, :, 0] = True
        flag_array[inds2, 0, :, 0] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test antenna type objects
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        uvm = UVFlag(uvc, history='I made this')
        uvf = UVFlag(uvc, mode='flag')

        # set metric and flag arrays to specific values
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[0, 0, 0, 1, 0] = 7.
        uvf.flag_array[0, 0, 0, 0, 0] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[0, 0, 0, :2, 0] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test flagging channels adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[:, :, 1, :, :] = 1.
        uvf.flag_array[:, :, 0, :, :] = True

        # run watershed flag
        uvf2 = xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=False)

        # check answer
        flag_array = np.zeros_like(uvf2.flag_array, dtype=np.bool)
        flag_array[:, :, :2, :, :] = True
        nt.assert_true(np.allclose(uvf2.flag_array, flag_array))
        del(uvf2)

        # test flagging times adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[:, :, :, 1, :] = 1.
        uvf.flag_array[:, :, :, 0, :] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :, :, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test waterfall types
        uv = UVData()
        uv.read_miriad(test_d_file)
        uvm = UVFlag(uv, history='I made this', waterfall=True)
        uvf = UVFlag(uv, mode='flag', waterfall=True)

        # set metric and flag arrays to specific values
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[0, 1, 0] = 7.
        uvf.flag_array[0, 0, 0] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[0, :2, 0] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test flagging channels adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[:, 1, :] = 1.
        uvf.flag_array[:, 0, :] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test flagging times adjacent to fully flagged ones
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvm.metric_array[1, :, :] = 1.
        uvf.flag_array[0, :, :] = True

        # run watershed flag
        xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

        # check answer
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:2, :, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

    def test_watershed_flag_errors(self):
        # setup
        uv = UVData()
        uv.read_miriad(test_d_file)
        uvm = UVFlag(uv, history='I made this')
        uvf = UVFlag(uv, mode='flag')
        uvf2 = UVFlag(uv, mode='flag', waterfall=True)

        # pass in objects besides UVFlag
        nt.assert_raises(ValueError, xrfi.watershed_flag, 1, 2)
        nt.assert_raises(ValueError, xrfi.watershed_flag, uvm, 2)
        nt.assert_raises(ValueError, xrfi.watershed_flag, uvm, uvf2)

        # use a bogus average_method
        nt.assert_raises(KeyError, xrfi.watershed_flag, uvm, uvf, avg_method='blah')

        # set the UVFlag object to have a bogus type
        uvm.type = 'blah'
        nt.assert_raises(ValueError, xrfi.watershed_flag, uvm, uvf)

    def test_ws_flag_waterfall(self):
        # test 1d
        d = np.zeros((10,))
        f = np.zeros((10,), dtype=np.bool)
        d[1] = 3.
        f[0] = True
        f_out = xrfi._ws_flag_waterfall(d, f, nsig=2.)
        ans = np.zeros_like(f_out, dtype=np.bool)
        ans[:2] = True
        nt.assert_true(np.allclose(f_out, ans))

        # test 2d
        d = np.zeros((10, 10))
        f = np.zeros((10, 10), dtype=np.bool)
        d[0, 1] = 3.
        d[1, 0] = 3.
        f[0, 0] = True
        f_out = xrfi._ws_flag_waterfall(d, f, nsig=2.)
        ans = np.zeros_like(f_out, dtype=np.bool)
        ans[:2, 0] = True
        ans[0, :2] = True
        nt.assert_true(np.allclose(f_out, ans))

        # catch errors
        d1 = np.zeros((10,))
        f2 = np.zeros((10, 10), dtype=np.bool)
        nt.assert_raises(ValueError, xrfi._ws_flag_waterfall, d1, f2)
        d3 = np.zeros((5, 4, 3))
        f3 = np.zeros((5, 4, 3), dtype=np.bool)
        nt.assert_raises(ValueError, xrfi._ws_flag_waterfall, d3, f3)

    def test_flag(self):
        # setup
        uv = UVData()
        uv.read_miriad(test_d_file)
        uvm = UVFlag(uv, history='I made this')

        # initialize array with specific values
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[0, 0, 0, 0] = 7.
        uvf = xrfi.flag(uvm, nsig_p=6.)
        nt.assert_true(uvf.mode == 'flag')
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[0, 0, 0, 0] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test channel flagging in baseline type
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[:, :, 0, :] = 7.
        uvm.metric_array[:, :, 1, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=6., nsig_f=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test time flagging in baseline type
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        times = np.unique(uvm.time_array)
        inds1 = np.where(uvm.time_array == times[0])[0]
        inds2 = np.where(uvm.time_array == times[1])[0]
        uvm.metric_array[inds1, :, :, :] = 7.
        uvm.metric_array[inds2, :, :, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[inds1, :, :, :] = True
        flag_array[inds2, :, :, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test channel flagging in antenna type
        uv = UVCal()
        uv.read_calfits(test_c_file)
        uvm = UVFlag(uv, history='I made this')
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[:, :, 0, :, :] = 7.
        uvm.metric_array[:, :, 1, :, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=7., nsig_f=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :, :2, :, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test time flagging in antenna type
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[:, :, :, 0, :] = 7.
        uvm.metric_array[:, :, :, 1, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :, :, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test channel flagging in waterfall type
        uv = UVData()
        uv.read_miriad(test_d_file)
        uvm = UVFlag(uv, history='I made this', waterfall=True)
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[:, 0, :] = 7.
        uvm.metric_array[:, 1, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=6., nsig_f=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:, :2, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # test time flagging in waterfall type
        uvm.metric_array = np.zeros_like(uvm.metric_array)
        uvm.metric_array[0, :, :] = 7.
        uvm.metric_array[1, :, :] = 3.
        uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
        flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        flag_array[:2, :, :] = True
        nt.assert_true(np.allclose(uvf.flag_array, flag_array))

        # catch errors
        nt.assert_raises(ValueError, xrfi.flag, 2)
        nt.assert_raises(KeyError, xrfi.flag, uvm, avg_method='blah')
        uvm.type = 'blah'
        nt.assert_raises(ValueError, xrfi.flag, uvm)

    def test_unflag(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_flag_apply(self):
        # test applying to UVData
        uv = UVData()
        uv.read_miriad(test_d_file)
        uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool)
        uvf = UVFlag(uv, mode='flag')
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvf.flag_array[:, :, 0, :] = True
        uvf2 = xrfi.flag_apply(uvf, uv, return_net_flags=True)
        nt.assert_true(np.allclose(uv.flag_array, uvf2.flag_array))

        # test applying to UVCal
        uv = UVCal()
        uv.read_calfits(test_c_file)
        uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool)
        uvf = UVFlag(uv, mode='flag')
        uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool)
        uvf.flag_array[:, :, 0, :, :] = True
        uvf2 = xrfi.flag_apply(uvf, uv, return_net_flags=True)
        nt.assert_true(np.allclose(uv.flag_array, uvf2.flag_array))

        # test applying to waterfalls
        uv = UVData()
        uv.read_miriad(test_d_file)
        uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool)
        uvf = UVFlag(uv, mode='flag', waterfall=True)
        uvf.flag_array[:, 0, :] = True
        xrfi.flag_apply(uvf, uv)
        nt.assert_true(np.allclose(uv.flag_array[:, :, 0, :], True))
        nt.assert_true(np.allclose(uv.flag_array[:, :, 1:, :], False))

        uv = UVCal()
        uv.read_calfits(test_c_file)
        uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool)
        uvf = UVFlag(uv, mode='flag', waterfall=True)
        uvf.flag_array[:, 0, :] = True
        xrfi.flag_apply(uvf, uv)
        nt.assert_true(np.allclose(uv.flag_array[:, :, 0, :, :], True))
        nt.assert_true(np.allclose(uv.flag_array[:, :, 1:, :, :], False))

        # catch errors
        nt.assert_raises(ValueError, xrfi.flag_apply, uvf, 2)
        nt.assert_raises(ValueError, xrfi.flag_apply, 2, uv)
        uvf.mode = 'metric'
        nt.assert_raises(ValueError, xrfi.flag_apply, uvf, uv)


class TestHighLevelFunctions():

    def test_calculate_metric_vis(self):
        # setup
        uv = UVData()
        uv.read_miriad(test_d_file)
        # Use Kt=3 because test file only has three times
        uvf = xrfi.calculate_metric(uv, 'detrend_medfilt', Kt=3)
        nt.assert_equal(uvf.mode, 'metric')
        nt.assert_equal(uvf.type, 'baseline')
        inds = uv.antpair2ind(uv.ant_1_array[0], uv.ant_2_array[0])
        wf = uv.get_data(uv.ant_1_array[0], uv.ant_2_array[0])
        filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3)
        nt.assert_true(np.allclose(filtered, uvf.metric_array[inds, 0, :, 0]))

    def test_calculate_metric_gains(self):
        # Cal gains version
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', Kt=3, Kf=3)
        nt.assert_equal(uvf.mode, 'metric')
        nt.assert_equal(uvf.type, 'antenna')
        wf = uvc.gain_array[0, 0, :, :, 0]
        filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3, Kf=3)
        nt.assert_true(np.allclose(filtered, uvf.metric_array[0, 0, :, :, 0]))

    def test_calculate_metric_chisq(self):
        # Cal chisq version
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', cal_mode='chisq',
                                    Kt=3, Kf=3)
        nt.assert_equal(uvf.mode, 'metric')
        nt.assert_equal(uvf.type, 'antenna')
        wf = uvc.quality_array[0, 0, :, :, 0]
        filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3, Kf=3)
        nt.assert_true(np.allclose(filtered, uvf.metric_array[0, 0, :, :, 0]))

    def test_calculate_metric_tot_chisq(self):
        # Cal total chisq version
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', cal_mode='tot_chisq',
                                    Kt=3, Kf=3)
        nt.assert_equal(uvf.mode, 'metric')
        nt.assert_equal(uvf.type, 'waterfall')
        filtered = xrfi.detrend_medfilt(np.abs(uvc.total_quality_array[0, :, :, 0]).T,
                                        Kt=3, Kf=3)
        nt.assert_true(np.allclose(filtered, uvf.metric_array[:, :, 0]))

    def test_calculate_metric_errors(self):
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        nt.assert_raises(ValueError, xrfi.calculate_metric, 5, 'detrend_medfilt')
        nt.assert_raises(KeyError, xrfi.calculate_metric, uvc, 'my_awesome_algorithm')
        nt.assert_raises(ValueError, xrfi.calculate_metric, uvc, 'detrend_medfilt',
                         cal_mode='foo')


class TestPipelines():

    def test_xrfi_h1c_pipe(self):
        # Do a test, add more tests as needed
        nt.assert_true(True)

    def test_xrfi_h1c_idr2_2_pipe(self):
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        uvf = xrfi.xrfi_pipe(uvc, Kt=3)
        nt.assert_equal(uvf.mode, 'metric')
        nt.assert_equal(uvf.type, 'waterfall')
        nt.assert_equal(len(uvf.polarization_array), 1)
        nt.assert_equal(uvf.weights_array.max(), 1.)


class TestWrappers():

    def test_xrfi_cal_h1c_idr2_2_run(self):
        # Run in nicest way possible
        uvtest.checkWarnings(xrfi.cal_xrfi_run, [test_c_file, test_c_file,
                                                 test_uvh5_file, 'Just a test'],
                             {'xrfi_path': xrfi_path, 'kt_size': 3},
                             nwarnings=2, message='This object is already a waterfall')

        basename = utils.strip_extension(os.path.basename(test_c_file))
        basename = utils.strip_extension(os.path.basename(basename))
        out1 = '.'.join([basename, 'xrfi_cal_metrics.h5'])
        out1 = os.path.join(xrfi_path, out1)
        print(out1)
        nt.assert_true(os.path.exists(out1))
        out2 = '.'.join([basename, 'cal_flags.h5'])
        out2 = os.path.join(xrfi_path, out2)
        print(out2)
        nt.assert_true(os.path.exists(out2))

    def test_xrfi_h1c_run(self):
        # run with bad antennas specified
        xrfi.xrfi_h1c_run(test_d_file, filename=test_d_file,
                          history='Just a test.', ex_ants='1,2', xrfi_path=xrfi_path,
                          kt_size=3)

        # catch no provided data file for flagging
        uvtest.checkWarnings(xrfi.xrfi_h1c_run, [None],
                             {'filename': test_d_file, 'history': 'Just a test.',
                              'model_file': test_d_file, 'model_file_format': 'miriad',
                              'xrfi_path': xrfi_path}, nwarnings=191,
                             message=['indata is None'] + 190 * ['Kt value 8'])

    def test_xrfi_h1c_run_no_indata(self):
        # test no indata provided
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_run, None,
                         'Just as test.', filename=test_d_file + '.h1c_run')

    def test_xrfi_h1c_run_no_filename(self):
        # test no filename provided
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_run, uvd,
                         'Just as test.', filename=None)

    def test_xrfi_h1c_run_filename_not_string(self):
        # filename is not a string
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_run, uvd,
                         'Just a test.', filename=5)

    def test_xrfi_h1c_run_uvfits_no_xrfi_path(self):
        # test uvfits file and no xrfi path
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        basename = utils.strip_extension(test_uvfits_file)
        outtest = basename + '.flags.h5'
        if os.path.exists(outtest):
            os.remove(outtest)
        if os.path.exists(basename + '.waterfall.flags.h5'):
            os.remove(basename + '.waterfall.flags.h5')
        cbasename = utils.strip_extension(test_c_file)
        g_temp = os.path.join(cbasename + '.g.flags.h5')
        x_temp = os.path.join(cbasename + '.x.flags.h5')
        xrfi.xrfi_h1c_run(test_uvfits_file, infile_format='uvfits',
                          history='Just a test.', kt_size=3, model_file=test_d_file,
                          model_file_format='miriad', calfits_file=test_c_file)
        nt.assert_true(os.path.exists(outtest))
        os.remove(outtest)
        if os.path.exists(basename + '.waterfall.flags.h5'):
            os.remove(basename + '.waterfall.flags.h5')
        if os.path.exists(utils.strip_extension(test_d_file) + '.flags.h5'):
            os.remove(utils.strip_extension(test_d_file) + '.flags.h5')
        os.remove(g_temp)
        os.remove(x_temp)

    def test_xrfi_h1c_run_uvfits_xrfi_path(self):
        # test uvfits file with xrfi path
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        basename = utils.strip_extension(os.path.basename(test_uvfits_file))
        outtest = os.path.join(xrfi_path, basename) + '.flags.h5'
        if os.path.exists(outtest):
            os.remove(outtest)
        xrfi.xrfi_h1c_run(test_uvfits_file, infile_format='uvfits',
                          history='Just a test.', kt_size=3, xrfi_path=xrfi_path)
        nt.assert_true(os.path.exists(outtest))
        os.remove(outtest)

    def test_xrfi_h1c_run_miriad_model(self):
        # miriad model file test
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        ext = 'flag'
        uvd.read_miriad(test_d_file)
        basename = os.path.basename(utils.strip_extension(test_d_file))
        outtest = '.'.join([os.path.join(xrfi_path, basename), ext])
        if os.path.exists(outtest):
            os.remove(outtest)
        xrfi.xrfi_h1c_run(uvd, history='Just a test.', filename=test_d_file,
                          extension=ext, summary=True, model_file=test_d_file,
                          model_file_format='miriad', xrfi_path=xrfi_path, kt_size=3)
        nt.assert_true(os.path.exists(outtest))

    def test_xrfi_h1c_run_uvfits_model(self):
        # uvfits model file test
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        ext = 'flag'
        basename = os.path.basename(utils.strip_extension(test_uvfits_file))
        outtest = '.'.join([os.path.join(xrfi_path, basename), ext])
        if os.path.exists(outtest):
            os.remove(outtest)
        xrfi.xrfi_h1c_run(uvd, history='Just a test.', filename=test_d_file,
                          extension=ext, summary=True, model_file=test_uvfits_file,
                          model_file_format='uvfits', xrfi_path=xrfi_path, kt_size=3)
        nt.assert_true(os.path.exists(outtest))

    def test_xrfi_h1c_run_incorrect_model(self):
        # incorrect model
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        bad_uvfits_test = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA.uvfits')
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_run, uvd, 'Just a test.',
                         filename=test_d_file, model_file=bad_uvfits_test,
                         model_file_format='uvfits', xrfi_path=xrfi_path, kt_size=3)

    def test_xrfi_h1c_run_input_calfits(self):
        # input calfits
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        ext = 'flag'
        cbasename = os.path.basename(utils.strip_extension(test_c_file))
        outtest1 = '.'.join([os.path.join(xrfi_path, cbasename), 'x', ext])
        outtest2 = '.'.join([os.path.join(xrfi_path, cbasename), 'g', ext])
        if os.path.exists(outtest1):
            os.remove(outtest1)
        if os.path.exists(outtest2):
            os.remove(outtest2)
        xrfi.xrfi_h1c_run(uvd, history='Just a test.', filename=test_d_file,
                          extension=ext, summary=True, model_file=test_d_file,
                          model_file_format='miriad', calfits_file=test_c_file,
                          xrfi_path=xrfi_path, kt_size=3)
        nt.assert_true(os.path.exists(outtest1))
        nt.assert_true(os.path.exists(outtest2))

    def test_xrfi_h1c_run_incorrect_calfits(self):
        # check for calfits with incorrect time/freq axes
        uvd = UVData()
        uvd.read_miriad(test_d_file)
        bad_calfits = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_run, uvd, 'Just a test.',
                         filename=test_d_file, calfits_file=bad_calfits,
                         xrfi_path=xrfi_path, kt_size=3)

    def test_xrfi_h1c_run_indata_string_filename_not_string(self):
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_run, 'foo', 'Just a test.',
                         filename=3)

    def test_xrfi_h1c_apply(self):
        xrfi_path = os.path.join(DATA_PATH, 'test_output')
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.h5')
        wf_file2 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.h5')
        waterfalls = wf_file1 + ',' + wf_file2
        history = 'history stuff'

        # test running on our test data
        basename, ext = utils.strip_extension(os.path.basename(test_d_file), return_ext=True)
        dest_file = os.path.join(xrfi_path, basename + '.R.uv')
        dest_flag = os.path.join(xrfi_path, basename + '.R.flags.h5')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        if os.path.exists(dest_flag):
            os.remove(dest_flag)
        xrfi.xrfi_h1c_apply([test_d_file], history, xrfi_path=xrfi_path,
                            flag_file=test_f_file_flags, waterfalls=waterfalls)
        nt.assert_true(os.path.exists(dest_file))
        nt.assert_true(os.path.exists(dest_flag))
        shutil.rmtree(dest_file)  # clean up

        # uvfits output
        basename = os.path.basename(utils.strip_extension(test_d_file))
        dest_file = os.path.join(xrfi_path, basename + '.R.uvfits')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path, flag_file=test_f_file_flags,
                            outfile_format='uvfits', extension='R', output_uvflag=False)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

        # uvh5 output
        basename = os.path.basename(utils.strip_extension(test_d_file))
        dest_file = os.path.join(xrfi_path, basename + '.R.uvh5')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path, flag_file=test_f_file_flags,
                            outfile_format='uvh5', extension='R', output_uvflag=False)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

    def test_xrfi_h1c_apply_errors(self):
        xrfi_path = os.path.join(DATA_PATH, 'test_output')
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.h5')
        wf_file2 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.h5')
        waterfalls = wf_file1 + ',' + wf_file2
        history = 'history stuff'
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_apply, [], history)

        # test running with two files
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_apply, ['file1', 'file2'], history)

        # Outfile error
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, outfile_format='bla')

        basename = utils.strip_extension(os.path.basename(test_d_file))
        dest_file = os.path.join(xrfi_path, basename + '.R.uvfits')
        if not os.path.exists(dest_file):
            open(dest_file, 'a').close()
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, xrfi_path=xrfi_path,
                         flag_file=test_f_file_flags, outfile_format='uvfits', extension='R')
