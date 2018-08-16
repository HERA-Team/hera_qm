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
test_f_file_flags = test_d_file + '.testuvflag.flags.h5'  # version in 'flag' mode
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
        d_filt = uvtest.checkWarnings(xrfi.medminfilt, [data, Kt, Kf], nwarnings=2,
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
        nt.assert_raises(ValueError, xrfi.detrend_deriv, data, False, False)

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
        dm = uvtest.checkWarnings(xrfi.detrend_medfilt, [data, Kt, Kf], nwarnings=2,
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

    def test_xrfi_h1c_apply(self):
        xrfi_path = os.path.join(DATA_PATH, 'test_output')
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.h5')
        wf_file2 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.h5')
        waterfalls = wf_file1 + ',' + wf_file2
        history = 'history stuff'

        # test running on our test data
        dest_file = os.path.join(xrfi_path, os.path.basename(test_d_file) + 'R')
        dest_flag = os.path.join(xrfi_path, os.path.basename(test_d_file) + 'R.flags.h5')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        if os.path.exists(dest_flag):
            os.remove(dest_flag)
        xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path,
                            flag_file=test_f_file_flags, waterfalls=waterfalls)
        nt.assert_true(os.path.exists(dest_file))
        nt.assert_true(os.path.exists(dest_flag))
        shutil.rmtree(dest_file)  # clean up

        # uvfits output
        dest_file = os.path.join(xrfi_path, os.path.basename(test_d_file) + 'R.uvfits')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path, flag_file=test_f_file_flags,
                            outfile_format='uvfits', extension='R.uvfits', output_uvflag=False)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

    def test_xrfi_apply_errors(self):
        xrfi_path = os.path.join(DATA_PATH, 'test_output')
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.h5')
        wf_file2 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.h5')
        waterfalls = wf_file1 + ',' + wf_file2
        history = 'history stuff'
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_apply, [], history)

        # test running with two files
        nt.assert_raises(AssertionError, xrfi.xrfi_h1c_apply, ['file1', 'file2'], history)

        # Conflicting file formats
        nt.assert_raises(IOError, xrfi.xrfi_h1c_apply, test_d_file, history, infile_format='uvfits')
        nt.assert_raises(Exception, xrfi.xrfi_h1c_apply, test_d_file, history, infile_format='fhd')
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, infile_format='bla')

        # Outfile error
        nt.assert_raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, outfile_format='bla')
