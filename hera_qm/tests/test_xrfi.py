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
import pylab as plt
import hera_qm.tests as qmtest
from inspect import getargspec
import pyuvdata.tests as uvtest
from pyuvdata import UVData
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH


class TestComplex(object):

    def test_detrend_medfilt(self):
        RFI = 50
        snr = 10
        nsig = 4
        np.random.seed(0)
        data = np.array(qmtest.noise((SIZE, SIZE)))
        rfi = (np.random.randint(SIZE, size=RFI), np.random.randint(SIZE, size=RFI))
        data[rfi] = snr
        f = xrfi.detrend_medfilt(data)
        f = np.where(f > nsig, 1, 0)
        cf, fp = get_accuracy(f, rfi)
        nt.assert_greater_equal(cf, 0.9)
        nt.assert_less_equal(fp, 0.01)


class TestXrfiRun(object):
    def test_xrfi_run_xrfi(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--infile_format=miriad"
        arg1 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg2 = "--algorithm=xrfi"
        arg3 = "--kt_size=2"
        arg4 = "--kf_size=2"
        arg5 = "--sig_init=6"
        arg6 = "--sig_adj=2"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args.filename, args, history)

        # test running with filename as None
        nt.assert_raises(AssertionError, xrfi.xrfi_run, None, args, history)

        # test running with too many files
        cmd = ' '.join([arguments, 'file1', 'file2'])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args.filename, args, history)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.flags.npz')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        xrfi.xrfi_run(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        # load data and test all arrays exist in file
        d = np.load(dest_file)
        nt.assert_true(np.array(map(lambda n: n in d, ['baseline_array', 'waterfall', 'flag_array', 'history'])).all())
        # test baseline array has same shape as flag_array axis 0
        nt.assert_equal(len(d['baseline_array']), d['flag_array'].shape[0])
        os.remove(dest_file)

        # test running with UVData object
        uv = UVData()
        uv.read_miriad(xx_file)
        if os.path.exists(dest_file):
            os.remove(dest_file)
        xrfi.xrfi_run(uv, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

        # Test missing filename
        cmd = ' '.join([arguments])
        args = a.parse_args(cmd.split())
        nt.assert_raises(AssertionError, xrfi.xrfi_run, uv, args, cmd)

    def test_xrfi_run_xrfi_simple(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--infile_format=miriad"
        arg1 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg2 = "--algorithm=xrfi_simple"
        arg3 = "--nsig_dt=6"
        arg4 = "--nsig_df=6"
        arg5 = "--nsig_all=0"
        arg6 = "--summary"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args.filename, args, cmd)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.flags.npz')
        sum_file = os.path.join(DATA_PATH, 'test_output',
                                'zen.2457698.40355.xx.HH.uvcAA.flag_summary.npz')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        if os.path.exists(sum_file):
            os.remove(sum_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        nt.assert_true(os.path.exists(sum_file))
        os.remove(dest_file)
        os.remove(sum_file)

    def test_xrfi_run_model_and_cal(self):

        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg1 = "--algorithm=xrfi_simple"
        model_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits')
        arg2 = "--model_file=" + model_file
        calfits_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
        arg3 = "--calfits_file=" + calfits_file
        arguments = ' '.join([arg0, arg1, arg2, arg3])

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_files = []
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvcAA.flags.npz'))
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvc.vis.uvfits.flags.npz'))
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.npz'))
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.npz'))
        for f in dest_files:
            if os.path.exists(f):
                os.remove(f)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.filename, args, cmd)
        for f in dest_files:
            nt.assert_true(os.path.exists(f))
            os.remove(f)

        # test running without a filename
        dest_files = []
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvc.vis.uvfits.flags.npz'))
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.npz'))
        dest_files.append(os.path.join(DATA_PATH, 'test_output',
                                       'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.npz'))
        for f in dest_files:
            if os.path.exists(f):
                os.remove(f)
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        nt.assert_true
        uvtest.checkWarnings(xrfi.xrfi_run, [None, args, cmd], nwarnings=1, message='indata is none,')
        for f in dest_files:
            nt.assert_true(os.path.exists(f))
            os.remove(f)

        # Test model_file_format
        uv = UVData()
        uv.read_uvfits(model_file)
        model_file = os.path.join(DATA_PATH, 'test_output',
                                  'zen.2457698.40355.xx.HH.uvc.vis')
        uv.write_miriad(model_file, clobber=True)
        dest_file = model_file + '.flags.npz'
        if os.path.exists(dest_file):
            os.remove(dest_file)
        arg2 = "--model_file=" + model_file
        arg4 = "--model_file_format=miriad"
        cmd = ' '.join([arg0, arg1, arg2, arg4, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)
        shutil.rmtree(model_file)

    def test_xrfi_run_model_and_cal_errors(self):

        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg1 = "--algorithm=xrfi_simple"
        model_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits')
        arg2 = "--model_file=" + model_file
        calfits_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
        arg3 = "--calfits_file=" + calfits_file
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        arguments = ' '.join([arg0, arg1, arg2, arg3])

        arg4 = "--model_file_format=fhd"
        cmd = ' '.join([arguments, arg4, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(Exception, xrfi.xrfi_run, args.filename, args, cmd)

        arg4 = "--model_file_format=blah"
        cmd = ' '.join([arguments, arg4, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.filename, args, cmd)

        # Model file with wrong freq/time axes
        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))
        uv.select(freq_chans=[0, 1])
        bad_model_file = os.path.join(DATA_PATH, 'test_output', 'bad_model.uvfits')
        uv.write_uvfits(bad_model_file)
        arg2 = "--model_file=" + bad_model_file
        arguments = ' '.join([arg0, arg1, arg2, arg3])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.filename, args, cmd)
        # Again for times
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))
        uv.select(times=uv.time_array[0])
        bad_model_file = os.path.join(DATA_PATH, 'test_output', 'bad_model.uvfits')
        uv.write_uvfits(bad_model_file)
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.filename, args, cmd)
        os.remove(bad_model_file)

    def test_xrfi_run_errors(self):
        # test code to read different file formats
        # these will raise errors
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--infile_format=uvfits"
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(IOError, xrfi.xrfi_run, args.filename, args, cmd)

        arg0 = "--infile_format=fhd"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(Exception, xrfi.xrfi_run, args.filename, args, cmd)

        arg0 = "--infile_format=blah"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.filename, args, cmd)

        # choose an invalid alrogithm
        arg0 = "--infile_format=miriad"
        arg1 = "--algorithm=foo"
        arguments = ' '.join([arg0, arg1])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.filename, args, cmd)

    def test_xrfi_run_output_args(self):
        # test different output arguments
        a = utils.get_metrics_ArgumentParser('xrfi_run')

        # test writing to same directory
        arg0 = "--infile_format=miriad"
        arg1 = "--algorithm=xrfi_simple"
        arg2 = "--extension=.testflags.npz"
        arguments = ' '.join([arg0, arg1, arg2])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testflags.npz')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        # clean up
        os.remove(dest_file)

    def test_xrfi_run_exants(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        arg0 = "--infile_format=miriad"
        arg1 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg2 = "--algorithm=xrfi_simple"
        arg3 = "--nsig_dt=6"
        arg4 = "--nsig_df=6"
        arg5 = "--nsig_all=0"
        arg6 = "--ex_ants=72"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6])

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.flags.npz')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        # clean up
        os.remove(dest_file)


class TestXrfiApply(object):
    def test_xrfi_apply(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi_apply')
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=miriad"
        arg2 = "--extension=R"
        arg3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        flag_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.flags.npz')
        arg4 = "--flag_file=" + flag_file
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.npz')
        wf_file2 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.x.flags.npz')
        arg5 = "--waterfalls=" + wf_file1 + "," + wf_file2
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_apply, args.filename, args, history)

        # test running with two files
        cmd = ' '.join([arguments, 'file1', 'file2'])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_apply, args.filename, args, history)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        xrfi.xrfi_apply(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        shutil.rmtree(dest_file)  # clean up

        # uvfits output
        arg1 = "--outfile_format=uvfits"
        arg2 = "--extension=R.uvfits"
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR.uvfits')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        xrfi.xrfi_apply(args.filename, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

    def test_xrfi_apply_errors(self):
        # test code to read different file formats
        # these will raise errors
        a = utils.get_metrics_ArgumentParser('xrfi_apply')
        arg0 = "--infile_format=uvfits"
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(IOError, xrfi.xrfi_apply, args.filename, args, cmd)

        arg0 = "--infile_format=fhd"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(Exception, xrfi.xrfi_apply, args.filename, args, cmd)

        arg0 = "--infile_format=blah"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_apply, args.filename, args, cmd)

        # Outfile error
        arg0 = "--outfile_format=bla"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(ValueError, xrfi.xrfi_apply, args.filename, args, cmd)

        # array size checks
        arg0 = "--infile_format=miriad"
        arg1 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        flag_file = os.path.join(DATA_PATH, 'test_output', 'bad.flags.npz')
        np.savez(flag_file, flag_array=np.zeros((3, 4)))
        arg2 = "--flag_file=" + flag_file
        arguments = ' '.join([arg0, arg1, arg2])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        # Flag array wrong size
        nt.assert_raises(ValueError, xrfi.xrfi_apply, args.filename, args, cmd)
        os.remove(flag_file)

        flag_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.flags.npz')
        arg2 = "--flag_file=" + flag_file
        wf_file1 = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits.g.flags.npz')
        wf_file2 = os.path.join(DATA_PATH, 'test_output', 'bad_wf.npz')
        np.savez(wf_file2, waterfall=np.zeros((3, 4)))
        arg3 = "--waterfalls=" + wf_file1 + "," + wf_file2
        arguments = ' '.join([arg0, arg1, arg2, arg3])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        # Waterfall wrong size
        nt.assert_raises(ValueError, xrfi.xrfi_apply, args.filename, args, cmd)
        os.remove(wf_file2)


class TestSummary(unittest.TestCase):
    def test_summarize_flags(self):
        from hera_qm.version import hera_qm_version_str

        infile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        uv = UVData()
        uv.read_miriad(infile)
        outfile = os.path.join(DATA_PATH, 'test_output', 'rfi_summary.npz')
        if os.path.exists(outfile):
            os.remove(outfile)
        xrfi.summarize_flags(uv, outfile)
        self.assertTrue(os.path.exists(outfile))
        data = np.load(outfile)
        nt, nf, npol = (3, 256, 1)
        self.assertEqual(data['waterfall'].shape, (nt, nf, npol))
        self.assertEqual(data['waterfall'].min(), 0)
        self.assertEqual(data['waterfall'].max(), 0)
        self.assertEqual(data['tmax'].shape, (nf, npol))
        self.assertEqual(data['tmin'].shape, (nf, npol))
        self.assertEqual(data['tmean'].shape, (nf, npol))
        self.assertEqual(data['tstd'].shape, (nf, npol))
        self.assertEqual(data['tmedian'].shape, (nf, npol))
        self.assertEqual(data['fmax'].shape, (nt, npol))
        self.assertEqual(data['fmin'].shape, (nt, npol))
        self.assertEqual(data['fmean'].shape, (nt, npol))
        self.assertEqual(data['fstd'].shape, (nt, npol))
        self.assertEqual(data['fmedian'].shape, (nt, npol))

        self.assertEqual(data['times'].shape, (nt,))
        self.assertEqual(data['freqs'].shape, (nf,))
        self.assertEqual(data['pols'], ['XX'])
        self.assertEqual(data['version'], hera_qm_version_str)
        os.remove(outfile)  # cleanup

    def test_summarize_flags_with_prior(self):

        infile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        uv = UVData()
        uv.read_miriad(infile)
        outfile = os.path.join(DATA_PATH, 'test_output', 'rfi_summary.npz')
        if os.path.exists(outfile):
            os.remove(outfile)
        prior_flags = np.zeros_like(uv.flag_array)
        prior_flags[0, 0, 100, 0] = True
        flags = np.zeros_like(uv.flag_array)
        flags[0, 0, 100, 0] = True
        flags[:, 0, 101, 0] = True
        xrfi.summarize_flags(uv, outfile, flag_array=flags, prior_flags=prior_flags)
        self.assertTrue(os.path.exists(outfile))
        data = np.load(outfile)
        self.assertTrue((data['waterfall'][:, 100, 0] == 0).all())
        self.assertTrue((data['waterfall'][:, 101, 0] == 1).all())
        os.remove(outfile)  # cleanup


class TestVisFlag(object):
    def test_vis_flag(self):

        a = utils.get_metrics_ArgumentParser('xrfi_run')
        args = a.parse_args([''])

        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))
        flag_array = xrfi.vis_flag(uv, args)
        nt.assert_equal(uv.flag_array.shape, flag_array.shape)
        nt.assert_equal(flag_array.dtype, bool)

        # run xrfi (not simple)
        arg0 = "--algorithm=xrfi"
        arg1 = "--kt_size=2"
        cmd = ' '.join([arg0, arg1, ''])
        args = a.parse_args(cmd.split())
        flag_array = xrfi.vis_flag(uv, args)
        nt.assert_equal(uv.flag_array.shape, flag_array.shape)
        nt.assert_equal(flag_array.dtype, bool)

    def test_vis_flag_errors(self):
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        args = a.parse_args([''])
        # First argument must be UVData object.
        nt.assert_raises(ValueError, xrfi.vis_flag, 4, args)


class TestCalFlag(object):
    def test_cal_flag(self):
        from pyuvdata import UVCal

        a = utils.get_metrics_ArgumentParser('xrfi_run')
        args = a.parse_args([''])

        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits'))
        gf, xf = xrfi.cal_flag(uvc, args)
        nt.assert_equal(uvc.flag_array.shape, gf.shape)
        nt.assert_equal(uvc.flag_array.shape, xf.shape)
        nt.assert_equal(gf.dtype, bool)
        nt.assert_equal(xf.dtype, bool)

        # run xrfi (not simple)
        arg0 = "--algorithm=xrfi"
        arg1 = "--kt_size=2"
        cmd = ' '.join([arg0, arg1, ''])
        args = a.parse_args(cmd.split())
        gf, xf = xrfi.cal_flag(uvc, args)
        nt.assert_equal(uvc.flag_array.shape, gf.shape)
        nt.assert_equal(uvc.flag_array.shape, xf.shape)
        nt.assert_equal(gf.dtype, bool)
        nt.assert_equal(xf.dtype, bool)

    def test_cal_flag_errors(self):
        from pyuvdata import UVCal
        a = utils.get_metrics_ArgumentParser('xrfi_run')
        args = a.parse_args([''])
        # First argument must be UVData object.
        nt.assert_raises(ValueError, xrfi.cal_flag, 4, args)

        # Must be type 'gain'
        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457678.16694.yy.HH.uvc.good.first.calfits'))
        nt.assert_raises(ValueError, xrfi.cal_flag, uvc, args)

        # unknown algorithm
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits'))
        arg0 = "--algorithm=bla"
        cmd = ' '.join([arg0, ''])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.cal_flag, uvc, args)


class TestxrfiErrorHandling(object):
    def test_kernel_size_exceeds_data(self):
        d = np.random.normal((7, 9))
        f = uvtest.checkWarnings(xrfi.xrfi, [d], {'Kt': 8, 'Kf': 8}, nwarnings=1,
                                 message='Kernel size exceeds data.')
        nt.assert_true(np.all(f))


class TestFlags2Waterfall(object):
    def test_flags2waterfall(self):
        from pyuvdata import UVCal

        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))

        np.random.seed(0)
        uv.flag_array = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
        wf = xrfi.flags2waterfall(uv)
        nt.assert_almost_equal(np.mean(wf), np.mean(uv.flag_array))
        nt.assert_equal(wf.shape, (uv.Ntimes, uv.Nfreqs))

        # Test external flag_array
        uv.flag_array = np.zeros_like(uv.flag_array)
        f = np.random.randint(0, 2, size=uv.flag_array.shape, dtype=bool)
        wf = xrfi.flags2waterfall(uv, flag_array=f)
        nt.assert_almost_equal(np.mean(wf), np.mean(f))
        nt.assert_equal(wf.shape, (uv.Ntimes, uv.Nfreqs))

        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits'))

        uvc.flag_array = np.random.randint(0, 2, size=uvc.flag_array.shape, dtype=bool)
        wf = xrfi.flags2waterfall(uvc)
        nt.assert_almost_equal(np.mean(wf), np.mean(uvc.flag_array))
        nt.assert_equal(wf.shape, (uvc.Ntimes, uvc.Nfreqs))

    def test_flags2waterfall_errors(self):

        # First argument must be UVData or UVCal object
        nt.assert_raises(ValueError, xrfi.flags2waterfall, 5)

        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))
        # Flag array must have same shape as uv.flag_array
        nt.assert_raises(ValueError, xrfi.flags2waterfall, uv, np.array([4, 5]))


class TestWaterfall2Flags(object):
    def test_waterfall2flags(self):
        from pyuvdata import UVCal

        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))

        np.random.seed(0)
        wf = np.random.randint(0, 2, size=(uv.Ntimes, uv.Nfreqs), dtype=bool)
        flags = xrfi.waterfall2flags(wf, uv)
        nt.assert_equal(flags.shape, uv.flag_array.shape)
        wf_spectrum = np.mean(wf, axis=0)
        f_spectrum = np.mean(flags, axis=(0, 1, 3))
        nt.assert_true(np.allclose(wf_spectrum, f_spectrum))

        # UVCal version
        uvc = UVCal()
        uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits'))
        wf = np.random.randint(0, 2, size=(uvc.Ntimes, uvc.Nfreqs), dtype=bool)
        flags = xrfi.waterfall2flags(wf, uvc)
        nt.assert_equal(flags.shape, uvc.flag_array.shape)
        for ai in range(uvc.Nants_data):
            nt.assert_true(np.all(wf == flags[ai, 0, :, :, 0].T))

    def test_waterfall2flags_errors(self):

        uv = UVData()
        uv.read_uvfits(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvc.vis.uvfits'))

        # Waterfall must have dims (uv.Ntimes, uv.Nfreqs)
        wf = np.zeros((2, 2))
        nt.assert_raises(ValueError, xrfi.waterfall2flags, wf, uv)

        # Second argument must be UVData or UVCal object
        nt.assert_raises(ValueError, xrfi.waterfall2flags, np.array([4, 5]), 5)


class TestThresholdFlags(object):
    def test_threshold_flags(self):
        Nt = 20
        Nf = 15
        wf = np.zeros((Nt, Nf))

        wf_t = xrfi.threshold_flags(wf)
        nt.assert_true(wf_t.sum() == 0)
        wf[0, 0] = 0.5
        wf_t = xrfi.threshold_flags(wf)
        nt.assert_true(wf_t.sum() == 1)
        wf_t = xrfi.threshold_flags(wf, time_threshold=0.4 / Nt)
        nt.assert_true(wf_t.sum() == Nt)
        wf_t = xrfi.threshold_flags(wf, freq_threshold=.4 / Nf)
        nt.assert_true(wf_t.sum() == Nf)

        # Test NaN mapping
        wf[0, 0] = np.nan
        wf_t = xrfi.threshold_flags(wf)
        nt.assert_true(wf_t[0, 0])


class TestNormalizeWf(object):
    def test_normalize_wf(self):
        Nt = 20
        Nf = 15
        wf = 0.5 * np.ones((Nt, Nf))
        wfp = 0.25 * np.ones((Nt, Nf))

        wfn = xrfi.normalize_wf(wf, wfp)
        nt.assert_true(np.all(wfn == 1. / 3.))
        nt.assert_true(wfn.shape == wf.shape)

    def test_normalize_wf_fail(self):
        Nt = 20
        Nf = 15
        wf = 0.5 * np.ones((Nt, Nf))
        wfp = 0.25 * np.ones((Nt, Nf + 1))
        nt.assert_raises(AssertionError, xrfi.normalize_wf, wf, wfp)


class TestFlagXants(object):
    def test_flag_xants(self):
        # Raise an error by passing in something besides a UVData object
        nt.assert_raises(ValueError, xrfi.flag_xants, 7, [0])

        # Read in a data file and flag some antennas
        infile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        uv = UVData()
        uv.read_miriad(infile)

        # Make sure nothing is flagged
        uv.flag_array[:, :, :, :] = False

        # Specify list of antennas to flag
        all_ants = uv.get_ants()
        xants = all_ants[:2]
        uv = xrfi.flag_xants(uv, xants)

        # Check that the list of xants was flagged for all visibilities
        for xant in xants:
            for ant in all_ants:
                blts = uv.antpair2ind(ant, xant)
                flags = uv.flag_array[blts, :, :, :]
                nt.assert_true(np.allclose(flags, True))
                blts = uv.antpair2ind(xant, ant)
                flags = uv.flag_array[blts, :, :, :]
                nt.assert_true(np.allclose(flags, True))


class TestInputFlagWatershed(object):
    def test_input_flag(self):
        # create some fake data for input to watershed
        SIZE = 10
        sigmas = np.ones((SIZE, SIZE))
        sigmas[:, 3] = 7
        sigmas[1::2, 5:7] = 3
        # create some input flags
        input_flags = np.zeros((SIZE, SIZE), dtype=bool)
        input_flags[:, 4] = 1

        # flag using watershed
        w_input_flags = xrfi.watershed_flag(sigmas, f=input_flags)

        # create array to test against
        flag_check = np.zeros((SIZE, SIZE), dtype=bool)
        flag_check[:, 3:5] = 1
        flag_check[1::2, 5:7] = 1
        nt.assert_true(np.all(w_input_flags == flag_check))
