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
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH

np.random.seed(0)

SIZE = 100
VERBOSE = False
PLOT = False


def get_accuracy(f, rfi, verbose=VERBOSE):
    correctly_flagged = np.average(f[rfi])
    m = f.copy()
    m[rfi] = 0
    false_positive = float(np.sum(m)) / (m.size - len(rfi[0]))
    if verbose:
        print '\t Found RFI: %1.3f\n\t False Positive: %1.3f' % (correctly_flagged, false_positive)
    return correctly_flagged, false_positive


def fake_flags(SIZE):
    fakeflags = np.random.randint(0, 2, size=(SIZE, SIZE)).astype(bool)
    return fakeflags


def plot_waterfall(data, f, mx=10, drng=10, mode='lin'):
    if not PLOT:
        return
    plt.subplot(121)
    plt.imshow(np.abs(data), aspect='auto', cmap='jet')
    capo.plot.waterfall(data, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.subplot(122)
    capo.plot.waterfall(f, mode='lin', mx=10, drng=10)
    plt.imshow(f, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()


def plot_result(f, rfi):
    if not PLOT:
        return
    plt.plot(rfi[0], rfi[1], 'ko')
    fi = np.where(f)
    plt.plot(fi[0], fi[1], 'r.')
    plt.show()


class Template():

    def setUp(self):
        raise unittest.SkipTest  # setUp has to be overridden to actually run a test
    rfi_gen = None  # Need to override this for each TestCase, usually in setUp

    def _run_test(self, func, arg, correct_flag, false_positive, nsig=5, fmode=False):
        for data, rfi in self.rfi_gen():
            try:
                f = func(data, *arg)
            except:
                # ValueError check to make sure kernel size isn't too big
                self.assertRaises(ValueError, func, data, *arg)
                f = fake_flags(SIZE)
            if VERBOSE:
                print self.__class__, func.__name__
            f = np.where(f > nsig, 1, 0)
            cf, fp = get_accuracy(f, rfi)
            if PLOT:
                plot_waterfall(data, f)
                plot_result(f, rfi)
            if fmode:
                if VERBOSE:
                    print 'In failure mode now.'
                try:
                    self.assertLessEqual(cf, correct_flag)
                except AssertionError:
                    self.assertGreaterEqual(fp, false_positive)

            else:
                self.assertGreaterEqual(cf, correct_flag)
                self.assertLessEqual(fp, false_positive)
    ans = {
        'detrend_deriv': [(.9, .6, .6), (.1, .005, .005)],
        'detrend_medfilt': [(.9, .9, .9, 0.), (.01, .01, .01, .8)],
        'detrend_medminfilt': [(.97, .95, .95, 0.), (.5, .5, .5, .8)],
        'xrfi_simple': [(.99, .99), (.01, .01)],
        'xrfi': (.99, .01),
        'watershed': [(.9, .9), (.1, .1)],
    }

    # mode determines whether a specific test should fail, and inverts the assertions
    # for percentage of correct and false positive flags

    mode = {
        'detrend_deriv': [False, False, False],
        'detrend_medfilt': [False, False, False, True],
        'detrend_medminfilt': [False, False, False, True],
        'xrfi_simple': [False, False],
        'xrfi': False,
        'watershed': [False, True],
    }
    # detrend_medfilt,detrend_medminfilt fails in final case because kernel size > data size
    # watershed fails by default in the second case due to purposefully being supplied incorrect flags

    def test_detrend_deriv(self):
        cf, fp = self.ans['detrend_deriv']
        args = [(True, True), (True, False), (False, True)]
        mode = self.mode['detrend_deriv']
        for i in range(3):
            self._run_test(xrfi.detrend_deriv, args[i], cf[i], fp[i], nsig=4,
                           fmode=mode[i])

    def test_detrend_medfilt(self):
        cf, fp = self.ans['detrend_medfilt']
        argsList = [(8, 8), (7, 9), (9, 7), (1000, 1000)]
        for i in range(4):
            self._run_test(xrfi.detrend_medfilt, argsList[i], cf[i], fp[i], nsig=4)

    def test_detrend_medminfilt(self):
        cf, fp = self.ans['detrend_medminfilt']
        argsList = [(8, 8), (7, 9), (9, 7), (1000, 1000)]
        mode = self.mode['detrend_medminfilt']
        for i in range(4):
            self._run_test(xrfi.detrend_medminfilt, argsList[i], cf[i], fp[i], nsig=4, fmode=mode[i])

    def test_xrfi_simple(self):
        cf, fp = self.ans['xrfi_simple']
        args = getargspec(xrfi.xrfi_simple).defaults
        fflags = fake_flags(SIZE)
        argsList = [args, (fflags, 6, 6, 1)]
        fmode = [False, True]
        for i in range(2):
            self._run_test(xrfi.xrfi_simple, argsList[i], cf[i], fp[i], nsig=.5, fmode=fmode[i])

    def test_xrfi(self):
        cf, fp = self.ans['xrfi']
        args = getargspec(xrfi.xrfi).defaults
        self._run_test(xrfi.xrfi, args, cf, fp, nsig=.5)

    def test_watershed(self):
        cf, fp = self.ans['watershed']
        args = getargspec(xrfi.watershed_flag).defaults
        fflags = fake_flags(SIZE)
        argsList = [args, (fflags, 6, 2)]
        mode = self.mode['watershed']
        for i in range(2):
            self._run_test(xrfi.watershed_flag, argsList[i], cf[i], fp[i], nsig=.5, fmode=mode[i])


class TestSparseScatter(Template, unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = np.array(qmtest.real_noise((SIZE, SIZE)))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.mode['detrend_deriv'] = [False, False, False]
        self.mode['detrend_medminfilt'] = [False, False, False, True]
        self.mode['watershed'] = [False, True]


class TestDenseScatter(Template, unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        RFI = 1000
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = qmtest.real_noise((SIZE, SIZE))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = [(.33, .33, .33), (.1, .1, .1)]
        self.ans['xrfi_simple'] = [(.90, .90), (.1, .1)]
        self.mode['detrend_deriv'] = [False, False, False]
        self.mode['detrend_medminfilt'] = [False, False, False, True]
        self.mode['watershed'] = [False, True]


class TestCluster(Template, unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        RFI = 10
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = qmtest.real_noise((SIZE, SIZE))
                x, y = (np.random.randint(SIZE - 1, size=RFI),
                        np.random.randint(SIZE - 1, size=RFI))
                x = np.concatenate([x, x, x + 1, x + 1])
                y = np.concatenate([y, y + 1, y, y + 1])
                rfi = (np.array(x), np.array(y))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['xrfi_simple'] = [(.39, .39), (.1, .1)]
        self.ans['detrend_deriv'] = [(-.05, -.05, -.05), (.1, .1, .1)]
        self.mode['detrend_deriv'] = [False, False, False]
        self.mode['detrend_medminfilt'] = [False, False, False, True]
        self.mode['watershed'] = [False, True]


class TestLines(Template, unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        RFI = 3
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                data = qmtest.real_noise((SIZE, SIZE))
                x, y = (np.random.randint(SIZE, size=RFI),
                        np.random.randint(SIZE, size=RFI))
                mask = np.zeros_like(data)
                mask[x] = 1
                mask[:, y] = 1
                data += mask * NSIG
                yield data, np.where(mask)
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = [(.9, .3, .3), (0.1, 0.1, 0.1)]
        self.ans['xrfi_simple'] = [(.75, .75), (.1, .1)]
        self.ans['xrfi'] = (.97, .01)
        self.mode['detrend_deriv'] = [True, True, True]
        self.mode['detrend_medminfilt'] = [False, False, False, True]
        self.mode['watershed'] = [False, True]


class TestBackground(Template, unittest.TestCase):

    def setUp(self):
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for i in xrange(NTRIALS):
                sin_t = np.sin(np.linspace(0, 2 * np.pi, SIZE))
                sin_t.shape = (-1, 1)
                sin_f = np.sin(np.linspace(0, 4 * np.pi, SIZE))
                sin_f.shape = (1, -1)
                data = 5 * sin_t * sin_f + qmtest.real_noise((SIZE, SIZE))
                rfi = (np.random.randint(SIZE, size=RFI),
                       np.random.randint(SIZE, size=RFI))
                data[rfi] = NSIG
                yield data, rfi
            return
        self.rfi_gen = rfi_gen
        self.ans['detrend_deriv'] = [(.83, .83, .83), (.1, .1, .1)]
        self.ans['detrend_medminfilt'] = [(.2, .2, .3, 0.), (.1, .1, .1, .8)]
        self.ans['xrfi'] = (.7, .1)
        self.ans['xrfi_simple'] = [(.90, .90), (.1, .1)]
        self.mode['detrend_deriv'] = [False, False, True]
        self.mode['detrend_medminfilt'] = [False, False, False, True]
        self.mode['watershed'] = [True, True]


class TestXrfiRun(object):
    def test_xrfi_run_xrfi(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi')
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=miriad"
        arg2 = "--extension=R"
        arg3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg4 = "--algorithm=xrfi"
        arg5 = "--kt_size=2"
        arg6 = "--kf_size=2"
        arg7 = "--sig_init=6"
        arg8 = "--sig_adj=2"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args.files, args, history)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        xrfi.xrfi_run(args.files, args, cmd)
        nt.assert_true(os.path.exists(dest_file))

    def test_xrfi_run_xrfi_simple(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('xrfi')
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=miriad"
        arg2 = "--extension=R"
        arg3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg4 = "--algorithm=xrfi_simple"
        arg5 = "--nsig_dt=6"
        arg6 = "--nsig_df=6"
        arg7 = "--nsig_all=0"
        arg8 = "--summary"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args.files, args, cmd)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR')
        sum_file = os.path.join(DATA_PATH, 'test_output',
                                'zen.2457698.40355.xx.HH.uvcAA.flag_summary.npz')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        if os.path.exists(sum_file):
            os.remove(sum_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.files, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        nt.assert_true(os.path.exists(sum_file))

    def test_xrfi_run_errors(self):
        # test code to read different file formats
        # these will raise errors
        a = utils.get_metrics_ArgumentParser('xrfi')
        arg0 = "--infile_format=uvfits"
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(IOError, xrfi.xrfi_run, args.files, args, cmd)

        arg0 = "--infile_format=fhd"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(StandardError, xrfi.xrfi_run, args.files, args, cmd)

        arg0 = "--infile_format=blah"
        cmd = ' '.join([arg0, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.files, args, cmd)

        # choose an invalid alrogithm
        arg0 = "--infile_format=miriad"
        arg1 = "--algorithm=foo"
        arguments = ' '.join([arg0, arg1])
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.files, args, cmd)

        # choose an invalid output format
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=blah"
        arg2 = "--algorithm=xrfi_simple"
        arguments = ' '.join([arg0, arg1, arg2])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args.files, args, cmd)

    def test_xrfi_run_output_args(self):
        # test different output arguments
        a = utils.get_metrics_ArgumentParser('xrfi')

        # test writing uvfits
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=uvfits"
        arg2 = "--extension=.uvfits"
        arg3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg4 = "--algorithm=xrfi_simple"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.uvfits')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.files, args, cmd)
        nt.assert_true(os.path.exists(dest_file))

        # test writing to same directory
        arg0 = "--infile_format=miriad"
        arg1 = "--outfile_format=miriad"
        arg2 = "--extension=R"
        arg3 = "--algorithm=xrfi_simple"
        arguments = ' '.join([arg0, arg1, arg2, arg3])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        xrfi.xrfi_run(args.files, args, cmd)
        nt.assert_true(os.path.exists(dest_file))
        # clean up
        shutil.rmtree(dest_file)


class TestSummary(unittest.TestCase):
    def test_summarize_flags(self):
        from hera_qm.version import hera_qm_version_str
        from pyuvdata import UVData

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


class TestBroadcast(unittest.TestCase):
    def test_summarize_flags(self):
        from pyuvdata import UVData

        infile = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        uv = UVData()
        uv.read_miriad(infile)
        uv.flag_array[0, 0, uv.Nfreqs / 2, 0] = True
        bflags = xrfi.broadcast_flags(uv, threshold=0.)
        nbl = np.sum(uv.time_array == uv.time_array[0])
        self.assertEqual(bflags.mean(), float(nbl) / (uv.Nblts * uv.Nfreqs))
        # Check thresholding works correctly
        bflags = xrfi.broadcast_flags(uv, threshold=0.5)
        self.assertEqual(bflags.sum(), 1)


if __name__ == '__main__':
    unittest.main()
