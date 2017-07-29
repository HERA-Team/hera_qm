import unittest
import nose.tools as nt
import glob
import os
import shutil
import hera_qm.xrfi as xrfi
import numpy as np
import pylab as plt
import hera_qm.tests as qmtest
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH

np.random.seed(0)

SIZE = 100
VERBOSE = False
PLOT = False

FILES = {
    'paper': glob.glob('xrfi_data/paper/chisq0*.npz'),
    'hera': glob.glob('xrfi_data/hera/chisq0*.npz'),
}


def get_accuracy(f, rfi, verbose=VERBOSE):
    correctly_flagged = np.average(f[rfi])
    m = f.copy()
    m[rfi] = 0
    false_positive = float(np.sum(m)) / (m.size - len(rfi[0]))
    if verbose:
        print '\t Found RFI: %1.3f\n\t False Positive: %1.3f' % (correctly_flagged, false_positive)
    return correctly_flagged, false_positive


def plot_waterfall(data, f, mx=10, drng=10, mode='lin'):
    if not PLOT:
        return
    plt.subplot(121)
    capo.plot.waterfall(data, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.subplot(122)
    capo.plot.waterfall(f, mode='lin', mx=10, drng=10)
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

    def _run_test(self, func, correct_flag, false_positive, nsig=4):
        for data, rfi in self.rfi_gen():
            f = func(data)
            if VERBOSE:
                print self.__class__, func.__name__
            # plot_waterfall(data, f)
            f = np.where(f > nsig, 1, 0)
            cf, fp = get_accuracy(f, rfi)
            # plot_result(f, rfi)
            self.assertGreater(cf, correct_flag)
            self.assertLess(fp, false_positive)
    ans = {
        'detrend_deriv': (.9, .1),
        'detrend_medfilt': (.99, .01),
        'detrend_medminfilt': (.97, .05),
        'xrfi_simple': (.99, .1),
        'xrfi': (.99, .01),
    }

    def test_detrend_deriv(self):
        cf, fp = self.ans['detrend_deriv']
        self._run_test(xrfi.detrend_deriv, cf, fp, nsig=4)

    def test_detrend_medfilt(self):
        cf, fp = self.ans['detrend_medfilt']
        self._run_test(xrfi.detrend_medfilt, cf, fp, nsig=4)

    def test_detrend_medminfilt(self):
        cf, fp = self.ans['detrend_medminfilt']
        self._run_test(xrfi.detrend_medminfilt, cf, fp, nsig=6)

    def test_xrfi_simple(self):
        cf, fp = self.ans['xrfi_simple']
        self._run_test(xrfi.xrfi_simple, cf, fp, nsig=.5)

    def test_xrfi(self):
        cf, fp = self.ans['xrfi']
        self._run_test(xrfi.xrfi, cf, fp, nsig=.5)


class TestSparseScatter(Template, unittest.TestCase):

    def setUp(self):
        RFI = 50
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


class TestDenseScatter(Template, unittest.TestCase):

    def setUp(self):
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
        self.ans['detrend_deriv'] = (.33, .1)
        self.ans['xrfi_simple'] = (.90, .1)


class TestCluster(Template, unittest.TestCase):

    def setUp(self):
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
        self.ans['xrfi_simple'] = (.39, .1)
        self.ans['detrend_deriv'] = (-.05, .1)


class TestLines(Template, unittest.TestCase):

    def setUp(self):
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
        self.ans['detrend_deriv'] = (.0, .1)
        self.ans['xrfi_simple'] = (.75, .1)
        self.ans['xrfi'] = (.97, .01)


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
        self.ans['detrend_deriv'] = (.83, .1)
        self.ans['detrend_medminfilt'] = (.2, .1)
        self.ans['xrfi'] = (.75, .1)
        self.ans['xrfi_simple'] = (.90, .1)

class TestXrfiRun(object):
    def test_xrfi_run_xrfi(self):
        # get options object
        o = utils.get_metrics_OptionParser('xrfi')
        opt0 = "--infile_format=miriad"
        opt1 = "--outfile_format=miriad"
        opt2 = "--extension=R"
        opt3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        opt4 = "--algorithm=xrfi"
        opt5 = "--k_size=2"
        opt6 = "--sig_init=6"
        opt7 = "--sig_adj=2"
        options = ' '.join([opt0, opt1, opt2, opt3, opt4, opt5, opt6, opt7])

        # test running with no files
        cmd = ' '.join([options, ''])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args, opts, history)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        xrfi.xrfi_run(args, opts, cmd)
        nt.assert_true(os.path.exists(dest_file))

    def test_xrfi_run_xrfi_simple(self):
        # get options object
        o = utils.get_metrics_OptionParser('xrfi')
        opt0 = "--infile_format=miriad"
        opt1 = "--outfile_format=miriad"
        opt2 = "--extension=R"
        opt3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        opt4 = "--algorithm=xrfi_simple"
        opt5 = "--nsig_dt=6"
        opt6 = "--nsig_df=6"
        options = ' '.join([opt0, opt1, opt2, opt3, opt4, opt5, opt6])

        # test running with no files
        cmd = ' '.join([options, ''])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(AssertionError, xrfi.xrfi_run, args, opts, cmd)

        # test running on our test data
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        xrfi.xrfi_run(args, opts, cmd)
        nt.assert_true(os.path.exists(dest_file))

    def test_xrfi_run_errors(self):
        # test code to read different file formats
        # these will raise errors
        o = utils.get_metrics_OptionParser('xrfi')
        opt0 = "--infile_format=uvfits"
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([opt0, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(IOError, xrfi.xrfi_run, args, opts, cmd)

        opt0 = "--infile_format=fhd"
        cmd = ' '.join([opt0, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(StandardError, xrfi.xrfi_run, args, opts, cmd)

        opt0 = "--infile_format=ms"
        cmd = ' '.join([opt0, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(RuntimeError, xrfi.xrfi_run, args, opts, cmd)

        opt0 = "--infile_format=blah"
        cmd = ' '.join([opt0, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args, opts, cmd)

        # choose an invalid alrogithm
        opt0 = "--infile_format=miriad"
        opt1 = "--algorithm=foo"
        options = ' '.join([opt0, opt1])
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args, opts, cmd)

        # choose an invalid output format
        opt0 = "--infile_format=miriad"
        opt1 = "--outfile_format=blah"
        opt2 = "--algorithm=xrfi_simple"
        options = ' '.join([opt0, opt1, opt2])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        nt.assert_raises(ValueError, xrfi.xrfi_run, args, opts, cmd)

    def test_xrfi_run_output_options(self):
        # test different output options
        o = utils.get_metrics_OptionParser('xrfi')

        # test writing uvfits
        opt0 = "--infile_format=miriad"
        opt1 = "--outfile_format=uvfits"
        opt2 = "--extension=.uvfits"
        opt3 = "--xrfi_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        opt4 = "--algorithm=xrfi_simple"
        options = ' '.join([opt0, opt1, opt2, opt3, opt4])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'test_output', 'zen.2457698.40355.xx.HH.uvcAA.uvfits')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        xrfi.xrfi_run(args, opts, cmd)
        nt.assert_true(os.path.exists(dest_file))

        # test writing to same directory
        opt0 = "--infile_format=miriad"
        opt1 = "--outfile_format=miriad"
        opt2 = "--extension=R"
        opt3 = "--algorithm=xrfi_simple"
        options = ' '.join([opt0, opt1, opt2, opt3])
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        dest_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAAR')
        if os.path.exists(dest_file):
            shutil.rmtree(dest_file)
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        xrfi.xrfi_run(args, opts, cmd)
        nt.assert_true(os.path.exists(dest_file))
        # clean up
        shutil.rmtree(dest_file)


if __name__ == '__main__':
    unittest.main()
