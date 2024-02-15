# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import unittest
import hera_qm.xrfi as xrfi
import numpy as np
import pylab as plt
import hera_qm.tests as qmtest
from inspect import getargspec

np.random.seed(0)

SIZE = 100
VERBOSE = False
PLOT = False


def get_accuracy(flags, rfi, verbose=VERBOSE):
    correctly_flagged = np.average(flags[rfi])
    m = flags.copy()
    m[rfi] = 0
    false_positive = float(np.sum(m)) / (m.size - len(rfi[0]))
    if verbose:
        print('\t Found RFI: %1.3f\n\t False Positive: %1.3f' % (correctly_flagged, false_positive))
    return correctly_flagged, false_positive


def fake_flags(SIZE):
    fakeflags = np.random.randint(0, 2, size=(SIZE, SIZE)).astype(bool)
    return fakeflags


def plot_waterfall(data, flags, mx=10, drng=10, mode='lin'):
    if not PLOT:
        return
    plt.subplot(121)
    plt.imshow(np.abs(data), aspect='auto', cmap='jet')
    capo.plot.waterfall(data, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.subplot(122)
    capo.plot.waterfall(flags, mode='lin', mx=10, drng=10)
    plt.imshow(f, aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()


def plot_result(flags, rfi):
    if not PLOT:
        return
    plt.plot(rfi[0], rfi[1], 'ko')
    fi = np.where(flags)
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
            except BaseException:
                # AssertionError check to make sure kernel size isn't too big
                self.assertRaises(AssertionError, func, data, *arg)
                f = fake_flags(SIZE)
            if VERBOSE:
                print(self.__class__, func.__name__)
            f = np.where(f > nsig, 1, 0)
            cf, fp = get_accuracy(f, rfi)
            if PLOT:
                plot_waterfall(data, f)
                plot_result(f, rfi)
            if fmode:
                if VERBOSE:
                    print('In failure mode now.')
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
        for ind in range(3):
            self._run_test(xrfi.detrend_deriv, args[ind], cf[ind], fp[ind], nsig=4,
                           fmode=mode[ind])

    def test_detrend_medfilt(self):
        cf, fp = self.ans['detrend_medfilt']
        argsList = [(8, 8), (7, 9), (9, 7), (1000, 1000)]
        for ind in range(4):
            self._run_test(xrfi.detrend_medfilt, argsList[ind], cf[ind], fp[ind], nsig=4)

    def test_detrend_medminfilt(self):
        cf, fp = self.ans['detrend_medminfilt']
        argsList = [(8, 8), (7, 9), (9, 7), (1000, 1000)]
        mode = self.mode['detrend_medminfilt']
        for ind in range(4):
            self._run_test(xrfi.detrend_medminfilt, argsList[ind], cf[ind], fp[ind], nsig=4, fmode=mode[ind])

    def test_xrfi_simple(self):
        cf, fp = self.ans['xrfi_simple']
        args = getargspec(xrfi.xrfi_simple).defaults
        fflags = fake_flags(SIZE)
        argsList = [args, (fflags, 6, 6, 1)]
        fmode = [False, True]
        for ind in range(2):
            self._run_test(xrfi.xrfi_simple, argsList[ind], cf[ind], fp[ind], nsig=.5, fmode=fmode[ind])

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
        for ind in range(2):
            self._run_test(xrfi.watershed_flag, argsList[ind], cf[ind], fp[ind], nsig=.5, fmode=mode[ind])


class TestSparseScatter(Template, unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for ind in xrange(NTRIALS):
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
            for ind in xrange(NTRIALS):
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
            for ind in xrange(NTRIALS):
                data = qmtest.real_noise((SIZE, SIZE))
                xpos, ypos = (np.random.randint(SIZE - 1, size=RFI),
                              np.random.randint(SIZE - 1, size=RFI))
                xpos = np.concatenate([xpos, xpos, xpos + 1, xpos + 1])
                ypos = np.concatenate([ypos, ypos + 1, ypos, ypos + 1])
                rfi = (np.array(xpos), np.array(ypos))
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
            for ind in xrange(NTRIALS):
                data = qmtest.real_noise((SIZE, SIZE))
                xpos, ypos = (np.random.randint(SIZE, size=RFI),
                              np.random.randint(SIZE, size=RFI))
                mask = np.zeros_like(data)
                mask[xpos] = 1
                mask[:, ypos] = 1
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
        np.random.seed(0)
        RFI = 50
        NTRIALS = 10
        NSIG = 10

        def rfi_gen():
            for ind in xrange(NTRIALS):
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


if __name__ == '__main__':
    unittest.main()
