import unittest
import glob
import hera_qm.xrfi as xrfi
import numpy as np
import pylab as plt
import hera_qm.tests as qmtest
from inspect import getargspec


np.random.seed(0)

SIZE = 100
VERBOSE = False
LOT = False

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

def fake_flags():
    RFI = 10
    SIZE = 100
    fakeflags = np.random.randint(0,2,size=(SIZE,SIZE)).astype(bool)
    return fakeflags
    

def plot_waterfall(data, f, mx=10, drng=10, mode='lin'):
    #if not PLOT:
    #    return
    plt.subplot(121)
    plt.imshow(np.abs(data),aspect='auto',cmap='jet')
#    capo.plot.waterfall(data, mode='lin', mx=10, drng=10)
    plt.colorbar()
    plt.subplot(122)
    #capo.plot.waterfall(f, mode='lin', mx=10, drng=10)
    plt.imshow(f,aspect='auto',cmap='jet')
    plt.colorbar()
    plt.show()


def plot_result(f, rfi):
    #if not PLOT:
    #    return
    plt.plot(rfi[0], rfi[1], 'ko')
    fi = np.where(f)
    plt.plot(fi[0], fi[1], 'r.')
    plt.show()


class Template():

    def setUp(self):
        raise unittest.SkipTest  # setUp has to be overridden to actually run a test
    rfi_gen = None  # Need to override this for each TestCase, usually in setUp

    def _run_test(self, func, arg, correct_flag, false_positive, nsig=5):
        for data, rfi in self.rfi_gen():      
            f = func(data,*arg)
            if VERBOSE:
                print self.__class__, func.__name__
            #plot_waterfall(data, f)
            f = np.where(f > nsig, 1, 0)
            cf, fp = get_accuracy(f, rfi)
            print cf, fp
            #plot_result(f, rfi)
            self.assertGreater(cf, correct_flag)
            self.assertLess(fp, false_positive)
    ans = {
        'detrend_deriv': [(.9, .9, .9),(.1,.1,.1)],
        'detrend_medfilt': (.99, .01),
        'detrend_medminfilt': (.97, .05),
        'xrfi_simple': [(.99, .99),(.01, .01)],
        'xrfi': (.99, .01),
    }

    def test_detrend_deriv(self):
        cf, fp = self.ans['detrend_deriv']
        args = [(True,True),(True,False),(False,True)]
        #for arg in argsList:
        for i in range(3):
            print args[i]
            print cf[i],fp[i]
            self._run_test(xrfi.detrend_deriv, args[i],  cf[i], fp[i], nsig=4)

    def test_detrend_medfilt(self):
        cf, fp = self.ans['detrend_medfilt']
        argsList = [(8,8),(7,9),(9,7)]
        for arg in argsList:
            self._run_test(xrfi.detrend_medfilt, arg, cf, fp, nsig=4)

    def test_detrend_medminfilt(self):
        cf, fp = self.ans['detrend_medminfilt']
        argsList = [(8,8),(7,9),(9,7)]
        for arg in argsList:
            self._run_test(xrfi.detrend_medminfilt, arg, cf, fp, nsig=6)

    def test_xrfi_simple(self):
        cf, fp = self.ans['xrfi_simple']
        args = getargspec(xrfi.xrfi_simple).defaults
        fflags = fake_flags()
        argsList = [args,(fflags, 6, 6, 1)]
        for i in range(2):
            self._run_test(xrfi.xrfi_simple, argsList[i], cf[i], fp[i], nsig=.5)

    def test_xrfi(self):
        cf, fp = self.ans['xrfi']
        args = getargspec(xrfi.xrfi_simple).defaults
        self._run_test(xrfi.xrfi, args, cf, fp, nsig=.5)


class TestSparseScatter(Template, unittest.TestCase):

    def setUp(self):
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
        self.ans['detrend_deriv'] = [(.33, .33, .33),(.1, .1, .1)]
        self.ans['xrfi_simple'] = [(.90, .90),(.1, .1)]


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
        self.ans['xrfi_simple'] = [(.39, .39),(.1, .1)]
        self.ans['detrend_deriv'] = [(-.05, -.05, -.05),(.1, .1, .1)]

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
        self.ans['detrend_deriv'] = [(.01, .01, .01),(0.1,0.1,0.1)]
        self.ans['xrfi_simple'] = [(.75, .75),(.1, .1)]
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
        self.ans['detrend_deriv'] = [(.83, .83, .83),(.1,.1,.1)]
        self.ans['detrend_medminfilt'] = (.2, .1)
        self.ans['xrfi'] = (.75, .1)
        self.ans['xrfi_simple'] = [(.90, .90),(.1, .1)]

# class TestHERA(Template, unittest.TestCase):
#    def setUp(self):
#        def rfi_gen():
#            for f in FILES['hera']:
#                data = np.load(f)['chisq']
#                rfi = np.where(xrfi.xrfi(data)) # XXX actual answers?
#                yield data, rfi
#            return
#        self.rfi_gen = rfi_gen
#        self.ans['detrend_deriv'] = (.05, .1)
#        self.ans['detrend_medfilt'] = (.5, .1)
#        self.ans['detrend_medminfilt'] = (.30, .1)
#        self.ans['xrfi_simple'] = (.40, .3)
#
# class TestPAPER(Template, unittest.TestCase):
#    def setUp(self):
#        def rfi_gen():
#            for f in FILES['paper']:
#                data = np.load(f)['chisq']
#                rfi = np.where(xrfi.xrfi(data)) # XXX actual answers?
#                yield data, rfi
#            return
#        self.rfi_gen = rfi_gen
#        self.ans['detrend_deriv'] = (.0, .1)
#        self.ans['detrend_medfilt'] = (.1, .1)
#        self.ans['detrend_medminfilt'] = (.0, .35)
#        self.ans['xrfi_simple'] = (.3, .5)


if __name__ == '__main__':
    unittest.main()
