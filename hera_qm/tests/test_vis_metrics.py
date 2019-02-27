# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
import hera_qm.tests as qmtest
import unittest
from hera_qm import vis_metrics
import numpy as np
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_qm.data import DATA_PATH
import hera_qm.tests as qmtest
import os
import pyuvdata.tests as uvtest
import copy
import nose.tools as nt
import matplotlib.pyplot as plt
from scipy import stats


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data = UVData()
        filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        self.data.read_miriad(filename)
        # massage the object to make it work with check_noise_variance
        self.data.select(antenna_nums=self.data.get_ants()[0:10])
        self.data.select(freq_chans=range(100))
        # Data file only has three times... need more.
        while self.data.Ntimes < 90:
            d2 = copy.deepcopy(self.data)
            d2.time_array += d2.time_array.max() + d2.integration_time / (24 * 3600)
            self.data += d2
        ntimes = self.data.Ntimes
        nchan = self.data.Nfreqs
        self.data1 = qmtest.noise(size=(ntimes, nchan))
        self.data2 = qmtest.noise(size=(ntimes, nchan))
        ant_dat = {}
        for i in self.data.get_ants():
            ant_dat[i] = qmtest.noise(size=(ntimes, nchan)) + 0.1 * self.data1
        for key in self.data.get_antpairpols():
            ind = self.data._key2inds(key)[0]
            self.data.data_array[ind, 0, :, 0] = ant_dat[key[0]] * ant_dat[key[1]].conj()

    def test_check_noise_variance(self):
        nos = vis_metrics.check_noise_variance(self.data)
        for bl in self.data.get_antpairs():
            inds = self.data.antpair2ind(*bl)
            n = nos[bl + (uvutils.parse_polstr('xx'),)]
            self.assertEqual(n.shape, (self.data.Nfreqs - 1,))
            nsamp = self.data.channel_width * self.data.integration_time[inds][0]
            np.testing.assert_almost_equal(n, np.ones_like(n) * nsamp,
                                           -np.log10(nsamp))

    def test_check_noise_variance_inttime_error(self):
        self.data.integration_time = (self.data.integration_time
                                      * np.arange(self.data.integration_time.size))
        self.assertRaises(NotImplementedError,
                          vis_metrics.check_noise_variance, self.data)


def test_vis_bl_cov():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA'))

    # test basic execution
    bls = [(0, 1), (11, 12), (12, 13), (13, 14), (23, 24), (24, 25)]
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls)
    nt.assert_equal(cov.shape, (6, 6, 1, 1))
    nt.assert_equal(cov.dtype, np.complex128)
    nt.assert_almost_equal(cov[0, 0, 0, 0], (51.06967733738634 + 0j))

    # test iterax
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, iterax='freq')
    nt.assert_equal(cov.shape, (6, 6, 1, 1024))
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, iterax='time')
    nt.assert_equal(cov.shape, (6, 6, 1, 1))

    # test corr
    corr = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, return_corr=True)
    nt.assert_true(np.isclose(np.abs(corr).max(), 1.0))
    nt.assert_almost_equal(corr[1, 0, 0, 0], (0.4204243425812837 - 0.3582194575457562j))


@qmtest.skipIf_no_matplotlib
def test_plot_bl_cov():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA'))

    # basic execution
    fig, ax = plt.subplots()
    bls = [(0, 1), (11, 12), (12, 13), (13, 14), (23, 24), (24, 25)]
    vis_metrics.plot_bl_bl_cov(uvd, uvd, bls, ax=ax, component='abs', colorbar=True,
                               freqs=np.unique(uvd.freq_array)[:10])
    plt.close('all')
    fig = vis_metrics.plot_bl_bl_cov(uvd, uvd, bls, component='real', plot_corr=True)
    plt.close('all')
    fig = vis_metrics.plot_bl_bl_cov(uvd, uvd, bls, component='imag')
    plt.close('all')


@qmtest.skipIf_no_matplotlib
def test_plot_bl_bl_scatter():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA'))

    # basic execution
    bls = uvd.get_antpairs()[:3]  # should use redundant bls, but this is just a test...
    Nbls = len(bls)
    fig, axes = plt.subplots(Nbls, Nbls, figsize=(8, 8))
    vis_metrics.plot_bl_bl_scatter(uvd, uvd, bls, axes=axes, component='real', colorax='freq',
                                   freqs=np.unique(uvd.freq_array)[100:900], axfontsize=10)
    plt.close('all')
    fig = vis_metrics.plot_bl_bl_scatter(uvd, uvd, bls, component='abs', colorax='time', whiten=False)
    plt.close('all')
    fig = vis_metrics.plot_bl_bl_scatter(uvd, uvd, bls, component='imag', colorax='time', whiten=True)
    plt.close('all')
    fig = vis_metrics.plot_bl_bl_scatter(uvd, uvd, bls, component='angle')
    plt.close('all')

    # test exceptions
    nt.assert_raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, bls, component='foo')
    nt.assert_raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, uvd.get_antpairs())
    nt.assert_raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, bls, colorax='foo')
    uv = copy.deepcopy(uvd)
    # test flagged first bl data and no xylim fails
    uv.flag_array[uv.antpair2ind(bls[0], ordered=False)] = True
    nt.assert_raises(ValueError, vis_metrics.plot_bl_bl_scatter, uv, uv, bls)
    # once xylim specified, should pass
    fig = vis_metrics.plot_bl_bl_scatter(uv, uv, bls, xylim=(-50, 50))
    plt.close('all')


def test_sequential_diff():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))

    # diff across time
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=0, pad=False)
    nt.assert_equal(uvd_diff.Ntimes, uvd.Ntimes - 1)
    nt.assert_equal(uvd_diff.Nfreqs, uvd.Nfreqs)

    # diff across freq
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=1, pad=False)
    nt.assert_equal(uvd_diff.Ntimes, uvd.Ntimes)
    nt.assert_equal(uvd_diff.Nfreqs, uvd.Nfreqs - 1)

    # diff across both
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0, 1), pad=False)
    nt.assert_equal(uvd_diff.Ntimes, uvd.Ntimes - 1)
    nt.assert_equal(uvd_diff.Nfreqs, uvd.Nfreqs - 1)

    # switch diff and test closeness to within 5 decimals
    uvd_diff2 = vis_metrics.sequential_diff(uvd, axis=(1, 0), pad=False)
    nt.assert_true(np.isclose(uvd_diff.data_array, uvd_diff2.data_array, atol=1e-5).all())

    # test flag propagation
    uvd.flag_array[uvd.antpair2ind(89, 96, ordered=False)[:1]] = True
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0,), pad=False)
    nt.assert_true(uvd_diff.get_flags(89, 96)[0].all())
    nt.assert_false(uvd_diff.get_flags(89, 96)[1:].any())

    # test exception
    nt.assert_raises(ValueError, vis_metrics.sequential_diff, uvd, axis=3)
    nt.assert_raises(ValueError, vis_metrics.sequential_diff, 'foo')
    nt.assert_raises(ValueError, vis_metrics.sequential_diff, np.arange(10), t_int='foo')

    # fake noise test
    uvn = copy.deepcopy(uvd)
    uvn.flag_array[:] = False
    f = np.arange(uvn.Nfreqs)
    t = np.arange(uvn.Ntimes)
    np.random.seed(0)
    for bl in uvn.get_antpairs():
        # generate random noise
        n = (stats.norm.rvs(0, 1 / np.sqrt(2), uvn.Ntimes * uvn.Nfreqs)
             + 1j * stats.norm.rvs(0, 1 / np.sqrt(2), uvn.Ntimes * uvn.Nfreqs)).reshape(uvn.Ntimes, uvn.Nfreqs)

        # generate smooth signal
        s = np.exp(1j * f[None, :] / 100.0 + 1j * t[:, None] / 10.0)

        # add into data
        uvn.data_array[uvn.antpair2ind(bl, ordered=False), 0, :, 0] = s + n

    # run sequential diff
    uvn_diff1 = vis_metrics.sequential_diff(uvn, axis=(0, ), pad=False)
    uvn_diff2 = vis_metrics.sequential_diff(uvn, axis=(1, ), pad=False)
    uvn_diff3 = vis_metrics.sequential_diff(uvn, axis=(0, 1), pad=False)

    # assert noise std is equal to 1 within sampling error
    nt.assert_almost_equal(np.std(uvn_diff1.data_array), 1.0,
                           delta=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))
    nt.assert_almost_equal(np.std(uvn_diff2.data_array), 1.0,
                           delta=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))
    nt.assert_almost_equal(np.std(uvn_diff3.data_array), 1.0,
                           delta=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))

    # test pad
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0, 1), pad=True)
    nt.assert_equal(uvd_diff.Ntimes, uvd.Ntimes)
    nt.assert_equal(uvd_diff.Nfreqs, uvd.Nfreqs)
    nt.assert_true(uvd_diff.flag_array[:, 0, -1, 0].all())
    nt.assert_true(uvd_diff.select(times=np.unique(uvd_diff.time_array)[-1:], inplace=False).flag_array.all())


if __name__ == '__main__':
    unittest.main()
