# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
import hera_qm.tests as qmtest
from hera_qm import vis_metrics
import numpy as np
from pyuvdata import UVData
from pyuvdata import utils as uvutils
from hera_qm.data import DATA_PATH
import os
import copy
import pytest
from scipy import stats

pytestmark = pytest.mark.filterwarnings(
    "ignore:The uvw_array does not match the expected values given the antenna positions.",
)

@pytest.fixture(scope='function')
def vismetrics_data():
    data = UVData()
    filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
    data.read_miriad(filename)
    # massage the object to make it work with check_noise_variance
    data.select(antenna_nums=data.get_ants()[0:10])
    data.select(freq_chans=range(100))
    # Data file only has three times... need more.
    while data.Ntimes < 90:
        d2 = copy.deepcopy(data)
        d2.time_array += d2.time_array.max() + d2.integration_time / (24 * 3600)
        data += d2
    ntimes = data.Ntimes
    nchan = data.Nfreqs
    data1 = qmtest.noise(size=(ntimes, nchan))
    data2 = qmtest.noise(size=(ntimes, nchan))
    ant_dat = {}
    for i in data.get_ants():
        ant_dat[i] = qmtest.noise(size=(ntimes, nchan)) + 0.1 * data1
    for key in data.get_antpairpols():
        ind = data._key2inds(key)[0]
        data.data_array[ind, 0, :, 0] = ant_dat[key[0]] * ant_dat[key[1]].conj()

    class DataHolder(object):
        def __init__(self, data, data1, data2):
            self.data = data
            self.data1 = data1
            self.data2 = data2

    vismetrics_data = DataHolder(data, data1, data2)
    # yield lets us return the data and then continue with clean up after
    yield vismetrics_data

    # post test clean up
    del(vismetrics_data)

    return


def test_check_noise_variance(vismetrics_data):
    nos = vis_metrics.check_noise_variance(vismetrics_data.data)
    for bl in vismetrics_data.data.get_antpairs():
        inds = vismetrics_data.data.antpair2ind(*bl)
        n = nos[bl + (uvutils.parse_polstr('xx'),)]
        assert n.shape == (vismetrics_data.data.Nfreqs - 1,)
        nsamp = vismetrics_data.data.channel_width * vismetrics_data.data.integration_time[inds][0]
        np.testing.assert_almost_equal(n, np.ones_like(n) * nsamp,
                                       -np.log10(nsamp))


def test_check_noise_variance_inttime_error(vismetrics_data):
    vismetrics_data.data.integration_time = (vismetrics_data.data.integration_time
                                             * np.arange(vismetrics_data.data.integration_time.size))
    pytest.raises(NotImplementedError,
                  vis_metrics.check_noise_variance, vismetrics_data.data)


def test_vis_bl_cov():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA'))

    # test basic execution
    bls = [(0, 1), (11, 12), (12, 13), (13, 14), (23, 24), (24, 25)]
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls)
    assert cov.shape == (6, 6, 1, 1)
    assert cov.dtype == np.complex128
    assert np.isclose(cov[0, 0, 0, 0], (51.06967733738634 + 0j))

    # test iterax
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, iterax='freq')
    assert cov.shape == (6, 6, 1, 1024)
    cov = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, iterax='time')
    assert cov.shape == (6, 6, 1, 1)

    # test corr
    corr = vis_metrics.vis_bl_bl_cov(uvd, uvd, bls, return_corr=True)
    assert np.isclose(np.abs(corr).max(), 1.0)
    assert np.isclose(corr[1, 0, 0, 0], (0.4204243425812837 - 0.3582194575457562j))


def test_plot_bl_cov():
    plt = pytest.importorskip("matplotlib.pyplot")
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


def test_plot_bl_bl_scatter():
    plt = pytest.importorskip("matplotlib.pyplot")
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
    pytest.raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, bls, component='foo')
    pytest.raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, uvd.get_antpairs())
    pytest.raises(ValueError, vis_metrics.plot_bl_bl_scatter, uvd, uvd, bls, colorax='foo')
    uv = copy.deepcopy(uvd)
    # test flagged first bl data and no xylim fails
    uv.flag_array[uv.antpair2ind(bls[0], ordered=False)] = True
    pytest.raises(ValueError, vis_metrics.plot_bl_bl_scatter, uv, uv, bls)
    # once xylim specified, should pass
    fig = vis_metrics.plot_bl_bl_scatter(uv, uv, bls, xylim=(-50, 50))
    plt.close('all')


def test_sequential_diff():
    uvd = UVData()
    uvd.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))

    # diff across time
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=0, pad=False)
    assert uvd_diff.Ntimes == uvd.Ntimes - 1
    assert uvd_diff.Nfreqs == uvd.Nfreqs

    # diff across freq
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=1, pad=False)
    assert uvd_diff.Ntimes == uvd.Ntimes
    assert uvd_diff.Nfreqs == uvd.Nfreqs - 1

    # diff across both
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0, 1), pad=False)
    assert uvd_diff.Ntimes == uvd.Ntimes - 1
    assert uvd_diff.Nfreqs == uvd.Nfreqs - 1

    # switch diff and test closeness to within 5 decimals
    uvd_diff2 = vis_metrics.sequential_diff(uvd, axis=(1, 0), pad=False)
    assert np.isclose(uvd_diff.data_array, uvd_diff2.data_array, atol=1e-5).all()

    # test flag propagation
    uvd.flag_array[uvd.antpair2ind(89, 96, ordered=False)[:1]] = True
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0,), pad=False)
    assert uvd_diff.get_flags(89, 96)[0].all()
    assert not uvd_diff.get_flags(89, 96)[1:].any()

    # test exception
    pytest.raises(ValueError, vis_metrics.sequential_diff, uvd, axis=3)
    pytest.raises(ValueError, vis_metrics.sequential_diff, 'foo')
    pytest.raises(ValueError, vis_metrics.sequential_diff, np.arange(10), t_int='foo')

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
    assert np.isclose(np.std(uvn_diff1.data_array), 1.0, atol=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))
    assert np.isclose(np.std(uvn_diff2.data_array), 1.0, atol=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))
    assert np.isclose(np.std(uvn_diff3.data_array), 1.0, atol=1 / np.sqrt(uvn.Ntimes * uvn.Nfreqs))

    # test pad
    uvd_diff = vis_metrics.sequential_diff(uvd, axis=(0, 1), pad=True)
    assert uvd_diff.Ntimes == uvd.Ntimes
    assert uvd_diff.Nfreqs == uvd.Nfreqs
    assert uvd_diff.flag_array[:, 0, -1, 0].all()
    assert uvd_diff.select(times=np.unique(uvd_diff.time_array)[-1:], inplace=False).flag_array.all()
