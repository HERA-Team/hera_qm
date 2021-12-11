# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
import pytest
import os
import shutil
import hera_qm.xrfi as xrfi
import numpy as np
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
import hera_qm.utils as utils
from hera_qm.data import DATA_PATH
from pyuvdata import UVFlag
import glob


test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_uvfits_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.uvfits')
test_uvh5_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvh5')
test_c_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'
test_f_file_flags = test_d_file + '.testuvflag.flags.h5'  # version in 'flag' mode
test_outfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')
xrfi_path = os.path.join(DATA_PATH, 'test_output')
test_flag_integrations= os.path.join(DATA_PATH, 'a_priori_flags_integrations.yaml')
test_flag_jds= os.path.join(DATA_PATH, 'a_priori_flags_jds.yaml')
test_flag_lsts= os.path.join(DATA_PATH, 'a_priori_flags_lsts.yaml')

test_uvh5_files = ['zen.2457698.40355191.xx.HH.uvh5',
                   'zen.2457698.40367619.xx.HH.uvh5',
                   'zen.2457698.40380046.xx.HH.uvh5']
test_c_files = ['zen.2457698.40355191.xx.HH.uvcAA.omni.calfits',
                'zen.2457698.40367619.xx.HH.uvcAA.omni.calfits',
                'zen.2457698.40380046.xx.HH.uvcAA.omni.calfits']
for cnum, cf, uvf in zip(range(3), test_c_files, test_uvh5_files):
    test_c_files[cnum] = os.path.join(DATA_PATH, cf)
    test_uvh5_files[cnum] = os.path.join(DATA_PATH, uvf)

pytestmark = pytest.mark.filterwarnings(
    "ignore:The uvw_array does not match the expected values given the antenna positions.",
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA."
)

def test_uvdata():
    uv = UVData()
    uv.read_miriad(test_d_file)
    xant = uv.get_ants()[0]
    xrfi.flag_xants(uv, xant)
    assert np.all(uv.flag_array[uv.ant_1_array == xant, :, :, :])
    assert np.all(uv.flag_array[uv.ant_2_array == xant, :, :, :])


def test_uvcal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    xant = uvc.ant_array[0]
    xrfi.flag_xants(uvc, xant)
    assert np.all(uvc.flag_array[0, :, :, :, :])


def test_uvflag():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    xant = uvf.ant_1_array[0]
    xrfi.flag_xants(uvf, xant)
    assert np.all(uvf.flag_array[uvf.ant_1_array == xant, :, :, :])
    assert np.all(uvf.flag_array[uvf.ant_2_array == xant, :, :, :])


def test_input_error():
    pytest.raises(ValueError, xrfi.flag_xants, 4, 0)


def test_uvflag_waterfall_error():
    uvf = UVFlag(test_f_file)
    uvf.to_waterfall()
    uvf.to_flag()
    pytest.raises(ValueError, xrfi.flag_xants, uvf, 0)


def test_uvflag_not_flag_error():
    uvf = UVFlag(test_f_file)
    pytest.raises(ValueError, xrfi.flag_xants, uvf, 0)


def test_not_inplace_uvflag():
    uvf = UVFlag(test_f_file)
    xant = uvf.ant_1_array[0]
    uvf2 = xrfi.flag_xants(uvf, xant, inplace=False)
    assert np.all(uvf2.flag_array[uvf2.ant_1_array == xant, :, :, :])
    assert np.all(uvf2.flag_array[uvf2.ant_2_array == xant, :, :, :])


def test_not_inplace_uvdata():
    uv = UVData()
    uv.read_miriad(test_d_file)
    xant = uv.get_ants()[0]
    uv2 = xrfi.flag_xants(uv, xant, inplace=False)
    assert np.all(uv2.flag_array[uv2.ant_1_array == xant, :, :, :])
    assert np.all(uv2.flag_array[uv2.ant_2_array == xant, :, :, :])


def test_resolve_xrfi_path_given():
    dirname = xrfi.resolve_xrfi_path(xrfi_path, test_d_file)
    assert xrfi_path == dirname


def test_resolve_xrfi_path_empty():
    dirname = xrfi.resolve_xrfi_path('', test_d_file)
    assert os.path.dirname(os.path.abspath(test_d_file)) == dirname


def test_resolve_xrfi_path_does_not_exist():
    dirname = xrfi.resolve_xrfi_path(os.path.join(xrfi_path, 'foogoo'), test_d_file)
    assert os.path.dirname(os.path.abspath(test_d_file)) == dirname


def test_resolve_xrfi_path_jd_subdir():
    dirname = xrfi.resolve_xrfi_path('', test_d_file, jd_subdir=True)
    expected_dir = os.path.join(os.path.dirname(os.path.abspath(test_d_file)),
                                '.'.join(os.path.basename(test_d_file).split('.')[0:3])
                                + '.xrfi')
    assert dirname == expected_dir
    assert os.path.exists(expected_dir)
    shutil.rmtree(expected_dir)


def test_check_convolve_dims_3D():
    # Error if d.ndims != 2
    pytest.raises(ValueError, xrfi._check_convolve_dims, np.ones((3, 2, 3)), 1, 2)


def test_check_convolve_dims_1D():
    size = 10
    d = np.ones(size)
    with uvtest.check_warnings(
            UserWarning,
            match=f"K1 value {size + 1} is larger than the data",
            nwarnings=1
    ):
        K = xrfi._check_convolve_dims(d, size + 1)
    assert K == size


def test_check_convolve_dims_kernel_not_given():
    size = 10
    d = np.ones((size, size))
    with uvtest.check_warnings(
            UserWarning,
            match=["No K1 input provided.", "No K2 input provided"],
            nwarnings=2
    ):
        K1, K2 = xrfi._check_convolve_dims(d)
    assert K1 == size
    assert K2 == size


def test_check_convolve_dims_Kt_too_big():
    size = 10
    d = np.ones((size, size))
    with uvtest.check_warnings(
            UserWarning,
            match=f"K1 value {size + 1} is larger than the data",
            nwarnings=1,
    ):
        Kt, Kf = xrfi._check_convolve_dims(d, size + 1, size)
    assert Kt == size
    assert Kf == size


def test_check_convolve_dims_Kf_too_big():
    size = 10
    d = np.ones((size, size))
    with uvtest.check_warnings(
            UserWarning,
            match=f"K2 value {size + 1} is larger than the data",
            nwarnings=1,
    ):
        Kt, Kf = xrfi._check_convolve_dims(d, size, size + 1)
    assert Kt == size
    assert Kf == size


def test_check_convolve_dims_K1K2_lt_one():
    size = 10
    data = np.ones((size, size))
    pytest.raises(ValueError, xrfi._check_convolve_dims, data, 0, 2)
    pytest.raises(ValueError, xrfi._check_convolve_dims, data, 2, 0)


def test_robus_divide():
    a = np.array([1., 1., 1.], dtype=np.float32)
    b = np.array([2., 0., 1e-9], dtype=np.float32)
    c = xrfi.robust_divide(a, b)
    assert np.array_equal(c, np.array([1. / 2., np.inf, np.inf]))


@pytest.fixture(scope='function')
def fake_data():
    size = 100
    fake_data = np.zeros((size, size))

    # yield returns the data and lets us do post test clean up after
    yield fake_data

    # post-test clean up
    del(fake_data)

    return


def test_medmin(fake_data):
    # make fake data
    for i in range(fake_data.shape[1]):
        fake_data[:, i] = i * np.ones_like(fake_data[:, i])
    # medmin should be .size - 1 for these data
    medmin = xrfi.medmin(fake_data)
    assert np.allclose(medmin, fake_data.shape[0] - 1)

    # Test error when wrong dimensions are passed
    pytest.raises(ValueError, xrfi.medmin, np.ones((5, 4, 3)))


def test_medminfilt(fake_data):
    # make fake data
    for i in range(fake_data.shape[1]):
        fake_data[:, i] = i * np.ones_like(fake_data[:, i])
    # run medmin filt
    Kt = 8
    Kf = 8
    d_filt = xrfi.medminfilt(fake_data, Kt=Kt, Kf=Kf)

    # build up "answer" array
    ans = np.zeros_like(fake_data)
    for i in range(fake_data.shape[1]):
        if i < fake_data.shape[0] - Kf:
            ans[:, i] = i + (Kf - 1)
        else:
            ans[:, i] = fake_data.shape[0] - 1
    assert np.allclose(d_filt, ans)


def test_detrend_deriv(fake_data):
    # make fake data
    for i in range(fake_data.shape[0]):
        for j in range(fake_data.shape[1]):
            fake_data[i, j] = j * i**2 + j**3
    # run detrend_deriv in both dimensions
    dtdf = xrfi.detrend_deriv(fake_data, df=True, dt=True)
    ans = np.ones_like(dtdf)
    assert np.allclose(dtdf, ans)

    # only run along frequency
    for i in range(fake_data.shape[0]):
        for j in range(fake_data.shape[1]):
            fake_data[i, j] = j**3
    df = xrfi.detrend_deriv(fake_data, df=True, dt=False)
    ans = np.ones_like(df)
    assert np.allclose(df, ans)

    # only run along time
    for i in range(fake_data.shape[0]):
        for j in range(fake_data.shape[1]):
            fake_data[i, j] = i**3
    dt = xrfi.detrend_deriv(fake_data, df=False, dt=True)
    ans = np.ones_like(dt)
    assert np.allclose(dt, ans)

    # catch error of df and dt both being False
    pytest.raises(ValueError, xrfi.detrend_deriv, fake_data, dt=False, df=False)

    # Test error when wrong dimensions are passed
    pytest.raises(ValueError, xrfi.detrend_deriv, np.ones((5, 4, 3)))


def test_detrend_medminfilt(fake_data):
    # make fake data
    for i in range(fake_data.shape[1]):
        fake_data[:, i] = i * np.ones_like(fake_data[:, i])
    # run detrend_medminfilt
    Kt = 8
    Kf = 8
    dm = xrfi.detrend_medminfilt(fake_data, Kt=Kt, Kf=Kf)

    # read in "answer" array
    # this is output that corresponds to .size==100, Kt==8, Kf==8
    ans_fn = os.path.join(DATA_PATH, 'test_detrend_medminfilt_ans.txt')
    ans = np.loadtxt(ans_fn)
    assert np.allclose(ans, dm)


def test_detrend_medfilt():
    # make fake data
    x = np.sin(np.linspace(0, 2.1 * np.pi, 100))
    y = np.cos(np.linspace(0, 5.3 * np.pi, 100))
    fake_data = np.outer(x,y)
    # run detrend medfilt
    Kt = 101
    Kf = 101
    with uvtest.check_warnings(
            UserWarning,
            match=[
                f"K1 value {Kt} is larger than the data",
                f"K2 value {Kf} is larger than the data",
            ],
            nwarnings=2,
    ):
        dm = xrfi.detrend_medfilt(fake_data, None, Kt, Kf)

    # read in "answer" array
    # this is output that corresponds to .size==100, Kt==101, Kf==101
    ans_fn = os.path.join(DATA_PATH, 'test_detrend_medfilt_ans_v2.txt')
    ans = np.loadtxt(ans_fn)
    np.testing.assert_array_almost_equal(ans, dm)


def test_detrend_medfilt_complex():
    # use complex data
    x = np.sin(np.linspace(0, 2.1 * np.pi, 100)) + 1.0j * np.cos(np.linspace(0, 1.3 * np.pi, 100))
    y = np.cos(np.linspace(0, 5.3 * np.pi, 100)) + 1.0j * np.sin(np.linspace(0, 2.9 * np.pi, 100))
    fake_data = np.outer(x,y)

    # run detrend_medfilt
    Kt = 8
    Kf = 8
    dm = xrfi.detrend_medfilt(fake_data, Kt=Kt, Kf=Kf)

    # read in "answer" array
    # this is output that corresponds to .size=100, Kt=8, Kf=8
    ans_fn = os.path.join(DATA_PATH, 'test_detrend_medfilt_complex_ans_v2.txt')
    ans = np.loadtxt(ans_fn).view('complex')
    np.testing.assert_array_almost_equal(ans, dm)


def test_detrend_medfilt_3d_error():
    # Test error when wrong dimensions are passed
    pytest.raises(ValueError, xrfi.detrend_medfilt, np.ones((5, 4, 3)))


def test_detrend_meanfilt(fake_data):
    # make fake data
    for i in range(fake_data.shape[1]):
        fake_data[:, i] = i**2 * np.ones_like(fake_data[:, i])
    # run detrend medfilt
    Kt = 8
    Kf = 8
    dm = xrfi.detrend_meanfilt(fake_data, Kt=Kt, Kf=Kf)

    # read in "answer" array
    # this is output that corresponds to .size==100, Kt==8, Kf==8
    ans_fn = os.path.join(DATA_PATH, 'test_detrend_meanfilt_ans.txt')
    ans = np.loadtxt(ans_fn)
    assert np.allclose(ans, dm)


def test_detrend_meanfilt_flags(fake_data):
    # make fake data
    for i in range(fake_data.shape[1]):
        fake_data[:, i] = i * np.ones_like(fake_data[:, i])
    ind = int(fake_data.shape[0] / 2)
    fake_data[ind, :] = 10000.
    flags = np.zeros(fake_data.shape, dtype=np.bool_)
    flags[ind, :] = True
    # run detrend medfilt
    Kt = 8
    Kf = 8
    dm1 = xrfi.detrend_meanfilt(fake_data, flags=flags, Kt=Kt, Kf=Kf)

    # Compare with drastically different flagged values
    fake_data[ind, :] = 0
    dm2 = xrfi.detrend_meanfilt(fake_data, flags=flags, Kt=Kt, Kf=Kf)
    dm2[ind, :] = dm1[ind, :]  # These don't have valid values, so don't compare them.
    assert np.allclose(dm1, dm2)


def test_zscore_full_array(fake_data):
    # Make some fake data
    np.random.seed(182)
    fake_data[...] = np.random.randn(fake_data.shape[0], fake_data.shape[1])
    out = xrfi.zscore_full_array(fake_data)
    fake_mean = np.mean(fake_data)
    fake_std = np.std(fake_data)
    assert np.all(out == (fake_data - fake_mean) / fake_std)


def test_zscore_full_array_flags(fake_data):
    # Make some fake data
    np.random.seed(182)
    fake_data[...] = np.random.randn(fake_data.shape[0], fake_data.shape[1])
    flags = np.zeros(fake_data.shape, dtype=np.bool_)
    flags[45, 33] = True
    out = xrfi.zscore_full_array(fake_data, flags=flags)
    fake_mean = np.mean(np.ma.masked_array(fake_data, flags))
    fake_std = np.std(np.ma.masked_array(fake_data, flags))
    out_exp = (fake_data - fake_mean) / fake_std
    out_exp[45, 33] = np.inf
    assert np.all(out == out_exp)


def test_zscore_full_array_modified(fake_data):
    # Make some fake data
    np.random.seed(182)
    fake_data[...] = np.random.randn(fake_data.shape[0], fake_data.shape[1])
    out = xrfi.zscore_full_array(fake_data, modified=True)
    fake_med = np.median(fake_data)
    fake_mad = np.median(np.abs(fake_data - fake_med))
    assert np.all(out == (fake_data - fake_med) / (1.486 * fake_mad))


def test_zscore_full_array_modified_complex(fake_data):
    # Make some fake data
    np.random.seed(182)
    rands = np.random.randn(100, 100)
    fake_data = rands + 1j * rands
    out = xrfi.zscore_full_array(fake_data, modified=True)
    fake_med = np.median(rands)
    fake_mad = np.sqrt(2) * np.median(np.abs(rands - fake_med))
    assert np.allclose(out, (fake_data - fake_med - 1j * fake_med) / (1.486 * fake_mad))


def test_modzscore_1d_no_detrend():
    npix = 1000
    np.random.seed(182)
    data = np.random.randn(npix)
    data[50] = 500
    out = xrfi.modzscore_1d(data, detrend=False)
    assert out.shape == (npix,)
    assert np.isclose(out[50], 500, rtol=.2)
    assert np.isclose(np.median(np.abs(out)), .67, rtol=.1)


def test_modzscore_1d():
    npix = 1000
    np.random.seed(182)
    data = np.random.randn(npix)
    data[50] = 500
    data += .1 * np.arange(npix)
    out = xrfi.modzscore_1d(data)
    assert out.shape == (npix,)
    assert np.isclose(out[50], 500, rtol=.2)
    assert np.isclose(np.median(np.abs(out)), .67, rtol=.1)


def test_watershed_flag():
    # generate a metrics and flag UVFlag object
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvm = UVFlag(uv, history='I made this')
    uvf = UVFlag(uv, mode='flag')

    # set metric and flag arrays to specific values
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[0, 0, 1, 0] = 7.
    uvf.flag_array[0, 0, 0, 0] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[0, 0, :2, 0] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test flagging channels adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[:, :, 1, :] = 1.
    uvf.flag_array[:, :, 0, :] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test flagging times adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    times = np.unique(uv.time_array)
    inds1 = np.where(uv.time_array == times[0])[0]
    inds2 = np.where(uv.time_array == times[1])[0]
    uvm.metric_array[inds2, 0, :, 0] = 1.
    uvf.flag_array[inds1, 0, :, 0] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[inds1, 0, :, 0] = True
    flag_array[inds2, 0, :, 0] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test antenna type objects
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvm = UVFlag(uvc, history='I made this')
    uvf = UVFlag(uvc, mode='flag')

    # set metric and flag arrays to specific values
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[0, 0, 0, 1, 0] = 7.
    uvf.flag_array[0, 0, 0, 0, 0] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[0, 0, 0, :2, 0] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test flagging channels adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[:, :, 1, :, :] = 1.
    uvf.flag_array[:, :, 0, :, :] = True

    # run watershed flag
    uvf2 = xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=False)

    # check answer
    flag_array = np.zeros_like(uvf2.flag_array, dtype=np.bool_)
    flag_array[:, :, :2, :, :] = True
    assert np.allclose(uvf2.flag_array, flag_array)
    del(uvf2)

    # test flagging times adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[:, :, :, 1, :] = 1.
    uvf.flag_array[:, :, :, 0, :] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :, :, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test waterfall types
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvm = UVFlag(uv, history='I made this', waterfall=True)
    uvf = UVFlag(uv, mode='flag', waterfall=True)

    # set metric and flag arrays to specific values
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[0, 1, 0] = 7.
    uvf.flag_array[0, 0, 0] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[0, :2, 0] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test flagging channels adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[:, 1, :] = 1.
    uvf.flag_array[:, 0, :] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_f=0.5, inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test flagging times adjacent to fully flagged ones
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvm.metric_array[1, :, :] = 1.
    uvf.flag_array[0, :, :] = True

    # run watershed flag
    xrfi.watershed_flag(uvm, uvf, nsig_p=2., nsig_t=0.5, inplace=True)

    # check answer
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:2, :, :] = True
    assert np.allclose(uvf.flag_array, flag_array)


def test_watershed_flag_errors():
    # setup
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvm = UVFlag(uv, history='I made this')
    uvf = UVFlag(uv, mode='flag')
    uvf2 = UVFlag(uv, mode='flag', waterfall=True)

    # pass in objects besides UVFlag
    pytest.raises(ValueError, xrfi.watershed_flag, 1, 2)
    pytest.raises(ValueError, xrfi.watershed_flag, uvm, 2)
    pytest.raises(ValueError, xrfi.watershed_flag, uvm, uvf2)

    # set the UVFlag object to have a bogus type
    uvm.type = 'blah'
    pytest.raises(ValueError, xrfi.watershed_flag, uvm, uvf)


def test_ws_flag_waterfall():
    # test 1d
    d = np.zeros((10,))
    f = np.zeros((10,), dtype=np.bool_)
    d[1] = 3.
    f[0] = True
    f_out = xrfi._ws_flag_waterfall(d, f, nsig=2.)
    ans = np.zeros_like(f_out, dtype=np.bool_)
    ans[:2] = True
    assert np.allclose(f_out, ans)

    # another 1D test
    metric = np.array([2., 2., 5., 0., 2., 0., 5.])
    fin = (metric >= 5.)
    fout = xrfi._ws_flag_waterfall(metric, fin)
    np.testing.assert_array_equal(fout, [True, True, True, False, False, False, True])

    # test 2d
    d = np.zeros((10, 10))
    f = np.zeros((10, 10), dtype=np.bool_)
    d[0, 1] = 3.
    d[1, 0] = 3.
    f[0, 0] = True
    f_out = xrfi._ws_flag_waterfall(d, f, nsig=2.)
    ans = np.zeros_like(f_out, dtype=np.bool_)
    ans[:2, 0] = True
    ans[0, :2] = True
    assert np.allclose(f_out, ans)

    # catch errors
    d1 = np.zeros((10,))
    f2 = np.zeros((10, 10), dtype=np.bool_)
    pytest.raises(ValueError, xrfi._ws_flag_waterfall, d1, f2)
    d3 = np.zeros((5, 4, 3))
    f3 = np.zeros((5, 4, 3), dtype=np.bool_)
    pytest.raises(ValueError, xrfi._ws_flag_waterfall, d3, f3)


def test_xrfi_waterfall():
    # test basic functions
    np.random.seed(21)
    data = 100 * np.ones((10, 10))
    data += np.random.randn(10, 10)
    data[3, 3] += 100
    data[3, 4] += 3
    flags = xrfi.xrfi_waterfall(data)
    assert np.sum(flags) == 2
    assert flags[3, 3]
    assert flags[3, 4]
    flags = xrfi.xrfi_waterfall(data, nsig_adj=6.)
    assert np.sum(flags) == 1
    assert flags[3, 3]


def test_xrfi_waterfall_prior_flags():
    # test with prior flags
    np.random.seed(21)
    data = 100 * np.ones((10, 10))
    data += np.random.randn(10, 10)
    prior_flags = np.zeros((10, 10), dtype=bool)
    prior_flags[3, 3] = True
    data[3, 4] += 3
    flags = xrfi.xrfi_waterfall(data, flags=prior_flags)
    assert np.sum(flags) == 2
    assert flags[3, 3]
    assert flags[3, 4]
    flags = xrfi.xrfi_waterfall(data, flags=prior_flags, nsig_adj=6.)
    assert np.sum(flags) == 1
    assert flags[3, 3]


def test_xrfi_waterfall_error():
    # test errors
    data = np.ones((10, 10))
    with pytest.raises(KeyError):
        xrfi.xrfi_waterfall(data, algorithm='not_an_algorithm')


def test_flag():
    # setup
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvm = UVFlag(uv, history='I made this')

    # initialize array with specific values
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[0, 0, 0, 0] = 7.
    uvf = xrfi.flag(uvm, nsig_p=6.)
    assert uvf.mode == 'flag'
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[0, 0, 0, 0] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test channel flagging in baseline type
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[:, :, 0, :] = 7.
    uvm.metric_array[:, :, 1, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=6., nsig_f=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test time flagging in baseline type
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    times = np.unique(uvm.time_array)
    inds1 = np.where(uvm.time_array == times[0])[0]
    inds2 = np.where(uvm.time_array == times[1])[0]
    uvm.metric_array[inds1, :, :, :] = 7.
    uvm.metric_array[inds2, :, :, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[inds1, :, :, :] = True
    flag_array[inds2, :, :, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test channel flagging in antenna type
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvm = UVFlag(uv, history='I made this')
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[:, :, 0, :, :] = 7.
    uvm.metric_array[:, :, 1, :, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=7., nsig_f=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :, :2, :, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test time flagging in antenna type
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[:, :, :, 0, :] = 7.
    uvm.metric_array[:, :, :, 1, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :, :, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test channel flagging in waterfall type
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvm = UVFlag(uv, history='I made this', waterfall=True)
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[:, 0, :] = 7.
    uvm.metric_array[:, 1, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=6., nsig_f=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:, :2, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # test time flagging in waterfall type
    uvm.metric_array = np.zeros_like(uvm.metric_array)
    uvm.metric_array[0, :, :] = 7.
    uvm.metric_array[1, :, :] = 3.
    uvf = xrfi.flag(uvm, nsig_p=6., nsig_t=2.)
    flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    flag_array[:2, :, :] = True
    assert np.allclose(uvf.flag_array, flag_array)

    # catch errors
    pytest.raises(ValueError, xrfi.flag, 2)
    uvm.type = 'blah'
    pytest.raises(ValueError, xrfi.flag, uvm)


def test_unflag():
    # Do a test, add more tests as needed
    assert True


def test_flag_apply():
    # test applying to UVData
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool_)
    uvf = UVFlag(uv, mode='flag')
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvf.flag_array[:, :, 0, :] = True
    uvf2 = xrfi.flag_apply(uvf, uv, return_net_flags=True)
    assert np.allclose(uv.flag_array, uvf2.flag_array)

    # test applying to UVCal
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool_)
    uvf = UVFlag(uv, mode='flag')
    uvf.flag_array = np.zeros_like(uvf.flag_array, dtype=np.bool_)
    uvf.flag_array[:, :, 0, :, :] = True
    uvf2 = xrfi.flag_apply(uvf, uv, return_net_flags=True)
    assert np.allclose(uv.flag_array, uvf2.flag_array)

    # test applying to waterfalls
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool_)
    uvf = UVFlag(uv, mode='flag', waterfall=True)
    uvf.flag_array[:, 0, :] = True
    xrfi.flag_apply(uvf, uv)
    assert np.allclose(uv.flag_array[:, :, 0, :], True)
    assert np.allclose(uv.flag_array[:, :, 1:, :], False)

    uv = UVCal()
    uv.read_calfits(test_c_file)
    uv.flag_array = np.zeros_like(uv.flag_array, dtype=np.bool_)
    uvf = UVFlag(uv, mode='flag', waterfall=True)
    uvf.flag_array[:, 0, :] = True
    xrfi.flag_apply(uvf, uv)
    assert np.allclose(uv.flag_array[:, :, 0, :, :], True)
    assert np.allclose(uv.flag_array[:, :, 1:, :, :], False)

    # catch errors
    pytest.raises(ValueError, xrfi.flag_apply, uvf, 2)
    pytest.raises(ValueError, xrfi.flag_apply, 2, uv)
    uvf.mode = 'metric'
    pytest.raises(ValueError, xrfi.flag_apply, uvf, uv)


def test_simple_flag_waterfall():
    np.random.seed(21)
    data = np.random.randn(100, 100)
    data[3:, 10] += 200  # this tests spectral thresholding
    data[70, 20:] += 100  # this tests temporal thresholding
    data[50:55, 50] += 10  # this tests chanchan_thresh_frac
    flags = xrfi.simple_flag_waterfall(data, edge_cut=2, chan_thresh_frac=.01)

    assert not np.all(flags)
    np.testing.assert_array_equal(flags[:, :2], True)
    np.testing.assert_array_equal(flags[:, -2:], True)
    np.testing.assert_array_equal(flags[70, :], True)
    np.testing.assert_array_equal(flags[:, 10], True)
    np.testing.assert_array_equal(flags[:, 50], True)


def test_calculate_metric_vis():
    # setup
    uv = UVData()
    uv.read_miriad(test_d_file)
    # Use Kt=3 because test file only has three times
    uvf = xrfi.calculate_metric(uv, 'detrend_medfilt', Kt=3)
    assert uvf.mode == 'metric'
    assert uvf.type == 'baseline'
    inds = uv.antpair2ind(uv.ant_1_array[0], uv.ant_2_array[0])
    wf = uv.get_data(uv.ant_1_array[0], uv.ant_2_array[0])
    filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3)
    assert np.allclose(filtered, uvf.metric_array[inds, 0, :, 0])


def test_calculate_metric_gains():
    # Cal gains version
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', Kt=3, Kf=3)
    assert uvf.mode == 'metric'
    assert uvf.type == 'antenna'
    wf = uvc.gain_array[0, 0, :, :, 0]
    filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3, Kf=3)
    assert np.allclose(filtered, uvf.metric_array[0, 0, :, :, 0])


def test_calculate_metric_chisq():
    # Cal chisq version
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', cal_mode='chisq',
                                Kt=3, Kf=3)
    assert uvf.mode == 'metric'
    assert uvf.type == 'antenna'
    wf = uvc.quality_array[0, 0, :, :, 0]
    filtered = xrfi.detrend_medfilt(np.abs(wf), Kt=3, Kf=3)
    assert np.allclose(filtered, uvf.metric_array[0, 0, :, :, 0])


def test_calculate_metric_tot_chisq():
    # Cal total chisq version
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = xrfi.calculate_metric(uvc, 'detrend_medfilt', cal_mode='tot_chisq',
                                Kt=3, Kf=3)
    assert uvf.mode == 'metric'
    assert uvf.type == 'waterfall'
    filtered = xrfi.detrend_medfilt(np.abs(uvc.total_quality_array[0, :, :, 0]).T,
                                    Kt=3, Kf=3)
    assert np.allclose(filtered, uvf.metric_array[:, :, 0])


def test_calculate_metric_errors():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    pytest.raises(ValueError, xrfi.calculate_metric, 5, 'detrend_medfilt')
    pytest.raises(KeyError, xrfi.calculate_metric, uvc, 'my_awesome_algorithm')
    pytest.raises(ValueError, xrfi.calculate_metric, uvc, 'detrend_medfilt',
                  cal_mode='foo')


def test_xrfi_h1c_pipe_no_summary():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf_f, uvf_wf = xrfi.xrfi_h1c_pipe(uvc, Kt=3, return_summary=False)
    assert uvf_f.mode == 'flag'
    assert uvf_f.type == 'antenna'
    assert uvf_f.flag_array.shape == uvc.flag_array.shape
    assert uvf_wf.mode == 'flag'
    assert uvf_wf.type == 'waterfall'


def test_xrfi_h1c_pipe_summary():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf_f, uvf_wf, uvf_w = xrfi.xrfi_h1c_pipe(uvc, Kt=3, return_summary=True)
    assert uvf_f.mode == 'flag'
    assert uvf_f.type == 'antenna'
    assert uvf_f.flag_array.shape == uvc.flag_array.shape
    assert uvf_wf.mode == 'flag'
    assert uvf_wf.type == 'waterfall'
    assert uvf_w.mode == 'metric'
    assert uvf_w.type == 'waterfall'
    assert uvf_w.metric_array.max() <= 1.0


def test_xrfi_h1c_idr2_2_pipe():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf_m, uvf_f = xrfi.xrfi_pipe(uvc, Kt=3)
    assert uvf_m.mode == 'metric'
    assert uvf_f.mode == 'flag'
    assert uvf_m.type == 'waterfall'
    assert len(uvf_m.polarization_array) == 1
    assert uvf_m.weights_array.max() == 1.

def test_xrfi_run_step(tmpdir):
    # setup files in tmpdir.
    tmp_path = tmpdir.strpath
    fake_obs = 'zen.2457698.40355.HH'
    ocal_file = os.path.join(tmp_path, fake_obs + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obs + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obs + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    model_file = os.path.join(tmp_path, fake_obs + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    a_priori_flag_integrations = os.path.join(tmp_path, 'a_priori_flags_integrations.yaml')
    shutil.copyfile(test_flag_integrations, a_priori_flag_integrations)
    # if run_filter is false, then uv should not be None but everything else should be None
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile, run_filter=False, dtype='uvdata')
    assert issubclass(uv1.__class__, UVData)
    assert uvf1 is None
    assert uvf_f1 is None
    assert uvf_a1 is None
    assert len(metrics1) == 0
    assert len(flags1) == 0


    # test expected output formats if run_filter is True.
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile, run_filter=True, dtype='uvdata')
    assert len(flags1) == 1
    assert len(metrics1) == 1
    assert uvf_a1 is None

    # test expected output formats when calculate_uvf_apriori is True
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', wf_method='mean')
    assert len(flags1) == 2
    assert len(metrics1) == 1
    assert uvf_a1 is not None

    # now test partial i/o
    uv2, uvf2, uvf_f2, uvf_a2, metrics2, flags2 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', Nwf_per_load=1, wf_method='mean')
    assert len(flags2) == 2
    assert len(metrics2) == 1
    assert np.all(np.isclose(uvf_f1.flag_array, uvf_f2.flag_array))
    assert np.all(np.isclose(uvf_a1.flag_array, uvf_a2.flag_array))
    assert np.all(np.isclose(uvf1.metric_array, uvf2.metric_array))


    # comparison for autos only.
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', wf_method='quadmean', alg='detrend_medfilt', correlations='auto')
    assert len(flags1) == 2
    assert len(metrics1) == 1
    assert uvf_a1 is not None

    uv2, uvf2, uvf_f2, uvf_a2, metrics2, flags2 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', Nwf_per_load=1, wf_method='quadmean', alg='detrend_medfilt', correlations='auto')
    assert len(flags2) == 2
    assert len(metrics2) == 1
    assert np.all(np.isclose(uvf_f1.flag_array, uvf_f2.flag_array))
    assert np.all(np.isclose(uvf_a1.flag_array, uvf_a2.flag_array))
    assert np.all(np.isclose(uvf1.metric_array, uvf2.metric_array))

    #now try autos, then crosses
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile,
                                                                     calculate_uvf_apriori=True,
                                                                     run_filter=True, dtype='uvdata',
                                                                     wf_method='quadmean',
                                                                     alg='detrend_medfilt',
                                                                     correlations='cross')

    uv1x, uvf1x, uvf_f1x, uvf_a1x, metrics1x, flags1x = xrfi.xrfi_run_step(uv_files=raw_dfile,
                                                                         uv = uv1,
                                                                         apply_uvf_apriori=True,
                                                                         uvf_apriori=uvf_a1,
                                                                         run_filter=True, dtype='uvdata',
                                                                         wf_method='quadmean',
                                                                         alg='detrend_medfilt',
                                                                         correlations='auto')

    uv2, uvf2, uvf_f2, uvf_a2, metrics2, flags2 = xrfi.xrfi_run_step(uv_files=raw_dfile,
                                                                    calculate_uvf_apriori=True,
                                                                    run_filter=True, dtype='uvdata',
                                                                    Nwf_per_load=1,
                                                                    wf_method='quadmean',
                                                                    alg='detrend_medfilt',
                                                                    correlations='cross')

    uv2, uvf2x, uvf_f2x, uvf_a2x, metrics2x, flags2x = xrfi.xrfi_run_step(uv_files=raw_dfile,
                                                                    uv=uv2,
                                                                    apply_uvf_apriori=True,
                                                                    uvf_apriori=uvf_a2,
                                                                    run_filter=True, dtype='uvdata',
                                                                    Nwf_per_load=1,
                                                                    wf_method='quadmean',
                                                                    alg='detrend_medfilt',
                                                                    correlations='auto')
    assert np.all(np.isclose(uvf_f1.flag_array, uvf_f2.flag_array))
    assert np.all(np.isclose(uvf_a1.flag_array, uvf_a2.flag_array))
    assert np.all(np.isclose(uvf1.metric_array, uvf2.metric_array))
    assert np.all(np.isclose(uvf1x.metric_array, uvf2x.metric_array))
    assert np.all(np.isclose(uvf_f1x.flag_array, uvf_f2x.flag_array))
    assert np.all(np.isclose(uvf_a1x.flag_array, uvf_a2x.flag_array))

    # test use_cross_pol_vis (since the file is all a single pol, the results should be the same)
    uv3, uvf3, uvf_f3, uvf_a3, metrics3, flags3 = xrfi.xrfi_run_step(uv_files=raw_dfile,
                                                                     calculate_uvf_apriori=True,
                                                                     run_filter=True, dtype='uvdata',
                                                                     use_cross_pol_vis=False,
                                                                     wf_method='quadmean',
                                                                     alg='detrend_medfilt',
                                                                     correlations='cross')
    assert np.all(np.isclose(uvf_f1.flag_array, uvf_f3.flag_array))
    assert np.all(np.isclose(uvf_a1.flag_array, uvf_a3.flag_array))
    assert np.all(np.isclose(uvf1.metric_array, uvf3.metric_array))

    # comparison for meanfilter.
    uv1, uvf1, uvf_f1, uvf_a1, metrics1, flags1 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', wf_method='mean', alg='detrend_meanfilt')
    assert len(flags1) == 2
    assert len(metrics1) == 1
    assert uvf_a1 is not None

    uv2, uvf2, uvf_f2, uvf_a2, metrics2, flags2 = xrfi.xrfi_run_step(uv_files=raw_dfile,
    calculate_uvf_apriori=True, run_filter=True, dtype='uvdata', Nwf_per_load=1, wf_method='mean', alg='detrend_meanfilt')
    assert len(flags2) == 2
    assert len(metrics2) == 1
    assert np.all(np.isclose(uvf_f1.flag_array, uvf_f2.flag_array))
    assert np.all(np.isclose(uvf_a1.flag_array, uvf_a2.flag_array))
    assert np.all(np.isclose(uvf1.metric_array, uvf2.metric_array))
    # hit one line involving uvcal reinitialization and include yaml file.
    xrfi.xrfi_run_step(uv_files=ocal_file, calculate_uvf_apriori=True, run_filter=True, reinitialize=True, a_priori_flag_yaml=a_priori_flag_integrations)
    # test invalid data type error
    with pytest.raises(ValueError):
        xrfi.xrfi_run_step(uv_files=ocal_file, calculate_uvf_apriori=True, run_filter=True, reinitialize=True, dtype='uvwhatever')

def test_xrfi_run_yaml_flags(tmpdir):
    # test xrfi_run with yaml pre-flagging.
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    tmp_path = tmpdir.strpath
    fake_obs = 'zen.2457698.40355.HH'
    ocal_file = os.path.join(tmp_path, fake_obs + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obs + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obs + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    model_file = os.path.join(tmp_path, fake_obs + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    a_priori_flag_integrations = os.path.join(tmp_path, 'a_priori_flags_integrations.yaml')
    shutil.copyfile(test_flag_integrations, a_priori_flag_integrations)
    a_priori_flag_jds = os.path.join(tmp_path, 'a_priori_flags_jds.yaml')
    shutil.copyfile(test_flag_jds, a_priori_flag_jds)
    a_priori_flag_lsts = os.path.join(tmp_path, 'a_priori_flags_lsts.yaml')
    shutil.copyfile(test_flag_lsts, a_priori_flag_lsts)
    outdir = os.path.join(tmp_path, 'zen.2457698.40355.xrfi')
    ext_labels = {'ag_flags1': 'Abscal gains, median filter. Flags.',
                  'ag_flags2': 'Abscal gains, mean filter. Flags.',
                  'ag_metrics1': 'Abscal gains, median filter.',
                  'ag_metrics2': 'Abscal gains, mean filter.',
                  'apriori_flags': 'A priori flags.',
                  'ax_flags1': 'Abscal chisq, median filter. Flags.',
                  'ax_flags2': 'Abscal chisq, mean filter. Flags.',
                  'ax_metrics1': 'Abscal chisq, median filter.',
                  'ax_metrics2': 'Abscal chisq, mean filter.',
                  'omnical_chi_sq_flags1': 'Omnical overall modified z-score of chisq. Flags.',
                  'omnical_chi_sq_flags2': 'Omnical overall z-score of chisq. Flags.',
                  'omnical_chi_sq_renormed_metrics1': 'Omnical overall modified z-score of chisq.',
                  'omnical_chi_sq_renormed_metrics2': 'Omnical overall z-score of chisq.',
                  'abscal_chi_sq_flags1': 'Abscal overall modified z-score of chisq. Flags.',
                  'abscal_chi_sq_flags2': 'Abscal overall z-score of chisq. Flags.',
                  'abscal_chi_sq_renormed_metrics1': 'Abscal overall modified z-score of chisq.',
                  'abscal_chi_sq_renormed_metrics2': 'Abscal overall z-score of chisq.',
                  'combined_flags1': 'Flags from combined metrics, round 1.',
                  'combined_flags2': 'Flags from combined metrics, round 2.',
                  'combined_metrics1': 'Combined metrics, round 1.',
                  'combined_metrics2': 'Combined metrics, round 2.',
                  'cross_flags1': 'Crosscorr, median filter. Flags.',
                  'cross_flags2': 'Crosscorr, mean filter. Flags.',
                  'auto_flags1': 'Autocorr, median filter. Flags.',
                  'auto_flags2': 'Autocorr, mean filter. Flags.',
                  'auto_metrics2': 'Autocorr, mean filter.',
                  'auto_metrics1': 'Autocorr, median filter.',
                  'cross_metrics2': 'Crosscorr, mean filter.',
                  'cross_metrics1': 'Crosscorr, median filter.',
                  'flags1': 'ORd flags, round 1.',
                  'flags2': 'ORd flags, round 2.',
                  'og_flags1': 'Omnical gains, median filter. Flags.',
                  'og_flags2': 'Omnical gains, mean filter. Flags.',
                  'og_metrics1': 'Omnical gains, median filter.',
                  'og_metrics2': 'Omnical gains, mean filter.',
                  'ox_flags1': 'Omnical chisq, median filter. Flags.',
                  'ox_flags2': 'Omnical chisq, mean filter. Flags.',
                  'ox_metrics1': 'Omnical chisq, median filter.',
                  'ox_metrics2': 'Omnical chisq, mean filter.',
                  'v_flags1': 'Omnical visibility solutions, median filter. Flags.',
                  'v_flags2': 'Omnical visibility solutions, mean filter. Flags.',
                  'v_metrics1': 'Omnical visibility solutions, median filter.',
                  'v_metrics2': 'Omnical visibility solutions, mean filter.'}
    freq_regions = [(0, 110e6), (150e6, 155e6), (190e6, 200e6)] # frequencies from yaml file.
    channel_flags = [0, 1, 60] + list(range(10, 21)) # channels from yaml file.
    integration_flags = [0, 1] # integrations from yaml file that should be flagged.

    # now test apriori flag file.
    # test for different integrations modes (lsts, jds, integrations)
    for test_flag in [a_priori_flag_integrations, a_priori_flag_jds, a_priori_flag_lsts]:
        with pytest.warns(None) as record:
            xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile,
                          a_priori_flag_yaml=test_flag, history='Just a test', kt_size=3, throw_away_edges=False)
        assert len(record) >= len(messages)
        n_matched_warnings = 0
        for i in range(len(record)):
            if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
                n_matched_warnings += 1
        assert n_matched_warnings == 8
        for ext, label in ext_labels.items():
            # by default, only cross median filter / mean filter is not performed.
            if not ext in['cross_metrics1', 'cross_flags1']:
                out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
                assert os.path.exists(out)
                uvf = UVFlag(out)
                assert uvf.label == label
                # check that all flags in frequency regions are set to True.
                # only check init flags, apriori flags, and round 2 flags.
                # other round 1 flags are derived from metrics and won't necessarily
                # have data /cal flags in them.
                if 'flag' in ext and ('apriori' in ext or '2' in ext or 'flags1' == ext):
                    for region in freq_regions:
                        selection = (uvf.freq_array[0] >= region[0]) & (uvf.freq_array[0] <= region[-1])
                        assert np.all(uvf.flag_array[:, selection, :])
                    for chan in channel_flags:
                        assert np.all(uvf.flag_array[:, chan, :])
                    for integration in integration_flags:
                        assert np.all(uvf.flag_array[integration, :, :])
        # cleanup
        for ext, label in ext_labels.items():
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            if os.path.exists(out):
                os.remove(out)

def test_xrfi_run(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    tmp_path = tmpdir.strpath
    fake_obs = 'zen.2457698.40355.HH'
    ocal_file = os.path.join(tmp_path, fake_obs + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obs + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obs + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    model_file = os.path.join(tmp_path, fake_obs + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile, history='Just a test', kt_size=3)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    outdir = os.path.join(tmp_path, 'zen.2457698.40355.xrfi')
    ext_labels = {'ag_flags1': 'Abscal gains, median filter. Flags.',
                  'ag_flags2': 'Abscal gains, mean filter. Flags.',
                  'ag_metrics1': 'Abscal gains, median filter.',
                  'ag_metrics2': 'Abscal gains, mean filter.',
                  'apriori_flags': 'A priori flags.',
                  'ax_flags1': 'Abscal chisq, median filter. Flags.',
                  'ax_flags2': 'Abscal chisq, mean filter. Flags.',
                  'ax_metrics1': 'Abscal chisq, median filter.',
                  'ax_metrics2': 'Abscal chisq, mean filter.',
                  'omnical_chi_sq_flags1': 'Omnical overall modified z-score of chisq. Flags.',
                  'omnical_chi_sq_flags2': 'Omnical overall z-score of chisq. Flags.',
                  'omnical_chi_sq_renormed_metrics1': 'Omnical overall modified z-score of chisq.',
                  'omnical_chi_sq_renormed_metrics2': 'Omnical overall z-score of chisq.',
                  'abscal_chi_sq_flags1': 'Abscal overall modified z-score of chisq. Flags.',
                  'abscal_chi_sq_flags2': 'Abscal overall z-score of chisq. Flags.',
                  'abscal_chi_sq_renormed_metrics1': 'Abscal overall modified z-score of chisq.',
                  'abscal_chi_sq_renormed_metrics2': 'Abscal overall z-score of chisq.',
                  'combined_flags1': 'Flags from combined metrics, round 1.',
                  'combined_flags2': 'Flags from combined metrics, round 2.',
                  'combined_metrics1': 'Combined metrics, round 1.',
                  'combined_metrics2': 'Combined metrics, round 2.',
                  'cross_flags1': 'Crosscorr, median filter. Flags.',
                  'cross_flags2': 'Crosscorr, mean filter. Flags.',
                  'auto_flags1': 'Autocorr, median filter. Flags.',
                  'auto_flags2': 'Autocorr, mean filter. Flags.',
                  'auto_metrics2': 'Autocorr, mean filter.',
                  'auto_metrics1': 'Autocorr, median filter.',
                  'cross_metrics2': 'Crosscorr, mean filter.',
                  'cross_metrics1': 'Crosscorr, median filter.',
                  'flags1': 'ORd flags, round 1.',
                  'flags2': 'ORd flags, round 2.',
                  'og_flags1': 'Omnical gains, median filter. Flags.',
                  'og_flags2': 'Omnical gains, mean filter. Flags.',
                  'og_metrics1': 'Omnical gains, median filter.',
                  'og_metrics2': 'Omnical gains, mean filter.',
                  'ox_flags1': 'Omnical chisq, median filter. Flags.',
                  'ox_flags2': 'Omnical chisq, mean filter. Flags.',
                  'ox_metrics1': 'Omnical chisq, median filter.',
                  'ox_metrics2': 'Omnical chisq, mean filter.',
                  'v_flags1': 'Omnical visibility solutions, median filter. Flags.',
                  'v_flags2': 'Omnical visibility solutions, mean filter. Flags.',
                  'v_metrics1': 'Omnical visibility solutions, median filter.',
                  'v_metrics2': 'Omnical visibility solutions, mean filter.'}
    for ext, label in ext_labels.items():
        # by default, only cross median filter / mean filter is not performed.
        if not ext in['cross_metrics1', 'cross_flags1']:
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)

    # now really do everything.
    uvf_list1 = []
    uvf_list1_names = []
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile,
                      history='Just a test', kt_size=3, cross_median_filter=True)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        assert os.path.exists(out)
        uvf = UVFlag(out)
        uvf_list1.append(uvf)
        uvf_list1_names.append(out)
        assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # now do partial i/o and check equality of outputs.
    uvf_list2 = []
    uvf_list2_names = []
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile, Nwf_per_load=1,
                      history='Just a test', kt_size=3, cross_median_filter=True)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        assert os.path.exists(out)
        uvf = UVFlag(out)
        uvf_list2.append(uvf)
        uvf_list2_names.append(out)
        assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # compare
    for uvf1, uvf2 in zip(uvf_list1, uvf_list2):
        if uvf1.mode == 'flag':
            assert np.all(np.isclose(uvf1.flag_array, uvf2.flag_array))
        elif uvf1.mode == 'metric':
            assert np.all(np.isclose(uvf1.metric_array, uvf2.metric_array))
    # test cross correlations.
    xrfi.xrfi_run(history='data cross corrs.', data_files=raw_dfile,
                  cross_median_filter=True, cross_mean_filter=True, auto_mean_filter=False, auto_median_filter=False)
    for ext, label in ext_labels.items():
      out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
      if 'cross' in ext or 'combined' in ext:
          assert os.path.exists(out)
          uvf = UVFlag(out)
          assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
          os.remove(out)
    # test auto correlations.
    xrfi.xrfi_run(history='data autocorrs.', data_files=raw_dfile,
                  cross_mean_filter=False, cross_median_filter=False)
    for ext, label in ext_labels.items():
      out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
      if 'auto' in ext or 'combined' in ext:
          assert os.path.exists(out)
          uvf = UVFlag(out)
          assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
      out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
      if os.path.exists(out):
          os.remove(out)
    # test xrfi_run errors when providing no data inputs
    with pytest.raises(ValueError):
         xrfi.xrfi_run(None, None, None, None, None)
    # test error when no data file or output prefix provided.
    with pytest.raises(ValueError):
        xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file, history='fail', output_prefixes=None)
    # test run with only ocal and acal files.
    xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file, history='calibration only flags.',
                  output_prefixes=raw_dfile)
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if 'cross' not in ext and 'v_' not in ext and 'auto' not in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # abscal ONLY
    xrfi.xrfi_run(acalfits_files=acal_file, history='abscal only flags.',
                  output_prefixes=raw_dfile)
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if 'cross' not in ext and 'v_' not in ext and 'auto' not in ext\
         and 'ox_' not in ext and 'og_' not in ext and not 'omnical' in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # test run with only data files. Median/mean filter on autos and crosses.
    xrfi.xrfi_run(data_files=raw_dfile, history='data only flags.', cross_median_filter=True)
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if 'cross' in ext or 'combined' in ext or 'auto' in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # Make sure switches work, provide all files but set all filters to false except data.
    xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file, data_files=raw_dfile,
                  model_files=model_file,
                  history='data only flags.', cross_median_filter=True,
                  abscal_mean_filter=False, abscal_median_filter=False,
                  abscal_chi2_median_filter=False, abscal_chi2_mean_filter=False,
                  abscal_zscore_filter=False,
                  omnical_mean_filter=False, omnical_median_filter=False,
                  omnical_chi2_median_filter=False, omnical_chi2_mean_filter=False,
                  omnical_zscore_filter=False,
                  omnivis_mean_filter=False, omnivis_median_filter=False)
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if 'cross' in ext or 'combined' in ext or 'auto' in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # Don't do any median filters.
    xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file, data_files=raw_dfile,
                  model_files=model_file,
                  history='data only flags.',
                  abscal_mean_filter=False, abscal_median_filter=False,
                  abscal_chi2_median_filter=False, abscal_chi2_mean_filter=False,
                  abscal_zscore_filter=False,
                  omnical_mean_filter=False, omnical_median_filter=False,
                  omnical_chi2_median_filter=False, omnical_chi2_mean_filter=False,
                  omnical_zscore_filter=False,
                  omnivis_mean_filter=False, omnivis_median_filter=False,
                  cross_median_filter=False, auto_median_filter=False)
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if ('cross' in ext or 'combined' in ext or 'auto' in ext) and '1' not in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # test run with only omnivis files
    xrfi.xrfi_run(model_files=model_file, output_prefixes=raw_dfile, history='omnivis only flags.')
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if 'v_' in ext or 'combined' in ext:
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # test run with auto data and omnivis files
    xrfi.xrfi_run(data_files=raw_dfile, model_files=model_file, history='omnivis and cross flags.', cross_mean_filter=False)
    for ext, label in ext_labels.items():
        if 'v_' in ext or 'combined' in ext or 'auto' in ext:
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # test data with data and omnical
    xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file,
                  data_files=raw_dfile, history='data and omni/abs cal.', cross_median_filter=True)
    for ext, label in ext_labels.items():
        if not 'v_' in ext:
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # test omnivis with omnical
    xrfi.xrfi_run(acalfits_files=acal_file, ocalfits_files=ocal_file,
                  model_files=model_file, history='model and omni/abs cal.', output_prefixes=raw_dfile)
    for ext, label in ext_labels.items():
        if 'cross' not in ext and 'auto' not in ext:
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # remove spoofed files
    shutil.rmtree(outdir)  # cleanup
    for fname in [ocal_file, acal_file, model_file, raw_dfile]:
        os.remove(fname)

def test_xrfi_run_edgeflag(tmpdir):
    # test that flags within kt_size are flagged.
    # first try out a single file.
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    tmp_path = tmpdir.strpath
    fake_obs = 'zen.2457698.40355.HH'
    ocal_file = os.path.join(tmp_path, fake_obs + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obs + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obs + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    model_file = os.path.join(tmp_path, fake_obs + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile, history='Just a test', kt_size=2)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8
    outdir = os.path.join(tmp_path, 'zen.2457698.40355.xrfi')
    ext_labels = {'ag_flags1': 'Abscal gains, median filter. Flags.',
                  'ag_flags2': 'Abscal gains, mean filter. Flags.',
                  'ag_metrics1': 'Abscal gains, median filter.',
                  'ag_metrics2': 'Abscal gains, mean filter.',
                  'apriori_flags': 'A priori flags.',
                  'ax_flags1': 'Abscal chisq, median filter. Flags.',
                  'ax_flags2': 'Abscal chisq, mean filter. Flags.',
                  'ax_metrics1': 'Abscal chisq, median filter.',
                  'ax_metrics2': 'Abscal chisq, mean filter.',
                  'omnical_chi_sq_flags1': 'Omnical overall modified z-score of chisq. Flags.',
                  'omnical_chi_sq_flags2': 'Omnical overall z-score of chisq. Flags.',
                  'omnical_chi_sq_renormed_metrics1': 'Omnical overall modified z-score of chisq.',
                  'omnical_chi_sq_renormed_metrics2': 'Omnical overall z-score of chisq.',
                  'abscal_chi_sq_flags1': 'Abscal overall modified z-score of chisq. Flags.',
                  'abscal_chi_sq_flags2': 'Abscal overall z-score of chisq. Flags.',
                  'abscal_chi_sq_renormed_metrics1': 'Abscal overall modified z-score of chisq.',
                  'abscal_chi_sq_renormed_metrics2': 'Abscal overall z-score of chisq.',
                  'combined_flags1': 'Flags from combined metrics, round 1.',
                  'combined_flags2': 'Flags from combined metrics, round 2.',
                  'combined_metrics1': 'Combined metrics, round 1.',
                  'combined_metrics2': 'Combined metrics, round 2.',
                  'cross_flags1': 'Crosscorr, median filter. Flags.',
                  'cross_flags2': 'Crosscorr, mean filter. Flags.',
                  'auto_flags1': 'Autocorr, median filter. Flags.',
                  'auto_flags2': 'Autocorr, mean filter. Flags.',
                  'auto_metrics2': 'Autocorr, mean filter.',
                  'auto_metrics1': 'Autocorr, median filter.',
                  'cross_metrics2': 'Crosscorr, mean filter.',
                  'cross_metrics1': 'Crosscorr, median filter.',
                  'flags1': 'ORd flags, round 1.',
                  'flags2': 'ORd flags, round 2.',
                  'og_flags1': 'Omnical gains, median filter. Flags.',
                  'og_flags2': 'Omnical gains, mean filter. Flags.',
                  'og_metrics1': 'Omnical gains, median filter.',
                  'og_metrics2': 'Omnical gains, mean filter.',
                  'ox_flags1': 'Omnical chisq, median filter. Flags.',
                  'ox_flags2': 'Omnical chisq, mean filter. Flags.',
                  'ox_metrics1': 'Omnical chisq, median filter.',
                  'ox_metrics2': 'Omnical chisq, mean filter.',
                  'v_flags1': 'Omnical visibility solutions, median filter. Flags.',
                  'v_flags2': 'Omnical visibility solutions, mean filter. Flags.',
                  'v_metrics1': 'Omnical visibility solutions, median filter.',
                  'v_metrics2': 'Omnical visibility solutions, mean filter.'}
    for ext, label in ext_labels.items():
        # by default, only cross median filter / mean filter is not performed.
        if not ext in['cross_metrics1', 'cross_flags1']:
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            if ext == 'flags2.h5':
                # check that two within edges are flagged
                assert np.all(uvf.flag_array[:2])
                assert np.all(uvf.flag_array[-2:])
                # check that there are is some unflagged data in between
                assert not np.all(uvf.flag_array[2:-2])

    # cleanup
    for ext, label in ext_labels.items():
        out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
        if os.path.exists(out):
            os.remove(out)
    # now try two different files.
    fake_obses = ['zen.2457698.40355191.HH', 'zen.2457698.40367619.HH', 'zen.2457698.40380046.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355191.HH', 'zen.2457698.40367619.HH', 'zen.2457698.40380046.HH']
    ocal_files = []
    acal_files = []
    model_files =[]
    raw_dfiles = []
    for uvf, cf, fo in zip(test_uvh5_files, test_c_files, fake_obses):
        ocal_file = os.path.join(tmp_path, fo + '.omni.calfits')
        shutil.copyfile(cf, ocal_file)
        ocal_files.append(ocal_file)
        acal_file = os.path.join(tmp_path, fo + '.abs.calfits')
        shutil.copyfile(cf, acal_file)
        acal_files.append(acal_file)
        raw_dfile = os.path.join(tmp_path, fo + '.uvh5')
        shutil.copyfile(uvf, raw_dfile)
        raw_dfiles.append(raw_dfile)
        model_file = os.path.join(tmp_path, fo + '.omni_vis.uvh5')
        shutil.copyfile(uvf, model_file)
        model_files.append(model_file)
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_files, acal_files, model_files, raw_dfiles, history='Just a test', kt_size=1)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8
    flags2 = sorted(glob.glob(tmp_path + '/*.xrfi/*.HH.flags2.h5'))
    assert len(flags2) == 3
    uvf = UVFlag(flags2)
    # check that two within edges are flagged
    assert np.all(uvf.flag_array[:1])
    assert np.all(uvf.flag_array[-1:])
    # check that there are is some unflagged data in between
    assert not np.all(uvf.flag_array[1:-1])



def test_xrfi_run_multifile(tmpdir):
    # test xrfi_run with multiple files
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355191.HH', 'zen.2457698.40367619.HH', 'zen.2457698.40380046.HH']
    ocal_files = []
    acal_files = []
    model_files =[]
    raw_dfiles = []
    for uvf, cf, fo in zip(test_uvh5_files, test_c_files, fake_obses):
        ocal_file = os.path.join(tmp_path, fo + '.omni.calfits')
        shutil.copyfile(cf, ocal_file)
        ocal_files.append(ocal_file)
        acal_file = os.path.join(tmp_path, fo + '.abs.calfits')
        shutil.copyfile(cf, acal_file)
        acal_files.append(acal_file)
        raw_dfile = os.path.join(tmp_path, fo + '.uvh5')
        shutil.copyfile(uvf, raw_dfile)
        raw_dfiles.append(raw_dfile)
        model_file = os.path.join(tmp_path, fo + '.omni_vis.uvh5')
        shutil.copyfile(uvf, model_file)
        model_files.append(model_file)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_files, acal_files, model_files, raw_dfiles,
                      history='Just a test', kt_size=3, cross_median_filter=True)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8
    ext_labels = {'ag_flags1': 'Abscal gains, median filter. Flags.',
                  'ag_flags2': 'Abscal gains, mean filter. Flags.',
                  'ag_metrics1': 'Abscal gains, median filter.',
                  'ag_metrics2': 'Abscal gains, mean filter.',
                  'apriori_flags': 'A priori flags.',
                  'ax_flags1': 'Abscal chisq, median filter. Flags.',
                  'ax_flags2': 'Abscal chisq, mean filter. Flags.',
                  'ax_metrics1': 'Abscal chisq, median filter.',
                  'ax_metrics2': 'Abscal chisq, mean filter.',
                  'omnical_chi_sq_flags1': 'Omnical overall modified z-score of chisq. Flags.',
                  'omnical_chi_sq_flags2': 'Omnical overall z-score of chisq. Flags.',
                  'omnical_chi_sq_renormed_metrics1': 'Omnical overall modified z-score of chisq.',
                  'omnical_chi_sq_renormed_metrics2': 'Omnical overall z-score of chisq.',
                  'abscal_chi_sq_flags1': 'Abscal overall modified z-score of chisq. Flags.',
                  'abscal_chi_sq_flags2': 'Abscal overall z-score of chisq. Flags.',
                  'abscal_chi_sq_renormed_metrics1': 'Abscal overall modified z-score of chisq.',
                  'abscal_chi_sq_renormed_metrics2': 'Abscal overall z-score of chisq.',
                  'combined_flags1': 'Flags from combined metrics, round 1.',
                  'combined_flags2': 'Flags from combined metrics, round 2.',
                  'combined_metrics1': 'Combined metrics, round 1.',
                  'combined_metrics2': 'Combined metrics, round 2.',
                  'cross_flags1': 'Crosscorr, median filter. Flags.',
                  'cross_flags2': 'Crosscorr, mean filter. Flags.',
                  'auto_flags1': 'Autocorr, median filter. Flags.',
                  'auto_flags2': 'Autocorr, mean filter. Flags.',
                  'auto_metrics2': 'Autocorr, mean filter.',
                  'auto_metrics1': 'Autocorr, median filter.',
                  'cross_metrics2': 'Crosscorr, mean filter.',
                  'cross_metrics1': 'Crosscorr, median filter.',
                  'flags1': 'ORd flags, round 1.',
                  'flags2': 'ORd flags, round 2.',
                  'og_flags1': 'Omnical gains, median filter. Flags.',
                  'og_flags2': 'Omnical gains, mean filter. Flags.',
                  'og_metrics1': 'Omnical gains, median filter.',
                  'og_metrics2': 'Omnical gains, mean filter.',
                  'ox_flags1': 'Omnical chisq, median filter. Flags.',
                  'ox_flags2': 'Omnical chisq, mean filter. Flags.',
                  'ox_metrics1': 'Omnical chisq, median filter.',
                  'ox_metrics2': 'Omnical chisq, mean filter.',
                  'v_flags1': 'Omnical visibility solutions, median filter. Flags.',
                  'v_flags2': 'Omnical visibility solutions, mean filter. Flags.',
                  'v_metrics1': 'Omnical visibility solutions, median filter.',
                  'v_metrics2': 'Omnical visibility solutions, mean filter.'}
    # check that the number of outdirs is 1
    outdirs = sorted(glob.glob(tmp_path + '/*.xrfi'))
    assert len(outdirs) == 3
    # check all the metrics and flags.
    for outdir, fake_obs in zip(outdirs, fake_obses):
        for ext, label in ext_labels.items():
            # by default, only cross median filter / mean filter is not performed.
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
            # all of flags2 should be flagged with this kt_size.
            if ext == 'flags2.h5':
                assert np.all(uvf.flag_array)
    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_files, acal_files, model_files, raw_dfiles,
                      history='Just a test', kt_size=3, cross_median_filter=True,
                      throw_away_edges=False, clobber=True)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    # check that the number of outdirs is 1
    outdirs = sorted(glob.glob(tmp_path + '/*.xrfi'))
    assert len(outdirs) == 3
    # check all the metrics and flags.
    for outdir, fake_obs in zip(outdirs, fake_obses):
        outdir = os.path.join(tmp_path, outdir)
        for ext, label in ext_labels.items():
            # by default, only cross median filter / mean filter is not performed.
            out = os.path.join(outdir, '.'.join([fake_obs, ext, 'h5']))
            assert os.path.exists(out)
            uvf = UVFlag(out)
            assert uvf.label == label
            # if we don't throw away the edges, then there shouldn't be flags
            # at the edge.
            if ext == 'flags2.h5':
                assert not np.all(uvf.flag_array)

def test_day_threshold_run(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof the files - run xrfi_run twice on spoofed files.
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355.HH', 'zen.2457698.41101.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    ocal_file = os.path.join(tmp_path, fake_obses[0] + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[0] + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obses[0] + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    data_files = [raw_dfile]
    model_file = os.path.join(tmp_path, fake_obses[0] + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile, history='Just a test', kt_size=3, throw_away_edges=False)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    # Need to adjust time arrays when duplicating files
    uvd = UVData()
    uvd.read_uvh5(data_files[0])
    dt = (uvd.time_array.max() - uvd.time_array.min()) + uvd.integration_time.mean() / (24. * 3600.)
    uvd.time_array += dt
    uvd.set_lsts_from_time_array()
    data_files += [os.path.join(tmp_path, fake_obses[1] + '.uvh5')]
    uvd.write_uvh5(data_files[1])
    model_file = os.path.join(tmp_path, fake_obses[1] + '.omni_vis.uvh5')
    uvd.write_uvh5(model_file)
    uvc = UVCal()
    uvc.read_calfits(ocal_file)
    dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
    uvc.time_array += dt
    ocal_file = os.path.join(tmp_path, fake_obses[1] + '.omni.calfits')
    uvc.write_calfits(ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[1] + '.abs.calfits')
    uvc.write_calfits(acal_file)
    a_priori_flag_integrations = os.path.join(tmp_path, 'a_priori_flags_integrations.yaml')
    shutil.copyfile(test_flag_integrations, a_priori_flag_integrations)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, data_files[1], history='Just a test', kt_size=3, clobber=True,
                      throw_away_edges=False, a_priori_flag_yaml=a_priori_flag_integrations, a_priori_ants_only=True)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    xrfi.day_threshold_run(data_files, history='just a test', a_priori_flag_yaml=a_priori_flag_integrations)
    types = ['og', 'ox', 'ag', 'ax', 'v', 'cross', 'auto', 'omnical_chi_sq_renormed',
             'abscal_chi_sq_renormed', 'combined', 'total']
    for type in types:
        basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.' + type + '_threshold_flags.h5'
        outfile = os.path.join(tmp_path, basename)
        assert os.path.exists(outfile)

    basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.total_threshold_and_a_priori_flags.h5'
    outfile = os.path.join(tmp_path, basename)
    assert os.path.exists(outfile)
        
    for fake_obs in fake_obses:
        calfile = os.path.join(tmp_path, fake_obs + '.flagged_abs.calfits')
        assert os.path.exists(calfile)


def test_day_threshold_run_yaml(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 8 * mess1
    cat1 = [UserWarning]
    categories = 8 * cat1
    # Spoof the files - run xrfi_run twice on spoofed files.
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355.HH', 'zen.2457698.41101.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    ocal_file = os.path.join(tmp_path, fake_obses[0] + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[0] + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obses[0] + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    data_files = [raw_dfile]
    model_file = os.path.join(tmp_path, fake_obses[0] + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, raw_dfile, history='Just a test', kt_size=3, throw_away_edges=False)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    # Need to adjust time arrays when duplicating files
    uvd = UVData()
    uvd.read_uvh5(data_files[0])
    dt = (uvd.time_array.max() - uvd.time_array.min()) + uvd.integration_time.mean() / (24. * 3600.)
    uvd.time_array += dt
    uvd.set_lsts_from_time_array()
    data_files += [os.path.join(tmp_path, fake_obses[1] + '.uvh5')]
    uvd.write_uvh5(data_files[1])
    model_file = os.path.join(tmp_path, fake_obses[1] + '.omni_vis.uvh5')
    uvd.write_uvh5(model_file)
    uvc = UVCal()
    uvc.read_calfits(ocal_file)
    dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
    uvc.time_array += dt
    ocal_file = os.path.join(tmp_path, fake_obses[1] + '.omni.calfits')
    uvc.write_calfits(ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[1] + '.abs.calfits')
    uvc.write_calfits(acal_file)

    # check warnings
    with pytest.warns(None) as record:
        xrfi.xrfi_run(ocal_file, acal_file, model_file, data_files[1], history='Just a test', kt_size=3, clobber=True, throw_away_edges=False)
    assert len(record) >= len(messages)
    n_matched_warnings = 0
    for i in range(len(record)):
        if mess1[0] in str(record[i].message) and cat1[0] == record[i].category:
            n_matched_warnings += 1
    assert n_matched_warnings == 8

    xrfi.day_threshold_run(data_files, history='just a test')
    types = ['og', 'ox', 'ag', 'ax', 'v', 'cross', 'auto', 'omnical_chi_sq_renormed',
             'abscal_chi_sq_renormed', 'combined', 'total']
    for type in types:
        basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.' + type + '_threshold_flags.h5'
        outfile = os.path.join(tmp_path, basename)
        assert os.path.exists(outfile)

    for fake_obs in fake_obses:
        calfile = os.path.join(tmp_path, fake_obs + '.flagged_abs.calfits')
        assert os.path.exists(calfile)


def test_day_threshold_run_data_only(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 6 * mess1
    cat1 = [UserWarning]
    categories = 6 * cat1
    # Spoof the files - run xrfi_run twice on spoofed files.
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355.HH', 'zen.2457698.41101.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    ocal_file = os.path.join(tmp_path, fake_obses[0] + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[0] + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obses[0] + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    data_files = [raw_dfile]
    model_file = os.path.join(tmp_path, fake_obses[0] + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    xrfi.xrfi_run(None, None, None, raw_dfile, history='Just a test', kt_size=3, throw_away_edges=False)
    # Need to adjust time arrays when duplicating files
    uvd = UVData()
    uvd.read_uvh5(data_files[0])
    dt = (uvd.time_array.max() - uvd.time_array.min()) + uvd.integration_time.mean() / (24. * 3600.)
    uvd.time_array += dt
    uvd.set_lsts_from_time_array()
    data_files += [os.path.join(tmp_path, fake_obses[1] + '.uvh5')]
    uvd.write_uvh5(data_files[1])
    model_file = os.path.join(tmp_path, fake_obses[1] + '.omni_vis.uvh5')
    uvd.write_uvh5(model_file)
    uvc = UVCal()
    uvc.read_calfits(ocal_file)
    dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
    uvc.time_array += dt
    ocal_file = os.path.join(tmp_path, fake_obses[1] + '.omni.calfits')
    uvc.write_calfits(ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[1] + '.abs.calfits')
    uvc.write_calfits(acal_file)
    # only perform median filter on autocorrs to hit lines where only first round exists.
    xrfi.xrfi_run(None, None, None, data_files[1], history='Just a test', kt_size=3, auto_mean_filter=False, throw_away_edges=False)

    xrfi.day_threshold_run(data_files, history='just a test')
    types = ['cross', 'auto', 'combined', 'total']
    for t in types:
        basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.' + t + '_threshold_flags.h5'
        outfile = os.path.join(tmp_path, basename)
        assert os.path.exists(outfile)

    for fake_obs in fake_obses:
        calfile = os.path.join(tmp_path, fake_obs + '.flagged_abs.calfits')
        assert os.path.exists(calfile)

def test_day_threshold_run_cal_only(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    metadata_messages = [
        "telescope_location is not set. Using known values for HERA.",
        "antenna_positions is not set. Using known values for HERA."
    ]
    mess1 = ['This object is already a waterfall']
    messages = metadata_messages + 2 * mess1 + metadata_messages + 6 * mess1  + metadata_messages
    # Spoof the files - run xrfi_run twice on spoofed files.
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355.HH', 'zen.2457698.41101.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    ocal_file = os.path.join(tmp_path, fake_obses[0] + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[0] + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obses[0] + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    data_files = [raw_dfile]
    model_file = os.path.join(tmp_path, fake_obses[0] + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    with uvtest.check_warnings(
            UserWarning, match=messages, nwarnings=len(messages)
    ):
        xrfi.xrfi_run(
            acal_file,
            ocal_file,
            None,
            None,
            history="Just a test",
            kt_size=3,
            output_prefixes=raw_dfile,
            throw_away_edges=False,
        )

    # Need to adjust time arrays when duplicating files
    uvd = UVData()
    uvd.read_uvh5(data_files[0])
    dt = (uvd.time_array.max() - uvd.time_array.min()) + uvd.integration_time.mean() / (24. * 3600.)
    uvd.time_array += dt
    uvd.set_lsts_from_time_array()
    data_files += [os.path.join(tmp_path, fake_obses[1] + '.uvh5')]
    uvd.write_uvh5(data_files[1])
    model_file = os.path.join(tmp_path, fake_obses[1] + '.omni_vis.uvh5')
    uvd.write_uvh5(model_file)
    uvc = UVCal()
    uvc.read_calfits(ocal_file)
    dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
    uvc.time_array += dt
    ocal_file = os.path.join(tmp_path, fake_obses[1] + '.omni.calfits')
    uvc.write_calfits(ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[1] + '.abs.calfits')
    uvc.write_calfits(acal_file)
    messages = mess1 * 8
    with uvtest.check_warnings(
            UserWarning, match=messages, nwarnings=len(messages)
    ):
        xrfi.xrfi_run(
            acal_file,
            ocal_file,
            None,
            None,
            history="Just a test",
            kt_size=3,
            output_prefixes=data_files[1],
            throw_away_edges=False,
        )

    xrfi.day_threshold_run(data_files, history='just a test')
    types = ['ox', 'og', 'ax', 'ag', 'omnical_chi_sq_renormed', 'abscal_chi_sq_renormed',
             'combined', 'total']
    for t in types:
        basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.' + t + '_threshold_flags.h5'
        outfile = os.path.join(tmp_path, basename)
        assert os.path.exists(outfile)

    for fake_obs in fake_obses:
        calfile = os.path.join(tmp_path, fake_obs + '.flagged_abs.calfits')
        assert os.path.exists(calfile)

def test_day_threshold_run_omnivis_only(tmpdir):
    # The warnings are because we use UVFlag.to_waterfall() on the total chisquareds
    # This doesn't hurt anything, and lets us streamline the pipe
    mess1 = ['This object is already a waterfall']
    messages = 6 * mess1
    cat1 = [UserWarning]
    categories = 6 * cat1
    # Spoof the files - run xrfi_run twice on spoofed files.
    tmp_path = tmpdir.strpath
    fake_obses = ['zen.2457698.40355.HH', 'zen.2457698.41101.HH']
    # Spoof a couple files to use as extra inputs (xrfi_run needs two cal files and two data-like files)
    ocal_file = os.path.join(tmp_path, fake_obses[0] + '.omni.calfits')
    shutil.copyfile(test_c_file, ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[0] + '.abs.calfits')
    shutil.copyfile(test_c_file, acal_file)
    raw_dfile = os.path.join(tmp_path, fake_obses[0] + '.uvh5')
    shutil.copyfile(test_uvh5_file, raw_dfile)
    data_files = [raw_dfile]
    model_file = os.path.join(tmp_path, fake_obses[0] + '.omni_vis.uvh5')
    shutil.copyfile(test_uvh5_file, model_file)
    xrfi.xrfi_run(None, None, model_file, None, history='Just a test', kt_size=3, output_prefixes=raw_dfile, throw_away_edges=False)
    # Need to adjust time arrays when duplicating files
    uvd = UVData()
    uvd.read_uvh5(data_files[0])
    dt = (uvd.time_array.max() - uvd.time_array.min()) + uvd.integration_time.mean() / (24. * 3600.)
    uvd.time_array += dt
    uvd.set_lsts_from_time_array()
    data_files += [os.path.join(tmp_path, fake_obses[1] + '.uvh5')]
    uvd.write_uvh5(data_files[1])
    model_file = os.path.join(tmp_path, fake_obses[1] + '.omni_vis.uvh5')
    uvd.write_uvh5(model_file)
    uvc = UVCal()
    uvc.read_calfits(ocal_file)
    dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
    uvc.time_array += dt
    ocal_file = os.path.join(tmp_path, fake_obses[1] + '.omni.calfits')
    uvc.write_calfits(ocal_file)
    acal_file = os.path.join(tmp_path, fake_obses[1] + '.abs.calfits')
    uvc.write_calfits(acal_file)
    xrfi.xrfi_run(None, None, model_file, None, history='Just a test', kt_size=3, output_prefixes=data_files[1], throw_away_edges=False)
    xrfi.day_threshold_run(data_files, history='just a test')
    types = ['v', 'combined']
    for t in types:
        basename = '.'.join(fake_obses[0].split('.')[0:-2]) + '.' + t + '_threshold_flags.h5'
        outfile = os.path.join(tmp_path, basename)
        assert os.path.exists(outfile)

    for fake_obs in fake_obses:
        calfile = os.path.join(tmp_path, fake_obs + '.flagged_abs.calfits')
        assert os.path.exists(calfile)

def test_xrfi_h1c_run():
    # run with bad antennas specified
    xrfi.xrfi_h1c_run(test_d_file, filename=test_d_file,
                      history='Just a test.', ex_ants='1,2', xrfi_path=xrfi_path,
                      kt_size=3)

    # catch no provided data file for flagging
    with pytest.warns(None) as record:
        xrfi.xrfi_h1c_run(None, **{'filename': test_d_file, 'history': 'Just a test.',
                                   'model_file': test_d_file, 'model_file_format': 'miriad',
                                   'xrfi_path': xrfi_path})
    assert len(record) >= 191
    n_matched_warnings = 0
    for i in range(len(record)):
        if 'indata is None' in str(record[i].message) or 'K1 value 8' in str(record[i].message):
            n_matched_warnings += 1
    assert n_matched_warnings == 191


def test_xrfi_h1c_run_no_indata():
    # test no indata provided
    pytest.raises(AssertionError, xrfi.xrfi_h1c_run, None,
                  'Just as test.', filename=test_d_file + '.h1c_run')


def test_xrfi_h1c_run_no_filename():
    # test no filename provided
    uvd = UVData()
    uvd.read_miriad(test_d_file)
    pytest.raises(AssertionError, xrfi.xrfi_h1c_run, uvd,
                  'Just as test.', filename=None)


def test_xrfi_h1c_run_filename_not_string():
    # filename is not a string
    uvd = UVData()
    uvd.read_miriad(test_d_file)
    pytest.raises(ValueError, xrfi.xrfi_h1c_run, uvd,
                  history='Just a test.', filename=5)


def test_xrfi_h1c_run_uvfits_no_xrfi_path():
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
    assert os.path.exists(outtest)
    os.remove(outtest)
    if os.path.exists(basename + '.waterfall.flags.h5'):
        os.remove(basename + '.waterfall.flags.h5')
    if os.path.exists(utils.strip_extension(test_d_file) + '.flags.h5'):
        os.remove(utils.strip_extension(test_d_file) + '.flags.h5')
    os.remove(g_temp)
    os.remove(x_temp)


def test_xrfi_h1c_run_uvfits_xrfi_path():
    # test uvfits file with xrfi path
    uvd = UVData()
    uvd.read_miriad(test_d_file)
    basename = utils.strip_extension(os.path.basename(test_uvfits_file))
    outtest = os.path.join(xrfi_path, basename) + '.flags.h5'
    if os.path.exists(outtest):
        os.remove(outtest)
    xrfi.xrfi_h1c_run(test_uvfits_file, infile_format='uvfits',
                      history='Just a test.', kt_size=3, xrfi_path=xrfi_path)
    assert os.path.exists(outtest)
    os.remove(outtest)


def test_xrfi_h1c_run_miriad_model():
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
    assert os.path.exists(outtest)


def test_xrfi_h1c_run_uvfits_model():
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
    assert os.path.exists(outtest)


def test_xrfi_h1c_run_incorrect_model():
    # incorrect model
    uvd = UVData()
    uvd.read_miriad(test_d_file)
    bad_uvfits_test = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA.uvfits')
    pytest.raises(ValueError, xrfi.xrfi_h1c_run, uvd, history='Just a test.',
                  filename=test_d_file, model_file=bad_uvfits_test,
                  model_file_format='uvfits', xrfi_path=xrfi_path, kt_size=3)


def test_xrfi_h1c_run_input_calfits():
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
    assert os.path.exists(outtest1)
    assert os.path.exists(outtest2)


def test_xrfi_h1c_run_incorrect_calfits():
    # check for calfits with incorrect time/freq axes
    uvd = UVData()
    uvd.read_miriad(test_d_file)
    bad_calfits = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
    pytest.raises(ValueError, xrfi.xrfi_h1c_run, uvd, history='Just a test.',
                  filename=test_d_file, calfits_file=bad_calfits,
                  xrfi_path=xrfi_path, kt_size=3)


def test_xrfi_h1c_run_indata_string_filename_not_string():
    pytest.raises(ValueError, xrfi.xrfi_h1c_run, 'foo', history='Just a test.',
                  filename=3)


def test_xrfi_h1c_apply():
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
    assert os.path.exists(dest_file)
    assert os.path.exists(dest_flag)
    shutil.rmtree(dest_file)  # clean up

    # uvfits output
    basename = os.path.basename(utils.strip_extension(test_d_file))
    dest_file = os.path.join(xrfi_path, basename + '.R.uvfits')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path, flag_file=test_f_file_flags,
                        outfile_format='uvfits', extension='R', output_uvflag=False)
    assert os.path.exists(dest_file)
    os.remove(dest_file)

    # uvh5 output
    basename = os.path.basename(utils.strip_extension(test_d_file))
    dest_file = os.path.join(xrfi_path, basename + '.R.uvh5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    xrfi.xrfi_h1c_apply(test_d_file, history, xrfi_path=xrfi_path, flag_file=test_f_file_flags,
                        outfile_format='uvh5', extension='R', output_uvflag=False)
    assert os.path.exists(dest_file)
    os.remove(dest_file)


def test_xrfi_h1c_apply_errors():
    xrfi_path = os.path.join(DATA_PATH, 'test_output')
    history = 'history stuff'
    pytest.raises(AssertionError, xrfi.xrfi_h1c_apply, [], history)

    # test running with two files
    pytest.raises(AssertionError, xrfi.xrfi_h1c_apply, ['file1', 'file2'], history)

    # Outfile error
    pytest.raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, outfile_format='bla')

    basename = utils.strip_extension(os.path.basename(test_d_file))
    dest_file = os.path.join(xrfi_path, basename + '.R.uvfits')
    if not os.path.exists(dest_file):
        open(dest_file, 'a').close()
    pytest.raises(ValueError, xrfi.xrfi_h1c_apply, test_d_file, history, xrfi_path=xrfi_path,
                  flag_file=test_f_file_flags, outfile_format='uvfits', extension='R')


@pytest.fixture(scope='function')
def fake_waterfall():
    # generate a dummy metric waterfall
    np.random.seed(0)
    uvm = UVFlag(test_f_file)
    uvm.to_waterfall()

    # populate with noise and add bad times / channels
    # Note we're breaking the object here to get larger time dimension. But should be ok for test.
    uvm.metric_array = np.random.chisquare(100, 10000).reshape((100, 100, 1)) / 100

    # The fake data changes properties of the object.
    # We need to make the object self consistent
    uvm.Ntimes = 100
    # since this is a waterfall object Nblts = Ntimes
    uvm.Nblts = 100
    uvm.Nfreqs = 100
    uvm.freq_array = np.linspace(np.amin(uvm.freq_array),
                                 np.amax(uvm.freq_array),
                                 uvm.Nfreqs)
    uvm.time_array = np.linspace(np.amin(uvm.time_array),
                                 np.amax(uvm.time_array),
                                 uvm.Ntimes)
    uvm.lst_array = np.linspace(np.amin(uvm.lst_array),
                                np.amax(uvm.lst_array),
                                uvm.Ntimes)
    uvm.weights_array = np.ones_like(uvm.metric_array)

    # yield to allow the test to compute
    yield uvm

    # clean up
    del uvm


def test_threshold_wf_no_rfi(fake_waterfall):
    uvm = fake_waterfall
    # no flags should exist
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=5, detrend=False)
    assert not uvf.flag_array.any()


def test_threshold_wf_time_broadcast(fake_waterfall):
    uvm = fake_waterfall

    # time broadcasting tests
    uvm.metric_array[:, 50] = 6.0  # should get this
    uvm.metric_array[0, 75] = 100.0  # should not get this
    uvm.metric_array[:50, 30] = 5.0
    uvm.metric_array[80:, 30] = 5.0  # should get this
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=100, detrend=False)
    assert uvf.flag_array[:, 50].all()
    assert not uvf.flag_array[:, 75].any()
    assert uvf.flag_array[:, 30].all()


def test_threshold_wf_freq_broadcast(fake_waterfall):
    uvm = fake_waterfall

    # freq broadcasting tests
    uvm.metric_array[10, ::3] = 6.0  # should get this
    uvm.metric_array[1, 50] = 100  # should not get this
    uvf = xrfi.threshold_wf(uvm, nsig_f=100, nsig_t=5, detrend=False)
    assert uvf.flag_array[10].all()
    assert not uvf.flag_array[1].any()


def test_threshold_wf_detrend(fake_waterfall):
    uvm = fake_waterfall

    # test with detrend
    uvm.metric_array[50, :] += .5  # should get this
    uvm.metric_array += .01 * np.arange(100).reshape((100, 1, 1))
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=5, detrend=True)
    assert uvf.flag_array[50].all()
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=5, detrend=False)
    assert not uvf.flag_array[50].all()


def test_threshold_wf_detrend_no_check():
    # generate a dummy metric waterfall
    np.random.seed(0)
    uvm = UVFlag(test_f_file)
    uvm.to_waterfall()

    # populate with noise and add bad times / channels
    # Note we're breaking the object here to get larger time dimension. But should be ok for test.
    uvm.metric_array = np.random.chisquare(100, 10000).reshape((100, 100, 1)) / 100

    # test with detrend
    uvm.metric_array[50, :] += .5  # should get this
    uvm.metric_array += .01 * np.arange(100).reshape((100, 1, 1))
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=5, detrend=True,
                            run_check=False)
    assert uvf.flag_array[50].all()
    uvf = xrfi.threshold_wf(uvm, nsig_f=5, nsig_t=5, detrend=False,
                            run_check=False)
    assert not uvf.flag_array[50].all()


def test_threshold_wf_exceptions():
    # generate a dummy metric waterfall
    np.random.seed(0)
    uvf = UVFlag(test_f_file)

    # exceptions
    pytest.raises(ValueError, xrfi.threshold_wf, uvf)  # UVFlag object but not a waterfall
    uvf.to_flag()
    pytest.raises(ValueError, xrfi.threshold_wf, uvf)  # UVFlag object but not a metric
    pytest.raises(ValueError, xrfi.threshold_wf, 'foo')  # not a UVFlag object


@pytest.mark.filterwarnings("ignore:This object is already a waterfall")
def test_xrfi_h3c_idr2_1_run(tmp_path):

    dec_jds = ['40355', '41101', '41847', '42593', '43339', '44085', '44831']
    fake_obses = [f'zen.2457698.{dec_jd}' for dec_jd in dec_jds]

    ocalfits_files = [tmp_path / f'{fake_obs}.omni.calfits' for fake_obs in fake_obses]
    acalfits_files = [tmp_path / f'{fake_obs}.abs.calfits' for fake_obs in fake_obses]
    model_files = [tmp_path / f'{fake_obs}.omni_vis.uvh5' for fake_obs in fake_obses]
    data_files = [tmp_path / f'{fake_obs}.uvh5' for fake_obs in fake_obses]

    for obsi in range(len(fake_obses)):
        uvc = UVCal()
        uvc.read_calfits(test_c_file)
        dt = (uvc.time_array.max() - uvc.time_array.min()) + uvc.integration_time / (24. * 3600.)
        uvc.time_array += obsi * dt
        uvc.write_calfits(ocalfits_files[obsi])
        uvc.write_calfits(acalfits_files[obsi])

        uv = UVData()
        uv.read(test_uvh5_file)
        uv.time_array += obsi * dt
        uv.set_lsts_from_time_array()
        uv.write_uvh5(model_files[obsi])
        uv.write_uvh5(data_files[obsi])

    ext_labels = {'ag_flags1': 'Abscal gains, round 1. Flags.',
                  'ag_flags2': 'Abscal gains, round 2. Flags.',
                  'ag_metrics1': 'Abscal gains, round 1.',
                  'ag_metrics2': 'Abscal gains, round 2.',
                  'apriori_flags': 'A priori flags.',
                  'ax_flags1': 'Abscal chisq, round 1. Flags.',
                  'ax_flags2': 'Abscal chisq, round 2. Flags.',
                  'ax_metrics1': 'Abscal chisq, round 1.',
                  'ax_metrics2': 'Abscal chisq, round 2.',
                  'chi_sq_flags1': 'Renormalized chisq, round 1. Flags.',
                  'chi_sq_flags2': 'Renormalized chisq, round 2. Flags.',
                  'chi_sq_renormed1': 'Renormalized chisq, round 1.',
                  'chi_sq_renormed2': 'Renormalized chisq, round 2.',
                  'combined_flags1': 'Flags from combined metrics, round 1.',
                  'combined_flags2': 'Flags from combined metrics, round 2.',
                  'combined_metrics1': 'Combined metrics, round 1.',
                  'combined_metrics2': 'Combined metrics, round 2.',
                  'data_flags2': 'Data, round 2. Flags.',
                  'data_metrics2': 'Data, round 2.',
                  'flags1': 'ORd flags, round 1.',
                  'flags2': 'ORd flags, round 2.',
                  'og_flags1': 'Omnical gains, round 1. Flags.',
                  'og_flags2': 'Omnical gains, round 2. Flags.',
                  'og_metrics1': 'Omnical gains, round 1.',
                  'og_metrics2': 'Omnical gains, round 2.',
                  'ox_flags1': 'Omnical chisq, round 1. Flags.',
                  'ox_flags2': 'Omnical chisq, round 2. Flags.',
                  'ox_metrics1': 'Omnical chisq, round 1.',
                  'ox_metrics2': 'Omnical chisq, round 2.',
                  'v_flags1': 'Omnical visibility solutions, round 1. Flags.',
                  'v_flags2': 'Omnical visibility solutions, round 2. Flags.',
                  'v_metrics1': 'Omnical visibility solutions, round 1.',
                  'v_metrics2': 'Omnical visibility solutions, round 2.'}

    # Run with first few obses, should create output for first two
    xrfi.xrfi_h3c_idr2_1_run(ocalfits_files[0:3], acalfits_files[0:3],
                             model_files[0:3], data_files[0:3], 'Just a test',
                             kt_size=3)

    for obsi, obs in enumerate(fake_obses):
        outdir = tmp_path / (obs + '.xrfi')
        if obsi not in [0, 1]:
            # Should not exist
            assert not os.path.exists(outdir)
        else:
            for ext, label in ext_labels.items():
                out = outdir / '.'.join([obs, ext, 'h5'])
                assert os.path.exists(out)
                uvf = UVFlag(str(out))
                assert uvf.label == label
            shutil.rmtree(outdir)  # cleanup

    # Run for three middle obses. Should create output for just the middle.
    xrfi.xrfi_h3c_idr2_1_run(ocalfits_files[2:5], acalfits_files[2:5],
                             model_files[2:5], data_files[2:5], 'Just a test',
                             kt_size=3)

    for obsi, obs in enumerate(fake_obses):
        outdir = tmp_path / (obs + '.xrfi')
        if obsi not in [3]:
            # Should not exist
            assert not os.path.exists(outdir)
        else:
            for ext, label in ext_labels.items():
                out = outdir / '.'.join([obs, ext, 'h5'])
                assert os.path.exists(out)
                uvf = UVFlag(str(out))
                assert uvf.label == label
            shutil.rmtree(outdir)  # cleanup

    # Run for end few, should create output for last two
    xrfi.xrfi_h3c_idr2_1_run(ocalfits_files[4:], acalfits_files[4:],
                             model_files[4:], data_files[4:], 'Just a test',
                             kt_size=3)

    for obsi, obs in enumerate(fake_obses):
        outdir = tmp_path / (obs + '.xrfi')
        if obsi not in [5, 6]:
            # Should not exist
            assert not os.path.exists(outdir)
        else:
            for ext, label in ext_labels.items():
                out = outdir / '.'.join([obs, ext, 'h5'])
                assert os.path.exists(out)
                uvf = UVFlag(str(out))
                assert uvf.label == label
            shutil.rmtree(outdir)  # cleanup
