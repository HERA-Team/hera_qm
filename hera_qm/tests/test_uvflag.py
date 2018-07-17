from __future__ import division
import unittest
import nose.tools as nt
import os
import numpy as np
import pyuvdata.tests as uvtest
from pyuvdata import UVData
from pyuvdata import UVCal
from hera_qm.data import DATA_PATH
from hera_qm import UVFlag
from hera_qm.uvflag import lst_from_uv
from hera_qm.version import hera_qm_version_str
import copy

test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_c_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'


def test_init_UVData():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    nt.assert_true(uvf.metric_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 0))
    nt.assert_true(uvf.type == 'baseline')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true('Flag object with type "baseline"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)

def test_init_UVCal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    nt.assert_true(uvf.metric_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 0))
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uvc.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uvc.jones_array))
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true('Flag object with type "antenna"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)

def test_init_wf_uvd():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))
    uvf = UVFlag(uv, wf=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.weights_array == 0))
    nt.assert_true(uvf.type == 'wf')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.lst_array == np.unique(uv.lst_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true('Flag object with type "wf"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)

def test_init_wf_uvc():
    uv = UVCal()
    uv.read_calfits(os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits'))
    uvf = UVFlag(uv, wf=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.weights_array == 0))
    nt.assert_true(uvf.type == 'wf')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "wf"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)

def test_read_write_loop():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))
    uvf = UVFlag(uv)
    testfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')
    uvf.write(testfile, clobber=True)
    uvf2 = UVFlag(testfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))

def test_read_write_ant():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv)
    testfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')
    uvf.write(testfile, clobber=True)
    uvf2 = UVFlag(testfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))

def test_write_no_clobber():
    fi = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5')
    uvf = UVFlag(fi)
    nt.assert_raises(ValueError, uvf.write, fi)

def test_lst_from_uv():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA'))
    lst_array = lst_from_uv(uv)
    nt.assert_true(np.allclose(uv.lst_array, lst_array))

def test_lst_from_uv_error():
    nt.assert_raises(ValueError, lst_from_uv, 4)

def test_add():
    uv1 = UVFlag(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5'))
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along time axis with ' + hera_qm_version_str in uv3.history)

def test_add_baseline():
    uv1 = UVFlag(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5'))
    uv2 = copy.deepcopy(uv1)
    uv2.baseline_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='baseline')
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along baseline axis with ' + hera_qm_version_str in uv3.history)

def test_add_antenna():
    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits'))
    uv1 = UVFlag(uvc)
    uv2 = copy.deepcopy(uv1)
    uv2.ant_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='antenna')
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_array, uv2.ant_array)),
                                  uv3.ant_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=0),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(uv3.type == 'antenna')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along antenna axis with ' + hera_qm_version_str in uv3.history)

def test_add_frequency():
    uv1 = UVFlag(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5'))
    uv2 = copy.deepcopy(uv1)
    uv2.freq_array += 1e4  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='frequency')
    nt.assert_true(np.array_equal(np.concatenate((uv1.freq_array, uv2.freq_array)),
                                  uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=2),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=2),
                                  uv3.weights_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along frequency axis with ' + hera_qm_version_str in uv3.history)

def test_add_pol():
    uv1 = UVFlag(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5'))
    uv2 = copy.deepcopy(uv1)
    uv2.polarization_array += 1  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='polarization')
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.lst_array, uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.metric_array, uv2.metric_array), axis=3),
                                  uv3.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=3),
                                  uv3.weights_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'metric')
    nt.assert_true(np.array_equal(np.concatenate((uv1.polarization_array, uv2.polarization_array)),
                                  uv3.polarization_array))
    nt.assert_true('Data combined along polarization axis with ' + hera_qm_version_str in uv3.history)

def test_add_flag():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))
    uv1 = UVFlag(uv, mode='flag')
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.lst_array, uv2.lst_array)),
                                  uv3.lst_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.flag_array, uv2.flag_array), axis=0),
                                  uv3.flag_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.weights_array, uv2.weights_array), axis=0),
                                  uv3.weights_array))
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(uv3.type == 'baseline')
    nt.assert_true(uv3.mode == 'flag')
    nt.assert_true(np.array_equal(uv1.polarization_array, uv3.polarization_array))
    nt.assert_true('Data combined along time axis with ' + hera_qm_version_str in uv3.history)

def test_add_errors():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA'))
    uvc = UVCal()
    uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits'))
    uv1 = UVFlag(uv)
    # Mismatched classes
    nt.assert_raises(ValueError, uv1.__add__, 3)
    # Mismatched types
    uv2 = UVFlag(uvc)
    nt.assert_raises(ValueError, uv1.__add__, uv2)
    # Mismatched modes
    uv3 = UVFlag(uv, mode='flag')
    nt.assert_raises(ValueError, uv1.__add__, uv3)
    # Invalid axes
    nt.assert_raises(ValueError, uv1.__add__, uv1, axis='antenna')
    nt.assert_raises(ValueError, uv2.__add__, uv2, axis='baseline')

def test_inplace_add():
    uv1a = UVFlag(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.testuvflag.h5'))
    uv1b = copy.deepcopy(uv1a)
    uv2 = copy.deepcopy(uv1a)
    uv2.time_array += 1
    uv1a += uv2
    nt.assert_true(uv1a.__eq__(uv1b + uv2))
