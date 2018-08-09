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
from hera_qm.utils import lst_from_uv
from hera_qm.version import hera_qm_version_str
import copy

test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
test_c_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
test_f_file = test_d_file + '.testuvflag.h5'
test_outfile = os.path.join(DATA_PATH, 'test_output', 'uvflag_testout.h5')


def test_init_UVData():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, history='I made a UVFlag object')
    nt.assert_true(uvf.metric_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'baseline')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.ant_1_array == uv.ant_1_array))
    nt.assert_true(np.all(uvf.ant_2_array == uv.ant_2_array))
    nt.assert_true('I made a UVFlag object' in uvf.history)
    nt.assert_true('Flag object with type "baseline"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_UVData_copy_flags():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'metric'},
                               nwarnings=1, message='Copying flags to type=="baseline"')
    nt.assert_false(hasattr(uvf, 'metric_array'))  # Should be flag due to copy flags
    nt.assert_true(np.array_equal(uvf.flag_array, uv.flag_array))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'baseline')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == uv.time_array))
    nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true(np.all(uvf.baseline_array == uv.baseline_array))
    nt.assert_true(np.all(uvf.ant_1_array == uv.ant_1_array))
    nt.assert_true(np.all(uvf.ant_2_array == uv.ant_2_array))
    nt.assert_true('Flag object with type "baseline"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_UVCal():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    nt.assert_true(uvf.metric_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    lst = lst_from_uv(uvc)
    nt.assert_true(np.all(uvf.lst_array == lst))
    nt.assert_true(np.all(uvf.freq_array == uvc.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uvc.jones_array))
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true('Flag object with type "antenna"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_cal_copy_flags():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'metric'},
                               nwarnings=1, message='Copying flags to type=="antenna"')
    nt.assert_false(hasattr(uvf, 'metric_array'))  # Should be flag due to copy flags
    nt.assert_true(np.array_equal(uvf.flag_array, uv.flag_array))
    nt.assert_true(uvf.weights_array.shape == uv.flag_array.shape)
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_waterfall_uvd():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, waterfall=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Npols))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.lst_array == np.unique(uv.lst_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_waterfall_uvc():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True)
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_waterfall_flag():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, waterfall=True, mode='flag')
    nt.assert_true(uvf.flag_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(not np.any(uvf.flag_array))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(np.all(uvf.weights_array == 1))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'flag')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_init_waterfall_copy_flags():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = uvtest.checkWarnings(UVFlag, [uv], {'copy_flags': True, 'mode': 'flag', 'waterfall': True},
                               nwarnings=1, message='Copying flags into waterfall')
    nt.assert_false(hasattr(uvf, 'flag_array'))  # Should be metric due to copy flags
    nt.assert_true(uvf.metric_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(uvf.weights_array.shape == (uv.Ntimes, uv.Nfreqs, uv.Njones))
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == np.unique(uv.time_array)))
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.jones_array))
    nt.assert_true('Flag object with type "waterfall"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)


def test_read_write_loop():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


def test_read_write_ant():
    uv = UVCal()
    uv.read_calfits(test_c_file)
    uvf = UVFlag(uv, mode='flag')
    uvf.write(test_outfile, clobber=True)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


def test_read_write_nocompress():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv)
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


def test_read_write_nocompress_flag():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, mode='flag')
    uvf.write(test_outfile, clobber=True, data_compression=None)
    uvf2 = UVFlag(test_outfile)
    # Update history to match expected additions that were made
    uvf.history += 'Written by ' + hera_qm_version_str
    uvf.history += ' Read by ' + hera_qm_version_str
    nt.assert_true(uvf.__eq__(uvf2, check_history=True))


def test_init_list():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.time_array -= 1
    uvf = UVFlag([uv, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    nt.assert_true(np.array_equal(np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0),
                                  uvf.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
                                  uvf.weights_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.time_array, uvf2.time_array)),
                                  uvf.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.baseline_array, uvf2.baseline_array)),
                                  uvf.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)),
                                  uvf.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)),
                                  uvf.ant_2_array))
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))


def test_read_list():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uv.time_array -= 1
    uvf = UVFlag(uv)
    uvf.write(test_outfile, clobber=True)
    uvf.read([test_outfile, test_f_file])
    uvf1 = UVFlag(uv)
    uvf2 = UVFlag(test_f_file)
    nt.assert_true(np.array_equal(np.concatenate((uvf1.metric_array, uvf2.metric_array), axis=0),
                                  uvf.metric_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.weights_array, uvf2.weights_array), axis=0),
                                  uvf.weights_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.time_array, uvf2.time_array)),
                                  uvf.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.baseline_array, uvf2.baseline_array)),
                                  uvf.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_1_array, uvf2.ant_1_array)),
                                  uvf.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uvf1.ant_2_array, uvf2.ant_2_array)),
                                  uvf.ant_2_array))
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.freq_array == uv.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uv.polarization_array))


def test_read_error():
    nt.assert_raises(IOError, UVFlag, 'foo')


def test_read_change_type():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.write(test_outfile, clobber=True)
    nt.assert_true(hasattr(uvf, 'ant_array'))
    uvf.read(test_f_file)
    nt.assert_false(hasattr(uvf, 'ant_array'))
    nt.assert_true(hasattr(uvf, 'baseline_array'))
    nt.assert_true(hasattr(uvf, 'ant_1_array'))
    nt.assert_true(hasattr(uvf, 'ant_2_array'))
    uvf.read(test_outfile)
    nt.assert_true(hasattr(uvf, 'ant_array'))
    nt.assert_false(hasattr(uvf, 'baseline_array'))
    nt.assert_false(hasattr(uvf, 'ant_1_array'))
    nt.assert_false(hasattr(uvf, 'ant_2_array'))


def test_read_change_mode():
    uv = UVData()
    uv.read_miriad(test_d_file)
    uvf = UVFlag(uv, mode='flag')
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))
    uvf.write(test_outfile, clobber=True)
    uvf.read(test_f_file)
    nt.assert_true(hasattr(uvf, 'metric_array'))
    nt.assert_false(hasattr(uvf, 'flag_array'))
    uvf.read(test_outfile)
    nt.assert_true(hasattr(uvf, 'flag_array'))
    nt.assert_false(hasattr(uvf, 'metric_array'))


def test_write_no_clobber():
    uvf = UVFlag(test_f_file)
    nt.assert_raises(ValueError, uvf.write, test_f_file)


def test_lst_from_uv():
    uv = UVData()
    uv.read_miriad(test_d_file)
    lst_array = lst_from_uv(uv)
    nt.assert_true(np.allclose(uv.lst_array, lst_array))


def test_lst_from_uv_error():
    nt.assert_raises(ValueError, lst_from_uv, 4)


def test_add():
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
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
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.baseline_array += 100  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='baseline')
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
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
    uvc.read_calfits(test_c_file)
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
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.freq_array += 1e4  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='frequency')
    nt.assert_true(np.array_equal(np.concatenate((uv1.freq_array, uv2.freq_array)),
                                  uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.ant_1_array, uv3.ant_1_array))
    nt.assert_true(np.array_equal(uv1.ant_2_array, uv3.ant_2_array))
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
    uv1 = UVFlag(test_f_file)
    uv2 = copy.deepcopy(uv1)
    uv2.polarization_array += 1  # Arbitrary
    uv3 = uv1.__add__(uv2, axis='polarization')
    nt.assert_true(np.array_equal(uv1.freq_array, uv3.freq_array))
    nt.assert_true(np.array_equal(uv1.time_array, uv3.time_array))
    nt.assert_true(np.array_equal(uv1.baseline_array, uv3.baseline_array))
    nt.assert_true(np.array_equal(uv1.ant_1_array, uv3.ant_1_array))
    nt.assert_true(np.array_equal(uv1.ant_2_array, uv3.ant_2_array))
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
    uv.read_miriad(test_d_file)
    uv1 = UVFlag(uv, mode='flag')
    uv2 = copy.deepcopy(uv1)
    uv2.time_array += 1  # Add a day
    uv3 = uv1 + uv2
    nt.assert_true(np.array_equal(np.concatenate((uv1.time_array, uv2.time_array)),
                                  uv3.time_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.baseline_array, uv2.baseline_array)),
                                  uv3.baseline_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_1_array, uv2.ant_1_array)),
                                  uv3.ant_1_array))
    nt.assert_true(np.array_equal(np.concatenate((uv1.ant_2_array, uv2.ant_2_array)),
                                  uv3.ant_2_array))
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
    uv.read_miriad(test_d_file)
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
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
    uv1a = UVFlag(test_f_file)
    uv1b = copy.deepcopy(uv1a)
    uv2 = copy.deepcopy(uv1a)
    uv2.time_array += 1
    uv1a += uv2
    nt.assert_true(uv1a.__eq__(uv1b + uv2))


def test_clear_unused_attributes():
    uv = UVFlag(test_f_file)
    nt.assert_true(hasattr(uv, 'baseline_array') & hasattr(uv, 'ant_1_array')
                   & hasattr(uv, 'ant_2_array'))
    uv.type = 'antenna'
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'baseline_array') | hasattr(uv, 'ant_1_array')
                    | hasattr(uv, 'ant_2_array'))
    uv.mode = 'flag'
    nt.assert_true(hasattr(uv, 'metric_array'))
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'metric_array'))

    # Start over
    uv = UVFlag(test_f_file)
    uv.ant_array = np.array([4])
    uv.flag_array = np.array([5])
    uv.clear_unused_attributes()
    nt.assert_false(hasattr(uv, 'ant_array'))
    nt.assert_false(hasattr(uv, 'flag_array'))


def test_not_equal():
    uvf1 = UVFlag(test_f_file)
    # different class
    nt.assert_false(uvf1.__eq__(5))
    # different mode
    uvf2 = uvf1.copy()
    uvf2.mode = 'flag'
    nt.assert_false(uvf1.__eq__(uvf2))
    # different type
    uvf2 = uvf1.copy()
    uvf2.type = 'antenna'
    nt.assert_false(uvf1.__eq__(uvf2))
    # array different
    uvf2 = uvf1.copy()
    uvf2.freq_array += 1
    nt.assert_false(uvf1.__eq__(uvf2))
    # history different
    uvf2 = uvf1.copy()
    uvf2.history += 'hello'
    nt.assert_false(uvf1.__eq__(uvf2, check_history=True))


def test_to_waterfall_bl():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)


def test_to_waterfall_bl_multi_pol():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf2 = uvf.copy()
    uvf2.polarization_array[0] = -4
    uvf.__add__(uvf2, inplace=True, axis='pol')  # Concatenate to form multi-pol object
    uvf2 = uvf.copy()  # Keep a copy to run with keep_pol=False
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)
    nt.assert_true(len(uvf.polarization_array) == 2)
    # Repeat with keep_pol=False
    uvf2.to_waterfall(keep_pol=False)
    nt.assert_true(uvf2.type == 'waterfall')
    nt.assert_true(uvf2.metric_array.shape == (len(uvf2.time_array), len(uvf.freq_array), 1))
    nt.assert_true(uvf2.weights_array.shape == uvf2.metric_array.shape)
    nt.assert_true(len(uvf2.polarization_array) == 1)
    nt.assert_true(uvf2.polarization_array[0] == ','.join(map(str, uvf.polarization_array)))


def test_to_waterfall_bl_flags():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)


def test_to_waterfall_ant():
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    uvf = UVFlag(uvc)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    nt.assert_true(uvf.type == 'waterfall')
    nt.assert_true(uvf.metric_array.shape == (len(uvf.time_array), len(uvf.freq_array),
                                              len(uvf.polarization_array)))
    nt.assert_true(uvf.weights_array.shape == uvf.metric_array.shape)


def test_to_waterfall_waterfall():
    uvf = UVFlag(test_f_file)
    uvf.weights_array = np.ones_like(uvf.weights_array)
    uvf.to_waterfall()
    uvtest.checkWarnings(uvf.to_waterfall, [], {}, nwarnings=1,
                         message='This object is already a waterfall')


def test_copy():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    nt.assert_true(uvf == uvf2)
    # Make sure it's a copy and not just pointing to same object
    uvf.to_waterfall()
    nt.assert_false(uvf == uvf2)


def test_or():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf3 = uvf | uvf2
    nt.assert_true(np.all(uvf3.flag_array[0]))
    nt.assert_false(np.any(uvf3.flag_array[1]))
    nt.assert_true(np.all(uvf3.flag_array[2:]))


def test_or_error():
    uvf = UVFlag(test_f_file)
    uvf2 = uvf.copy()
    uvf.to_flag()
    nt.assert_raises(ValueError, uvf.__or__, uvf2)


def test_or_add_history():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.history = 'Different history'
    uvf3 = uvf | uvf2
    nt.assert_true(uvf.history in uvf3.history)
    nt.assert_true(uvf2.history in uvf3.history)
    nt.assert_true("Flags OR'd with:" in uvf3.history)


def test_ior():
    uvf = UVFlag(test_f_file)
    uvf.to_flag()
    uvf2 = uvf.copy()
    uvf2.flag_array = np.ones_like(uvf2.flag_array)
    uvf.flag_array[0] = True
    uvf2.flag_array[0] = False
    uvf2.flag_array[1] = False
    uvf |= uvf2
    nt.assert_true(np.all(uvf.flag_array[0]))
    nt.assert_false(np.any(uvf.flag_array[1]))
    nt.assert_true(np.all(uvf.flag_array[2:]))
