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
from hera_qm.version import hera_qm_version_str


def test_init_UVData():
    uv = UVData()
    uv.read_miriad(os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA'))
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
    uvc.read_calfits(os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits'))
    uvf = UVFlag(uvc)
    nt.assert_true(uvf.metric_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.metric_array == 0))
    nt.assert_true(uvf.weights_array.shape == uvc.flag_array.shape)
    nt.assert_true(np.all(uvf.weights_array == 0))
    nt.assert_true(uvf.type == 'antenna')
    nt.assert_true(uvf.mode == 'metric')
    nt.assert_true(np.all(uvf.time_array == uvc.time_array))
    # TODO: Need this when lst_array is implemented from cal data
    # nt.assert_true(np.all(uvf.lst_array == uv.lst_array))
    nt.assert_true(np.all(uvf.freq_array == uvc.freq_array[0]))
    nt.assert_true(np.all(uvf.polarization_array == uvc.jones_array))
    nt.assert_true(np.all(uvf.ant_array == uvc.ant_array))
    nt.assert_true('Flag object with type "antenna"' in uvf.history)
    nt.assert_true(hera_qm_version_str in uvf.history)

def test_init_wf():
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

def test_write_read_loop():
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
