# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
"""Tests for metrics_io module."""

from __future__ import print_function
import unittest
import nose.tools as nt
import numpy as np
import json
import os
import h5py
import warnings
from hera_qm.data import DATA_PATH
from hera_qm import metrics_io
import pyuvdata.tests as uvtest
from hera_qm.version import hera_qm_version_str


class test_class(object):
    """A dummy class to break h5py object types."""

    def __init__(self):
        """Create blank object."""
        pass


def test_recursive_error_for_object_arrays():
    """Test a TypeError is raised if dictionary items are np.arrays of objecsts."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    path = '/'
    bad_dict = {'1': np.array(['123', 1, np.pi], dtype='object')}
    with h5py.File(test_file, 'w') as h5_test:
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_error_for_dict_of_object():
    """Test a TypeError is raised if dictionary items are objects."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    path = '/'
    bad_dict = {'0': test_class()}
    with h5py.File(test_file, 'w') as h5_test:
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_error_for_object_in_nested_dict():
    """Test TypeError is raised if object is nested in dict."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    path = '/'
    bad_dict = {'0': {'0.0': test_class()}}
    with h5py.File(test_file, 'w') as h5_test:
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_adds_numpy_array_to_h5file():
    """Test that proper numpy arrays are added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_array = np.arange(10)
    path = '/'
    good_dict = {'0': test_array}
    with h5py.File(test_file, 'w') as h5_test:
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_true(np.allclose(test_array, h5_test['0'].value))
    os.remove(test_file)


def test_recursive_adds_nested_numpy_array_to_h5file():
    """Test that a numpy array nested in a dict is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_array = np.arange(10)
    path = '/'
    good_dict = {'0': {'1': test_array}}
    with h5py.File(test_file, 'w') as h5_test:
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)
        nt.assert_true(np.allclose(test_array, h5_test['0/1'].value))
    os.remove(test_file)


def test_recursive_adds_scalar_to_h5file():
    """Test that a scalar type is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    path = '/'
    good_dict = {'0': test_scalar}
    with h5py.File(test_file, 'w') as h5_test:
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_equal(test_scalar, h5_test['0'].value)
    os.remove(test_file)


def test_recursive_adds_nested_scalar_to_h5file():
    """Test that a scalar nested in a dict is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    good_dict = {'0': {'1': test_scalar}}
    path = '/'
    with h5py.File(test_file, 'w') as h5_test:
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_equal(test_scalar, h5_test['0/1'].value)
    os.remove(test_file)


def test_write_metric_file_hdf5():
    """Test that correct hdf4 structure created from write_metric_file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    test_array = np.arange(10)
    test_dict = {'0': test_scalar, 1: {'0': test_scalar, '1': test_array}}
    path = '/'
    metrics_io.write_metric_file(test_file, test_dict)

    with h5py.File(test_file, 'r') as test_h5:
        nt.assert_equal(test_scalar, test_h5['/Metrics/0'].value)
        nt.assert_equal(test_scalar, test_h5['Metrics/1/0'].value)
        nt.assert_true(np.allclose(test_array, test_h5['Metrics/1/1'].value))
    os.remove(test_file)


def test_write_metric_file_hdf5_no_appelation_exists():
    """Test hdf5 file created from write_metric_file if no appelation given."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test')
    test_scalar = 'hello world'
    test_array = np.arange(10)
    test_dict = {'0': test_scalar, 1: {'0': test_scalar, '1': test_array}}
    path = '/'
    metrics_io.write_metric_file(test_file, test_dict)
    test_file += '.hdf5'
    nt.assert_true(os.path.exists(test_file))
    os.remove(test_file)


def test_write_metric_warning_json():
    """Test the known warning is issued when writing to json."""
    json_file = os.path.join(DATA_PATH, 'test_output', 'test_save.json')
    test_scalar = 'hello world'
    test_array = np.arange(10)
    test_dict = {'0': test_scalar, 1: {'0': test_scalar, '1': test_array}}
    warn_message = ["JSON-type files can still be written but are no longer "
                    "writen by default.\n"
                    "Write to HDF5 format for future compatibility."]
    json_dict = uvtest.checkWarnings(metrics_io.write_metric_file,
                                     func_args=[json_file, test_dict],
                                     category=UserWarning, nwarnings=1,
                                     message=warn_message)
    nt.assert_true(os.path.exists(json_file))
    os.remove(json_file)


def test_write_then_recursive_load_dict_to_group_no_nested_dicts():
    """Test recursive load can gather dictionary from a group."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    path = '/'
    good_dict = {'0': test_scalar, 'history': "this is a test",
                 'version': hera_qm_version_str}
    metrics_io.write_metric_file(test_file, good_dict)
    with h5py.File(test_file, 'r') as h5file:
        read_dict = metrics_io._recursively_load_dict_to_group(h5file, '/Header/')
        read_dict.update(metrics_io._recursively_load_dict_to_group(h5file, '/Metrics/'))
    nt.assert_dict_equal(good_dict, read_dict)
    os.remove(test_file)


def test_write_then_recursive_load_dict_to_group_with_nested_dicts():
    """Test recursive load can gather dictionary from a  nested group."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    path = '/'
    good_dict = {'0': test_scalar, 'history': "this is a test",
                 'version': hera_qm_version_str, '1': {'0': test_scalar}}
    metrics_io.write_metric_file(test_file, good_dict)
    with h5py.File(test_file, 'r') as h5file:
        read_dict = metrics_io._recursively_load_dict_to_group(h5file, '/Header/')
        read_dict.update(metrics_io._recursively_load_dict_to_group(h5file, '/Metrics/'))
    for key in good_dict:
        if isinstance(good_dict[key], dict):
            nt.assert_dict_equal(good_dict[key], read_dict[key])
        else:
            nt.assert_equal(good_dict[key], read_dict[key])
    os.remove(test_file)


def test_write_then_load_metric_file_hdf5():
    """Test loaded in map is same as written one from hdf5."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    path = '/'
    good_dict = {'0': test_scalar, 'history': "this is a test",
                 'version': hera_qm_version_str, '1': {'0': test_scalar}}
    metrics_io.write_metric_file(test_file, good_dict)
    read_dict = metrics_io.load_metric_file(test_file)
    for key in good_dict:
        if isinstance(good_dict[key], dict):
            nt.assert_dict_equal(good_dict[key], read_dict[key])
        else:
            nt.assert_equal(good_dict[key], read_dict[key])
    os.remove(test_file)


def test_write_then_load_metric_warning_json_():
    """Test the known warning is issued when writing to json."""
    json_file = os.path.join(DATA_PATH, 'test_output', 'test_save.json')
    test_scalar = "hello world"
    test_array = np.arange(10)
    test_dict = {'history': 'Test case', 'version': '0.0.0',
                 '0': test_scalar, '1': {'0': str(test_scalar),
                                         '1': test_array}}
    warn_message = ["JSON-type files can still be read but are no longer "
                    "writen by default.\n"
                    "Write to HDF5 format for future compatibility.",
                    "The key: 0 could not be parsed, added "
                    "the value as a string"]
    output_json = metrics_io.write_metric_file(json_file, test_dict)
    json_dict = uvtest.checkWarnings(metrics_io.load_metric_file,
                                     func_args=[json_file],
                                     category=UserWarning, nwarnings=2,
                                     message=warn_message)
    for key in test_dict:
        if isinstance(test_dict[key], dict):
            for key1 in test_dict[key]:
                if isinstance(test_dict[key][key1], np.ndarray):
                    nt.assert_true(np.allclose(test_dict[key][key1],
                                               json_dict[key][key1]))
                else:
                    nt.assert_equal(test_dict[key][key1], json_dict[key][key1])

        else:
            nt.assert_equal(test_dict[key], json_dict[key])
    os.remove(json_file)
