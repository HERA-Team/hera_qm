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


class test_class(object):
    """A dummy class to break h5py object types."""

    def __init__(self):
        """Create blank object."""
        pass


def test_recursive_error_for_object_arrays():
    """Test a TypeError is raised if dictionary items are np.arrays of objecsts."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        bad_dict = {'1': np.array(['123', 1, np.pi], dtype='object')}
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_error_for_dict_of_object():
    """Test a TypeError is raised if dictionary items are objects."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        bad_dict = {'0': test_class()}
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_error_for_object_in_nested_dict():
    """Test TypeError is raised if object is nested in dict."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        bad_dict = {'0': {'0.0': test_class()}}
        nt.assert_raises(TypeError, metrics_io._recursively_save_dict_to_group,
                         h5_test, path, bad_dict)
    os.remove(test_file)


def test_recursive_adds_numpy_array_to_h5file():
    """Test that proper numpy arrays are added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_array = np.arange(10)
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        good_dict = {'0': test_array}
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_true(np.allclose(test_array, h5_test['0'].value))
    os.remove(test_file)


def test_recursive_adds_nested_numpy_array_to_h5file():
    """Test that a numpy array nested in a dict is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_array = np.arange(10)
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        good_dict = {'0': {'1': test_array}}
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)
        for key in h5_test:
            print(key, h5_test[key])
        nt.assert_true(np.allclose(test_array, h5_test['0/1'].value))
    os.remove(test_file)


def test_recursive_adds_scalar_to_h5file():
    """Test that a scalar type is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        good_dict = {'0': test_scalar}
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_equal(test_scalar, h5_test['0'].value)
    os.remove(test_file)


def test_recursive_adds_nested_scalar_to_h5file():
    """Test that a scalar nested in a dict is added to h5file."""
    test_file = os.path.join(DATA_PATH, 'test_output', 'test.h5')
    test_scalar = 'hello world'
    with h5py.File(test_file, 'w') as h5_test:
        path = '/'
        good_dict = {'0': {'1': test_scalar}}
        metrics_io._recursively_save_dict_to_group(h5_test, path, good_dict)

        nt.assert_equal(test_scalar, h5_test['0/1'].value)
    os.remove(test_file)
