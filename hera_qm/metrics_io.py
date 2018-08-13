# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
"""I/O Handler for all metrics that can be stored as (nested) dictionaries."""

from __future__ import print_function, division, absolute_import
import json
import os
import h5py
import warnings
import numpy as np
import copy


def _recursively_save_dict_to_group(h5file, path, in_dict):
    """Recursively walks a dictionary to save to hdf5.

    Adds allowed types to the current group as datasets
    creates new subgroups if a given item in a dictionary is a dictionary.
    """
    allowed_types = (np.ndarray, np.float, np.int, bytes, str, list)
    compressable_types = (np.ndarray, list)
    for key in in_dict:
        if isinstance(in_dict[key], allowed_types):
            if isinstance(in_dict[key], compressable_types):
                try:
                    _ = h5file[path].create_dataset(str(key),
                                                    data=in_dict[key],
                                                    compression='lzf')
                except TypeError as e:
                    raise TypeError("Input dictionary key: {0} does not have "
                                    "compatible dtype. Received this error: "
                                    "{1}".format(key, e))
            else:
                _ = h5file[path].create_dataset(str(key), data=in_dict[key])
        elif isinstance(in_dict[key], dict):
            _ = h5file[path].create_group(str(key))
            _recursively_save_dict_to_group(h5file, path + str(key) + '/',
                                            in_dict[key])
        else:
            raise TypeError("Cannot save key {0} with type {1}"
                            .format(key, type(in_dict[key])))


def write_metric_file(filename, input_dict):
    """Convert the input dictionary into an HDF5 File."""
    input_dict = copy.deepcopy(input_dict)
    if filename.split('.')[-1] == 'json':
        warnings.warn("JSON-type files can still be written "
                      "but are no longer writen by default.\n"
                      "Write to HDF5 format for future compatibility.")
        for key in input_dict:
            input_dict[key] = str(input_dict[key])
        with open(filename, 'w') as outfile:
            json.dump(input_dict, outfile, indent=4)
    else:
        if filename.split('.')[-1] not in ('hdf5', 'h5'):
            filename += '.hdf5'

        with h5py.File(filename, 'w') as f:
            header = f.create_group('Header')

            header['history'] = input_dict.pop('history',
                                               'No History Provided')

            header['version'] = input_dict.pop('version',
                                               'No Version information')

            # Create group for metrics data in file
            mgrp = f.create_group('Metrics')
            _recursively_save_dict_to_group(f, '/Metrics/', input_dict)


def _recursively_load_dict_to_group(h5file, path):
    """Recursively read the hdf5 file and create sub-dictionaries."""
    out_dict = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            out_dict[str(key)] = item.value
        elif isinstance(item, h5py.Group):
            out_dict[str(key)] = _recursively_load_dict_to_group(h5file,
                                                                 (path + key
                                                                  + '/'))
    return out_dict


def load_json_metrics(json_file):
    """Load cut decisions and metrics from a JSON into python dictionary."""
    with open(json_file, 'r') as infile:
        jsonMetrics = json.load(infile)
    gvars = {'nan': np.nan, 'inf': np.inf, '-inf': -np.inf, 'array': np.array}

    metric_dict = {}
    for key, val in jsonMetrics.items():
        if (key == 'version') or (key == 'history'):
            metric_dict[key] = str(val)
        else:
            try:
                metric_dict[key] = eval(str(val), gvars)
            except SyntaxError as sne:
                if str(sne).strip() == ('unexpected EOF while parsing '
                                        '(<string>, line 1)'):
                    warnings.warn("The key: {0} could not be parsed, added "
                                  "the value as a string".format(key))
                    metric_dict[key] = str(val)
    return metric_dict


def load_metric_file(filename):
    """Load the given hdf5 files name into a dictionary."""
    if filename.split('.')[-1] == 'json':
        warnings.warn("JSON-type files can still be read but are no longer "
                      "writen by default.\n"
                      "Write to HDF5 format for future compatibility.")
        return load_json_metrics(filename)
    with h5py.File(filename, 'r') as f:
        metric_dict = _recursively_load_dict_to_group(f, '/Header/')
        metric_dict.update(_recursively_load_dict_to_group(f, '/Metrics/'))
    return metric_dict
