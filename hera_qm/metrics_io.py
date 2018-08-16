# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License
"""I/O Handler for all metrics that can be stored as (nested) dictionaries."""

from __future__ import print_function, division, absolute_import

from six.moves import range, map
import json
import os
import h5py
import warnings
import numpy as np
import copy
import six
from collections import OrderedDict
from hera_qm.version import hera_qm_version_str

# HDF5 casts all inputs to numpy arrays.
# Define a custom numpy dtype for tuples we wish to preserve shape
# antpol items look like (antenna, antpol) with types (int, string)
antpol_dtype = np.dtype([('ant', np.int32),
                         ('pol', np.dtype('S1'))])
# antpair items look like (ant1, ant2) with types (int, int)
antpair_dtype = np.dtype([('ant1', np.int32),
                          ('ant2', np.int32)])

antpair_keys = ['reds']
antpol_keys = ['xants', 'dead_ants', 'crossed_ants']
known_string_keys = ['history', 'version', 'filedir', 'cut_edges',
                     'fc_filename', 'filename',  'fc_filestem', 'filestem',
                     'pol', 'ant_pol', 'chisq_good_sol', 'good_sol',
                     'ant_phs_std_good_sol']


def _reds_list_to_dict(reds):
    """Convert nested list of lists to ordered dict.

    HDF5 will not save lists whose sub-lists vary in size.
    Converts lists of redundant basleine groups to ordered dict to allow
    redundant groups to be saved separately and still preserve order.

    Argument
        reds: List of list of tuples. e.g. list of list of antenna pairs for each baseline group
    Returns
        reds: OrderdDict of baseline groups able to save to HDF5
    """
    return OrderedDict([(i, np.array(reds[i], dtype=antpair_dtype))
                        for i in range(len(reds))])


def _reds_dict_to_list(reds):
    """Convert dict of redundant baseline groups to list.

    Arguments
        reds: OrderedDict of redundant baseline groups.
    Returns
        reds: List of lists of baseline groups.
    """
    if isinstance(reds, dict):
        reds = list(reds.values())

    return [list(map(tuple, red)) for red in reds]


def _recursively_save_dict_to_group(h5file, path, in_dict):
    """Recursively walks a dictionary to save to hdf5.

    Adds allowed types to the current group as datasets
    creates new subgroups if a given item in a dictionary is a dictionary.


    Arguments
        h5file: H5py file object into which data is written, this is edited in place.
        path: absolute path in HDF5 to store the current dictionary
        in_dict: Dictionary to be recursively walked and stored in h5file
    Returns
        None
    """
    allowed_types = (np.ndarray, np.float, np.int, bytes, six.text_type, list)
    compressable_types = (np.ndarray, list)
    for key in in_dict:
        key_str = str(key)
        if key == 'reds':
            in_dict[key] = _reds_list_to_dict(in_dict[key])

        if isinstance(in_dict[key], allowed_types):

            if key in antpol_keys:
                in_dict[key] = np.array(in_dict[key],
                                        dtype=antpol_dtype)
            if key in antpair_keys:
                in_dict[key] = np.array(in_dict[key],
                                        dtype=antpair_dtype)

            if isinstance(in_dict[key], compressable_types):
                try:
                    dset = h5file[path].create_dataset(key_str,
                                                       data=in_dict[key],
                                                       compression='lzf')
                    dset.attrs['key_is_string'] = isinstance(key, str)
                except TypeError as e:
                    raise TypeError("Input dictionary key: {0} does not have "
                                    "compatible dtype. Received this error: "
                                    "{1}".format(key, e))
            else:
                dset = h5file[path].create_dataset(key_str, data=in_dict[key])
                dset.attrs['key_is_string'] = isinstance(key, str)

        elif isinstance(in_dict[key], dict):
            grp = h5file[path].create_group(key_str)
            grp.attrs['ordered'] = False
            if isinstance(in_dict[key], OrderedDict):
                grp.attrs['ordered'] = True

                key_order = list(map(str, list(in_dict[key].keys())))
                key_set = grp.create_dataset('key_order', data=key_order)
                key_set.attrs['key_is_string'] = True
            grp.attrs['key_is_string'] = isinstance(key, str)
            _recursively_save_dict_to_group(h5file, path + key_str + '/',
                                            in_dict[key])
        else:
            raise TypeError("Cannot save key {0} with type {1}"
                            .format(key, type(in_dict[key])))
    return


def _recursively_make_dict_arrays_strings(in_dict):
    """Recursively search dict for numpy array and cast as string.

    Numpy arrays by default are space delimited in strings, this makes them comma delimitedself.

    Arguments
        in_dict: Dictionary to recursively search for numpy array
    Returns
        out_dict: Copy of in_dict with numpy arrays cast as comma delimited strings
    """
    out_dict = {}
    for key in in_dict:
        key_str = str(key)
        if isinstance(in_dict[key], dict):
            out_dict[key_str] = _recursively_make_dict_arrays_strings(in_dict[key])
        elif isinstance(in_dict[key], np.ndarray):
            out_dict[key_str] = np.array2string(in_dict[key],  separator=',')
        else:
            out_dict[key_str] = str(in_dict[key])
    return out_dict


def write_metric_file(filename, input_dict):
    """Convert the input dictionary into an HDF5 File.

    Can write either HDF5 (recommended) or JSON (Depreciated in Future) types.
    Will try to guess which type based on the extension of the input filename.
    If not extension is given, HDF5 is used by default.

    Arguments
        filename: String of filename to which metrics will be written.
                  Can include either HDF5 (recommended) or JSON (Depreciated in Future) extension.

    Returns:
        None
    """
    input_dict = copy.deepcopy(input_dict)
    if filename.split('.')[-1] == 'json':
        warnings.warn("JSON-type files can still be written "
                      "but are no longer written by default.\n"
                      "Write to HDF5 format for future compatibility.",
                      PendingDeprecationWarning)

        json_write_dict = _recursively_make_dict_arrays_strings(input_dict)

        with open(filename, 'w') as outfile:
            json.dump(json_write_dict, outfile, indent=4)
    else:
        if filename.split('.')[-1] not in ('hdf5', 'h5'):
            filename += '.hdf5'

        with h5py.File(filename, 'w') as f:
            header = f.create_group('Header')
            header.attrs['key_is_string'] = True

            header['history'] = input_dict.pop('history',
                                               'No History Found. '
                                               'Written by '
                                               'hera_qm.metrics_io')
            header['history'].attrs['key_is_string'] = True

            header['version'] = input_dict.pop('version', hera_qm_version_str)
            header['version'].attrs['key_is_string'] = True

            # Create group for metrics data in file
            mgrp = f.create_group('Metrics')
            mgrp.attrs['key_is_string'] = True
            _recursively_save_dict_to_group(f, "/Metrics/", input_dict)

    return


def _recursively_load_dict_to_group(h5file, path, ordered=False):
    """Recursively read the hdf5 file and create sub-dictionaries.

    Performs opposite function of _recursively_save_dict_to_group

    Arguments
        h5file: H5py file object into which data is written, this is edited in place.
        path: absolute path in HDF5 to read the current dictionary
        ordered: Boolean value to set up collections.OrderedDict objects if flag is set in h5file group
    Returns
    """
    if ordered:
        out_dict = OrderedDict()
        key_list = h5file[path + 'key_order'].value
    else:
        out_dict = {}
        key_list = list(h5file[path].keys())
    for key in key_list:

        item = h5file[path + key]

        if isinstance(item, h5py.Dataset):
            if item.attrs['key_is_string']:
                out_dict[str(key)] = item.value
            else:
                out_dict[eval(str(key))] = item.value

        elif isinstance(item, h5py.Group):
            if item.attrs['key_is_string']:
                out_key = str(key)
            else:
                out_key = eval(str(key))
            out_dict[out_key] = _recursively_load_dict_to_group(h5file, (path + key + '/'),
                                                                ordered=item.attrs["ordered"])
    return out_dict


def parse_key(key, gvars):
    """Parse the input key into an int, tuple or string.

    Arguments
        key: input string to parse
        gvars: global variables for eval call
    Returns
        out_key: Parsed key as an int, tuple or string
    """
    try:
        out_key = int(str(key))
    except ValueError:
        try:
            out_key = eval(str(key), gvars)
            if isinstance(out_key, (np.float, np.complex)):
                out_key = str(key)
        except (SyntaxError, NameError):
            out_key = str(key)
    return out_key


def _recursively_evaluate_json(in_dict, gvars):
    """recursively walk dictionary from json and convert to proper types.

    Arguments
        in_dict: Dictionary of strings read from json file to convert
        gvars: Global variables used in eval.
    Returns
        out_dict: dictionary with arrays/list/int/float cast to proper type.
    """
    out_dict = {}
    for key in in_dict:
        out_key = parse_key(key, gvars)

        if isinstance(in_dict[key], dict):
            out_dict[out_key] = _recursively_evaluate_json(in_dict[key], gvars)
        elif isinstance(in_dict[key], (list, np.ndarray)):
                try:
                    if len(in_dict[key]) > 0:
                        if isinstance(in_dict[key][0], (six.text_type, np.int,
                                                        np.float, np.complex)):
                            out_dict[out_key] = [eval(str(val), gvars)
                                                 for val in in_dict[key]]
                    else:
                        out_dict[out_key] = eval(str(in_dict[key]), gvars)
                except (SyntaxError, NameError) as err:
                        warnings.warn("The key: {0} has a value which "
                                      "could not be parsed, added"
                                      " the value as a string: {1}"
                                      .format(key, str(in_dict[key])))
                        out_dict[out_key] = str(in_dict[key])
        else:
            if key in known_string_keys:
                out_dict[out_key] = str(in_dict[key])
            else:
                try:
                    out_dict[out_key] = eval(str(in_dict[key]), gvars)
                except (SyntaxError, NameError) as err:
                    warnings.warn("The key: {0} has a value which "
                                  "could not be parsed, added"
                                  " the value as a string: {1}"
                                  .format(key, str(in_dict[key])))
                    out_dict[out_key] = str(in_dict[key])
    return out_dict


def _load_json_metrics(filename):
    """Load cut decisions and metrics from a JSON into python dictionary.

    reads json and recursively walks dictionaries to cast values to proper types.
    Arguments
        filename: JSON file to be read and walked
    returns
        metric_dict: dictionary with values cast as types made from hera_qm.
    """
    with open(filename, 'r') as infile:
        jsonMetrics = json.load(infile)
    gvars = {'nan': np.nan, 'inf': np.inf, '-inf': -np.inf, '__builtins__': {},
             'array': np.array, 'OrderedDict': OrderedDict}

    metric_dict = _recursively_evaluate_json(jsonMetrics, gvars)

    return metric_dict


def _recusively_validate_dict(in_dict, iter=0):
    """Walk dictionary recursively and cast special types to know formats.

    Walks a dictionary recursively and searches for special types of items.
    Cast 'reds' from OrderdDict to list of list
    Cast antpair_key and antpol_key items to tuples

    Arguments
        in_dict: Dictionary to search and cast special items.
    Returns
        out_dict: Copy of input dictionary with antpairs and antpols cast as tuples
    """
    for key in in_dict:
        if key == 'reds':
            in_dict[key] = _reds_dict_to_list(in_dict[key])

        if key in antpair_keys and key != 'reds':
            in_dict[key] = list(map(tuple, in_dict[key]))

        if key in antpol_keys:
            in_dict[key] = list(map(tuple, in_dict[key]))

        if isinstance(in_dict[key], dict):
            _recusively_validate_dict(in_dict[key], iter=iter+1)


def load_metric_file(filename):
    """Load the given hdf5 files name into a dictionary.

    Loads either HDF5 (recommended) or JSON (Depreciated in Future) save files.
    Guesses which type to load based off the file extension.
    Arguments:
        filename: Filename of the metric to load.

    Returns:
        Dictionary of metrics stored in the input file.
    """
    if filename.split('.')[-1] == 'json':
        warnings.warn("JSON-type files can still be read "
                      "but are no longer written by default.\n"
                      "Write to HDF5 format for future compatibility.",
                      PendingDeprecationWarning)
        metric_dict = _load_json_metrics(filename)
    else:
        with h5py.File(filename, 'r') as f:
            metric_dict = _recursively_load_dict_to_group(f, "/Header/")
            metric_dict.update(_recursively_load_dict_to_group(f, "/Metrics/"))

    _recusively_validate_dict(metric_dict)
    return metric_dict
