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
import re
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
                     'fc_filename', 'filename', 'fc_filestem', 'filestem',
                     'pol', 'ant_pol', 'chisq_good_sol', 'good_sol',
                     'ant_phs_std_good_sol']
float_keys = ['dead_ant_z_cut', 'cross_pol_z_cut', 'always_dead_ant_z_cut']
antpol_dict_keys = ['removal_iteration']
list_of_strings_keys = ['datafile_list']
dict_of_dicts_keys = ['final_mod_z_scores', 'final_metrics']
dict_of_dict_of_dicts_keys = ['all_metrics', 'all_mod_z_scores']


def _reds_list_to_dict(reds):
    """Convert nested list of lists to ordered dict.

    HDF5 will not save lists whose sub-lists vary in size.
    Converts lists of redundant basleine groups to ordered dict to allow
    redundant groups to be saved separately and still preserve order.

    Argument
        reds: List of list of tuples. e.g. list of list of antenna pairs for each baseline group
    Returns
        reds: OrderedDict of baseline groups able to save to HDF5
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

    Will add attributes for HDF5 objects as follows
        Datasets:
            key_is_string: Boolean flag to determine if dictionary key is a string

        Groups:
            group_is_ordered: Boolean flag to determine if dict was an OrderedDict

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
                    if np.issubdtype(np.asarray(in_dict[key]).dtype, np.unicode_):
                        in_dict[key] = np.asarray(in_dict[key]).astype(np.string_)
                    dset = h5file[path].create_dataset(key_str,
                                                       data=in_dict[key],
                                                       compression='lzf')
                    # Add boolean attribute to determine if key is a string
                    # Used to parse keys saved to dicts when reading
                    dset.attrs['key_is_string'] = isinstance(key, str)
                except TypeError as err:
                    raise TypeError("Input dictionary key: {0} does not have "
                                    "compatible dtype. Received this error: "
                                    "{1}".format(key, err))
            else:
                dset = h5file[path].create_dataset(key_str, data=in_dict[key])
                # Add boolean attribute to determine if key is a string
                # Used to parse keys saved to dicts when reading
                dset.attrs['key_is_string'] = isinstance(key, str)

        elif isinstance(in_dict[key], dict):
            grp = h5file[path].create_group(key_str)
            # Add boolean attribute to determine if input dictionary
            # was an OrderedDict
            grp.attrs['group_is_ordered'] = False
            if isinstance(in_dict[key], OrderedDict):
                grp.attrs['group_is_ordered'] = True

                # Generate additional dataset in an ordered group to save
                # the order of the group
                key_order = np.array(list(in_dict[key].keys())).astype('S')
                key_set = grp.create_dataset('key_order', data=key_order)

                # Add boolean attribute to determine if key is a string
                # Used to parse keys saved to dicts when reading
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
            out_dict[key_str] = np.array2string(in_dict[key], separator=',')
        else:
            out_dict[key_str] = str(in_dict[key])
    return out_dict


def write_metric_file(filename, input_dict, overwrite=False):
    """Convert the input dictionary into an HDF5 File.

    Can write either HDF5 (recommended) or JSON (Depreciated in Future) types.
    Will try to guess which type based on the extension of the input filename.
    If not extension is given, HDF5 is used by default.

    If writing to HDF5 the following attributes will be added
        Datasets:
            key_is_string: Boolean flag to determine if dictionary key is a string

        Groups:
            group_is_ordered: Boolean flag to determine if dict was an OrderedDict


    Arguments
        filename: String of filename to which metrics will be written.
                  Can include either HDF5 (recommended) or JSON (Depreciated in Future) extension.
        input_dict: Dictionary to be recursively written to the given file
        overwrite: If file exists, overwrite instead of raising error.
                   Default False

    Returns:
        None
    """
    if os.path.exists(filename) and not overwrite:
        raise IOError('File exists and overwrite set to False.')
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
            if os.path.exists(filename) and not overwrite:
                raise IOError('File exists and overwrite set to False.')

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


def _recursively_load_dict_to_group(h5file, path, group_is_ordered=False):
    """Recursively read the hdf5 file and create sub-dictionaries.

    Performs opposite function of _recursively_save_dict_to_group

    Arguments
        h5file: H5py file object into which data is written, this is edited in place.
        path: absolute path in HDF5 to read the current dictionary
        group_is_ordered: Boolean value to set up collections.OrderedDict objects if flag is set in h5file group
    Returns
    """
    if group_is_ordered:
        out_dict = OrderedDict()
        key_list = h5file[path + 'key_order'].value
    else:
        out_dict = {}
        key_list = list(h5file[path].keys())

    for key in key_list:
        if isinstance(key, bytes):
            key = key.decode()

        item = h5file[path][key]

        if isinstance(item, h5py.Dataset):
            if item.attrs['key_is_string']:
                out_dict[str(key)] = item.value
            else:
                out_key = _parse_key(key)
                out_dict[out_key] = item.value

        elif isinstance(item, h5py.Group):
            if item.attrs['key_is_string']:
                out_key = str(key)
            else:
                out_key = _parse_key(key)
            out_dict[out_key] = _recursively_load_dict_to_group(h5file, (path + key + '/'),
                                                                group_is_ordered=item.attrs["group_is_ordered"])
        else:
            raise TypeError("The HDF5 path: {0} is not associated with either "
                            "a dataset or group object. "
                            "Please verify input file.".format(path))
    return out_dict


def _parse_key(key):
    """Parse the input key into an int, antpol tuple, or string.

    Arguments
        key: input string to parse
    Returns
        out_key: Parsed key as an int, antpol tuple, or string
    """
    try:
        # is key an int?
        return int(str(key))
    except ValueError:
        # is key an antpol tuple?
        antpol_regex = r"(\([0-9]*?, \'[xy]\'\))"
        matches = re.findall(antpol_regex, key)
        try:
            # split tuple into antenna number and polarization
            ant, pol = matches[0][1:-1].split(",")
            # extract just polarization character
            pol = pol[-2]
            return tuple((int(ant), str(pol)))
        except IndexError:
            # treat it as a string
            return str(key)


def _recursively_parse_json(in_dict):
    """Recursively walk dictionary from json and convert to proper types.

    Arguments
        in_dict: Dictionary of strings read from json file to convert
    Returns
        out_dict: dictionary with arrays/list/int/float cast to proper type.
    """
    def _pretty_print_dict(di):
        output = '{'
        for key, val in six.iteritems(di):
            if isinstance(val, dict):
                tmp = _pretty_print_dict(val)
                if key in ['meanVijXPol', 'meanVij', 'redCorr', 'redCorrXPol']:
                    output += "'{}': {}".format(key, tmp)
                else:
                    output += "{}: {}".format(key, tmp)
            else:
                output += "{}: {}".format(key, val)
            output += ', '
        output = output[:-2]
        output += '}'
        return output

    out_dict = {}
    for key in in_dict:
        out_key = _parse_key(key)

        # special handling mostly for ant_metrics json files
        if key in known_string_keys:
            out_dict[out_key] = str(in_dict[key])
        elif key in float_keys:
            out_dict[out_key] = float(in_dict[key])
        elif key in antpol_dict_keys:
            if isinstance(in_dict[key], dict):
                str_in = _pretty_print_dict(in_dict[key])
            else:
                str_in = str(in_dict[key])
            out_dict[out_key] = _parse_dict(str_in)
        elif key in list_of_strings_keys:
            out_dict[out_key] = _parse_list_of_strings(in_dict[key])
        elif key in antpol_keys:
            str_in = "{}".format(in_dict[key])
            out_dict[out_key] = _parse_list_of_antpols(str_in)
        elif key in antpair_keys:
            str_in = "{}".format(in_dict[key])
            out_dict[out_key] = _parse_list_of_list_of_antpairs(str_in)
        elif key in dict_of_dicts_keys:
            if isinstance(in_dict[key], dict):
                str_in = _pretty_print_dict(in_dict[key])
            else:
                str_in = str(in_dict[key])
            out_dict[out_key] = _parse_dict_of_dicts(str_in)
        elif key in dict_of_dict_of_dicts_keys:
            if isinstance(in_dict[key], dict):
                str_in = _pretty_print_dict(in_dict[key])
            else:
                str_in = str(in_dict[key])
            out_dict[out_key] = _parse_dict_of_dict_of_dicts(str_in)
        else:
            if isinstance(in_dict[key], dict):
                out_dict[out_key] = _recursively_parse_json(in_dict[key])
            elif isinstance(in_dict[key], (list, np.ndarray)):
                    try:
                        if len(in_dict[key]) > 0:
                            if isinstance(in_dict[key][0], (six.text_type, np.int,
                                                            np.float, np.complex)):
                                try:
                                    out_dict[out_key] = [float(val)
                                                         for val in in_dict[key]]
                                except ValueError:
                                    out_dict[out_key] = [complex(val)
                                                         for val in in_dict[key]]
                        else:
                            out_dict[out_key] = str(in_dict[key])
                    except (SyntaxError, NameError) as err:
                            warnings.warn("The key: {0} has a value which "
                                          "could not be parsed, added"
                                          " the value as a string: {1}"
                                          .format(key, str(in_dict[key])))
                            out_dict[out_key] = str(in_dict[key])
            else:
                # make a last attempt to cast the value to an int/float/complex
                # otherwise make it a string
                out_dict[out_key] = _parse_value(in_dict[key])
    return out_dict


def _parse_value(in_val):
    """Try to turn given input value into an int, float, or complex.

    Uses builtin types to make a last attempt to cast the value as an int, float, or complex.
    If all the types fail, returns a string.

    Arguments
        in_val: the unicode or string value to be parsed.
    Returns
        input value cast as an int, float or complex, otherwise a string.
    """
    try:
        return int(str(in_val))
    except ValueError:
        try:
            return float(str(in_val))
        except ValueError:
            try:
                return np.complex(str(in_val))
            except ValueError:
                return str(in_val)


def _parse_dict(input_str, value_type=int):
    """
    Parse a text string as a dictionary.

    Arguments
        input_str: string to be processed
        value_type: data type to cast value as. Default is int
    Returns
        output: dictionary containing key-value pairs

    Notes
        This function explicitly assumes that dictionary keys are antpol tuples, where
        the only allowed polarization values are x or y. `value_type` is typically
        either `int` or `float`, depending on the parent key.
    """
    # use regex to extract keys and values
    # assumes keys are antpols and values are numbers
    dict_regex = r'(\([0-9]*?, \'[xy]\'\)): (.*?)[,}]'
    key_vals = re.findall(dict_regex, input_str)

    # initialize output
    output = {}
    for entry in key_vals:
        # split key into individual elements
        ant, pol = entry[0][1:-1].split(",")
        # further refine pol -- get just the polarization, not the quotes
        pol = pol[-2]
        key = tuple((int(ant), str(pol)))
        value = value_type(entry[1])
        output[key] = value

    return output


def _parse_list_of_strings(input_str):
    """
    Parse a text string as a list of strings.

    Arguments
        input_str: string to be processed
    Returns
        files: list of strings representing input files
    """
    # use basic string processing to extract file paths
    # remove brackets
    files = input_str[1:-1]

    # split on commas
    files = files.split(',')

    # strip surrounding spaces and quotes; cast as str type
    files = [str(f.strip()[1:-1]) for f in files]

    return files


def _parse_list_of_antpols(input_str):
    """
    Parse a text string as a list of antpols.

    Arguments
        input_str: string to be processed
    Returns
        li: list of antpols
    Notes
        Assumes polarizations are only x or y
    """
    # use regex to extract entries
    # assumes list entries are antpols
    list_regex = r"(\([0-9]*?, \'[xy]\'\))"
    entries = re.findall(list_regex, input_str)

    # initialize output
    li = []
    for entry in entries:
        ant, pol = entry[1:-1].split(",")
        pol = pol[-2]
        li.append(tuple((int(ant), str(pol))))

    return li


def _parse_list_of_list_of_antpairs(input_str):
    """
    Parse a text string as a list of list of antpairs.

    Arguments
        input_str: string to be processed
    Returns
        output: list of list of antpairs
    Notes
        Used for the `reds` key to return groups of redundant baselines.
    """
    # use regex to extract lists
    list_regex = r"\[(.*?)\]"
    lists = re.findall(list_regex, input_str[1:-1])

    # initialize output
    output = []
    for li in lists:
        sublist = []
        tuple_regex = r"\((.*?)\)"
        tuples = re.findall(tuple_regex, li)
        for t in tuples:
            ant1, ant2 = t.split(",")
            sublist.append(tuple((int(ant1), int(ant2))))
        output.append(sublist)
    return output


def _parse_dict_of_dicts(input_str, value_type=float):
    """
    Parse a text string as a dictionary of dictionaries.

    Arguments
        input_str: string to be processed
        value_type: type to cast values in nested dictionaries to. Default
        is float.
    Returns
        output: dictionary of dictionaries. The keys of the outer dictionary
        should be the names of the associated metrics.
    """
    # use regex to extract dictionaries
    dict_regex = r"\'([a-zA-Z]*?)\': (\{.*?\})"
    dicts = re.findall(dict_regex, input_str)

    # initialize output
    output = {}
    for d in dicts:
        # key is first capture group
        key = str(d[0])
        # subdictionary is second capture group
        subdict = _parse_dict(d[1], value_type=value_type)
        output[key] = subdict

    return output


def _parse_dict_of_dict_of_dicts(input_str, value_type=float):
    """
    Parse a text string as a dictionary of dictionaries of dictionaries.

    Arguments
        input_str: string to be processed
        value_type: type to cast values in nested dictionaries to. Default
        is float.
    Returns
        output: dictionary of dictionary of dictionaries
    """
    # use regex to extract dictionaries
    dict_regex = r"([0-9]*?)\'?: \{(.*?\})\}"
    dicts = re.findall(dict_regex, input_str)

    # initialize output
    output = {}
    for d in dicts:
        # key is first capture group
        key = int(str(d[0]))
        # subdictionary is second capture group
        subdict = _parse_dict_of_dicts(d[1], value_type=value_type)
        output[key] = subdict

    return output


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
    metric_dict = _recursively_parse_json(jsonMetrics)

    return metric_dict


def _recusively_validate_dict(in_dict):
    """Walk dictionary recursively and cast special types to know formats.

    Walks a dictionary recursively and searches for special types of items.
    Cast 'reds' from OrderedDict to list of list
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
            in_dict[key] = list(map(tuple, np.array(in_dict[key], dtype=antpol_dtype)))

        if key in known_string_keys and isinstance(in_dict[key], bytes):
            in_dict[key] = in_dict[key].decode()

        if key in list_of_strings_keys:
            if np.issubdtype(np.asarray(in_dict[key]).dtype, np.bytes_):
                in_dict[key] = np.asarray(in_dict[key]).astype(np.str_).tolist()

        if isinstance(in_dict[key], dict):
            _recusively_validate_dict(in_dict[key])


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
