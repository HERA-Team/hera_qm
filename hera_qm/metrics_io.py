# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License
"""I/O Handler for all metrics that can be stored as (nested) dictionaries."""

import json
import os
import h5py
import warnings
import numpy as np
import pickle as pkl
import copy
import re
import yaml
from collections import OrderedDict
from . import __version__
from . import utils as qm_utils


# HDF5 casts all inputs to numpy arrays.
# Define a custom numpy dtype for tuples we wish to preserve shape
# antpol items look like (antenna, antpol) with types (int, string)
antpol_dtype = np.dtype([('ant', np.int32),
                         ('pol', np.dtype('S3'))])

# antpair items look like (ant1, ant2) with types (int, int)
antpair_dtype = np.dtype([('ant1', np.int32),
                          ('ant2', np.int32)])

antpair_keys = ['reds']
antpol_keys = ['xants', 'dead_ants', 'crossed_ants']
bool_keys = ['good_sol', 'chisq_good_sol', 'ant_phs_std_good_sol']
known_string_keys = ['history', 'version', 'filedir', 'cut_edges',
                     'fc_filename', 'filename', 'fc_filestem', 'filestem',
                     'pol', 'ant_pol']
float_keys = ['dead_ant_z_cut', 'cross_pol_z_cut', 'always_dead_ant_z_cut']
antpol_dict_keys = ['removal_iteration']
list_of_strings_keys = ['datafile_list', 'datafile_list_sum', 'datafile_list_diff']
dict_of_dicts_keys = ['final_mod_z_scores', 'final_metrics']
dict_of_dict_of_dicts_keys = ['all_metrics', 'all_mod_z_scores']
dict_of_dict_of_tuple_keys = ['meanVijXPol', 'meanVij', 'redCorr', 'redCorrXPol', 'corr', 'corrXPol']
dict_of_bl_dicts_keys = ['spectra', 'modzs']  # for auto_metrics


def _reds_list_to_dict(reds):
    """Convert nested list of lists to ordered dict.

    HDF5 will not save lists whose sub-lists vary in size.
    Converts lists of redundant basleine groups to ordered dict to allow
    redundant groups to be saved separately and still preserve order.

    Parameters
    ----------
    reds : list of (lists or tuples)
        List of list of tuples. e.g. list of list of antenna pairs for each baseline group.

    Returns
    -------
    reds : OrderedDict
        OrderedDict of baseline groups able to save to HDF5.
    """
    return OrderedDict([(i, np.array(reds[i], dtype=antpair_dtype))
                        for i in range(len(reds))])


def _reds_dict_to_list(reds):
    """Convert dict of redundant baseline groups to list.

    Parameters
    ----------
    reds : OrderedDict
        OrderedDict of redundant baseline groups.

    Returns
    -------
    reds : list of lists
        List of lists of baseline groups.
    """
    if isinstance(reds, dict):
        reds = list(reds.values())

    return [list(map(tuple, red)) for red in reds]


def _recursively_save_dict_to_group(h5file, path, in_dict):
    """Recursively walks a dictionary to save to hdf5.

    This function adds allowed types to the current group as datasets. It creates
    new subgroups if a given item in a dictionary is a dictionary.

    The function will add attributes for HDF5 objects as follows:
        Datasets:
            key_is_string: Boolean flag to determine if dictionary key is a string

        Groups:
            group_is_ordered: Boolean flag to determine if dict was an OrderedDict

    Parameters
    ----------
    h5file : file object
        An h5py file object into which data is written. This file is edited in place.
    path : str
        An absolute path in HDF5 to store the current dictionary.
    in_dict : dict
        A dictionary to be recursively walked and stored in h5file.

    Returns
    -------
    None


    Raises
    ------
    TypeError:
        If a dataset is to be written to file and the datatype is not compatible
        with h5py, a TypeError is raised.

    """
    allowed_types = (np.ndarray, np.float32, float, int,
                     bytes, str, list, bool, np.bool_)
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
                if isinstance(in_dict[key], str):
                    in_dict[key] = qm_utils._str_to_bytes(in_dict[key])
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
            raise TypeError("Cannot save key {0} with type {1} at path {2}"
                            .format(key, type(in_dict[key]), path))
    return


def _recursively_make_dict_arrays_strings(in_dict):
    """Recursively search dict for numpy array and cast as string.

    Numpy arrays by default are space delimited in strings, this makes them comma
    delimited.

    Parameters
    ----------
    in_dict : dict
        Dictionary to recursively search for numpy arrays.

    Returns
    -------
    out_dict : dict
        A copy of in_dict with numpy arrays cast as comma delimited strings.

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


    Parameters
    ----------
    filename : str
        The full path to the filename to which metrics will be written. Based on the
        extention, the file will be written as HDF5 (recommended), JSON (Depreciated
        in Future), or a python pickle (Deprecated in Future).
    input_dict : dict
        Dictionary to be recursively written to the given file.
    overwrite : bool, optional
        If True, overwrite an existing file instead of raising error. Default is False.

    Returns
    -------
    None

    Raises
    ------
    IOError:
        If the file at filename exists and overwrite is False, an IOError is raised.

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
    elif filename.split('.')[-1] == 'pkl':
        warnings.warn("Pickle-type files can still be written "
                      "but are no longer written by default.\n"
                      "Write to HDF5 format for future compatibility.",
                      PendingDeprecationWarning)

        with open(filename, 'wb') as outfile:
            out_pickle = pkl.Pickler(outfile)
            out_pickle.dump(input_dict)
    else:
        if filename.split('.')[-1] not in ('hdf5', 'h5'):
            filename += '.hdf5'
            if os.path.exists(filename) and not overwrite:
                raise IOError('File exists and overwrite set to False.')

        with h5py.File(filename, 'w') as outfile:
            header = outfile.create_group('Header')
            header.attrs['key_is_string'] = True

            header['history'] = qm_utils._str_to_bytes(input_dict.pop('history',
                                                                      'No History Found. '
                                                                      'Written by '
                                                                      'hera_qm.metrics_io'))
            header['history'].attrs['key_is_string'] = True

            header['version'] = qm_utils._str_to_bytes(input_dict.pop('version', __version__))
            header['version'].attrs['key_is_string'] = True

            # Create group for metrics data in file
            mgrp = outfile.create_group('Metrics')
            mgrp.attrs['group_is_ordered'] = False
            if isinstance(input_dict, OrderedDict):
                mgrp.attrs['group_is_ordered'] = True

                # Generate additional dataset in an ordered group to save
                # the order of the group
                key_order = np.array(list(input_dict.keys())).astype('S')
                key_set = mgrp.create_dataset('key_order', data=key_order)

                # Add boolean attribute to determine if key is a string
                # Used to parse keys saved to dicts when reading
                key_set.attrs['key_is_string'] = True
            mgrp.attrs['key_is_string'] = True
            _recursively_save_dict_to_group(outfile, "/Metrics/", input_dict)

    return


def _recursively_load_dict_to_group(h5file, path, group_is_ordered=False):
    """Recursively read the hdf5 file and create sub-dictionaries.

    This function performs the inverse operation of _recursively_save_dict_to_group.

    Parameters
    ----------
    h5file : file object
        An h5py file object into which data is written. This file is edited in place.
    path : str
        An absolute path in HDF5 to store the current dictionary.
    group_is_ordered : bool, optional
        If True, the dictionary is unpacked as an OrderedDictionary. Default is False.

    Returns
    -------
    out_dict : dict
        The dictionary as saved in the output file.

    Raises
    ------
    TypeError:
        If a key in the HDF5 file tree is not associated with a dataset or a
        group, a TypeError is raised. This indicates a file without a correct
        structure.

    """
    if group_is_ordered:
        out_dict = OrderedDict()
        key_list = h5file[path + 'key_order'][()]
    else:
        out_dict = {}
        key_list = list(h5file[path].keys())

    for key in key_list:
        if isinstance(key, bytes):
            key = key.decode()

        item = h5file[path][key]

        if isinstance(item, h5py.Dataset):
            if item.attrs['key_is_string']:
                out_dict[str(key)] = item[()]
            else:
                out_key = _parse_key(key)
                out_dict[out_key] = item[()]

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

    Parameters
    ----------
    key : str
        Input string to parse.

    Returns
    -------
    out_key : int or tuple or str
        Parsed key as an int, antpol tuple, or string.

    """
    try:
        # is key an int?
        return int(str(key))
    except ValueError:
        # is key an antpol tuple?
        antpol_regex = r"(\([0-9]*?, \'[xyne]\'\))"
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

    Parameters
    ----------
    in_dict : dict
        Dictionary of strings read from json file to convert.

    Returns
    -------
    out_dict : dict
        Dictionary with arrays/list/int/float cast to proper type.

    """
    def _pretty_print_dict(di):
        output = '{'
        for key, val in di.items():
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
        elif key in bool_keys:
            out_dict[out_key] = np.bool_(in_dict[key])
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
                        if isinstance(in_dict[key][0], (str, int,
                                                        float, complex)):
                            try:
                                out_dict[out_key] = [float(val)
                                                     for val in in_dict[key]]
                            except ValueError:
                                out_dict[out_key] = [complex(val)
                                                     for val in in_dict[key]]
                    else:
                        out_dict[out_key] = in_dict[key]
                except (SyntaxError, NameError):
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

    Parameters
    ----------
    in_val : unicode or str
        The unicode or string value to be parsed.

    Returns
    -------
    parsed_val : int or float or complex or str
        The input value cast as an int, float or complex, otherwise a string.
    """
    try:
        return int(str(in_val))
    except ValueError:
        try:
            return float(str(in_val))
        except ValueError:
            try:
                return np.complex128(str(in_val))
            except ValueError:
                return str(in_val)


def _parse_dict(input_str, value_type=int):
    """
    Parse a text string as a dictionary.

    Note
    ----
    This function explicitly assumes that dictionary keys are antpol tuples, where
    the only allowed polarization values are x or y. `value_type` is typically
    either `int` or `float`, depending on the parent key.

    Parameters
    ----------
    input_str : str
        The string to be processed.
    value_type : data-type, optional
        The data type to cast value as. Default is int.

    Returns
    -------
    output : dict
        Dictionary containing key-value pairs.

    """
    # use regex to extract keys and values
    # assumes keys are antpols and values are numbers
    dict_regex = r'(\([0-9]*?, \'[xyne]\'\)): (.*?)[,}]'
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

    Parameters
    ----------
    input_str : str
        The input string to be processed.

    Returns
    -------
    files : list of str
        The list of strings representing input files.

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

    Note
    ----
    This function assumes polarizations are only x or y.

    Parameters
    ----------
    input_str : str
        The input string to be processed.

    Returns
    -------
    li : list
        The parsed list of antpols.
    """
    # use regex to extract entries
    # assumes list entries are antpols
    list_regex = r"(\([0-9]*?, \'[xyne]\'\))"
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

    Note
    ----
    This function is used for the `reds` key to return groups
    of redundant baselines.

    Parameters
    ----------
    input_str : str
        The input string to be processed.

    Returns
    -------
    output : list
        A list containing the parsed set of antpairs.
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
        for tup in tuples:
            ant1, ant2 = tup.split(",")
            sublist.append(tuple((int(ant1), int(ant2))))
        output.append(sublist)
    return output


def _parse_dict_of_dicts(input_str, value_type=float):
    """
    Parse a text string as a dictionary of dictionaries.

    Parameters
    ----------
    input_str : str
        The input string to be processed.
    value_type : data-type, optional
        The type to cast values in nested dictionaries to. Default is float.

    Returns
    -------
    output : dict
        A dictionary of dictionaries. The keys of the outer dictionary should
        be the names of the associated metrics.

    """
    # use regex to extract dictionaries
    dict_regex = r"\'([a-zA-Z]*?)\': (\{.*?\})"
    dicts = re.findall(dict_regex, input_str)

    # initialize output
    output = {}
    for dicti in dicts:
        # key is first capture group
        key = str(dicti[0])
        # subdictionary is second capture group
        subdict = _parse_dict(dicti[1], value_type=value_type)
        output[key] = subdict

    return output


def _parse_list_of_dict_of_dicts(input_str, value_type=float):
    """Parse a test string as a list of dictionaries of dictionaries.

    This function assumes the input list of dicts of dicts is from an old
    ant_metrics json file and will convert the list of dict of dicts to a
    dict of dict of dicts to be consistent with the new format of ant_metrics.

    Parameters
    ----------
    input_str : str
        The input string to be processed.
    value_type : data-type, optional
        The type to cast values in nested dictionaries to. Default is float.

    Returns
    -------
    output : dict
        A dictionary of dictionary of dictionaries.

    """
    # use regex to extract dictionaries

    dict_of_dict_regex = r"(\{\'[A-Za-z0-9]*?\': \{.*?\}+\})"
    dicts = re.findall(dict_of_dict_regex, input_str)

    # initialize output
    output = OrderedDict()
    for num, dicti in enumerate(dicts):
        subdict = _parse_dict_of_dicts(dicti, value_type=value_type)
        output[num] = subdict

    return output


def _parse_dict_of_dict_of_dicts(input_str, value_type=float):
    """
    Parse a text string as a dictionary of dictionaries of dictionaries.

    Also tests if input is list of dictionaries of dictionairies for backwards
    compatibility with older ant_metrics json files.

    Parameters
    ----------
    input_str : str
        The input string to be processed.
    value_type : data-type, optional
        The type to cast values in nested dictionaries to. Default is float.

    Returns
    -------
    output : dict
        A dictionary of dictionary of dictionaries.

    """
    # use regex to extract dictionaries
    list_regex = r"\[.*\]"
    list_match = re.match(list_regex, input_str)

    if list_match:
        return _parse_list_of_dict_of_dicts(input_str, value_type=value_type)

    dict_regex = r"([0-9]*?)\'?: \{(.*?\})\}"
    dicts = re.findall(dict_regex, input_str)

    # initialize output
    output = {}
    for dicti in dicts:
        # key is first capture group
        key = _parse_key(dicti[0])
        # subdictionary is second capture group
        subdict = _parse_dict_of_dicts(dicti[1], value_type=value_type)
        output[key] = subdict

    return output


def _load_json_metrics(filename):
    """Load cut decisions and metrics from a JSON into python dictionary.

    This function reads json and recursively walks dictionaries to cast values
    to proper types.

    Parameters
    ----------
    filename : str
        Full path to a JSON file to be read and walked.

    Returns
    -------
    metric_dict : dict
        A dictionary with values cast as types made from hera_qm.

    """
    with open(filename, 'r') as infile:
        jsonMetrics = json.load(infile)
    metric_dict = _recursively_parse_json(jsonMetrics)

    return metric_dict


def _load_pickle_metrics(filename):
    """Load cut decisions and metrics from a Pickle file into python dictionary.

    Parameters
    ----------
    filename : str
        Full path to a pickle file to be read and walked.

    Returns
    -------
    metric_dict : dict
        A dictionary with values cast as types made from hera_qm.

    """
    with open(filename, 'rb') as infile:
        unpickler = pkl.Unpickler(infile)
        metric_dict = unpickler.load()

    return metric_dict


def _recursively_validate_dict(in_dict):
    """Walk dictionary recursively and cast special types to know formats.

    This function walks a dictionary recursively and searches for special types
    of items. It casts 'reds' from OrderedDict to a list of list. It also casts
    antpair_key and antpol_key items to tuples.

    Parameters
    ----------
    in_dict : dict
        Dictionary to search and cast special items.

    Returns
    -------
    out_dict : dict
        A copy of input dictionary with antpairs and antpols cast as tuples.

    """
    def _antpol_str_to_tuple(antpol_str):
        '''Turns a string of the form "(1, 'Jee')" into a int/string tuple e.g. (1, "Jee")'''
        if isinstance(antpol_str, tuple):
            return antpol_str
        aps = antpol_str.replace('(', '')
        aps = aps.replace(')', '')
        aps = aps.replace(' ', '')
        aps = aps.replace("'", "")
        aps = aps.replace('"', '')
        split = aps.split(',')
        return tuple([int(ant) for ant in split[0:-1]]) + (split[-1],)

    for key in in_dict:
        if key in ['history', 'version']:
            if isinstance(in_dict[key], bytes):
                in_dict[key] = qm_utils._bytes_to_str(in_dict[key])

        if key == 'reds':
            in_dict[key] = _reds_dict_to_list(in_dict[key])

        if key in antpair_keys and key != 'reds':
            in_dict[key] = [tuple((int(a1), int(a2)))
                            for pair in in_dict[key]
                            for a1, a2 in pair]

        if key in antpol_keys:
            in_dict[key] = [tuple((int(ant), qm_utils._bytes_to_str(pol)))
                            if isinstance(pol, bytes) else tuple((int(ant), (pol)))
                            for ant, pol in in_dict[key]]

        if key in antpol_dict_keys:
            in_dict[key] = {_antpol_str_to_tuple(k): in_dict[key][k] for k in in_dict[key]}

        if key in dict_of_dict_of_tuple_keys:
            in_dict[key] = {_antpol_str_to_tuple(k): in_dict[key][k] for k in in_dict[key]}

        if key in dict_of_bl_dicts_keys:
            in_dict[key] = {k: {_antpol_str_to_tuple(k2): in_dict[key][k][k2] for k2 in in_dict[key][k]} for k in in_dict[key]}

        if key in known_string_keys and isinstance(in_dict[key], bytes):
            in_dict[key] = in_dict[key].decode()

        if key in list_of_strings_keys:
            in_dict[key] = [qm_utils._bytes_to_str(n) if isinstance(n, bytes)
                            else n for n in in_dict[key]]

        if isinstance(in_dict[key], (np.int64, np.int32)):
            in_dict[key] = int(in_dict[key])

        if isinstance(in_dict[key], dict):
            _recursively_validate_dict(in_dict[key])


def load_metric_file(filename):
    """Load the given hdf5 files name into a dictionary.

    Loads either HDF5 (recommended) or JSON (Depreciated in Future) save files.
    Guesses which type to load based off the file extension.

    Parameters
    ----------
    filename : str
        Full path to the filename of the metric to load.

    Returns
    -------
    metric_dict : dict
        Dictionary of metrics stored in the input file.

    """
    if filename.split('.')[-1] == 'json':
        warnings.warn("JSON-type files can still be read "
                      "but are no longer written by default.\n"
                      "Write to HDF5 format for future compatibility.",
                      PendingDeprecationWarning)
        metric_dict = _load_json_metrics(filename)
    elif filename.split('.')[-1] == 'pkl':
        warnings.warn("Pickle-type files can still be read "
                      "but are no longer written by default.\n"
                      "Write to HDF5 format for future compatibility.",
                      PendingDeprecationWarning)
        metric_dict = _load_pickle_metrics(filename)
    else:
        with h5py.File(filename, 'r') as infile:
            metric_dict = _recursively_load_dict_to_group(infile, "/Header/")
            metric_item = infile["/Metrics/"]
            if hasattr(metric_item, 'attrs'):
                if 'group_is_ordered' in metric_item.attrs:
                    group_is_ordered = metric_item.attrs["group_is_ordered"]
                else:
                    group_is_ordered = False
            else:
                group_is_ordered = False
            metric_dict.update(
                _recursively_load_dict_to_group(infile, "/Metrics/",
                                                group_is_ordered=group_is_ordered
                                                )
            )

    _recursively_validate_dict(metric_dict)
    return metric_dict


def process_ex_ants(ex_ants=None, metrics_files=[]):
    """Make a list of excluded antennas from command line argument.

    Parameters
    ----------
    ex_ants : str
        A comma-separated value list of excluded antennas as a single string.
    metrics_file : str or list of str
        A full path to file(s) readable by load_metric_file

    Returns
    -------
    xants : list of ints
        A list of antenna numbers to be excluded from analysis.
    """
    xants = set([])
    if ex_ants is not None:
        if ex_ants != '':
            for ant in ex_ants.split(','):
                try:
                    if int(ant) not in xants:
                        xants.add(int(ant))
                except ValueError:
                    raise AssertionError(
                        "ex_ants must be a comma-separated list of ints")
    if metrics_files is not None:
        if isinstance(metrics_files, str):
            metrics_files = [metrics_files]
        if len(metrics_files) > 0:
            for mf in metrics_files:
                try: # try to get data out quickly via h5py
                    with h5py.File(mf, 'r') as infile:
                        # load from an ant_metrics file
                        if 'xants' in infile['Metrics']:
                            # Just take the antenna number, flagging both polarizations
                            xants |= set([ant[0] for ant in infile['Metrics']['xants']])
                        # load from an auto_metrics file                            
                        elif 'ex_ants' in infile['Metrics'] and 'r2_ex_ants' in infile['Metrics']['ex_ants']:
                            # Auto metrics reports just antenna numbers
                            xants |= set(infile['Metrics']['ex_ants']['r2_ex_ants'])
                
                except: # fallback for the old JSON style
                    metrics = load_metric_file(mf)
                    # load from an ant_metrics file
                    if 'xants' in metrics:
                        for ant in metrics['xants']:
                            xants.add(int(ant[0]))  # Just take the antenna number, flagging both polarizations
                    # load from an auto_metrics file
                    elif 'ex_ants' in metrics and 'r2_ex_ants' in metrics['ex_ants']:
                        for ant in metrics['ex_ants']['r2_ex_ants']:
                            xants.add(int(ant))  # Auto metrics reports just antenna numbers
    return sorted(list(xants))


def read_a_priori_chan_flags(a_priori_flags_yaml, freqs=None):
    '''Parse an a priori flag YAML file for a priori channel flags.

    Parameters
    ----------
    a_priori_flags_yaml : str
        Path to YAML file with a priori channel and/or frequency flags
    freqs : ndarray, optional
        1D numpy array containing all frequencies in Hz, required if freq_flags is not empty in the YAML

    Returns
    -------
    a_priori_channel_flags : ndarray
        Numpy array of integer a priori channel index flags.
    '''
    apcf = []
    apf = yaml.safe_load(open(a_priori_flags_yaml, 'r'))

    # Load channel flags
    if 'channel_flags' in apf:
        for cf in apf['channel_flags']:
            # add single channel flag
            if type(cf) == int:
                apcf.append(cf)
            # add range of channel flags
            elif (type(cf) == list) and (len(cf) == 2) and isinstance(cf[0], int) and isinstance(cf[1], int):
                if cf[0] > cf[1]:
                    raise ValueError(f'Channel flag ranges must be increasing. {cf} is not.')
                apcf += list(range(cf[0], cf[1] + 1))
            else:
                raise TypeError(f'channel_flags entries must be integers or len-2 lists of integers. {cf} is not.')

    # Load frequency flags
    if 'freq_flags' in apf:
        # check for presense of freqs
        if (len(apf['freq_flags']) > 0) and (freqs is None):
            raise ValueError('If freq_flags is present in the YAML and not empty, freqs must be specified.')

        for ff in apf['freq_flags']:
            # validate each frequency pair
            if (type(ff) != list) or (len(ff) != 2):
                raise TypeError(f'freq_flags entires must be len-2 lists of floats. {ff} is not.')
            try:
                ff = [float(ff[0]), float(ff[1])]
            except ValueError:
                raise TypeError(f'Both entries in freq_flags = {ff} must be convertable to floats.')
            if ff[0] > ff[1]:
                raise ValueError(f'Frequency flags ranges must be increasing. {ff} is not.')

            # add flagged channels
            apcf += list(np.argwhere((freqs >= ff[0]) & (freqs <= ff[1])).flatten())

    # Return unique channel indices
    return np.array(sorted(set(apcf)))


def read_a_priori_int_flags(a_priori_flags_yaml, times=None, lsts=None):
    '''Parse an a priori flag YAML file for a priori integration flags.

    Parameters
    ----------
    a_priori_flags_yaml : str
        Path to YAML file with a priori JD, LST, or integration flags
    times : ndarray, optional
        1D numpy array containing all JDs in units of days, required if JD_flags is not empty in the YAML
    lsts : ndarray, optional
        1D numpy array containing all lsts in units of hours, required if LST_flags is not empty in the YAML

    Returns
    -------
    a_priori_int_flags : ndarray
        Numpy array of integer a priori integration index flags.
    '''
    apif = []
    apf = yaml.safe_load(open(a_priori_flags_yaml, 'r'))

    # Load integration flags
    if 'integration_flags' in apf:
        for intf in apf['integration_flags']:
            if type(intf) == int:
                apif.append(intf)
            elif (type(intf) == list) and (len(intf) == 2) and isinstance(intf[0], int) and isinstance(intf[1], int):
                if intf[0] > intf[1]:
                    raise ValueError(f'Integration flag ranges must be increasing. {intf} is not.')
                apif += list(range(intf[0], intf[1] + 1))
            else:
                raise TypeError(f'integration_flags entries must be integers or len-2 lists of integers. {intf} is not.')

    if (times is not None) and (lsts is not None) and (len(times) != len(lsts)):
        raise ValueError(f'Length of times ({len(times)}) != length of lsts ({len(lsts)}).')

    # Load time flags
    if 'JD_flags' in apf:
        # check that times exists
        if (len(apf['JD_flags']) > 0) and (times is None):
            raise ValueError('If JD_flags is present in the YAML and not empty, times must be specified.')

        for tf in apf['JD_flags']:
            # validate time flag ranges
            if (type(tf) != list) or (len(tf) != 2):
                raise TypeError(f'JD_flags entires must be len-2 lists of floats. {tf} is not.')
            try:
                tf = [float(tf[0]), float(tf[1])]
            except ValueError:
                raise TypeError(f'Both entries in JD_flags = {tf} must be convertable to floats.')
            if tf[0] > tf[1]:
                raise ValueError(f'JD flag ranges must be increasing. {tf} is not.')

            # add integration flag indices
            apif += list(np.argwhere((times >= tf[0]) & (times <= tf[1])).flatten())

    # Load LST flags
    if 'LST_flags' in apf:
        # Check that lsts exists and is valid
        if (len(apf['LST_flags']) > 0) and (lsts is None):
            raise ValueError('If LST_flags is present in the YAML and not empty, lsts must be specified.')
        elif not (np.all(lsts >= 0) and np.all(lsts <= 24)):
            raise ValueError(f'All lsts must be between 0 and 24. This is violated in lsts = {lsts}.')

        for lf in apf['LST_flags']:
            # validate LST flag ranges
            if (type(lf) != list) or (len(lf) != 2):
                raise TypeError(f'LST_flags entires must be len-2 lists of floats. {lf} is not.')
            try:
                lf = [float(lf[0]), float(lf[1])]
            except ValueError:
                raise TypeError(f'Both entries in JD_flags = {lf} must be convertable to floats.')
            if (lf[0] < 0) or (lf[0] > 24) or (lf[1] < 0) or (lf[1] > 24):
                raise ValueError(f'Both entries in LST_flags must be between 0 and 24 hours. {lf} is not.')

            # add integration flag indices
            if lf[0] <= lf[1]:  # normal LST range
                apif += list(np.argwhere((lsts >= lf[0]) & (lsts <= lf[1])).flatten())
            else:  # LST range that spans the 24-hour branch cut
                apif += list(np.argwhere((lsts >= lf[0]) | (lsts <= lf[1])).flatten())

    # Return unique frequency indices
    return np.array(sorted(set(apif)))


def read_a_priori_ant_flags(a_priori_flags_yaml, ant_indices_only=False, by_ant_pol=False, ant_pols=None):
    '''Parse an a priori flag YAML file for a priori antenna flags.

    Parameters
    ----------
    a_priori_flags_yaml : str
        Path to YAML file with a priori antenna flags
    ant_indices_only : bool
        If True, ignore polarizations and flag entire antennas when they appear, e.g. (1, 'Jee') --> 1.
    by_ant_pol : bool
        If True, expand all integer antenna indices into per-antpol entries using ant_pols
    ant_pols : list of str
        List of antenna polarizations strings e.g. 'Jee'. If not empty, strings in
        the YAML must be in here or an error is raised. Required if by_ant_pol is True.

    Returns
    -------
    a_priori_antenna_flags : list
         List of a priori antenna flags, either integers or ant-pol tuples e.g. (0, 'Jee')
    '''

    if ant_indices_only and by_ant_pol:
        raise ValueError("ant_indices_only and by_ant_pol can't both be True.")
    apaf = []
    apf = yaml.safe_load(open(a_priori_flags_yaml, 'r'))

    # Load antenna flags
    if 'ex_ants' in apf:
        for ant in apf['ex_ants']:
            # flag antenna number
            if type(ant) == int:
                apaf.append(ant)
            # flag single antpol
            elif (type(ant) == list) and (len(ant) == 2) and (type(ant[0]) == int) and (type(ant[1]) == str):
                # check that antpol string is valid if ant_pols is not empty
                if (ant_pols is not None) and (ant[1] not in ant_pols):
                    raise ValueError(f'{ant[1]} is not a valid ant_pol in {ant_pols}.')
                if ant_indices_only:
                    apaf.append(ant[0])
                else:
                    apaf += [tuple(ant)]
            else:
                raise TypeError(f'ex_ants entires must be integers or a list of one int and one str. {ant} is not.')

        # Expand all integer antenna flags into antpol pairs
        if by_ant_pol:
            if ant_pols is None:
                raise ValueError('If by_ant_pol is True, then ant_pols must be specified.')
            apapf = []
            for ant in apaf:
                if type(ant) == int:
                    apapf += [(ant, pol) for pol in ant_pols]
                else:  # then it's already and antpol tuple
                    apapf.append(ant)
            return sorted(set(apapf))

    return list(set(apaf))
