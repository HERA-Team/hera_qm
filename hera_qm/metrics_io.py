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
