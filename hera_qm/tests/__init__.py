# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

import numpy as np
import nose.tools as nt
np.random.seed(0)


def noise(size):
    """Generage complex Gaussian Noise with amplitude 1."""
    sig = 1. / np.sqrt(2)
    return (np.random.normal(scale=sig, size=size)
            + 1j * np.random.normal(scale=sig, size=size))


def real_noise(size):
    """Generate real Gaussian noise with amplitude 1."""
    sig = 1.
    return np.random.normal(scale=sig, size=size)


def recursive_compare_dicts(d1, d2):
    """Recursively compare dictionaries.

    Keys of each dict must match.
    Walks through two input dicts and compares each key.
    Makes calls to nt.asser_type_equals and np.allclose to compare values.
    """
    nt.assert_equal(set(d1.keys()), set(d2.keys()))
    for key in d1:
        if isinstance(d1[key], (list)):
            try:
                nt.assert_list_equal(d1[key], list(d2[key]))
            except (AssertionError, TypeError) as err:
                print('d1:', key, 'key value type:', type(d1[key]),
                      'data type:', type(d1[key][0]), d1[key])
                print('d2:', key, 'key value type', type(d1[key]),
                      'data type:', type(d2[key][0]), d2[key])
                raise err
        elif isinstance(d1[key], (np.ndarray)):
            if np.issubdtype(d1[key].dtype, np.string_):
                nt.assert_true(np.array_equal(d1[key], np.asarray(d2[key])))
            else:
                try:
                    nt.assert_true(np.allclose(d1[key], np.asarray(d2[key])))
                except (AssertionError, TypeError) as err:
                    print('d1:', key, 'key value type:', type(d1[key]),
                          'data type:', type(d1[key][0]), d1[key])
                    print('d2:', key, 'key value type', type(d1[key]),
                          'data type:', type(d2[key][0]), d2[key])
                    raise err
        elif isinstance(d1[key], dict):
            recursive_compare_dicts(d1[key], d2[key])
        elif isinstance(d1[key], (float, np.float, np.float32)):
            nt.assert_true(np.allclose(d1[key], d2[key]))
        else:
            nt.assert_equal(d1[key], d2[key])
