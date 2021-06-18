# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import numpy as np

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
    Makes calls to nt.assert_type_equals and np.allclose to compare values.
    """
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        if isinstance(d1[key], (list)):
            assert d1[key] == list(d2[key]), ("key: {key} has type {key1_type} in d1 and {key2_type} in d2\n"
                                              "d1:  data has type {data1_type} and value {data1_val}\n"
                                              "d2:  data has type {data2_type} and value {data2_val}\n"
                                              .format(key=key,
                                                      key1_type=type(d1[key]),
                                                      key2_type=type(d2[key]),
                                                      data1_type=type(d1[key][0]),
                                                      data1_val=d1[key],
                                                      data2_type=type(d2[key][0]),
                                                      data2_val=d2[key]
                                                      )
                                              )

        elif isinstance(d1[key], (np.ndarray)):
            if np.issubdtype(d1[key].dtype, np.string_):
                assert np.array_equal(d1[key], np.asarray(d2[key]))
            else:
                assert np.allclose(d1[key], np.asarray(d2[key]), equal_nan=True), ("key: {key} has type {key1_type} in d1 and {key2_type} in d2\n"
                                                                                   "d1:  data has type {data1_type} and value {data1_val}\n"
                                                                                   "d2:  data has type {data2_type} and value {data2_val}\n"
                                                                                   .format(key=key,
                                                                                           key1_type=type(d1[key]),
                                                                                           key2_type=type(d2[key]),
                                                                                           data1_type=type(d1[key][0]),
                                                                                           data1_val=d1[key],
                                                                                           data2_type=type(d2[key][0]),
                                                                                           data2_val=d2[key]
                                                                                           )
                                                                                   )
        elif isinstance(d1[key], dict):
            recursive_compare_dicts(d1[key], d2[key])
        elif isinstance(d1[key], (float, np.float32, np.float64)):
            assert np.allclose(d1[key], d2[key], equal_nan=True), ("key: {key} has type {key1_type} in d1 and {key2_type} in d2\n"
                                                                   "d1:  data has type {data1_type} and value {data1_val}\n"
                                                                   "d2:  data has type {data2_type} and value {data2_val}\n"
                                                                   .format(key=key,
                                                                           key1_type=type(d1[key]),
                                                                           key2_type=type(d2[key]),
                                                                           data1_type=type(d1[key][0]),
                                                                           data1_val=d1[key],
                                                                           data2_type=type(d2[key][0]),
                                                                           data2_val=d2[key]
                                                                           )
                                                                   )
        else:
            assert d1[key] == d2[key], ("key: {key} has type {key1_type} in d1 and {key2_type} in d2\n"
                                        "d1:  data has type {data1_type} and value {data1_val}\n"
                                        "d2:  data has type {data2_type} and value {data2_val}\n"
                                        .format(key=key,
                                                key1_type=type(d1[key]),
                                                key2_type=type(d2[key]),
                                                data1_type=type(d1[key][0]),
                                                data1_val=d1[key],
                                                data2_type=type(d2[key][0]),
                                                data2_val=d2[key]
                                                )
                                        )
    return True
