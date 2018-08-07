# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

import numpy as np


def check_noise_variance(data):
    """Calculate the noise levels of each baseline/pol relative to the autos.

    Calculates the noise for each baseline/pol by differencing along frequency
    dimension and compares to the noise on the auto-spectra for each
    antenna in the baseline.

    Args:
        data (UVData): UVData object with data.

    Returns:
        Cij (dict): dictionary of variance measurements
                    has keywords of (ant1, ant2, pol)
    """
    Cij = {}
    for key, d in data.antpairpol_iter():
        w = data.get_nsamples(key)
        bl = (key[0], key[1])
        ai = data.get_data((key[0], key[0], key[2])).real
        aj = data.get_data((key[1], key[1], key[2])).real
        ww = w[1:, 1:] * w[1:, :-1] * w[:-1, 1:] * w[:-1, :-1]
        dd = (((d[:-1, :-1] - d[:-1, 1:]) - (d[1:, :-1] - d[1:, 1:])) * ww
              / np.sqrt(4))
        dai = (((ai[:-1, :-1] + ai[:-1, 1:]) + (ai[1:, :-1] + ai[1:, 1:])) * ww
               / 4)
        daj = (((aj[:-1, :-1] + aj[:-1, 1:]) + (aj[1:, :-1] + aj[1:, 1:])) * ww
               / 4)
        Cij[key] = (np.sum(np.abs(dd)**2, axis=0) / np.sum(dai * daj, axis=0) *
                    (data.channel_width * data.integration_time))
    return Cij
