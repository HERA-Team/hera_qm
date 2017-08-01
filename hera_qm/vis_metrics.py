import numpy as np
from hera_qm.datacontainer import DataContainer


def check_noise_variance(data, wgts, bandwidth, inttime):
    '''Function to calculate the noise levels of each baseline/pol combination,
    relative to the noise on the autos.

    Args:
        data (dict): dictionary of visibilities with keywords of pol/ant pair
            in any order.
        wgts (dict): dictionary of weights with keywords of pol/ant pair in any order
        bandwidth (float): Channel width, preferably Hz, but only restriction is
            bandwidth units are recipricol inttime units.
        inttime (float): Integration time, preferably seconds, but only restriction
            is inttime units are recipricol bandwidth units.

    Returns:
        Cij (dict): dictionary of variance measurements with keywords of ant pair/pol
    '''
    dc = DataContainer(data)
    wc = DataContainer(wgts)
    Cij = {}
    for pol in dc.pols():
        for bl in dc.bls(pol):
            i, j = bl
            d = dc.get(bl, pol)
            w = wc.get(bl, pol)
            ai, aj = dc.get((i, i), pol).real, dc.get((j, j), pol).real
            ww = w[1:, 1:] * w[1:, :-1] * w[:-1, 1:] * w[:-1, :-1]
            dd = ((d[:-1, :-1] - d[:-1, 1:]) - (d[1:, :-1] - d[1:, 1:])) * ww / np.sqrt(4)
            dai = ((ai[:-1, :-1] + ai[:-1, 1:]) + (ai[1:, :-1] + ai[1:, 1:])) * ww / 4
            daj = ((aj[:-1, :-1] + aj[:-1, 1:]) + (aj[1:, :-1] + aj[1:, 1:])) * ww / 4
            Cij[bl + (pol,)] = np.sum(np.abs(dd)**2, axis=0) / np.sum(dai * daj, axis=0) * (bandwidth * inttime)
    return Cij
