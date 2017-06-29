import numpy as np
from hera_qm.datacontainer import DataContainer

def check_noise_variance(data, wgts, bandwidth, inttime):
    """XXX"""
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
