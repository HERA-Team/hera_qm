import unittest
from hera_qm import vis_metrics
import numpy as np
import hera_qm.tests as qmtest
from pyuvdata import UVData
from hera_qm.data import DATA_PATH
import os
import pyuvdata.tests as uvtest
import copy


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data = UVData()
        filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        self.data.read_miriad(filename)
        # massage the object to make it work with check_noise_variance
        self.data.select(antenna_nums=self.data.get_ants()[0:10])
        self.data.select(freq_chans=range(100))
        # Data file only has three times... need more.
        while self.data.Ntimes < 90:
            d2 = copy.deepcopy(self.data)
            d2.time_array += d2.time_array.max() + d2.integration_time / (24 * 3600)
            self.data += d2
        ntimes = self.data.Ntimes
        nchan = self.data.Nfreqs
        self.data1 = qmtest.noise(size=(ntimes, nchan))
        self.data2 = qmtest.noise(size=(ntimes, nchan))
        ant_dat = {}
        for i in self.data.get_ants():
            ant_dat[i] = qmtest.noise(size=(ntimes, nchan)) + 0.1 * self.data1
        for key in self.data.get_antpairpols():
            ind = self.data._key2inds(key)[0]
            self.data.data_array[ind, 0, :, 0] = ant_dat[key[0]] * ant_dat[key[1]].conj()

    def test_check_noise_variance(self):
        nos = vis_metrics.check_noise_variance(self.data)
        print(self.data.Ntimes, self.data.Nfreqs)
        for bl in self.data.get_antpairs():
            n = nos[bl + ('XX',)]
            self.assertEqual(n.shape, (self.data.Nfreqs - 1,))
            nsamp = self.data.channel_width * self.data.integration_time
            np.testing.assert_almost_equal(n, np.ones_like(n) * nsamp, -np.log10(nsamp))


if __name__ == '__main__':
    unittest.main()
