import unittest
from hera_qm import vis_metrics
import numpy as np
import hera_qm.tests as qmtest
from pyuvdata import UVData
from hera_qm.data import DATA_PATH
import os
import pyuvdata.tests as uvtest


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data = UVData()
        filename = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        uvtest.checkWarnings(self.data.read_miriad, [filename], nwarnings=1,
                             message='antenna_diameters is not set')

    def test_check_noise_variance(self):
        nos = vis_metrics.check_noise_variance(self.data)
        for bl in self.data.get_antpairs():
            n = nos[bl + ('XX',)]
            self.assertEqual(n.shape, (self.data.Nfreqs - 1,))


if __name__ == '__main__':
    unittest.main()
