import unittest
from hera_qm import vis_metrics
import numpy as np

np.random.seed(0)


def noise(size):
    sig = 1. / np.sqrt(2)
    return np.random.normal(scale=sig, size=size) + 1j * np.random.normal(scale=sig, size=size)


class TestMethods(unittest.TestCase):

    def setUp(self):
        self.data1 = noise(size=(100, 100))
        self.data2 = noise(size=(100, 100))

    def test_check_ants(self):
        reds = [[(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
                [(1, 3), (2, 4), (3, 5), (4, 6)],
                [(1, 4), (2, 5), (3, 6)], [(1, 5), (2, 6)]]
        data = {'xx': {}}
        for bl in reduce(lambda x, y: x + y, reds):
            data['xx'][bl] = self.data1
        cnts = vis_metrics.check_ants(reds, data)
        for i in [1, 2, 3, 4, 5]:
            self.assertEqual(cnts[(i, 'xx')], 0)
        for bl in data['xx']:
            if 3 in bl:
                data['xx'][bl] = self.data2
        cnts = vis_metrics.check_ants(reds, data)
        for i in [1, 2, 4, 5]:
            self.assertLess(cnts[(i, 'xx')], 3)
        self.assertGreater(cnts[(3, 'xx')], 3)
        for bl in data['xx']:
            if 3 in bl:
                data['xx'][bl] = .8 * self.data1 + .2 * self.data2
        cnts = vis_metrics.check_ants(reds, data, flag_thresh=.3)
        for i in [1, 2, 3, 4, 5]:
            self.assertEqual(cnts[(i, 'xx')], 0)
        for bl in data['xx']:
            if 3 in bl:
                data['xx'][bl] = .8 * self.data2 + .2 * self.data1
        cnts = vis_metrics.check_ants(reds, data, flag_thresh=.3)
        for i in [1, 2, 4, 5]:
            self.assertLess(cnts[(i, 'xx')], 3)
        self.assertGreater(cnts[(3, 'xx')], 3)

    def test_check_noise_variance(self):
        data = {'xx': {}}
        wgts = {'xx': {}}
        ants = range(10)
        ant_dat = {}
        for i in ants:
            ant_dat[i] = noise(size=(100, 100)) + .1 * self.data1
        for i, ai in enumerate(ants):
            for j in ants[i:]:
                data['xx'][(i, j)] = ant_dat[i] * ant_dat[j].conj()
                wgts['xx'][(i, j)] = np.ones((100, 100), dtype=float)
        nos = vis_metrics.check_noise_variance(data, wgts, 1., 1.)
        for bl in data['xx']:
            n = nos[bl + ('xx',)]
            self.assertEqual(n.shape, (100 - 1,))
            np.testing.assert_almost_equal(n, np.ones_like(n), 0)
        nos = vis_metrics.check_noise_variance(data, wgts, 1., 10.)
        for bl in data['xx']:
            n = nos[bl + ('xx',)]
            self.assertEqual(n.shape, (100 - 1,))
            np.testing.assert_almost_equal(n, 10 * np.ones_like(n), -1)
        nos = vis_metrics.check_noise_variance(data, wgts, 10., 10.)
        for bl in data['xx']:
            n = nos[bl + ('xx',)]
            self.assertEqual(n.shape, (100 - 1,))
            np.testing.assert_almost_equal(n, 106 * np.ones_like(n), -2)


if __name__ == '__main__':
    unittest.main()