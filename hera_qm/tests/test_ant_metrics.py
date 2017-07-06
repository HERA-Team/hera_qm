import unittest
from hera_qm import ant_metrics
import numpy as np
from hera_qm.data import DATA_PATH

class fake_data():
    
    def __init__(self):
        self.data = {}
        for bl in [(0,1), (1,2), (0,2)]:
            self.data[bl] = {}
            for pol in ['xx','xy','yx','yy']:
                self.data[bl][pol] = np.ones((2,3),dtype=complex)

    def get_data(self,i,j,pol):
        return self.data[(i,j)][pol]


class Test_Low_Level_Functions(unittest.TestCase):

    def setUp(self):
        self.data = fake_data()
        self.ants = [0,1,2]
        self.reds = [[(0,1),(1,2)],[(0,2)]]
        self.pols = ['xx','xy','yx','yy']
        self.antpols = ['x','y']
        self.bls = [(0,1), (1,2), (0,2)]

    def test_mean_Vij_metrics(self):
        mean_Vij = ant_metrics.mean_Vij_metrics(self.data, self.pols, self.antpols, 
                   self.ants, self.bls, rawMetric=True)
        self.assertEqual(mean_Vij,{(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0, 
                                   (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_red_corr_metrics(self):
        red_corr = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols, 
                   self.ants, self.reds, rawMetric=True)
        self.assertEqual(red_corr,{(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0, 
                                   (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_mean_Vij_cross_pol_metrics(self):
        mean_Vij_cross_pol = ant_metrics.mean_Vij_cross_pol_metrics(self.data, self.pols, 
                             self.antpols, self.ants, self.bls, rawMetric=True)
        self.assertEqual(mean_Vij_cross_pol,{(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0, 
                                             (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_red_corr_cross_pol_metrics(self):
        red_corr_cross_pol = ant_metrics.red_corr_cross_pol_metrics(self.data, self.pols, 
                             self.antpols, self.ants, self.reds, rawMetric=True)
        self.assertEqual(red_corr_cross_pol,{(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0, 
                                             (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})


    def test_per_antenna_modified_z_scores(self):
        metric = {(0,'x'): 1, (50,'x'): 0, (2,'x'): 2, (2,'y'):2000, (0,'y'):-300}
        zscores = ant_metrics.per_antenna_modified_z_scores(metric)
        np.testing.assert_almost_equal(zscores[0,'x'], 0, 10)
        np.testing.assert_almost_equal(zscores[50,'x'], -0.6745, 10)
        np.testing.assert_almost_equal(zscores[2,'x'], 0.6745, 10)

    def test_exclude_partially_excluded_ants(self):
        before_xants = [(0,'x'),(0,'y'),(1,'x'),(2,'y')]
        after_xants = ant_metrics.exclude_partially_excluded_ants(['x','y'],before_xants)
        after_xants_truth = [(0,'x'),(0,'y'),(1,'x'),(1,'y'),(2,'x'),(2,'y')]
        self.assertEqual(set(after_xants), set(after_xants_truth))

    def test_average_abs_metrics(self):
        metric1 = {(0,'x'): 1.0, (0,'y'): 2.0}
        metric2 = {(0,'x'): 3.0, (0,'y'): -4.0}
        metricAbsAvg = ant_metrics.average_abs_metrics(metric1,metric2)
        self.assertAlmostEqual(2.0, metricAbsAvg[(0,'x')],10)
        self.assertAlmostEqual(3.0, metricAbsAvg[(0,'y')],10)
        metric3 = {(0,'x'): 1}
        with self.assertRaises(KeyError):
            ant_metrics.average_abs_metrics(metric1,metric3)

class Test_Antenna_Metrics(unittest.TestCase):

    def setUp(self):
        dataFileList = [DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA',
                        DATA_PATH + '/zen.2457698.40355.xy.HH.uvcA',
                        DATA_PATH + '/zen.2457698.40355.yx.HH.uvcA',
                        DATA_PATH + '/zen.2457698.40355.yy.HH.uvcA']
        reds = [[(9, 31), (20, 65), (22, 89), (53, 96), (64, 104), (72, 81), (112, 10), 
                 (105, 20), (81, 43), (88, 53)], [(65, 72), (96, 105)], [(31, 105), 
                 (43, 112), (96, 9), (65, 22), (89, 72), (104, 88)], [(20, 72), (43, 97), 
                 (53, 105), (65, 81), (80, 88), (89, 112), (104, 9), (96, 20), (31, 22)], 
                 [(22, 96), (72, 31), (112, 65), (105, 104)], [(9, 105), (10, 97), (20, 22), 
                 (22, 72), (64, 88), (65, 89), (81, 112), (53, 9), (43, 10), (31, 20), (96, 31), 
                 (104, 53), (80, 64), (89, 81)], [(96, 97), (104, 112), (80, 72)], [(10, 72), 
                 (31, 88), (89, 105), (65, 9), (43, 22), (96, 64)], [(10, 105), (43, 9), (65, 64), 
                 (89, 88)], [(9, 20), (20, 89), (22, 81), (31, 65), (72, 112), (80, 104), (88, 9), 
                 (81, 10), (105, 22), (53, 31), (89, 43), (64, 53), (104, 96), (112, 97)], 
                 [(31, 112), (53, 72), (65, 97), (80, 105), (104, 22), (96, 81)], [(72, 104), 
                 (112, 96)], [(64, 97), (80, 10)], [(10, 64), (43, 80), (97, 88)], [(9, 80), 
                 (10, 65), (20, 104), (22, 53), (89, 96), (72, 9), (112, 20), (81, 31), (105, 64), 
                 (97, 89)], [(80, 112), (104, 97)], [(43, 105), (65, 88)], [(10, 22), (20, 88), 
                 (31, 64), (81, 105), (89, 9), (43, 20), (65, 53), (97, 72), (96, 80)], [(43, 53), 
                 (65, 80), (81, 88), (97, 105), (10, 9), (89, 64)], [(53, 97), (64, 112), (80, 81), 
                 (104, 10)], [(9, 64), (10, 89), (20, 53), (31, 104), (43, 65), (53, 80), (65, 96), 
                 (72, 105), (22, 9), (81, 20), (112, 22), (89, 31), (97, 81), (105, 88)], [(9, 112), 
                 (20, 97), (53, 81), (31, 10), (80, 20), (64, 22), (96, 43), (88, 72), (104, 89)], 
                 [(9, 81), (22, 97), (31, 43), (53, 89), (105, 112), (20, 10), (64, 20), (88, 22), 
                 (80, 31), (104, 65)], [(43, 72), (65, 105), (96, 88)], [(31, 97), (53, 112), 
                 (64, 72), (96, 10), (80, 22), (104, 81)], [(10, 88), (43, 64)], [(9, 97), (64, 81), 
                 (80, 89), (88, 112), (53, 10), (104, 43)]]
        self.am = ant_metrics.Antenna_Metrics(dataFileList, reds, fileformat='miriad')
        with self.assertRaises(KeyError):
            self.am.save_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')
        self.am.iterative_antenna_metrics_and_flagging()
        # internal names for summary statistics
        self.summaryStats = ['xants', 'crossedAntsRemoved', 'deadAntsRemoved', 'removalIter', 
                             'finalMetrics', 'allMetrics', 'finalModzScores', 'allModzScores',
                             'crossCut', 'deadCut']

    def test_nPos(self):
        with self.assertRaises(ValueError):
            am2 = ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'], [], fileformat='miriad')
        with self.assertRaises(IOError):
            am2 = ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'], [], fileformat='uvfits')
        with self.assertRaises(ValueError):
            am2 = ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'], [], fileformat='not_a_format')

    def test_init(self):
        self.assertEqual(len(self.am.ants),19)
        self.assertEqual(set(self.am.pols), set(['xx','yy','xy','yx']))
        self.assertEqual(set(self.am.antpols), set(['x','y']))
        self.assertEqual(len(self.am.bls),19*18/2 + 19)        
        self.assertEqual(len(self.am.reds),27)

    def test_iterative_antenna_metrics_and_flagging(self):
        for stat in self.summaryStats:
            self.assertTrue(hasattr(self.am, stat))
        self.assertIn((81,'x'), self.am.xants)
        self.assertIn((81,'y'), self.am.xants)

    def test_save_and_load_antenna_metrics(self):
        self.am.save_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')
        loaded = ant_metrics.load_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')
        # json names for summary statistics
        jsonStats = ['xants', 'ants_removed_as_crossed', 'ants_removed_as_dead', 'removal_iteration',
                     'final_metrics', 'all_metrics', 'final_mod_z_scores', 'all_mod_z_scores', 
                     'cross_pol_z_cut', 'dead_ant_z_cut']
        for stat, jsonStat in zip(self.summaryStats, jsonStats):
            self.assertEqual(loaded[jsonStat], getattr(self.am, stat))

if __name__ == '__main__':
    unittest.main()
