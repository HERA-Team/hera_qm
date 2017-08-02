import unittest
import nose.tools as nt
from hera_qm import ant_metrics
import numpy as np
from hera_qm.data import DATA_PATH
import pyuvdata.tests as uvtest
from hera_qm import utils
import os
import sys


class fake_data():

    def __init__(self):
        self.data = {}
        for bl in [(0, 1), (1, 2), (0, 2)]:
            self.data[bl] = {}
            for pol in ['xx', 'xy', 'yx', 'yy']:
                self.data[bl][pol] = np.ones((2, 3), dtype=complex)

    def get_data(self, i, j, pol):
        return self.data[(i, j)][pol]


class TestLowLevelFunctions(unittest.TestCase):

    def setUp(self):
        self.data = fake_data()
        self.ants = [0, 1, 2]
        self.reds = [[(0, 1), (1, 2)], [(0, 2)]]
        self.pols = ['xx', 'xy', 'yx', 'yy']
        self.antpols = ['x', 'y']
        self.bls = [(0, 1), (1, 2), (0, 2)]

    def test_mean_Vij_metrics(self):
        mean_Vij = ant_metrics.mean_Vij_metrics(self.data, self.pols, self.antpols,
                                                self.ants, self.bls, rawMetric=True)
        self.assertEqual(mean_Vij, {(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0,
                                    (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_red_corr_metrics(self):
        red_corr = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols,
                                                self.ants, self.reds, rawMetric=True)
        self.assertEqual(red_corr, {(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0,
                                    (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_mean_Vij_cross_pol_metrics(self):
        mean_Vij_cross_pol = ant_metrics.mean_Vij_cross_pol_metrics(self.data, self.pols,
                                                                    self.antpols, self.ants,
                                                                    self.bls, rawMetric=True)
        self.assertEqual(mean_Vij_cross_pol, {(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0,
                                              (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_red_corr_cross_pol_metrics(self):
        red_corr_cross_pol = ant_metrics.red_corr_cross_pol_metrics(self.data, self.pols,
                                                                    self.antpols, self.ants,
                                                                    self.reds, rawMetric=True)
        self.assertEqual(red_corr_cross_pol, {(1, 'y'): 1.0, (1, 'x'): 1.0, (0, 'x'): 1.0,
                                              (0, 'y'): 1.0, (2, 'x'): 1.0, (2, 'y'): 1.0})

    def test_per_antenna_modified_z_scores(self):
        metric = {(0, 'x'): 1, (50, 'x'): 0, (2, 'x'): 2, (2, 'y'): 2000, (0, 'y'): -300}
        zscores = ant_metrics.per_antenna_modified_z_scores(metric)
        np.testing.assert_almost_equal(zscores[0, 'x'], 0, 10)
        np.testing.assert_almost_equal(zscores[50, 'x'], -0.6745, 10)
        np.testing.assert_almost_equal(zscores[2, 'x'], 0.6745, 10)

    def test_exclude_partially_excluded_ants(self):
        before_xants = [(0, 'x'), (0, 'y'), (1, 'x'), (2, 'y')]
        after_xants = ant_metrics.exclude_partially_excluded_ants(['x', 'y'], before_xants)
        after_xants_truth = [(0, 'x'), (0, 'y'), (1, 'x'), (1, 'y'), (2, 'x'), (2, 'y')]
        self.assertEqual(set(after_xants), set(after_xants_truth))

    def test_antpol_metric_sum_ratio(self):
        crossMetrics = {(0, 'x'): 1.0, (0, 'y'): 1.0, (1, 'x'): 1.0}
        sameMetrics = {(0, 'x'): 2.0, (0, 'y'): 2.0}
        xants = [(1, 'y')]
        crossPolRatio = ant_metrics.antpol_metric_sum_ratio([0, 1], ['x', 'y'],
                                                            crossMetrics, sameMetrics,
                                                            xants=xants)
        self.assertEqual(crossPolRatio, {(0, 'x'): .5, (0, 'y'): .5})

    def test_average_abs_metrics(self):
        metric1 = {(0, 'x'): 1.0, (0, 'y'): 2.0}
        metric2 = {(0, 'x'): 3.0, (0, 'y'): -4.0}
        metricAbsAvg = ant_metrics.average_abs_metrics(metric1, metric2)
        self.assertAlmostEqual(2.0, metricAbsAvg[(0, 'x')], 10)
        self.assertAlmostEqual(3.0, metricAbsAvg[(0, 'y')], 10)
        metric3 = {(0, 'x'): 1}
        with self.assertRaises(KeyError):
            ant_metrics.average_abs_metrics(metric1, metric3)


class TestAntennaMetrics(unittest.TestCase):

    def setUp(self):
        self.dataFileList = [DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA',
                             DATA_PATH + '/zen.2457698.40355.yy.HH.uvcA',
                             DATA_PATH + '/zen.2457698.40355.xy.HH.uvcA',
                             DATA_PATH + '/zen.2457698.40355.yx.HH.uvcA']
        if not os.path.exists(DATA_PATH + '/test_output/'):
            os.makedirs(DATA_PATH + '/test_output/')
        self.reds = [[(9, 31), (20, 65), (22, 89), (53, 96), (64, 104), (72, 81),
                      (112, 10), (105, 20), (81, 43), (88, 53)], [(65, 72), (96, 105)],
                     [(31, 105), (43, 112), (96, 9), (65, 22), (89, 72), (104, 88)],
                     [(20, 72), (43, 97), (53, 105), (65, 81), (80, 88), (89, 112),
                      (104, 9), (96, 20), (31, 22)],
                     [(22, 96), (72, 31), (112, 65), (105, 104)],
                     [(9, 105), (10, 97), (20, 22), (22, 72), (64, 88), (65, 89),
                      (81, 112), (53, 9), (43, 10), (31, 20), (96, 31), (104, 53),
                      (80, 64), (89, 81)], [(96, 97), (104, 112), (80, 72)],
                     [(10, 72), (31, 88), (89, 105), (65, 9), (43, 22), (96, 64)],
                     [(10, 105), (43, 9), (65, 64), (89, 88)],
                     [(9, 20), (20, 89), (22, 81), (31, 65), (72, 112), (80, 104),
                      (88, 9), (81, 10), (105, 22), (53, 31), (89, 43), (64, 53),
                      (104, 96), (112, 97)],
                     [(31, 112), (53, 72), (65, 97), (80, 105), (104, 22), (96, 81)],
                     [(72, 104), (112, 96)], [(64, 97), (80, 10)],
                     [(10, 64), (43, 80), (97, 88)],
                     [(9, 80), (10, 65), (20, 104), (22, 53), (89, 96), (72, 9),
                      (112, 20), (81, 31), (105, 64), (97, 89)],
                     [(80, 112), (104, 97)], [(43, 105), (65, 88)],
                     [(10, 22), (20, 88), (31, 64), (81, 105), (89, 9), (43, 20),
                      (65, 53), (97, 72), (96, 80)],
                     [(43, 53), (65, 80), (81, 88), (97, 105), (10, 9), (89, 64)],
                     [(53, 97), (64, 112), (80, 81), (104, 10)],
                     [(9, 64), (10, 89), (20, 53), (31, 104), (43, 65), (53, 80),
                      (65, 96), (72, 105), (22, 9), (81, 20), (112, 22), (89, 31),
                      (97, 81), (105, 88)], [(9, 112), (20, 97), (53, 81), (31, 10),
                                             (80, 20), (64, 22), (96, 43), (88, 72), (104, 89)],
                     [(9, 81), (22, 97), (31, 43), (53, 89), (105, 112), (20, 10),
                      (64, 20), (88, 22), (80, 31), (104, 65)],
                     [(43, 72), (65, 105), (96, 88)],
                     [(31, 97), (53, 112), (64, 72), (96, 10), (80, 22), (104, 81)],
                     [(10, 88), (43, 64)], [(9, 97), (64, 81), (80, 89), (88, 112),
                                            (53, 10), (104, 43)]]
        # internal names for summary statistics
        self.summaryStats = ['xants', 'crossedAntsRemoved', 'deadAntsRemoved', 'removalIter',
                             'finalMetrics', 'allMetrics', 'finalModzScores', 'allModzScores',
                             'crossCut', 'deadCut', 'dataFileList', 'reds']

    def test_load_errors(self):
        with self.assertRaises(ValueError):
            uvtest.checkWarnings(ant_metrics.Antenna_Metrics,
                                 [[DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'], []],
                                 {"fileformat": 'miriad'}, nwarnings=1,
                                 message='antenna_diameters is not set')
        with self.assertRaises(IOError):
            ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                        [], fileformat='uvfits')
        with self.assertRaises(StandardError):
            ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                        [], fileformat='fhd')
        with self.assertRaises(ValueError):
            ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
                                        [], fileformat='not_a_format')
        # if casacore is not installed, this test raises a skiptest
        # with self.assertRaises(IOError):
        #     ant_metrics.Antenna_Metrics([DATA_PATH + '/zen.2457698.40355.xx.HH.uvcA'],
        #                                 [], fileformat='ms')

    def test_init(self):
        # We will throw 4 warnings for unknown antenna diameters.
        # All warnings are UserWarnings
        messages = (['antenna_diameters is not set'] * 4)
        categories = ([UserWarning] * 4)
        am = uvtest.checkWarnings(ant_metrics.Antenna_Metrics, [self.dataFileList, self.reds],
                                  {"fileformat": 'miriad'}, nwarnings=4, message=messages,
                                  category=categories)
        self.assertEqual(len(am.ants), 19)
        self.assertEqual(set(am.pols), set(['xx', 'yy', 'xy', 'yx']))
        self.assertEqual(set(am.antpols), set(['x', 'y']))
        self.assertEqual(len(am.bls), 19 * 18 / 2 + 19)
        self.assertEqual(len(am.reds), 27)

    def test_iterative_antenna_metrics_and_flagging_and_saving_and_loading(self):
        # We will throw 4 warnings for unknown antenna diameters.
        # All warnings are UserWarnings
        messages = (['antenna_diameters is not set'] * 4)
        categories = ([UserWarning] * 4)
        am = uvtest.checkWarnings(ant_metrics.Antenna_Metrics, [self.dataFileList, self.reds],
                                  {"fileformat": 'miriad'}, nwarnings=4, message=messages,
                                  category=categories)
        with self.assertRaises(KeyError):
            am.save_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')

        am.iterative_antenna_metrics_and_flagging()
        for stat in self.summaryStats:
            self.assertTrue(hasattr(am, stat))
        self.assertIn((81, 'x'), am.xants)
        self.assertIn((81, 'y'), am.xants)
        self.assertIn((81, 'x'), am.deadAntsRemoved)
        self.assertIn((81, 'y'), am.deadAntsRemoved)

        am.save_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')
        loaded = ant_metrics.load_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')
        # json names for summary statistics
        jsonStats = ['xants', 'crossed_ants', 'dead_ants', 'removal_iteration',
                     'final_metrics', 'all_metrics', 'final_mod_z_scores', 'all_mod_z_scores',
                     'cross_pol_z_cut', 'dead_ant_z_cut', 'datafile_list', 'reds', 'version']
        for stat, jsonStat in zip(self.summaryStats, jsonStats):
            self.assertEqual(loaded[jsonStat], getattr(am, stat))

    def test_cross_detection(self):
        # We will throw 4 warnings for unknown antenna diameters.
        # All warnings are UserWarnings
        messages = (['antenna_diameters is not set'] * 4)
        categories = ([UserWarning] * 4)
        am2 = uvtest.checkWarnings(ant_metrics.Antenna_Metrics, [
            self.dataFileList, self.reds], {"fileformat": 'miriad'}, nwarnings=4,
                                   message=messages, category=categories)
        am2.iterative_antenna_metrics_and_flagging(crossCut=3, deadCut=10)
        for stat in self.summaryStats:
            self.assertTrue(hasattr(am2, stat))
        self.assertIn((81, 'x'), am2.xants)
        self.assertIn((81, 'y'), am2.xants)
        self.assertIn((81, 'x'), am2.crossedAntsRemoved)
        self.assertIn((81, 'y'), am2.crossedAntsRemoved)

class TestAntmetricsRun(object):
    def test_ant_metrics_run(self):
        # get options object
        o = utils.get_metrics_OptionParser('ant_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        calfile = 'heratest_calfile'
        opt0 = "-C {}".format(calfile)
        opt1 = "-p xx,yy,xy,yx"
        opt2 = "--crossCut=5"
        opt3 = "--deadCut=5"
        opt4 = "--extension=.ant_metrics.json"
        opt5 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        opt6 = "--vis_format=miriad"
        options = ' '.join([opt0, opt1, opt2, opt3, opt4, opt5, opt6])

        # test running with no files
        cmd = ' '.join([options, ''])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, ant_metrics.ant_metrics_run, args, opts, history)

        # test running with a lone file
        lone_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([options, lone_file])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        # this test raises a warning, then fails...
        uvtest.checkWarnings(nt.assert_raises, [
            AssertionError, ant_metrics.ant_metrics_run, args, opts, history], nwarnings=1,
                             message='Could not find')

        # test actually running metrics
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
        dest_file = os.path.join(DATA_PATH, 'test_output',
                                 'zen.2457698.40355.HH.uvcA.ant_metrics.json')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([options, xx_file])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        # We will throw 4 warnings for unknown antenna diameters.
        # All warnings are UserWarnings
        messages = (['antenna_diameters is not set'] * 4)
        categories = ([UserWarning] * 4)
        uvtest.checkWarnings(ant_metrics.ant_metrics_run, [args, opts, history],
                             nwarnings=4, message=messages, category=categories)
        nt.assert_true(os.path.exists(dest_file))

if __name__ == '__main__':
    unittest.main()
