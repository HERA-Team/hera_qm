import unittest
import nose.tools as nt
from hera_qm import ant_metrics
import numpy as np
from hera_qm.data import DATA_PATH
import pyuvdata.tests as uvtest
from hera_qm import utils
import os
import sys
import json
import copy


class fake_data():

    def __init__(self):
        self.data = {}
        for bl in [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]:
            self.data[bl] = {}
            for poli, pol in enumerate(['xx', 'xy', 'yx', 'yy']):
                np.random.seed(bl[0] * 10 + bl[1] + 100 * poli)  # Give each bl different data
                self.data[bl][pol] = np.random.randn(2, 3)

    def get_data(self, i, j, pol):
        return self.data[(i, j)][pol]


class TestLowLevelFunctions(unittest.TestCase):

    def setUp(self):
        self.data = fake_data()
        self.ants = [0, 1, 2, 3]
        self.reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3)], [(0, 3)]]
        self.pols = ['xx', 'xy', 'yx', 'yy']
        self.antpols = ['x', 'y']
        self.bls = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]

    def test_mean_Vij_metrics(self):
        mean_Vij = ant_metrics.mean_Vij_metrics(self.data, self.pols, self.antpols,
                                                self.ants, self.bls, rawMetric=True)
        # The reference dictionaries here and in other functions were determined
        # by running the metrics by hand with the random seeds defined in fake_data()
        ref = {(0, 'x'): 1.009, (0, 'y'): 0.938, (1, 'x'): 0.788, (1, 'y'): 0.797,
               (2, 'x'): 0.846, (2, 'y'): 0.774, (3, 'x'): 0.667, (3, 'y'): 0.755}
        for key, val in ref.items():
            self.assertAlmostEqual(val, mean_Vij[key], places=3)
        zs = ant_metrics.mean_Vij_metrics(self.data, self.pols, self.antpols,
                                          self.ants, self.bls)
        ref = {(0, 'x'): 1.443, (0, 'y'): 4.970, (1, 'x'): -0.218, (1, 'y'): 0.373,
               (2, 'x'): 0.218, (2, 'y'): -0.373, (3, 'x'): -1.131, (3, 'y'): -0.976}
        for key, val in ref.items():
            self.assertAlmostEqual(val, zs[key], places=3)

    def test_red_corr_metrics(self):
        red_corr = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols,
                                                self.ants, self.reds, rawMetric=True)
        ref = {(0, 'x'): 0.468, (0, 'y'): 0.479, (1, 'x'): 0.614, (1, 'y'): 0.472,
               (2, 'x'): 0.536, (2, 'y'): 0.623, (3, 'x'): 0.567, (3, 'y'): 0.502}
        for key, val in ref.items():
            self.assertAlmostEqual(val, red_corr[key], places=3)
        zs = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols,
                                          self.ants, self.reds)
        ref = {(0, 'x'): -1.445, (0, 'y'): -0.516, (1, 'x'): 1.088, (1, 'y'): -0.833,
               (2, 'x'): -0.261, (2, 'y'): 6.033, (3, 'x'): 0.261, (3, 'y'): 0.516}
        for key, val in ref.items():
            self.assertAlmostEqual(val, zs[key], places=3)

    def test_red_corr_metrics_NaNs(self):
        ''' Test that antennas not in reds return NaNs for redundant metrics '''
        ants = copy.copy(self.ants)
        ants.append(99)
        red_corr = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols,
                                                ants, self.reds, rawMetric=True)
        ref = {(0, 'x'): 0.468, (0, 'y'): 0.479, (1, 'x'): 0.614, (1, 'y'): 0.472,
               (2, 'x'): 0.536, (2, 'y'): 0.623, (3, 'x'): 0.567, (3, 'y'): 0.502,
               (99, 'x'): np.NaN, (99, 'y'): np.NaN}
        for key, val in ref.items():
            if np.isnan(val):
                self.assertTrue(np.isnan(red_corr[key]))
            else:
                self.assertAlmostEqual(val, red_corr[key], places=3)
        zs = ant_metrics.red_corr_metrics(self.data, self.pols, self.antpols,
                                          ants, self.reds)
        ref = {(0, 'x'): -1.445, (0, 'y'): -0.516, (1, 'x'): 1.088, (1, 'y'): -0.833,
               (2, 'x'): -0.261, (2, 'y'): 6.033, (3, 'x'): 0.261, (3, 'y'): 0.516,
               (99, 'x'): np.NaN, (99, 'y'): np.NaN}
        for key, val in ref.items():
            if np.isnan(val):
                self.assertTrue(np.isnan(zs[key]))
            else:
                self.assertAlmostEqual(val, zs[key], places=3)

    def test_mean_Vij_cross_pol_metrics(self):
        mean_Vij_cross_pol = ant_metrics.mean_Vij_cross_pol_metrics(self.data, self.pols,
                                                                    self.antpols, self.ants,
                                                                    self.bls, rawMetric=True)
        ref = {(0, 'x'): 0.746, (0, 'y'): 0.746, (1, 'x'): 0.811, (1, 'y'): 0.811,
               (2, 'x'): 0.907, (2, 'y'): 0.907, (3, 'x'): 1.091, (3, 'y'): 1.091}
        for key, val in ref.items():
            self.assertAlmostEqual(val, mean_Vij_cross_pol[key], places=3)
        zs = ant_metrics.mean_Vij_cross_pol_metrics(self.data, self.pols, self.antpols,
                                                    self.ants, self.bls)
        ref = {(0, 'x'): -0.948, (0, 'y'): -0.948, (1, 'x'): -0.401, (1, 'y'): -0.401,
               (2, 'x'): 0.401, (2, 'y'): 0.401, (3, 'x'): 1.944, (3, 'y'): 1.944}
        for key, val in ref.items():
            self.assertAlmostEqual(val, zs[key], places=3)

    def test_red_corr_cross_pol_metrics(self):
        red_corr_cross_pol = ant_metrics.red_corr_cross_pol_metrics(self.data, self.pols,
                                                                    self.antpols, self.ants,
                                                                    self.reds, rawMetric=True)
        ref = {(0, 'x'): 1.062, (0, 'y'): 1.062, (1, 'x'): 0.934, (1, 'y'): 0.934,
               (2, 'x'): 0.917, (2, 'y'): 0.917, (3, 'x'): 1.027, (3, 'y'): 1.027}
        for key, val in ref.items():
            self.assertAlmostEqual(val, red_corr_cross_pol[key], places=3)
        zs = ant_metrics.red_corr_cross_pol_metrics(self.data, self.pols, self.antpols,
                                                    self.ants, self.reds)
        ref = {(0, 'x'): 1.001, (0, 'y'): 1.001, (1, 'x'): -0.572, (1, 'y'): -0.572,
               (2, 'x'): -0.777, (2, 'y'): -0.777, (3, 'x'): 0.572, (3, 'y'): 0.572}
        for key, val in ref.items():
            self.assertAlmostEqual(val, zs[key], places=3)

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

    def test_compute_median_auto_power_dict(self):
        power = ant_metrics.compute_median_auto_power_dict(self.data, self.pols, self.reds)
        for key, p in power.items():
            testp = np.median(np.mean(np.abs(self.data.get_data(*key))**2, axis=0))
            self.assertEqual(p, testp)
        for key in self.data.data.keys():
            for pol in self.data.data[key].keys():
                self.assertIn((key[0], key[1], pol), power.keys())

    def test_load_antenna_metrics(self):
        # load a metrics file and check some values
        metrics_file = os.path.join(DATA_PATH, 'example_ant_metrics.json')
        metrics = ant_metrics.load_antenna_metrics(metrics_file)
        self.assertAlmostEqual(metrics['final_mod_z_scores']['meanVijXPol'][(72, 'x')], 0.17529333517595402)
        self.assertAlmostEqual(metrics['final_mod_z_scores']['meanVijXPol'][(72, 'y')], 0.17529333517595402)
        self.assertAlmostEqual(metrics['final_mod_z_scores']['meanVijXPol'][(31, 'y')], 0.7012786080508268)

        # change some values to FPE values, and write it out
        metrics['final_mod_z_scores']['meanVijXPol'][(72, 'x')] = np.nan
        metrics['final_mod_z_scores']['meanVijXPol'][(72, 'y')] = np.inf
        metrics['final_mod_z_scores']['meanVijXPol'][(31, 'y')] = -np.inf
        for key in metrics.keys():
            metrics[key] = str(metrics[key])
        outpath = os.path.join(DATA_PATH, 'test_output', 'ant_metrics_output.json')
        with open(outpath, 'w') as outfile:
            json.dump(metrics, outfile, indent=4)

        # test reading it back in, and that the values agree
        metrics_new = ant_metrics.load_antenna_metrics(outpath)
        self.assertTrue(np.isnan(metrics_new['final_mod_z_scores']['meanVijXPol'][(72, 'x')]))
        self.assertEqual(metrics_new['final_mod_z_scores']['meanVijXPol'][(72, 'y')], np.inf)
        self.assertEqual(metrics_new['final_mod_z_scores']['meanVijXPol'][(31, 'y')], -np.inf)

        # clean up after ourselves
        os.remove(outpath)


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

    def test_init(self):
        am = ant_metrics.Antenna_Metrics(self.dataFileList, self.reds,
                                         fileformat='miriad')
        self.assertEqual(len(am.ants), 19)
        self.assertEqual(set(am.pols), set(['xx', 'yy', 'xy', 'yx']))
        self.assertEqual(set(am.antpols), set(['x', 'y']))
        self.assertEqual(len(am.bls), 19 * 18 / 2 + 19)
        self.assertEqual(len(am.reds), 27)

    def test_iterative_antenna_metrics_and_flagging_and_saving_and_loading(self):
        am = ant_metrics.Antenna_Metrics(self.dataFileList, self.reds,
                                         fileformat='miriad')
        with self.assertRaises(KeyError):
            am.save_antenna_metrics(DATA_PATH + '/test_output/ant_metrics_output.json')

        am.iterative_antenna_metrics_and_flagging()
        for stat in self.summaryStats:
            self.assertTrue(hasattr(am, stat))
        self.assertIn((81, 'x'), am.xants)
        self.assertIn((81, 'y'), am.xants)
        self.assertIn((81, 'x'), am.deadAntsRemoved)
        self.assertIn((81, 'y'), am.deadAntsRemoved)

        outfile = os.path.join(DATA_PATH, 'test_output', 'ant_metrics_output.json')
        am.save_antenna_metrics(outfile)
        loaded = ant_metrics.load_antenna_metrics(outfile)
        # json names for summary statistics
        jsonStats = ['xants', 'crossed_ants', 'dead_ants', 'removal_iteration',
                     'final_metrics', 'all_metrics', 'final_mod_z_scores', 'all_mod_z_scores',
                     'cross_pol_z_cut', 'dead_ant_z_cut', 'datafile_list', 'reds', 'version']
        for stat, jsonStat in zip(self.summaryStats, jsonStats):
            self.assertEqual(loaded[jsonStat], getattr(am, stat))
        os.remove(outfile)

    def test_cross_detection(self):
        am2 = ant_metrics.Antenna_Metrics(self.dataFileList, self.reds,
                                          fileformat='miriad')
        am2.iterative_antenna_metrics_and_flagging(crossCut=3, deadCut=10)
        for stat in self.summaryStats:
            self.assertTrue(hasattr(am2, stat))
        self.assertIn((81, 'x'), am2.xants)
        self.assertIn((81, 'y'), am2.xants)
        self.assertIn((81, 'x'), am2.crossedAntsRemoved)
        self.assertIn((81, 'y'), am2.crossedAntsRemoved)

    def test_totally_dead_ants(self):
        am2 = ant_metrics.Antenna_Metrics(self.dataFileList, self.reds,
                                          fileformat='miriad')
        am2.data.data_array[am2.data.ant_1_array == 9, :, :, :] = 0.0
        am2.reset_summary_stats()
        am2.find_totally_dead_ants()
        self.assertIn((9, 'x'), am2.xants)
        self.assertIn((9, 'y'), am2.xants)
        self.assertIn((9, 'x'), am2.deadAntsRemoved)
        self.assertIn((9, 'y'), am2.deadAntsRemoved)
        self.assertEqual(am2.removalIter[(9, 'x')], -1)
        self.assertEqual(am2.removalIter[(9, 'y')], -1)


class TestAntmetricsRun(object):
    def test_ant_metrics_run(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('ant_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        calfile = 'heratest_calfile'
        arg0 = "-C {}".format(calfile)
        arg1 = "-p xx,yy,xy,yx"
        arg2 = "--crossCut=5"
        arg3 = "--deadCut=5"
        arg4 = "--extension=.ant_metrics.json"
        arg5 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg6 = "--vis_format=miriad"
        arg7 = "--alwaysDeadCut=10"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7])

        # test running with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        nt.assert_raises(AssertionError, ant_metrics.ant_metrics_run, args.files,
                         args, history)

        # test running with a lone file
        lone_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
        cmd = ' '.join([arguments, lone_file])
        args = a.parse_args(cmd.split())
        history = cmd
        # this test raises a warning, then fails...
        args = [AssertionError, ant_metrics.ant_metrics_run, args.files, args, history]
        uvtest.checkWarnings(nt.assert_raises, args, nwarnings=1,
                             message='Could not find')

        # test actually running metrics
        xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
        dest_file = os.path.join(DATA_PATH, 'test_output',
                                 'zen.2457698.40355.HH.uvcA.ant_metrics.json')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        ant_metrics.ant_metrics_run(args.files, args, history)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)

    def test_ant_metrics_run_nocalfile(self):
        # get arguments
        a = utils.get_metrics_ArgumentParser('ant_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        arg0 = "-p xx,yy,xy,yx"
        arg1 = "--crossCut=5"
        arg2 = "--deadCut=5"
        arg3 = "--extension=.ant_metrics.json"
        arg4 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arg5 = "--vis_format=miriad"
        arg6 = "--alwaysDeadCut=10"
        arguments = ' '.join([arg0, arg1, arg2, arg3, arg4, arg5, arg6])

        # test running with no calfile
        xx_file = os.path.join(DATA_PATH, 'zen.2458002.47754.xx.HH.uvA')
        dest_file = os.path.join(DATA_PATH, 'test_output',
                                 'zen.2458002.47754.HH.uvA.ant_metrics.json')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, xx_file])
        args = a.parse_args(cmd.split())
        history = cmd
        ant_metrics.ant_metrics_run(args.files, args, history)
        nt.assert_true(os.path.exists(dest_file))
        os.remove(dest_file)


if __name__ == '__main__':
    unittest.main()
