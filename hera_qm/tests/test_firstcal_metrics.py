"""
test_firstcal_metrics.py

"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from hera_qm import firstcal_metrics
import unittest
import os

class Test_FirstCal_Metrics(unittest.TestCase):

    def setUp(self):
        self.FC = firstcal_metrics.FirstCal_Metrics('../data/zen.2457678.16694.yy.HH.uvc.good.first.calfits')

    def test_init(self):
        self.assertEqual(self.FC.Nants, 18)
        self.assertEqual(len(self.FC.delays), 18)

    def test_run_metrics(self):
        result = self.FC.run_metrics(output=True, std_cut=1.0)
        self.assertEqual(result['full_sol'], 'good')
        self.assertEqual(result['bad_ant'], [])
        self.assertIn('9', result['z_scores'])
        self.assertIn('9', result['ant_std'])
        self.assertAlmostEqual(result['agg_std'], 0.08702395042454745)

    def test_write_load_metrics(self):
        # run metrics
        self.FC.run_metrics()
        num_keys = len(self.FC.result.keys())
        # write
        self.FC.write_metrics(filename='metrics', filetype='pkl')
        self.assertEqual(os.path.isfile('metrics.pkl'), True)
        # load
        self.FC.load_metrics(filename='metrics.pkl')
        self.assertEqual(len(self.FC.result.keys()), num_keys)
        # erase
        os.system('rm metrics.pkl')

    def test_plot_delays(self):
        self.FC.plot_delays(fname='dlys.png', save=True)
        self.assertEqual(os.path.isfile('dlys.png'), True)
        os.system('rm dlys.png')

    def test_plot_zscores(self):
        self.FC.plot_zscores(fname='zscrs.png', save=True)
        self.assertEqual(os.path.isfile('zscrs.png'), True)
        os.system('rm zscrs.png')

    def test_plot_stds(self):
        self.FC.plot_stds(fname='stds.png', save=True)
        self.assertEqual(os.path.isfile('stds.png'), True)
        os.system('rm stds.png')

if __name__ == "__main__":
    unittest.main()
