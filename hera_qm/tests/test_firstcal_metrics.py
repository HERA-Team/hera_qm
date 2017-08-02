"""
test_firstcal_metrics.py

"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
from hera_qm import firstcal_metrics
from hera_qm.data import DATA_PATH
import unittest
import os

class Test_FirstCal_Metrics(unittest.TestCase):

    def setUp(self):
        self.FC = firstcal_metrics.FirstCal_Metrics(DATA_PATH+'/zen.2457678.16694.yy.HH.uvc.good.first.calfits')

    def test_init(self):
        self.assertEqual(self.FC.Nants, 18)
        self.assertEqual(len(self.FC.delays), 18)

    def test_run_metrics(self):
        metrics = self.FC.run_metrics(output=True, std_cut=1.0)
        self.assertEqual(metrics['full_sol'], 'good')
        self.assertEqual(metrics['bad_ant'], [])
        self.assertIn(9, metrics['z_scores'])
        self.assertIn(9, metrics['ant_std'])
        self.assertAlmostEqual(metrics['agg_std'], 0.08702395042454745)

    def test_write_load_metrics(self):
        # run metrics
        self.FC.run_metrics()
        num_keys = len(self.FC.metrics.keys())
        # write
        self.FC.write_metrics(filename='metrics', filetype='json')
        self.assertEqual(os.path.isfile('metrics.json'), True)
        # load
        self.FC.load_metrics(filename='metrics.json')
        self.assertEqual(len(self.FC.metrics.keys()), num_keys)
        # erase
        os.system('rm metrics.json')

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
