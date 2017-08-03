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
        infile = os.path.join(DATA_PATH, 'zen.2457678.16694.yy.HH.uvc.good.first.calfits')
        self.FC = firstcal_metrics.FirstCal_Metrics(infile)
        self.out_dir = os.path.join(DATA_PATH, 'test_output')

    def test_init(self):
        self.assertEqual(self.FC.Nants, 18)
        self.assertEqual(len(self.FC.delays), 18)

    def test_run_metrics(self):
        metrics = self.FC.run_metrics(output=True, std_cut=1.0)
        self.assertEqual(metrics['full_sol'], 'good')
        self.assertEqual(metrics['bad_ants'], [])
        self.assertIn(9, metrics['z_scores'])
        self.assertIn(9, metrics['ant_std'])
        self.assertIn(9, metrics['ant_avg'])
        self.assertAlmostEqual(metrics['agg_std'], 0.088757931322363717)

        # Test bad ants detection
        self.FC.delay_offsets[0, :] *= 10
        self.FC.run_metrics()
        self.assertIn(self.FC.ants[0], self.FC.bad_ants)
        # Test bad full solution
        self.FC.delay_offsets[1:, :] *= 10
        self.FC.run_metrics()
        self.assertEqual(self.FC.full_sol, 'bad')

    def test_write_load_metrics(self):
        # run metrics
        self.FC.run_metrics()
        num_keys = len(self.FC.metrics.keys())
        outfile = os.path.join(self.out_dir, 'firstcal_metrics.json')
        if os.path.isfile(outfile):
            os.remove(outfile)
        # write json
        self.FC.write_metrics(filename=outfile, filetype='json')
        self.assertTrue(os.path.isfile(outfile))
        # load json
        self.FC.load_metrics(filename=outfile)
        self.assertEqual(len(self.FC.metrics.keys()), num_keys)
        # erase
        os.remove(outfile)
        # write pickle
        outfile = os.path.join(self.out_dir, 'firstcal_metrics.pkl')
        if os.path.isfile(outfile):
            os.remove(outfile)
        self.FC.write_metrics(filename=outfile, filetype='pkl')
        self.assertTrue(os.path.isfile(outfile))
        # load pickle
        self.FC.load_metrics(filename=outfile)
        self.assertEqual(len(self.FC.metrics.keys()), num_keys)
        os.remove(outfile)

        # Check some exceptions
        outfile = os.path.join(self.out_dir, 'firstcal_metrics.txt')
        self.assertRaises(IOError, self.FC.load_metrics, filename=outfile)
        outfile = self.FC.file_stem + '.first_metrics.json'
        self.FC.write_metrics(filetype='json')  # No filename
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)
        outfile = self.FC.file_stem + '.first_metrics.pkl'
        self.FC.write_metrics(filetype='pkl')  # No filename
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

    def test_plot_delays(self):
        fname = os.path.join(self.out_dir, 'dlys.png')
        if os.path.isfile(fname):
            os.remove(fname)
        self.FC.plot_delays(fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        # Check cm defaults to spectral
        self.FC.plot_delays(fname=fname, save=True, cm='foo')
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_plot_zscores(self):
        fname = os.path.join(self.out_dir, 'zscrs.png')
        if os.path.isfile(fname):
            os.remove(fname)
        self.FC.plot_zscores(fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        self.FC.plot_zscores(fname=fname, plot_type='time_avg', save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_plot_stds(self):
        fname = os.path.join(self.out_dir, 'stds.png')
        if os.path.isfile(fname):
            os.remove(fname)
        self.FC.plot_stds(fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        self.FC.plot_stds(fname=fname, xaxis='time', save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

if __name__ == "__main__":
    unittest.main()
