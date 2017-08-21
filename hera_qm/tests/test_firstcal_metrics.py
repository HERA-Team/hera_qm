"""
test_firstcal_metrics.py

"""
import matplotlib
import numpy as np
from hera_qm import firstcal_metrics
from hera_qm.data import DATA_PATH
import unittest
import os
from hera_qm import utils
import sys


class Test_FirstCal_Metrics(unittest.TestCase):

    def setUp(self):
        infile = os.path.join(DATA_PATH, 'zen.2457678.16694.yy.HH.uvc.good.first.calfits')
        self.FC = firstcal_metrics.FirstCal_Metrics(infile)
        self.out_dir = os.path.join(DATA_PATH, 'test_output')

    def test_init(self):
        self.assertEqual(self.FC.Nants, 18)
        self.assertEqual(len(self.FC.delays), 18)
        self.assertEqual(self.FC.fc_filename, 'zen.2457678.16694.yy.HH.uvc.good.first.calfits')
        self.assertEqual(self.FC.fc_filestem, 'zen.2457678.16694.yy.HH.uvc.good.first')

    def test_run_metrics(self):
        self.FC.run_metrics(std_cut=1.0)
        self.assertEqual(self.FC.metrics['good_sol'], True)
        self.assertEqual(self.FC.metrics['bad_ants'], [])
        self.assertIn(9, self.FC.metrics['z_scores'])
        self.assertIn(9, self.FC.metrics['ant_std'])
        self.assertIn(9, self.FC.metrics['ant_avg'])
        self.assertIn(9, self.FC.metrics['ants'])
        self.assertIn(9, self.FC.metrics['z_scores'])
        self.assertIn(9, self.FC.metrics['ant_z_scores'])
        self.assertEqual(str, type(self.FC.metrics['version']))
        self.assertAlmostEqual(1.0, self.FC.metrics['std_cut'])
        self.assertAlmostEqual(self.FC.metrics['agg_std'], 0.088757931322363717)
        self.assertEqual('y', self.FC.metrics['pol'])

        # Test bad ants detection
        self.FC.delay_offsets[0, :] *= 10
        self.FC.run_metrics()
        self.assertIn(self.FC.ants[0], self.FC.metrics['bad_ants'])
        # Test bad full solution
        self.FC.delay_offsets[1:, :] *= 10
        self.FC.run_metrics()
        self.assertEqual(self.FC.metrics['good_sol'], False)

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
        outfile = self.FC.fc_filestem + '.first_metrics.json'
        self.FC.write_metrics(filetype='json')  # No filename
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)
        outfile = self.FC.fc_filestem + '.first_metrics.pkl'
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
        self.FC.plot_delays(fname=fname, save=True, plot_type='solution')
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        self.FC.plot_delays(fname=fname, save=True, plot_type='offset')
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        # Check cm defaults to spectral
        self.FC.plot_delays(fname=fname, save=True, cmap='foo')
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

        # Check exception

    def test_plot_zscores(self):
        # check exception
        self.assertRaises(NameError, self.FC.plot_zscores)
        self.FC.run_metrics()
        self.assertRaises(NameError, self.FC.plot_zscores, plot_type='foo')
        # check output
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
        # check exception
        self.assertRaises(NameError, self.FC.plot_stds)
        self.FC.run_metrics()
        self.assertRaises(NameError, self.FC.plot_stds, xaxis='foo')
        # check output
        fname = os.path.join(self.out_dir, 'stds.png')
        if os.path.isfile(fname):
            os.remove(fname)
        self.FC.plot_stds(fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        self.FC.plot_stds(fname=fname, xaxis='time', save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)


class TestFirstcalMetricsRun(unittest.TestCase):
    def test_firstcal_metrics_run(self):
        # get options object
        o = utils.get_metrics_OptionParser('firstcal_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)
        opt0 = "--std_cut=0.5"
        opt1 = "--extension=.firstcal_metrics.json"
        opt2 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        options = ' '.join([opt0, opt1, opt2])

        # Test runing with no files
        cmd = ' '.join([options, ''])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        self.assertRaises(AssertionError, firstcal_metrics.firstcal_metrics_run,
                          args, opts, history)

        # Test running with file
        filename = os.path.join(DATA_PATH, 'zen.2457678.16694.yy.HH.uvc.good.first.calfits')
        dest_file = os.path.join(DATA_PATH, 'test_output',
                                 'zen.2457678.16694.yy.HH.uvc.good.first.calfits.' +
                                 'firstcal_metrics.json')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([options, filename])
        opts, args = o.parse_args(cmd.split())
        history = cmd
        firstcal_metrics.firstcal_metrics_run(args, opts, history)
        self.assertTrue(os.path.exists(dest_file))


if __name__ == "__main__":
    unittest.main()
