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
        infile = os.path.join(DATA_PATH, 'zen.2457555.50099.yy.HH.uvcA.first.calfits')
        self.FC = firstcal_metrics.FirstCal_Metrics(infile)
        self.out_dir = os.path.join(DATA_PATH, 'test_output')

    def test_init(self):
        self.assertEqual(self.FC.Nants, 18)
        self.assertEqual(len(self.FC.delays), 18)

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
        self.FC.delay_fluctuations[0, :] *= 10
        self.FC.run_metrics()
        self.assertIn(self.FC.ants[0], self.FC.metrics['bad_ants'])
        # Test bad full solution
        self.FC.delay_fluctuations[1:, :] *= 10
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
        self.FC.plot_delays(fname=fname, save=True, plot_type='fluctuation')
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

    def test_rotated_metrics(self):
        infile = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
        rotant_json = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits.rotated_metric.json')
        FC = firstcal_metrics.FirstCal_Metrics(infile)
        FC.run_metrics(rotant_json=rotant_json, std_cut=0.5)
        out_dir = os.path.join(DATA_PATH, 'test_output')
        # test pickup of rotant key
        self.assertIn('rot_ants', FC.metrics.keys())
        # test rotants is correct
        self.assertEqual([43], FC.metrics['rot_ants'])


class TestFirstcalMetricsRun(unittest.TestCase):
    def test_firstcal_metrics_run(self):
        # get argument object
        a = utils.get_metrics_ArgumentParser('firstcal_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)

        arg0 = "--std_cut=0.5"
        arg1 = "--extension=.firstcal_metrics.json"
        arg2 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
        arguments = ' '.join([arg0, arg1, arg2])

        # Test runing with no files
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        self.assertRaises(AssertionError, firstcal_metrics.firstcal_metrics_run,
                          args.files, args, history)

        # Test running with file
        filename = os.path.join(DATA_PATH, 'zen.2457555.50099.yy.HH.uvcA.first.calfits')
        dest_file = os.path.join(DATA_PATH, 'test_output',
                                 'zen.2457555.50099.yy.HH.uvcA.first.calfits.' +
                                 'firstcal_metrics.json')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        cmd = ' '.join([arguments, filename])
        args = a.parse_args(cmd.split())
        history = cmd
        firstcal_metrics.firstcal_metrics_run(args.files, args, history)
        self.assertTrue(os.path.exists(dest_file))

        # test w/ rotant json
        infile = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
        outfile = infile+'.firstcal_metrics.json'
        rotant_json = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits.rotated_metric.json')
        cmd = '--rotant_files={0} {1}'.format(rotant_json, infile)
        args = a.parse_args(cmd.split())
        if os.path.isfile(outfile):
            os.remove(outfile)
        firstcal_metrics.firstcal_metrics_run(args.files, args, history)
        self.assertTrue(os.path.isfile(outfile))
        # test rotant key exists
        metrics = firstcal_metrics.load_firstcal_metrics(outfile)
        self.assertIn('rot_ants', metrics.keys())


if __name__ == "__main__":
    unittest.main()
