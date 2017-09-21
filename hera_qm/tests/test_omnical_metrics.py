"""
Test Omnical Metrics
"""
import matplotlib.pyplot as plt
import numpy as np
from hera_qm import omnical_metrics
from hera_qm.data import DATA_PATH
import unittest
import os
from hera_qm import utils
import sys
from collections import OrderedDict
from pyuvdata import UVCal


class Test_OmniCal_Metrics(unittest.TestCase):

    def setUp(self):
        self.fc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
        self.oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.out_dir = os.path.join(DATA_PATH, 'test_output')
        self.OM = omnical_metrics.OmniCal_Metrics(self.oc_file)
        self.OM.run_metrics()

    def test_init(self):
        self.assertEqual(self.OM.Nants, 16)
        self.assertEqual(self.OM.filename, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.assertEqual(self.OM.omni_gains.shape, (16, 3, 1024, 1))

    def test_run_metrics(self):
        # no fc file
        self.OM.run_metrics()
        self.assertIs(self.OM.metrics['ant_phs_noise'], None)
        self.assertAlmostEqual(self.OM.metrics['chisq_ant_std_loc'], 0.16368077864231928)
        self.assertIs(type(self.OM.metrics), OrderedDict)
        # no cut band
        self.OM.run_metrics(cut_band=False)
        # fc file
        self.OM.run_metrics(firstcal_file=self.fc_file)
        self.assertAlmostEqual(self.OM.metrics['tot_phs_noise'], 0.32104988077232272)
        self.assertAlmostEqual(self.OM.metrics['tot_phs_std'], 0.050937336291433551)

    def test_write_load_metrics(self):
        # Run metrics
        nkeys = len(self.OM.metrics.keys())
        outfile = os.path.join(self.out_dir, 'omnical_metrics.json')
        if os.path.isfile(outfile):
            os.remove(outfile)
        # write json
        self.OM.write_metrics(filename=outfile, filetype='json')
        self.assertTrue(os.path.isfile(outfile))
        # load json
        self.OM.load_metrics(filename=outfile)
        self.assertEqual(len(self.OM.metrics.keys()), nkeys)
        # erase
        os.remove(outfile)
        outfile = os.path.join(self.out_dir, 'omnical_metrics.pkl')
        if os.path.isfile(outfile):
            os.remove(outfile)
        # write pkl
        self.OM.write_metrics(filename=outfile, filetype='pkl')
        self.assertTrue(os.path.isfile(outfile))
        # load
        self.OM.load_metrics(filename=outfile)
        self.assertEqual(len(self.OM.metrics.keys()), nkeys)
        ## check exceptions
        # load filetype
        os.remove(outfile)
        _ = open(outfile+'.wtf', 'a').close()
        self.assertRaises(IOError, self.OM.load_metrics, filename=outfile+'.wtf')
        os.remove(outfile+'.wtf')
        # write w/o filename
        self.OM.run_metrics()
        outfile = os.path.join(self.OM.filedir, self.OM.filestem+'.omni_metrics.json')
        self.OM.write_metrics(filetype='json')
        os.remove(outfile)
        outfile = os.path.join(self.OM.filedir, self.OM.filestem+'.omni_metrics.pkl')
        self.OM.write_metrics(filetype='pkl')
        os.remove(outfile)


    def test_plot_phs_metrics(self):
        self.OM.run_metrics(firstcal_file=self.fc_file)
        fname = os.path.join(self.OM.filedir, 'phs.png')
        if os.path.isfile(fname):
            os.remove(fname)
        # plot w/ fname
        self.OM.plot_phs_metric(plot_type='std', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        self.OM.plot_phs_metric(plot_type='ft', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/ fname and outpath
        self.OM.plot_phs_metric(plot_type='std', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        self.OM.plot_phs_metric(plot_type='ft', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.phs_std.png')
        self.OM.plot_phs_metric(plot_type='std', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.phs_ft.png')
        self.OM.plot_phs_metric(plot_type='ft', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot feeding ax object
        fig,ax = plt.subplots()
        self.OM.plot_phs_metric(ax=ax)
        plt.close()
        # exception
        del self.OM.metrics
        self.assertRaises(Exception, self.OM.plot_phs_metric)
        self.OM.run_metrics()

    def test_plot_chisq_metrics(self):
        self.OM.run_metrics(firstcal_file=self.fc_file)
        fname = os.path.join(self.OM.filedir, 'chisq.png')
        if os.path.isfile(fname):
            os.remove(fname)
        # plot w/ fname
        self.OM.plot_chisq_metric(fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/ fname and outpath
        self.OM.plot_chisq_metric(fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.chisq_std.png')
        self.OM.plot_chisq_metric(plot_type='std', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot feeding ax object
        fig,ax = plt.subplots()
        self.OM.plot_chisq_metric(ax=ax)
        plt.close()
        # exception
        del self.OM.metrics
        self.assertRaises(Exception, self.OM.plot_chisq_metric)
        self.OM.run_metrics()

    def test_plot_chisq_tavg(self):
        self.OM.run_metrics(firstcal_file=self.fc_file)
        fname = os.path.join(self.OM.filedir, 'chisq_tavg.png')
        # test execution
        if os.path.isfile(fname):
            os.remove(fname)
        self.OM.plot_chisq_tavg(fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        self.OM.plot_chisq_tavg(ants=self.OM.ant_array[:2], fname=fname, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_plot_gains(self):
        self.OM.run_metrics(firstcal_file=self.fc_file)
        fname = os.path.join(self.OM.filedir, 'gains.png')
        if os.path.isfile(fname):
            os.remove(fname)
        # plot w/ fname
        self.OM.plot_gains(plot_type='phs', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        self.OM.plot_gains(plot_type='amp', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # divide_fc = True
        self.OM.plot_gains(plot_type='phs', fname=fname, save=True, divide_fc=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        self.OM.plot_gains(plot_type='amp', fname=fname, save=True, divide_fc=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/ fname and outpath
        self.OM.plot_gains(plot_type='phs', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.gain_phs.png')
        self.OM.plot_gains(plot_type='phs', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.gain_amp.png')
        self.OM.plot_gains(plot_type='amp', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        # plot feeding ax object
        fig,ax = plt.subplots()
        self.OM.plot_gains(ax=ax)
        plt.close()
        # plot feeding ants
        fname = os.path.join(self.OM.filedir, 'gains.png')
        self.OM.plot_gains(ants=self.OM.ant_array[:2], fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)

    def test_plot_metrics(self):
        # test exception
        del self.OM.metrics
        self.assertRaises(Exception, self.OM.plot_metrics)
        self.OM.run_metrics()
        self.assertRaises(Exception, self.OM.plot_metrics)
        # test execution
        self.OM.run_metrics(firstcal_file=self.fc_file)
        fname1 = os.path.join(self.OM.filedir, self.OM.filename+'.chisq_std.png')
        fname2 = os.path.join(self.OM.filedir, self.OM.filename+'.phs_std.png')
        fname3 = os.path.join(self.OM.filedir, self.OM.filename+'.phs_ft.png')
        for f in [fname1, fname2, fname3]:
            if os.path.isfile(f) == True:
                os.remove(f)
        self.OM.plot_metrics()
        self.assertEqual(os.path.isfile(fname1), True)
        self.assertEqual(os.path.isfile(fname2), True)
        self.assertEqual(os.path.isfile(fname3), True)
        os.remove(fname1)
        os.remove(fname2)
        os.remove(fname3)

class Test_OmniCalMetrics_Run(unittest.TestCase):
    def setUp(self):
        self.fc_file     = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
        self.oc_file     = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.oc_filedir  = os.path.dirname(self.oc_file)
        self.oc_basename = os.path.basename(self.oc_file)
        self.out_dir     = os.path.join(DATA_PATH, 'test_output')

    def test_firstcal_metrics_run(self):
        # get arg parse
        a = utils.get_metrics_ArgumentParser('omnical_metrics')
        if DATA_PATH not in sys.path:
            sys.path.append(DATA_PATH)

        # test w/ no file
        arguments = ''
        cmd = ' '.join([arguments, ''])
        args = a.parse_args(cmd.split())
        history = cmd
        self.assertRaises(AssertionError, omnical_metrics.omnical_metrics_run,
                                            args.files, args, history)
        # test w/ no fc_file
        arguments = ''
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file+'.omni_metrics.json'
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ extension
        arguments = '--extension=.omni.json'
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file+'.omni.json'
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ metrics_path
        arguments = '--metrics_path={}'.format(self.out_dir)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = os.path.join(self.out_dir, self.oc_basename+'.omni_metrics.json')
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ options
        arguments = '--fc_file={0} --no_bandcut --phs_noise_cut=0.5 --phs_std_cut=0.5 '\
                        '--chisq_std_cut=5.0'.format(self.fc_file)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file + '.omni_metrics.json'
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test make plots
        arguments = '--fc_file={0} --make_plots'.format(self.fc_file)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file + '.omni_metrics.json'
        outpng1 = self.oc_file + '.chisq_std.png'
        outpng2 = self.oc_file + '.phs_std.png'
        outpng3 = self.oc_file + '.phs_ft.png'
        self.assertTrue(os.path.isfile(outfile))
        self.assertTrue(os.path.isfile(outpng1))
        self.assertTrue(os.path.isfile(outpng2))
        self.assertTrue(os.path.isfile(outpng3))
        os.remove(outfile)
        os.remove(outpng1)
        os.remove(outpng2)
        os.remove(outpng3)


if __name__ == "__main__":
    unittest.main()

    