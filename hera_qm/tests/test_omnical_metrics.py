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
        self.fc_file_bad = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.bad.first.calfits')
        self.fc_fileyy = os.path.join(DATA_PATH, 'zen.2457555.42443.yy.HH.uvcA.first.calfits')
        self.oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.out_dir = os.path.join(DATA_PATH, 'test_output')
        self.OM = omnical_metrics.OmniCal_Metrics(self.oc_file)

    def test_init(self):
        self.assertEqual(self.OM.Nants, 17)
        self.assertEqual(self.OM.filename, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.assertEqual(self.OM.omni_gains.shape, (17, 3, 1024, 1))

    def test_load_firstcal_gains(self):
        firstcal_delays, firstcal_gains, fc_pol = omnical_metrics.load_firstcal_gains(self.fc_file)
        self.assertEqual(firstcal_delays.shape, (17, 3))
        self.assertEqual(firstcal_gains.shape, (17, 3, 1024))
        self.assertEqual(fc_pol, -5)

    def test_omni_load_firstcal_gains(self):
        # test execution
        try:
            del self.gain_diff
        except:
            pass
        self.OM.load_firstcal_gains(self.fc_file)
        self.assertTrue(hasattr(self.OM, 'gain_diff'))
        # test exception
        self.assertRaises(ValueError, self.OM.load_firstcal_gains, self.fc_fileyy)

    def test_multiple_pol_fc_pickup(self):
        fc_files = [os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits'), os.path.join(DATA_PATH, 'zen.2457555.42443.yy.HH.uvcA.first.calfits')]
        oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
        OM = omnical_metrics.OmniCal_Metrics(oc_file)
        OM.load_firstcal_gains(fc_files)
        # make sure 'XX' pol first, then 'YY' pol
        self.assertAlmostEqual(OM.firstcal_delays[0, 0, 0], 1.0918842818239246e-09)
        # reverse pol and repeat 
        OM.load_firstcal_gains(fc_files[::-1])
        self.assertAlmostEqual(OM.firstcal_delays[0, 0, 0], 1.0918842818239246e-09)
        # hit exception
        OM.pols[0] = 'XY'
        self.assertRaises(ValueError, OM.load_firstcal_gains, fc_files)

    def test_run_metrics(self):
        # no fc file
        full_metrics = self.OM.run_metrics()
        metrics = full_metrics['XX']
        self.assertIs(metrics['ant_phs_std'], None)
        self.assertAlmostEqual(metrics['chisq_ant_std_loc'], 0.16499103579233421)
        self.assertEqual(len(full_metrics), 1)
        self.assertEqual(len(metrics), 36)
        self.assertIs(type(full_metrics), OrderedDict)
        self.assertIs(type(metrics), OrderedDict)
        # no cut band
        full_metrics = self.OM.run_metrics(cut_edges=False)
        metrics = full_metrics['XX']
        self.assertIs(metrics['ant_phs_std'], None)
        self.assertAlmostEqual(metrics['chisq_ant_std_loc'], 0.17762839199226452)
        self.assertEqual(len(full_metrics), 1)
        self.assertEqual(len(metrics), 36)
        self.assertIs(type(full_metrics), OrderedDict)
        self.assertIs(type(metrics), OrderedDict)
        # use fc file
        full_metrics = self.OM.run_metrics(fcfiles=self.fc_file)
        metrics = full_metrics['XX']
        self.assertAlmostEqual(metrics['ant_phs_std_max'], 0.11690779502486655)
        self.assertEqual(len(metrics['ant_phs_std']), 17)

    def test_write_load_metrics(self):
        # Run metrics
        full_metrics = self.OM.run_metrics()
        metrics = full_metrics['XX']
        nkeys = len(metrics.keys())
        outfile = os.path.join(self.out_dir, 'omnical_metrics.json')
        if os.path.isfile(outfile):
            os.remove(outfile)
        # write json
        omnical_metrics.write_metrics(full_metrics, filename=outfile, filetype='json')
        self.assertTrue(os.path.isfile(outfile))
        # load json
        full_metrics_loaded = omnical_metrics.load_omnical_metrics(outfile)
        metrics_loaded = full_metrics_loaded['XX']
        self.assertEqual(len(metrics_loaded.keys()), nkeys)
        # erase
        os.remove(outfile)
        outfile = os.path.join(self.out_dir, 'omnical_metrics.pkl')
        if os.path.isfile(outfile):
            os.remove(outfile)
        # write pkl
        omnical_metrics.write_metrics(full_metrics, filename=outfile, filetype='pkl')
        self.assertTrue(os.path.isfile(outfile))
        # load
        full_metrics_loaded = omnical_metrics.load_omnical_metrics(outfile)
        metrics_loaded = full_metrics_loaded['XX']
        self.assertEqual(len(metrics_loaded.keys()), nkeys)
        ## check exceptions
        # load filetype
        os.remove(outfile)
        _ = open(outfile+'.wtf', 'a').close()
        self.assertRaises(IOError, omnical_metrics.load_omnical_metrics, filename=outfile+'.wtf')
        os.remove(outfile+'.wtf')
        # write w/o filename
        outfile = os.path.join(self.OM.filedir, self.OM.filestem+'.omni_metrics.json')
        omnical_metrics.write_metrics(full_metrics, filetype='json')
        os.remove(outfile)
        outfile = os.path.join(self.OM.filedir, self.OM.filestem+'.omni_metrics.pkl')
        omnical_metrics.write_metrics(full_metrics, filetype='pkl')
        os.remove(outfile)

    def test_plot_phs_metrics(self):
        # run metrics w/ fc file
        full_metrics = self.OM.run_metrics(fcfiles=self.fc_file)
        metrics = full_metrics['XX']
        # plot w/ fname, w/o outpath
        fname = os.path.join(self.OM.filedir, 'phs.png')
        if os.path.isfile(fname):
            os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='std', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='ft', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='hist', fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/ fname and outpath
        omnical_metrics.plot_phs_metric(metrics, plot_type='std', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='hist', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='ft', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.phs_std.png')
        if os.path.isfile(fname):
            os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='std', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.phs_ft.png')
        if os.path.isfile(fname):
            os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='ft', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.phs_hist.png')
        if os.path.isfile(fname):
            os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, plot_type='hist', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot feeding ax object
        fig,ax = plt.subplots()
        fname = os.path.join(self.OM.filedir, 'phs.png')
        if os.path.isfile(fname):
            os.remove(fname)
        omnical_metrics.plot_phs_metric(metrics, fname='phs.png', ax=ax, save=True)
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)
        plt.close()
        # exception
        del self.OM.omni_gains
        self.assertRaises(Exception, self.OM.plot_gains)
        self.OM.run_metrics()
        # check return figs
        fig = omnical_metrics.plot_phs_metric(metrics)
        self.assertTrue(fig is not None)
        plt.close()

    def test_plot_chisq_metrics(self):
        full_metrics = self.OM.run_metrics(fcfiles=self.fc_file)
        metrics = full_metrics['XX']
        fname = os.path.join(self.OM.filedir, 'chisq.png')
        if os.path.isfile(fname):
            os.remove(fname)
        # plot w/ fname
        omnical_metrics.plot_chisq_metric(metrics, fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/ fname and outpath
        omnical_metrics.plot_chisq_metric(metrics, fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.chisq_std.png')
        omnical_metrics.plot_chisq_metric(metrics, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot feeding ax object
        fig,ax = plt.subplots()
        omnical_metrics.plot_chisq_metric(metrics, ax=ax)
        plt.close()
        # check return figs
        fig = omnical_metrics.plot_chisq_metric(metrics)
        self.assertTrue(fig is not None)
        plt.close()

    def test_plot_chisq_tavg(self):
        self.OM.run_metrics(fcfiles=self.fc_file)
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
        plt.close()
        # check return figs
        fig = self.OM.plot_chisq_tavg()
        self.assertTrue(fig is not None)
        plt.close()


    def test_plot_gains(self):
        self.OM.run_metrics(fcfiles=self.fc_file)
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
        plt.close()
        # divide_fc = True
        self.OM.plot_gains(plot_type='phs', fname=fname, save=True, divide_fc=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        self.OM.plot_gains(plot_type='amp', fname=fname, save=True, divide_fc=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/ fname and outpath
        self.OM.plot_gains(plot_type='phs', fname=fname, save=True, outpath=self.OM.filedir)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot w/o fname
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.gain_phs.png')
        self.OM.plot_gains(plot_type='phs', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        fname = os.path.join(self.OM.filedir, self.OM.filename+'.gain_amp.png')
        self.OM.plot_gains(plot_type='amp', save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # plot feeding ax object
        fig,ax = plt.subplots()
        self.OM.plot_gains(ax=ax)
        plt.close()
        plt.close()
        # plot feeding ants
        fname = os.path.join(self.OM.filedir, 'gains.png')
        self.OM.plot_gains(ants=self.OM.ant_array[:2], fname=fname, save=True)
        self.assertEqual(os.path.isfile(fname), True)
        os.remove(fname)
        plt.close()
        # check return figs
        fig = self.OM.plot_gains()
        self.assertTrue(fig is not None)
        plt.close()

    def test_plot_metrics(self):
        # test execution
        full_metrics = self.OM.run_metrics(fcfiles=self.fc_file)
        metrics = full_metrics['XX']
        fname1 = os.path.join(self.OM.filedir, self.OM.filename+'.chisq_std.png')
        fname2 = os.path.join(self.OM.filedir, self.OM.filename+'.phs_std.png')
        fname3 = os.path.join(self.OM.filedir, self.OM.filename+'.phs_ft.png')
        for f in [fname1, fname2, fname3]:
            if os.path.isfile(f) == True:
                os.remove(f)
        self.OM.plot_metrics(metrics)
        self.assertEqual(os.path.isfile(fname1), True)
        self.assertEqual(os.path.isfile(fname2), True)
        self.assertEqual(os.path.isfile(fname3), True)
        os.remove(fname1)
        os.remove(fname2)
        os.remove(fname3)
        plt.close()

class Test_OmniCalMetrics_Run(unittest.TestCase):
    def setUp(self):
        self.fc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
        self.oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
        self.oc_filedir  = os.path.dirname(self.oc_file)
        self.oc_basename = os.path.basename(self.oc_file)
        self.out_dir = os.path.join(DATA_PATH, 'test_output')

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
        outfile = self.oc_file+'.omni_metrics.json'
        if os.path.isfile(outfile):
            os.remove(outfile)
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file+'.omni_metrics.json'
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ extension
        arguments = '--extension=.omni.json'
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        outfile = self.oc_file+'.omni.json'
        if os.path.isfile(outfile):
            os.remove(outfile)
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ metrics_path
        arguments = '--metrics_path={}'.format(self.out_dir)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        outfile = os.path.join(self.out_dir, self.oc_basename+'.omni_metrics.json')
        if os.path.isfile(outfile):
            os.remove(outfile)
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test w/ options
        arguments = '--fc_files={0} --no_bandcut --phs_std_cut=0.5 --chisq_std_zscore_cut=4.0'.format(self.fc_file)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        outfile = self.oc_file + '.omni_metrics.json'
        if os.path.isfile(outfile):
            os.remove(outfile)
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        self.assertTrue(os.path.isfile(outfile))
        os.remove(outfile)

        # test make plots
        arguments = '--fc_files={0} --make_plots'.format(self.fc_file)
        cmd = ' '.join([arguments, self.oc_file])
        args = a.parse_args(cmd.split())
        history = cmd
        omnical_metrics.omnical_metrics_run(args.files, args, history)
        outfile = self.oc_file + '.omni_metrics.json'
        outpng1 = self.oc_file + '.chisq_std.png'
        outpng2 = self.oc_file + '.phs_std.png'
        outpng3 = self.oc_file + '.phs_ft.png'
        outpng4 = self.oc_file + '.phs_hist.png'
        self.assertTrue(os.path.isfile(outfile))
        self.assertTrue(os.path.isfile(outpng1))
        self.assertTrue(os.path.isfile(outpng2))
        self.assertTrue(os.path.isfile(outpng3))
        self.assertTrue(os.path.isfile(outpng4))
        os.remove(outfile)
        os.remove(outpng1)
        os.remove(outpng2)
        os.remove(outpng3)
        os.remove(outpng4)



if __name__ == "__main__":
    unittest.main()

    
