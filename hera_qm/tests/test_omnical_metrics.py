# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

from hera_qm import omnical_metrics
from hera_qm.data import DATA_PATH
import os
from hera_qm import utils
import sys
from collections import OrderedDict
import pytest
import numpy as np

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA."
)

@pytest.fixture(scope='function')
def omnical_data():
    fc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
    fc_file_bad = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.bad.first.calfits')
    fc_fileyy = os.path.join(DATA_PATH, 'zen.2457555.42443.yy.HH.uvcA.first.calfits')
    oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
    out_dir = os.path.join(DATA_PATH, 'test_output')
    OM = omnical_metrics.OmniCal_Metrics(oc_file)

    class DataHolder():
        def __init__(self, OM, fc_file, fc_file_bad, fc_fileyy, oc_file, out_dir):
            self.OM = OM
            self.fc_file = fc_file
            self.fc_file_bad = fc_file_bad
            self.fc_fileyy = fc_fileyy
            self.oc_file = oc_file
            self.out_dir = out_dir
    omnical_data = DataHolder(OM, fc_file, fc_file_bad, fc_fileyy, oc_file, out_dir)

    # yield returns the data we need but lets us continue after for cleanup
    yield omnical_data

    # post-test cleanup
    del(omnical_data)


def test_initialization(omnical_data):
    assert omnical_data.OM.Nants == 17
    assert omnical_data.OM.filename == 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits'
    assert omnical_data.OM.omni_gains.shape == (17, 3, 1024, 1)


def test_load_firstcal_gains(omnical_data):
    firstcal_delays, firstcal_gains, fc_pol = omnical_metrics.load_firstcal_gains(omnical_data.fc_file)
    assert firstcal_delays.shape == (17, 3)
    assert firstcal_gains.shape == (17, 3, 1024)
    assert fc_pol == -5


def test_omni_load_firstcal_gains(omnical_data):
    omnical_data.OM.load_firstcal_gains(omnical_data.fc_file)
    assert hasattr(omnical_data.OM, 'gain_diff')
    # test exception
    pytest.raises(ValueError, omnical_data.OM.load_firstcal_gains, omnical_data.fc_fileyy)


def test_multiple_pol_fc_pickup(omnical_data):
    fc_files = [os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits'), os.path.join(DATA_PATH, 'zen.2457555.42443.yy.HH.uvcA.first.calfits')]
    oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.HH.uvcA.omni.calfits')
    OM = omnical_metrics.OmniCal_Metrics(oc_file)
    OM.load_firstcal_gains(fc_files)
    # make sure 'XX' pol first, then 'YY' pol
    assert np.isclose(OM.firstcal_delays[0, 0, 0], 1.0918842818239246e-09)
    # reverse pol and repeat
    OM.load_firstcal_gains(fc_files[::-1])
    assert np.isclose(OM.firstcal_delays[0, 0, 0], 1.0918842818239246e-09)
    # hit exception
    OM.pols[0] = 'XY'
    pytest.raises(ValueError, OM.load_firstcal_gains, fc_files)


def test_run_metrics(omnical_data):
    # no fc file
    full_metrics = omnical_data.OM.run_metrics()
    metrics = full_metrics['XX']
    assert metrics['ant_phs_std'] is None
    assert np.isclose(metrics['chisq_ant_std_loc'], 0.16499103579233421)
    assert len(full_metrics) == 1
    assert len(metrics) == 36
    assert type(full_metrics) is OrderedDict
    assert type(metrics) is OrderedDict
    # no cut band
    full_metrics = omnical_data.OM.run_metrics(cut_edges=False)
    metrics = full_metrics['XX']
    assert metrics['ant_phs_std'] is None
    assert np.isclose(metrics['chisq_ant_std_loc'], 0.17762839199226452)
    assert len(full_metrics) == 1
    assert len(metrics) == 36
    assert type(full_metrics) is OrderedDict
    assert type(metrics) is OrderedDict
    # use fc file
    full_metrics = omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    metrics = full_metrics['XX']
    assert np.isclose(metrics['ant_phs_std_max'], 0.11690779502486655)
    assert len(metrics['ant_phs_std']) == 17


def test_write_load_metrics(omnical_data):
    # Run metrics
    full_metrics = omnical_data.OM.run_metrics()
    metrics = full_metrics['XX']
    nkeys = len(metrics.keys())
    outfile = os.path.join(omnical_data.out_dir, 'omnical_metrics.json')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # write json
    omnical_metrics.write_metrics(full_metrics, filename=outfile, filetype='json')
    assert os.path.isfile(outfile)
    # load json
    full_metrics_loaded = omnical_metrics.load_omnical_metrics(outfile)
    metrics_loaded = full_metrics_loaded['XX']
    assert len(metrics_loaded.keys()) == nkeys
    # erase
    os.remove(outfile)
    outfile = os.path.join(omnical_data.out_dir, 'omnical_metrics.pkl')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # write pkl
    omnical_metrics.write_metrics(full_metrics, filename=outfile, filetype='pkl')
    assert os.path.isfile(outfile)
    # load
    full_metrics_loaded = omnical_metrics.load_omnical_metrics(outfile)
    metrics_loaded = full_metrics_loaded['XX']
    assert len(metrics_loaded.keys()) == nkeys
    # check exceptions
    # load filetype
    os.remove(outfile)
    _ = open(outfile + '.wtf', 'a').close()
    pytest.raises(IOError, omnical_metrics.load_omnical_metrics,
                  filename=outfile + '.wtf')
    os.remove(outfile + '.wtf')
    # write w/o filename
    outfile = os.path.join(omnical_data.OM.filedir, omnical_data.OM.filestem + '.omni_metrics.json')
    omnical_metrics.write_metrics(full_metrics, filetype='json')
    os.remove(outfile)
    outfile = os.path.join(omnical_data.OM.filedir, omnical_data.OM.filestem + '.omni_metrics.pkl')
    omnical_metrics.write_metrics(full_metrics, filetype='pkl')
    os.remove(outfile)


def test_plot_phs_metrics(omnical_data):
    plt = pytest.importorskip("matplotlib.pyplot")

    # run metrics w/ fc file
    full_metrics = omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    metrics = full_metrics['XX']
    # plot w/ fname, w/o outpath
    fname = os.path.join(omnical_data.OM.filedir, 'phs.png')
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='std', fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='ft', fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='hist', fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/ fname and outpath
    omnical_metrics.plot_phs_metric(metrics, plot_type='std', fname=fname, save=True, outpath=omnical_data.OM.filedir)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='hist', fname=fname, save=True, outpath=omnical_data.OM.filedir)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='ft', fname=fname, save=True, outpath=omnical_data.OM.filedir)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/o fname
    basename = utils.strip_extension(omnical_data.OM.filename)
    fname = os.path.join(omnical_data.OM.filedir, basename + '.phs_std.png')
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='std', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    fname = os.path.join(omnical_data.OM.filedir, basename + '.phs_ft.png')
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='ft', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    fname = os.path.join(omnical_data.OM.filedir, basename + '.phs_hist.png')
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, plot_type='hist', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot feeding ax object
    fig, ax = plt.subplots()
    fname = os.path.join(omnical_data.OM.filedir, 'phs.png')
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_metrics.plot_phs_metric(metrics, fname='phs.png', ax=ax, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # exception
    del omnical_data.OM.omni_gains
    pytest.raises(Exception, omnical_data.OM.plot_gains)
    omnical_data.OM.run_metrics()
    # check return figs
    fig = omnical_metrics.plot_phs_metric(metrics)
    assert fig is not None
    plt.close('all')


def test_plot_chisq_metrics(omnical_data):
    plt = pytest.importorskip("matplotlib.pyplot")
    full_metrics = omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    metrics = full_metrics['XX']
    fname = os.path.join(omnical_data.OM.filedir, 'chisq.png')
    if os.path.isfile(fname):
        os.remove(fname)
    # plot w/ fname
    omnical_metrics.plot_chisq_metric(metrics, fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/ fname and outpath
    omnical_metrics.plot_chisq_metric(metrics, fname=fname, save=True, outpath=omnical_data.OM.filedir)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/o fname
    fname = os.path.join(omnical_data.OM.filedir, utils.strip_extension(omnical_data.OM.filename)
                         + '.chisq_std.png')
    omnical_metrics.plot_chisq_metric(metrics, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot feeding ax object
    fig, ax = plt.subplots()
    omnical_metrics.plot_chisq_metric(metrics, ax=ax)
    plt.close('all')
    # check return figs
    fig = omnical_metrics.plot_chisq_metric(metrics)
    assert fig is not None
    plt.close('all')


def test_plot_chisq_tavg(omnical_data):
    plt = pytest.importorskip("matplotlib.pyplot")
    omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    fname = os.path.join(omnical_data.OM.filedir, 'chisq_tavg.png')
    # test execution
    if os.path.isfile(fname):
        os.remove(fname)
    omnical_data.OM.plot_chisq_tavg(fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_data.OM.plot_chisq_tavg(ants=omnical_data.OM.ant_array[:2], fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # check return figs
    fig = omnical_data.OM.plot_chisq_tavg()
    assert fig is not None
    plt.close('all')


def test_plot_gains(omnical_data):
    plt = pytest.importorskip("matplotlib.pyplot")
    omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    fname = os.path.join(omnical_data.OM.filedir, 'gains.png')
    if os.path.isfile(fname):
        os.remove(fname)
    # plot w/ fname
    omnical_data.OM.plot_gains(plot_type='phs', fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_data.OM.plot_gains(plot_type='amp', fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # divide_fc = True
    omnical_data.OM.plot_gains(plot_type='phs', fname=fname, save=True, divide_fc=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    omnical_data.OM.plot_gains(plot_type='amp', fname=fname, save=True, divide_fc=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/ fname and outpath
    omnical_data.OM.plot_gains(plot_type='phs', fname=fname, save=True, outpath=omnical_data.OM.filedir)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot w/o fname
    fname = os.path.join(omnical_data.OM.filedir, utils.strip_extension(omnical_data.OM.filename)
                         + '.gain_phs.png')
    omnical_data.OM.plot_gains(plot_type='phs', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    fname = os.path.join(omnical_data.OM.filedir, utils.strip_extension(omnical_data.OM.filename)
                         + '.gain_amp.png')
    omnical_data.OM.plot_gains(plot_type='amp', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # plot feeding ax object
    fig, ax = plt.subplots()
    omnical_data.OM.plot_gains(ax=ax)
    plt.close('all')
    # plot feeding ants
    fname = os.path.join(omnical_data.OM.filedir, 'gains.png')
    omnical_data.OM.plot_gains(ants=omnical_data.OM.ant_array[:2], fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # check return figs
    fig = omnical_data.OM.plot_gains()
    assert fig is not None
    plt.close('all')


def test_plot_metrics(omnical_data):
    plt = pytest.importorskip("matplotlib.pyplot")
    # test execution
    full_metrics = omnical_data.OM.run_metrics(fcfiles=omnical_data.fc_file)
    metrics = full_metrics['XX']
    basename = utils.strip_extension(omnical_data.OM.filename)
    fname1 = os.path.join(omnical_data.OM.filedir, basename + '.chisq_std.png')
    fname2 = os.path.join(omnical_data.OM.filedir, basename + '.phs_std.png')
    fname3 = os.path.join(omnical_data.OM.filedir, basename + '.phs_ft.png')
    for f in [fname1, fname2, fname3]:
        if os.path.isfile(f) is True:
            os.remove(f)
    omnical_data.OM.plot_metrics(metrics)
    assert os.path.isfile(fname1)
    assert os.path.isfile(fname2)
    assert os.path.isfile(fname3)
    os.remove(fname1)
    os.remove(fname2)
    os.remove(fname3)
    plt.close('all')


@pytest.fixture(scope='function')
def omnicalrun_data():
    fc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.first.calfits')
    oc_file = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.good.omni.calfits')
    oc_filedir = os.path.dirname(oc_file)
    oc_basename = os.path.basename(oc_file)
    out_dir = os.path.join(DATA_PATH, 'test_output')

    class DataHolder(object):
        def __init__(self, fc_file, oc_file, oc_filedir, oc_basename, out_dir):
            self.fc_file = fc_file
            self.oc_file = oc_file
            self.oc_filedir = oc_filedir
            self.oc_basename = oc_basename
            self.out_dir = out_dir

    omnicalrun_data = DataHolder(fc_file, oc_file, oc_filedir, oc_basename, out_dir)
    # yield returns the data we need but lets us continue after for cleanup
    yield omnicalrun_data

    # post-test cleanup
    del(omnicalrun_data)


def test_firstcal_metrics_run(omnicalrun_data):
    pytest.importorskip('matplotlib.pyplot')
    # get arg parse
    a = utils.get_metrics_ArgumentParser('omnical_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)

    # test w/ no file
    arguments = ''
    cmd = ' '.join([arguments, ''])
    args = a.parse_args(cmd.split())
    history = cmd
    pytest.raises(AssertionError, omnical_metrics.omnical_metrics_run,
                  args.files, args, history)
    # test w/ no fc_file
    arguments = ''
    cmd = ' '.join([arguments, omnicalrun_data.oc_file])
    args = a.parse_args(cmd.split())
    history = cmd
    outfile = utils.strip_extension(omnicalrun_data.oc_file) + '.omni_metrics.json'
    if os.path.isfile(outfile):
        os.remove(outfile)
    omnical_metrics.omnical_metrics_run(args.files, args, history)
    assert os.path.isfile(outfile)
    os.remove(outfile)

    # test w/ extension
    arguments = '--extension=.omni.json'
    cmd = ' '.join([arguments, omnicalrun_data.oc_file])
    args = a.parse_args(cmd.split())
    history = cmd
    outfile = utils.strip_extension(omnicalrun_data.oc_file) + '.omni.json'
    if os.path.isfile(outfile):
        os.remove(outfile)
    omnical_metrics.omnical_metrics_run(args.files, args, history)
    assert os.path.isfile(outfile)
    os.remove(outfile)

    # test w/ metrics_path
    arguments = '--metrics_path={}'.format(omnicalrun_data.out_dir)
    cmd = ' '.join([arguments, omnicalrun_data.oc_file])
    args = a.parse_args(cmd.split())
    history = cmd
    outfile = os.path.join(omnicalrun_data.out_dir,
                           utils.strip_extension(omnicalrun_data.oc_basename)
                           + '.omni_metrics.json')
    if os.path.isfile(outfile):
        os.remove(outfile)
    omnical_metrics.omnical_metrics_run(args.files, args, history)
    assert os.path.isfile(outfile)
    os.remove(outfile)

    # test w/ options
    arguments = '--fc_files={0} --no_bandcut --phs_std_cut=0.5 --chisq_std_zscore_cut=4.0'.format(omnicalrun_data.fc_file)
    cmd = ' '.join([arguments, omnicalrun_data.oc_file])
    args = a.parse_args(cmd.split())
    history = cmd
    outfile = utils.strip_extension(omnicalrun_data.oc_file) + '.omni_metrics.json'
    if os.path.isfile(outfile):
        os.remove(outfile)
    omnical_metrics.omnical_metrics_run(args.files, args, history)
    assert os.path.isfile(outfile)
    os.remove(outfile)

    # test make plots
    arguments = '--fc_files={0} --make_plots'.format(omnicalrun_data.fc_file)
    cmd = ' '.join([arguments, omnicalrun_data.oc_file])
    args = a.parse_args(cmd.split())
    history = cmd
    omnical_metrics.omnical_metrics_run(args.files, args, history)
    basename = utils.strip_extension(omnicalrun_data.oc_file)
    outfile = basename + '.omni_metrics.json'
    outpng1 = basename + '.chisq_std.png'
    outpng2 = basename + '.phs_std.png'
    outpng3 = basename + '.phs_ft.png'
    outpng4 = basename + '.phs_hist.png'
    assert os.path.isfile(outfile)
    assert os.path.isfile(outpng1)
    assert os.path.isfile(outpng2)
    assert os.path.isfile(outpng3)
    assert os.path.isfile(outpng4)
    os.remove(outfile)
    os.remove(outpng1)
    os.remove(outpng2)
    os.remove(outpng3)
    os.remove(outpng4)
