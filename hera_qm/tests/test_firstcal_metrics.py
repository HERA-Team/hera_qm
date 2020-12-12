# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

"""
test_firstcal_metrics.py
"""
import numpy as np
from hera_qm import firstcal_metrics
from hera_qm.data import DATA_PATH
import os
from hera_qm import utils
import hera_qm.tests as qmtest
from hera_qm import metrics_io
import sys
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA."
)

@pytest.fixture(scope='function')
def firstcal_setup():
    infile = os.path.join(DATA_PATH, 'zen.2457555.50099.yy.HH.uvcA.first.calfits')
    FC = firstcal_metrics.FirstCalMetrics(infile)
    out_dir = os.path.join(DATA_PATH, 'test_output')

    class DataHolder():
        def __init__(self, FC, infile, out_dir):
            self.FC = FC
            self.infile = infile
            self.out_dir = out_dir
    firstcal_setup = DataHolder(FC, infile, out_dir)

    # yield returns the data we need but lets us continue after for cleanup
    yield firstcal_setup

    # post-test cleanup
    del(firstcal_setup)

    return


def test_init(firstcal_setup):
    assert firstcal_setup.FC.Nants == 17
    assert len(firstcal_setup.FC.delays) == 17


def test_run_metrics(firstcal_setup):
    firstcal_setup.FC.run_metrics(std_cut=1.0)
    assert firstcal_setup.FC.metrics['nn']['good_sol'] is True
    assert firstcal_setup.FC.metrics['nn']['bad_ants'] == []
    assert 9 in firstcal_setup.FC.metrics['nn']['z_scores']
    assert 9 in firstcal_setup.FC.metrics['nn']['ant_std']
    assert 9 in firstcal_setup.FC.metrics['nn']['ant_avg']
    assert 9 in firstcal_setup.FC.metrics['nn']['ants']
    assert 9 in firstcal_setup.FC.metrics['nn']['z_scores']
    assert 9 in firstcal_setup.FC.metrics['nn']['ant_z_scores']
    assert np.isclose(1.0, firstcal_setup.FC.metrics['nn']['std_cut'])
    assert np.isclose(firstcal_setup.FC.metrics['nn']['agg_std'], 0.044662349588061437)
    assert np.isclose(firstcal_setup.FC.metrics['nn']['max_std'], 0.089829821120782846)
    assert 'nn' == firstcal_setup.FC.metrics['nn']['pol']

    # Test bad ants detection
    firstcal_setup.FC.delay_fluctuations[0, :] *= 1000
    firstcal_setup.FC.run_metrics()
    assert firstcal_setup.FC.ants[0] == firstcal_setup.FC.metrics['nn']['bad_ants']
    # Test bad full solution
    firstcal_setup.FC.delay_fluctuations[1:, :] *= 1000
    firstcal_setup.FC.run_metrics()
    assert firstcal_setup.FC.metrics['nn']['good_sol'] is False


def test_write_error_bad_type(firstcal_setup):
    """Test an error is raised if bad filetype is given to write."""
    firstcal_setup.FC.run_metrics()
    outfile = os.path.join(firstcal_setup.out_dir, 'firstcal_metrics.npz')
    pytest.raises(ValueError, firstcal_setup.FC.write_metrics,
                  filename=outfile, filetype='npz')


def test_write_load_metrics(firstcal_setup):
    # run metrics
    firstcal_setup.FC.run_metrics()
    num_keys = len(firstcal_setup.FC.metrics.keys())
    outfile = os.path.join(firstcal_setup.out_dir, 'firstcal_metrics.json')
    if os.path.isfile(outfile):
        os.remove(outfile)
    # write json
    firstcal_setup.FC.write_metrics(filename=outfile, filetype='json')
    assert os.path.isfile(outfile)
    # load json
    firstcal_setup.FC.load_metrics(filename=outfile)
    assert len(firstcal_setup.FC.metrics.keys()) == num_keys
    # erase
    os.remove(outfile)
    # write pickle
    outfile = os.path.join(firstcal_setup.out_dir, 'firstcal_metrics.pkl')
    if os.path.isfile(outfile):
        os.remove(outfile)

    firstcal_setup.FC.write_metrics(filename=outfile, filetype='pkl')
    assert os.path.isfile(outfile)
    # load pickle
    firstcal_setup.FC.load_metrics(filename=outfile)
    assert len(firstcal_setup.FC.metrics.keys()) == num_keys
    os.remove(outfile)

    outfile = os.path.join(firstcal_setup.out_dir, 'firstcal_metrics.hdf5')
    if os.path.isfile(outfile):
        os.remove(outfile)
    firstcal_setup.FC.write_metrics(filename=outfile, filetype='hdf5')
    assert os.path.isfile(outfile)
    # load pickle
    firstcal_setup.FC.load_metrics(filename=outfile)
    # These are added by default in hdf5 writes but not necessary here
    firstcal_setup.FC.metrics.pop('history', None)
    firstcal_setup.FC.metrics.pop('version', None)
    assert len(firstcal_setup.FC.metrics.keys()) == num_keys
    os.remove(outfile)

    # Check some exceptions
    outfile = os.path.join(firstcal_setup.out_dir, 'firstcal_metrics.txt')
    pytest.raises(IOError, firstcal_setup.FC.load_metrics, filename=outfile)
    outfile = firstcal_setup.FC.fc_filestem + '.first_metrics.json'
    firstcal_setup.FC.write_metrics(filetype='json')  # No filename
    assert os.path.isfile(outfile)
    os.remove(outfile)

    outfile = firstcal_setup.FC.fc_filestem + '.first_metrics.pkl'
    firstcal_setup.FC.write_metrics(filetype='pkl')  # No filename
    assert os.path.isfile(outfile)
    os.remove(outfile)

    outfile = firstcal_setup.FC.fc_filestem + '.first_metrics.hdf5'
    firstcal_setup.FC.write_metrics(filetype='hdf5')  # No filename
    assert os.path.isfile(outfile)
    os.remove(outfile)


def test_plot_delays(firstcal_setup):
    plt = pytest.importorskip("matplotlib.pyplot")
    fname = os.path.join(firstcal_setup.out_dir, 'dlys.png')
    if os.path.isfile(fname):
        os.remove(fname)
    firstcal_setup.FC.plot_delays(fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    firstcal_setup.FC.plot_delays(fname=fname, save=True, plot_type='solution')
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    firstcal_setup.FC.plot_delays(fname=fname, save=True, plot_type='fluctuation')
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')

    # Check cm defaults to spectral
    firstcal_setup.FC.plot_delays(fname=fname, save=True, cmap='foo')
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # check return figs
    fig = firstcal_setup.FC.plot_delays()
    assert fig is not None
    plt.close('all')


def test_plot_zscores(firstcal_setup):
    plt = pytest.importorskip("matplotlib.pyplot")
    # check exception
    pytest.raises(NameError, firstcal_setup.FC.plot_zscores)
    firstcal_setup.FC.run_metrics()
    pytest.raises(NameError, firstcal_setup.FC.plot_zscores, plot_type='foo')
    # check output
    fname = os.path.join(firstcal_setup.out_dir, 'zscrs.png')
    if os.path.isfile(fname):
        os.remove(fname)
    firstcal_setup.FC.plot_zscores(fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    firstcal_setup.FC.plot_zscores(fname=fname, plot_type='time_avg', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # check return fig
    fig = firstcal_setup.FC.plot_zscores()
    assert fig is not None
    plt.close('all')


def test_plot_stds(firstcal_setup):
    plt = pytest.importorskip("matplotlib.pyplot")
    # check exception
    pytest.raises(NameError, firstcal_setup.FC.plot_stds)
    firstcal_setup.FC.run_metrics()
    pytest.raises(NameError, firstcal_setup.FC.plot_stds, xaxis='foo')
    # check output
    fname = os.path.join(firstcal_setup.out_dir, 'stds.png')
    if os.path.isfile(fname):
        os.remove(fname)
    firstcal_setup.FC.plot_stds(fname=fname, save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    firstcal_setup.FC.plot_stds(fname=fname, xaxis='time', save=True)
    assert os.path.isfile(fname)
    os.remove(fname)
    plt.close('all')
    # check return fig
    fig = firstcal_setup.FC.plot_stds()
    assert fig is not None
    plt.close('all')


def test_rotated_metrics():
    infile = os.path.join(DATA_PATH, 'zen.2457555.42443.xx.HH.uvcA.bad.first.calfits')
    FC = firstcal_metrics.FirstCalMetrics(infile)
    FC.run_metrics(std_cut=0.5)
    # test pickup of rotant key
    assert 'rot_ants' in FC.metrics['ee'].keys()
    # test rotants is correct
    assert [43] == FC.metrics['ee']['rot_ants']


def test_delay_smoothing():
    infile = os.path.join(DATA_PATH, 'zen.2457555.50099.yy.HH.uvcA.first.calfits')
    np.random.seed(0)
    FC = firstcal_metrics.FirstCalMetrics(infile, use_gp=False)
    assert np.isclose(FC.delay_fluctuations[0, 0], 0.043740587980040324, atol=0.000001)
    np.random.seed(0)
    FC = firstcal_metrics.FirstCalMetrics(infile, use_gp=True)
    assert np.isclose(FC.delay_fluctuations[0, 0], 0.024669144881121961, atol=0.000001)


@pytest.fixture(scope='function')
def firstcal_twopol():
    infile = os.path.join(DATA_PATH, 'zen.2458098.49835.HH.first.calfits')
    FC = firstcal_metrics.FirstCalMetrics(infile)
    out_dir = os.path.join(DATA_PATH, 'test_output')

    class DataHolder(object):
        def __init__(self, FC, infile, out_dir):
            self.FC = FC
            self.infile = infile
            self.out_dir = out_dir
    firstcal_twopol = DataHolder(FC, infile, out_dir)

    # yield returns the data we need but lets us continue after for cleanup
    yield firstcal_twopol

    # post-test cleanup
    del(firstcal_twopol)


def test_init_two_pol(firstcal_twopol):
    assert firstcal_twopol.FC.Nants == 11
    assert len(firstcal_twopol.FC.delays) == 11


def test_run_metrics_two_pols(firstcal_twopol):
    # These results were run with a seed of 0, the seed shouldn't matter
    # but you never know.
    two_pol_known_results = os.path.join(DATA_PATH, 'example_two_polarization_firstcal_results.hdf5')
    np.random.seed(0)
    firstcal_twopol.FC.run_metrics(std_cut=.5)
    known_output = metrics_io.load_metric_file(two_pol_known_results)

    known_output.pop('history', None)
    known_output.pop('version', None)
    # There are some full paths of files saved in the files
    # Perhaps for record keeping, but that messes up the test comparison
    for key in known_output:
        known_output[key].pop('fc_filename', None)
        known_output[key].pop('fc_filestem', None)
        known_output[key].pop('version', None)
    for key in firstcal_twopol.FC.metrics:
        firstcal_twopol.FC.metrics[key].pop('fc_filename', None)
        firstcal_twopol.FC.metrics[key].pop('fc_filestem', None)
        firstcal_twopol.FC.metrics[key].pop('version', None)
    assert qmtest.recursive_compare_dicts(firstcal_twopol.FC.metrics, known_output)


def test_firstcal_metrics_run():
    # get argument object
    a = utils.get_metrics_ArgumentParser('firstcal_metrics')
    if DATA_PATH not in sys.path:
        sys.path.append(DATA_PATH)

    arg0 = "--std_cut=0.5"
    arg1 = "--extension=.firstcal_metrics.hdf5"
    arg2 = "--metrics_path={}".format(os.path.join(DATA_PATH, 'test_output'))
    arg3 = "--filetype=h5"
    arguments = ' '.join([arg0, arg1, arg2, arg3])

    # Test runing with no files
    cmd = ' '.join([arguments, ''])
    args = a.parse_args(cmd.split())
    history = cmd
    pytest.raises(AssertionError, firstcal_metrics.firstcal_metrics_run,
                  args.files, args, history)

    # Test running with file
    filename = os.path.join(DATA_PATH, 'zen.2457555.50099.yy.HH.uvcA.first.calfits')
    dest_file = os.path.join(DATA_PATH, 'test_output',
                             'zen.2457555.50099.yy.HH.uvcA.'
                             + 'firstcal_metrics.hdf5')
    if os.path.exists(dest_file):
        os.remove(dest_file)
    cmd = ' '.join([arguments, filename])
    args = a.parse_args(cmd.split())
    history = cmd
    firstcal_metrics.firstcal_metrics_run(args.files, args, history)
    assert os.path.exists(dest_file)
    os.remove(dest_file)
