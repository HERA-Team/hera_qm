# -*- coding: utf-8 -*-
# Copyright (c) 2019 the HERA Project
# Licensed under the MIT License

import pytest
import os
import pyuvdata.tests as uvtest
from hera_qm import utils
from hera_qm.data import DATA_PATH
from hera_qm.ant_metrics import get_ant_metrics_dict
from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
from hera_qm.omnical_metrics import get_omnical_metrics_dict
from hera_qm.utils import get_metrics_dict
import numpy as np
from pyuvdata import UVData
from pyuvdata import UVCal
import pyuvdata.utils as uvutils

pytestmark = pytest.mark.filterwarnings(
    "ignore:The uvw_array does not match the expected values given the antenna positions.",
    "ignore:telescope_location is not set. Using known values for HERA.",
    "ignore:antenna_positions is not set. Using known values for HERA."
)

def test_get_metrics_ArgumentParser_ant_metrics():
    a = utils.get_metrics_ArgumentParser('ant_metrics')
    # First try defaults - test a few of them
    args = a.parse_args(['sum_file1', 'sum_file2'])
    assert args.sum_files == ['sum_file1', 'sum_file2']
    assert args.crossCut == 0.0
    assert args.metrics_path == ''
    assert args.verbose is True
    # try to set something
    args = a.parse_args(['sum_file1', '--extension', 'foo'])
    assert args.extension == 'foo'

def test_get_metrics_ArgumentParser_auto_metrics():
    a = utils.get_metrics_ArgumentParser('auto_metrics')
    # First try defaults - test a few of them
    args = a.parse_args(['out.h5', 'auto1.uvh5', 'auto2.uvh5'])
    assert args.metric_outfile == 'out.h5'
    assert args.raw_auto_files == ['auto1.uvh5', 'auto2.uvh5']
    assert args.median_round_modz_cut == 8.0
    assert args.chan_thresh_frac == .05
    # try to set something
    args = a.parse_args(['out.h5', 'auto1.uvh5', 'auto2.uvh5', '--sig_init', '10.3'])
    assert args.sig_init == 10.3

def test_get_metrics_ArgumentParser_firstcal_metrics():
    a = utils.get_metrics_ArgumentParser('firstcal_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.std_cut == 0.5
    assert args.extension == '.firstcal_metrics.hdf5'
    assert args.metrics_path == ''
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    assert args.extension == 'foo'


def test_get_metrics_ArgumentParser_omnical_metrics():
    a = utils.get_metrics_ArgumentParser('omnical_metrics')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.fc_files is None
    assert args.phs_std_cut == 0.3
    assert args.extension == '.omni_metrics.json'
    assert args.metrics_path == ''
    # try to set something
    args = a.parse_args(['--extension', 'foo'])
    assert args.extension, 'foo'


def test_get_metrics_ArgumentParser_xrfi_h1c_run():
    a = utils.get_metrics_ArgumentParser('xrfi_h1c_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.summary_ext == 'flag_summary.h5'
    assert args.algorithm == 'xrfi_simple'
    assert args.nsig_df == 6.0
    assert args.px_threshold == 0.2
    # try to set something
    args = a.parse_args(['--px_threshold', '4.0'])
    assert args.px_threshold == 4.0


def test_get_metrics_ArgumentParser_delay_xrfi_h1c_idr2_1_run():
    a = utils.get_metrics_ArgumentParser('delay_xrfi_h1c_idr2_1_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.algorithm == 'xrfi_simple'
    assert args.nsig_dt == 6.0
    assert args.px_threshold == 0.2
    assert args.filename is None
    assert args.tol == 1e-7
    assert args.waterfalls is None
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    assert args.waterfalls == 'a,g'


def test_get_metrics_ArgumentParser_xrfi_run():
    a = utils.get_metrics_ArgumentParser('xrfi_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init_mean == 5.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj_mean', '3.0'])
    assert args.sig_adj_mean == 3.0


def test_get_metrics_ArgumentParser_xrfi_run_data_only():
    a = utils.get_metrics_ArgumentParser('xrfi_run_data_only')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init_med == 10.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj_med', '3.0'])
    assert args.sig_adj_med == 3.0


def test_get_metrics_ArgumentParser_day_threshold_run():
    a = utils.get_metrics_ArgumentParser('day_threshold_run')
    # First try defaults - test a few of them
    args = a.parse_args(['fooey'])
    assert args.nsig_f_adj == 3.0
    assert args.nsig_f == 7.0
    assert args.data_files == ['fooey']
    # try to set something
    args = a.parse_args(['--nsig_t', '3.0', 'fooey'])
    assert args.nsig_t == 3.0


def test_get_metrics_ArgumentParser_xrfi_apply():
    a = utils.get_metrics_ArgumentParser('xrfi_apply')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.infile_format == 'miriad'
    assert args.extension == 'R'
    assert args.flag_file is None
    assert args.output_uvflag is True
    assert args.output_uvflag_ext == 'flags.h5'
    # try to set something
    args = a.parse_args(['--waterfalls', 'a,g'])
    assert args.waterfalls == 'a,g'


def test_get_metrics_ArgumentParser_error():
    # raise error for requesting unknown type of parser
    pytest.raises(AssertionError, utils.get_metrics_ArgumentParser, 'fake_method')


def test_get_metrics_ArgumentParser_xrfi_h3c_idr2_1_run():
    a = utils.get_metrics_ArgumentParser('xrfi_h3c_idr2_1_run')
    # First try defaults - test a few of them
    args = a.parse_args('')
    assert args.kt_size == 8
    assert args.sig_init == 6.0
    assert args.ex_ants is None
    # try to set something
    args = a.parse_args(['--sig_adj', '3.0', '--ocalfits_files', 'foo', 'boo'])
    assert args.sig_adj == 3.0
    assert len(args.ocalfits_files) == 2


def test_metrics2mc():
    # test ant metrics
    filename = os.path.join(DATA_PATH, 'example_ant_metrics.hdf5')
    d = utils.metrics2mc(filename, ftype='ant')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    assert len(d['array_metrics']) == 0
    ant_metrics_list = get_ant_metrics_dict()
    for k in set(d['ant_metrics'].keys()):
        assert k in ant_metrics_list

    # test firstcal metrics
    filename = os.path.join(DATA_PATH, 'example_firstcal_metrics.json')
    d = utils.metrics2mc(filename, ftype='firstcal')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    firstcal_array_metrics = set(['firstcal_metrics_agg_std_y',
                                  'firstcal_metrics_good_sol_y',
                                  'firstcal_metrics_max_std_y'])
    assert set(d['array_metrics'].keys()) == firstcal_array_metrics
    firstcal_metrics_list = get_firstcal_metrics_dict()
    firstcal_ant_metrics = set(firstcal_metrics_list.keys()) - firstcal_array_metrics
    # remove others not in this data file
    firstcal_ant_metrics -= {'firstcal_metrics_good_sol_x', 'firstcal_metrics_good_sol',
                             'firstcal_metrics_agg_std_x', 'firstcal_metrics_agg_std',
                             'firstcal_metrics_max_std_x'}
    assert set(d['ant_metrics']) == firstcal_ant_metrics

    # test omnical metrics
    filename = os.path.join(DATA_PATH, 'example_omnical_metrics.json')
    d = utils.metrics2mc(filename, ftype='omnical')
    assert set(d.keys()) == set(['ant_metrics', 'array_metrics'])
    om = 'omnical_metrics_'
    assert set(d['array_metrics'].keys()) == set([om + 'chisq_tot_avg_XX',
                                                  om + 'chisq_good_sol_XX',
                                                  om + 'chisq_tot_avg_YY',
                                                  om + 'chisq_good_sol_YY',
                                                  om + 'ant_phs_std_max_XX',
                                                  om + 'ant_phs_std_good_sol_XX',
                                                  om + 'ant_phs_std_max_YY',
                                                  om + 'ant_phs_std_good_sol_YY'])
    assert set(d['ant_metrics'].keys()) == set([om + 'chisq_ant_avg',
                                                om + 'chisq_ant_std',
                                                om + 'ant_phs_std'])
    assert len(d['ant_metrics'][om + 'chisq_ant_avg']) == 32

    # Hit the exceptions
    pytest.raises(ValueError, utils.metrics2mc, filename, ftype='foo')


def test_get_metrics_dict():
    ant_metrics_dict = get_ant_metrics_dict()
    firstcal_metrics_dict = get_firstcal_metrics_dict()
    omnical_metrics_dict = get_omnical_metrics_dict()
    metrics_dict = get_metrics_dict()
    for key in ant_metrics_dict:
        assert ant_metrics_dict[key] == metrics_dict[key]
    for key in firstcal_metrics_dict:
        assert firstcal_metrics_dict[key] == metrics_dict[key]
    for key in omnical_metrics_dict:
        assert omnical_metrics_dict[key] == metrics_dict[key]


def test_dynamic_slice():
    a = np.arange(10).reshape(2, 5)
    b = utils.dynamic_slice(a, slice(1, 3))
    assert b.shape == (2, 2)
    np.testing.assert_array_equal(b, np.array([[1, 2], [6, 7]]))
    b = utils.dynamic_slice(a, slice(1, 3), axis=1)
    assert b.shape == (2, 2)
    b = utils.dynamic_slice(a, slice(1, None), axis=0)
    assert b.shape == (1, 5)

    pytest.raises(ValueError, utils.dynamic_slice, 'foo', slice(0, None))


def test_strip_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root = utils.strip_extension(path)
    assert root == path[:-4]


def test_strip_extension_return_ext_basename():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    assert root == path[:-4]


def test_strip_extension_return_ext_extension():
    path = 'goo/foo.boo/hoo/woo.two'
    root, ext = utils.strip_extension(path, return_ext=True)
    assert ext == path[-3:]

@pytest.mark.parametrize(
    "filein",
    ["a_priori_flags_integrations.yaml",
    "a_priori_flags_jds.yaml",
    "a_priori_flags_lsts.yaml",
    "a_priori_flags_no_integrations.yaml",
    "a_priori_flags_no_chans.yaml",
    "a_priori_flags_no_ants.yaml",
    ],
)
@pytest.mark.parametrize("flag_freqs", [True, False])
@pytest.mark.parametrize("flag_times", [True, False])
@pytest.mark.parametrize("flag_ants", [True, False])
def test_apply_yaml_flags_uvdata(tmpdir, filein, flag_freqs, flag_times, flag_ants):
    tmp_path = tmpdir.strpath
    test_d_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvh5')
    test_flag = os.path.join(DATA_PATH, filein)
    # first test flagging uvdata object
    freq_regions = [(0, 110e6), (150e6, 155e6), (190e6, 200e6)] # frequencies from yaml file.
    channel_flags = [0, 1, 60] + list(range(10, 21)) # channels from yaml file.
    integration_flags = [0, 1] # integrations from yaml file that should be flagged.
    ant_flags = [0, 10, [1, 'Jee'], [3, 'Jnn']]
    uvd = UVData()
    uvd.read(test_d_file)
    uvd = utils.apply_yaml_flags(uvd, test_flag, flag_freqs=flag_freqs, flag_times=flag_times,
                                flag_ants=flag_ants, unflag_first=True)
    if 'no_integrations' not in test_flag:
        for tind in integration_flags:
            time = sorted(np.unique(uvd.time_array))[tind]
            if flag_times:
                assert np.all(uvd.flag_array[uvd.time_array == time, :, :, :])
            else:
                assert not np.all(uvd.flag_array[uvd.time_array == time, :, :, :])\
                    or np.count_nonzero(uvd.time_array == time) == 0
    if 'no_chans' not in test_flag:
        for region in freq_regions:
            selection = (uvd.freq_array[0] >= region[0]) & (uvd.freq_array[0] <= region[-1])
            if flag_freqs:
                assert np.all(uvd.flag_array[:, :, selection, :])
            else:
                assert not np.all(uvd.flag_array[:, :, selection, :])\
                    or np.count_nonzero(selection) == 0

        for chan in channel_flags:
            if flag_freqs:
                assert np.all(uvd.flag_array[:, :, chan, :])
            else:
                assert not np.all(uvd.flag_array[:, :, chan, :])
    if 'no_ants' not in test_flag:
        for ant in ant_flags:
            if isinstance(ant, int):
                antnum = ant
                pol_selection = np.ones(uvd.Npols, dtype=bool)
            elif isinstance(ant, (list, tuple)):
                antnum = ant[0]
                pol_num = uvutils.jstr2num(ant[1], x_orientation=uvd.x_orientation)
                pol_selection = np.where(uvd.polarization_array == pol_num)[0]
            blt_selection = np.logical_or(uvd.ant_1_array == antnum, uvd.ant_2_array == antnum)
            if flag_ants:
                assert np.all(uvd.flag_array[blt_selection][: , :, :, pol_selection])
            else:
                assert not np.all(uvd.flag_array[blt_selection][: , :, :, pol_selection])\
                or np.count_nonzero(blt_selection) == 0 or np.count_nonzero(pol_selection) == 0

        # test removing antennas from the data if we are not running no_ants:
        if flag_ants:
            all_ants = np.unique(np.hstack([uvd.ant_1_array, uvd.ant_2_array]))
            uvd = utils.apply_yaml_flags(uvd, test_flag, flag_freqs=flag_freqs, flag_times=flag_times,
                                        flag_ants=flag_ants, throw_away_flagged_ants=True, ant_indices_only=True)
            trimmed_ants = np.unique(np.hstack([uvd.ant_1_array, uvd.ant_2_array]))
            # make sure there was a flagged antenna in the original flagged antennas.
            assert np.any([a in all_ants for a in [0, 10, 1, 3]])
            for a in [0, 10, 1, 3]:
                if a in all_ants:
                    assert a not in trimmed_ants
            # test NotImplementedError when trying to throw away flagged antennas when ant_indices_only is False
            pytest.raises(NotImplementedError, utils.apply_yaml_flags, uv=uvd, a_priori_flag_yaml=test_flag,
                         flag_freqs=flag_freqs, flag_times=flag_times,
                         flag_ants=flag_ants, throw_away_flagged_ants=True, ant_indices_only=False)



@pytest.mark.parametrize(
    "filein",
    ["a_priori_flags_integrations.yaml",
    "a_priori_flags_jds.yaml",
    "a_priori_flags_lsts.yaml",
    "a_priori_flags_no_integrations.yaml",
    "a_priori_flags_no_chans.yaml",
    "a_priori_flags_no_ants.yaml",
    "a_priori_flags_no_flags.yaml"
    ],
)
@pytest.mark.parametrize("new_metadata", [True, False])
def test_apply_yaml_flags_uvcal(filein, new_metadata):
    test_flag = os.path.join(DATA_PATH, filein)
    test_c_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    if not new_metadata:
        uvc.telescope_location = None
        uvc.antenna_positions = None
        uvc.lst_array = None

    uvc = utils.apply_yaml_flags(uvc, test_flag, unflag_first=True)
    freq_regions = [(0, 110e6), (150e6, 155e6), (190e6, 200e6)] # frequencies from yaml file.
    channel_flags = [0, 1, 60] + list(range(10, 21)) # channels from yaml file.
    integration_flags = [0, 1] # integrations from yaml file that should be flagged.
    ant_flags = [0, 10, [1, 'Jee'], [3, 'Jnn']]
    if 'no_flags' in test_flag:
        # check that the uvcal is completely unflagged since we used unflag_first
        assert not np.any(uvc.flag_array)
    else:
        if 'no_integrations' not in test_flag:
            for tind in integration_flags:
                time = sorted(np.unique(uvc.time_array))[tind]
                assert np.all(uvc.flag_array[:, :, :, uvc.time_array == time, :])
        if 'no_chans' not in test_flag:
            for region in freq_regions:
                selection = (uvc.freq_array[0] >= region[0]) & (uvc.freq_array[0] <= region[-1])
                assert np.all(uvc.flag_array[:, :, selection, :, :])
            for chan in channel_flags:
                assert np.all(uvc.flag_array[:, :, chan, :, :])
        # check flagged antennas
        if 'no_ants' not in test_flag:
            for ant in ant_flags:
                if isinstance(ant, int):
                    antnum = ant
                    pol_selection = np.ones(uvc.Njones, dtype=bool)
                elif isinstance(ant, (list, tuple)):
                    antnum = ant[0]
                    pol_num = uvutils.jstr2num(ant[1], x_orientation=uvc.x_orientation)
                    pol_selection = np.where(uvc.jones_array == pol_num)[0]
                ant_selection = uvc.ant_array == antnum
                assert np.all(uvc.flag_array[ant_selection][:, :, :, :, pol_selection])

            # test removing antennas from the data:
            all_ants = uvc.ant_array
            uvc = utils.apply_yaml_flags(uvc, test_flag, throw_away_flagged_ants=True, ant_indices_only=True)
            trimmed_ants = uvc.ant_array
            # make sure there was a flagged antenna in the original flagged antennas.
            assert np.any([a in all_ants for a in [0, 10, 1, 3]])
            for a in [0, 10, 1, 3]:
                if a in all_ants:
                    assert a not in trimmed_ants
            # test NotImplementedError when trying to throw away flagged antennas when ant_indices_only is False
            pytest.raises(NotImplementedError, utils.apply_yaml_flags, uvc, a_priori_flag_yaml=test_flag,
                         throw_away_flagged_ants=True, ant_indices_only=False)

def test_apply_yaml_flags_errors():
    test_flag_jds = os.path.join(DATA_PATH, 'a_priori_flags_jds.yaml')
    test_c_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA.omni.calfits')
    # check NotImplementedErrors
    uvc = UVCal()
    uvc.read_calfits(test_c_file)
    # check that setting uv to an object that is not a subclass of UVCal or UVData throws a NotImplementedError
    pytest.raises(NotImplementedError, utils.apply_yaml_flags, 'uvdata', test_flag_jds)
    # check that not providing lat_lon_alt_degrees and a telescope location that is not in the pyuvdata.KNOWN_TELESCOPES dict
    # throws a NotImplementedError
    # must remove the `lst_array` from the cal object first to test this
    uvc2 = uvc.copy()
    uvc2.lst_array = None
    pytest.raises(NotImplementedError, utils.apply_yaml_flags, uvc2, test_flag_jds, None, 'MITEOR')
    # check that more then a single spw throws a NotImplementedError
    uvc.Nspws = 2
    pytest.raises(NotImplementedError, utils.apply_yaml_flags, uvc, test_flag_jds)
    uvc.Nspws = 1
    # check warning for negative integrations
    for warn_yaml in ['a_priori_flags_maximum_channels.yaml', 'a_priori_flags_maximum_integrations.yaml',
                      'a_priori_flags_negative_channels.yaml', 'a_priori_flags_negative_integrations.yaml']:
        yaml_path = os.path.join(DATA_PATH, warn_yaml)
        with pytest.warns(None) as record:
            utils.apply_yaml_flags(uvc, yaml_path, unflag_first=True)
