import nose.tools as nt
import os
import pyuvdata.tests as uvtest
from pyuvdata import UVCal
from hera_qm import utils
from hera_qm.data import DATA_PATH
from hera_qm.ant_metrics import get_ant_metrics_dict
from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
from hera_qm.omnical_metrics import get_omnical_metrics_dict
from hera_qm.utils import get_metrics_dict


def test_get_pol():
    filename = 'zen.2457698.40355.xx.HH.uvcA'
    nt.assert_equal(utils.get_pol(filename), 'xx')


def test_generate_fullpol_file_list():
    pol_list = ['xx', 'xy', 'yx', 'yy']
    xx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcA')
    xy_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xy.HH.uvcA')
    yx_file = os.path.join(DATA_PATH, 'zen.2457698.40355.yx.HH.uvcA')
    yy_file = os.path.join(DATA_PATH, 'zen.2457698.40355.yy.HH.uvcA')
    file_list = [xx_file, xy_file, yx_file, yy_file]

    # feed in one file at a time
    fullpol_file_list = utils.generate_fullpol_file_list([xx_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([xy_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([yx_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))
    fullpol_file_list = utils.generate_fullpol_file_list([yy_file], pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))

    # feed in all four files
    fullpol_file_list = utils.generate_fullpol_file_list(file_list, pol_list)
    nt.assert_equal(sorted(fullpol_file_list[0]), sorted(file_list))

    # checks that we have a list of lists with outer length of 1, and inner length of 4
    nt.assert_equal(len(fullpol_file_list), 1)
    nt.assert_equal(len(fullpol_file_list[0]), 4)

    # try to pass in a file that doesn't have all pols present
    lone_file = os.path.join(DATA_PATH, 'zen.2457698.40355.xx.HH.uvcAA')
    fullpol_file_list = uvtest.checkWarnings(utils.generate_fullpol_file_list,
                                             [[lone_file], pol_list], nwarnings=1,
                                             message='Could not find')
    nt.assert_equal(fullpol_file_list, [])


def test_get_metrics_ArgumentParser():
    # raise error for requesting unknown type of parser
    nt.assert_raises(AssertionError, utils.get_metrics_ArgumentParser, 'fake_method')


def test_metrics2mc():
    # test ant metrics
    filename = os.path.join(DATA_PATH, 'example_ant_metrics.json')
    d = utils.metrics2mc(filename, ftype='ant')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    nt.assert_equal(len(d['array_metrics']), 0)
    ant_metrics_list = get_ant_metrics_dict()
    nt.assert_equal(set(d['ant_metrics'].keys()), set(ant_metrics_list.keys()))

    # test firstcal metrics
    filename = os.path.join(DATA_PATH, 'example_firstcal_metrics.json')
    d = utils.metrics2mc(filename, ftype='firstcal')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    firstcal_array_metrics = set(['firstcal_metrics_agg_std_y',
                                  'firstcal_metrics_good_sol_y',
                                  'firstcal_metrics_max_std_y'])
    nt.assert_equal(set(d['array_metrics'].keys()), firstcal_array_metrics)
    firstcal_metrics_list = get_firstcal_metrics_dict()
    firstcal_ant_metrics = set(firstcal_metrics_list.keys()) - firstcal_array_metrics
    # remove others not in this data file
    firstcal_ant_metrics -= {'firstcal_metrics_good_sol_x', 'firstcal_metrics_good_sol',
                             'firstcal_metrics_agg_std_x', 'firstcal_metrics_agg_std',
                             'firstcal_metrics_max_std_x'}
    nt.assert_equal(set(d['ant_metrics']), firstcal_ant_metrics)

    # test omnical metrics
    filename = os.path.join(DATA_PATH, 'example_omnical_metrics.json')
    d = utils.metrics2mc(filename, ftype='omnical')
    nt.assert_equal(set(d.keys()), set(['ant_metrics', 'array_metrics']))
    om = 'omnical_metrics_'
    nt.assert_equal(set(d['array_metrics'].keys()), set([om+'chisq_tot_avg_XX', om+'chisq_good_sol_XX',
                                                         om+'chisq_tot_avg_YY', om+'chisq_good_sol_YY',
                                                         om+'ant_phs_std_max_XX', om+'ant_phs_std_good_sol_XX',
                                                         om+'ant_phs_std_max_YY', om+'ant_phs_std_good_sol_YY']))
    nt.assert_equal(set(d['ant_metrics'].keys()), set([om+'chisq_ant_avg', om+'chisq_ant_std', om+'ant_phs_std']))
    nt.assert_equal(len(d['ant_metrics'][om+'chisq_ant_avg']), 32)

    # Hit the exceptions
    nt.assert_raises(ValueError, utils.metrics2mc, filename, ftype='foo')


def test_get_metrics_dict():
    ant_metrics_dict = get_ant_metrics_dict()
    firstcal_metrics_dict = get_firstcal_metrics_dict()
    omnical_metrics_dict = get_omnical_metrics_dict()
    metrics_dict = get_metrics_dict()
    for key in ant_metrics_dict:
        nt.assert_equal(ant_metrics_dict[key], metrics_dict[key])
    for key in firstcal_metrics_dict:
        nt.assert_equal(firstcal_metrics_dict[key], metrics_dict[key])
    for key in omnical_metrics_dict:
        nt.assert_equal(omnical_metrics_dict[key], metrics_dict[key])
