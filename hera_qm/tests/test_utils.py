import nose.tools as nt
import os
import pyuvdata.tests as uvtest
from hera_qm import utils
from hera_qm.data import DATA_PATH


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

