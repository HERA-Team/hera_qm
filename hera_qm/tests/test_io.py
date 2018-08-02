from __future__ import absolute_import, division, print_function
import nose.tools as nt
import os
import numpy as np
from pyuvdata import UVData
from hera_qm.data import DATA_PATH
from hera_qm.io import UVDWrapper

#import copy
#import six
#import ephem
#import pyuvdata.utils as uvutils
#import pyuvdata.tests as uvtest
#from pyuvdata.data import DATA_PATH

#from hera_qm import utils as qmutils
#from hera_qm.ant_metrics import get_ant_metrics_dict
#from hera_qm.firstcal_metrics import get_firstcal_metrics_dict
#from hera_qm.omnical_metrics import get_omnical_metrics_dict
#from hera_qm.utils import get_metrics_dict

class TestWrapper():

    def setup(self):
        self.uvd = UVData()
        self.uvd.read_miriad(os.path.join(DATA_PATH, "zen.2457698.40355.xx.HH.uvcA"))
        self.qmind = UVDWrapper(self.uvd)
    
    def test_dict(self):

        for (i,j),k in self.qmind.bl_dict_spaced.iteritems():
            assert( (i,j) in self.qmind.bl_dict_enum )
            spaced = np.arange(k[0],k[1],k[2])
            assert( self.qmind.bl_dict_enum[(i,j)] == spaced )

    def test_get_data(self):
        #print(self.qmind.data.ant_1_array[:10])
        #print(self.qmind.data.ant_2_array[:10])
        #print(self.qmind.data.get_feedpols())
        inda = self.qmind.data.get_data(9,10,'xx')
        indb = self.qmind.get_data(9,10,'xx')
        assert np.array_equal(inda, indb)

