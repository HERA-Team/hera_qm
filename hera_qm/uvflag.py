from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from pyuvdata import utils as uvutils
from pyuvdata import telescopes as uvtel
from hera_qm.version import hera_qm_version_str
from hera_qm import xrfi
import warnings
import h5py
import copy


def lst_from_uv(uv):
    ''' Calculate the lst_array for a UVData or UVCal object.
    Args:
        uv: a UVData or UVCal object.
    Returns:
        lst_array: lst_array corresponding to time_array and at telecope location.
                   Units are radian.
    '''
    if not isinstance(uv, (UVCal, UVData)):
        raise ValueError('Function lst_from_uv can only operate on '
                         'UVCal or UVData object.')

    tel = uvtel.get_telescope(uv.telescope_name)
    lat, lon, alt = tel.telescope_location_lat_lon_alt_degrees
    lst_array = uvutils.get_lst_for_time(uv.time_array, lat, lon, alt)
    return lst_array

class UVFlag():
    ''' Object to handle flag arrays and waterfalls. Supports reading/writing,
    and stores all relevant information to combine flags and apply to data.
    '''
    def __init__(self, input, mode='metric', copy_flags=False, wf=False, history=''):
        # TODO: Docstring
        #
        # Mode can be 'flag' or 'metric'
        # TODO: Weight array - get from nsample_array?
        # TODO: get_flags similar to pyuvdata and/or convert to data container
        # TODO: list loading

        self.mode = mode.lower()  # Gets overwritten if reading file
        self.history = history
        if isinstance(input, (list, tuple)):
            self.__init__(input[0], mode=mode, copy_flags=copy_flags, wf=wf, history=history)
            if len(input) > 1:
                for i in input[1:]:
                    fobj = UVFlag(i, mode=mode, copy_flags=copy_flags, wf=wf, history=history)
                    self += fobj
                del(fobj)
        elif isinstance(input, str):
            # Given a path, read input
            self.read(input, history)
        elif wf and isinstance(input, (UVData, UVCal)):
            self.type = 'wf'
            self.history += 'Flag object with type "wf" created by ' + hera_qm_version_str
            self.time_array, ri = np.unique(input.time_array, return_index=True)
            self.freq_array = input.freq_array[0, :]
            if isinstance(input, UVData):
                self.polarization_array = input.polarization_array
                self.lst_array = input.lst_array[ri]
            else:
                self.polarization_array = input.jones_array
                self.lst_array = lst_from_uv(input)[ri]
            if copy_flags:
                self.metric_array = xrfi.flags2waterfall(input, keep_pol=True)
                self.history += ' WF generated from ' + str(input.__class__) + ' object.'
                if self.mode == 'flag':
                    warnings.warn('Copying flags into waterfall results in mode=="metric".')
                    self.mode = 'metric'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros((len(self.time_array),
                                               len(self.freq_array),
                                               len(self.polarization_array)), np.bool)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros((len(self.time_array),
                                                 len(self.freq_array),
                                                 len(self.polarization_array)))

        elif isinstance(input, UVData):
            self.type = 'baseline'
            self.history += 'Flag object with type "baseline" created by ' + hera_qm_version_str
            self.baseline_array = input.baseline_array
            self.ant_1_array = input.ant_1_array
            self.ant_2_array = input.ant_2_array
            self.time_array = input.time_array
            self.lst_array = input.lst_array
            self.freq_array = input.freq_array
            self.polarization_array = input.polarization_array
            if copy_flags:
                self.flag_array = input.flag_array
                self.history += ' Flags copied from ' + str(input.__class__) + ' object.'
                if self.mode == 'metric':
                    warnings.warn('Copying flags to type=="baseline" results in mode=="flag".')
                    self.mode = 'flag'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros_like(input.flag_array)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros_like(input.flag_array).astype(np.float)

        elif isinstance(input, UVCal):
            self.type = 'antenna'
            self.history += 'Flag object with type "antenna" created by ' + hera_qm_version_str
            self.ant_array = input.ant_array
            self.time_array = input.time_array
            self.lst_array = lst_from_uv(input)
            self.freq_array = input.freq_array
            self.polarization_array = input.jones_array
            if copy_flags:
                self.flag_array = input.flag_array
                self.history += ' Flags copied from ' + str(input.__class__) + ' object.'
                if self.mode == 'metric':
                    warnings.warn('Copying flags to type=="antenna" results in mode=="flag".')
                    self.mode = 'flag'
            else:
                if self.mode == 'flag':
                    self.flag_array = np.zeros_like(input.flag_array)
                elif self.mode == 'metric':
                    self.metric_array = np.zeros_like(input.flag_array).astype(np.float)

        if not isinstance(input, str):
            if self.mode == 'flag':
                self.weights_array = np.zeros(self.flag_array.shape)
            else:
                self.weights_array = np.zeros(self.metric_array.shape)

        self.clear_unused_attributes()

    def __eq__(self, other, check_history=False):
        """ Function to check equality of two UVFlag objects
        Args:
            other: UVFlag object to check against
        """
        if not isinstance(other, self.__class__):
            return False
        if (self.type != other.type) or (self.mode != other.mode):
            return False

        array_list = ['weights_array', 'time_array', 'lst_array', 'freq_array',
                      'polarization_array']
        if self.type == 'antenna':
            array_list += ['ant_array']
        elif self.type == 'baseline':
            array_list += ['baseline_array', 'ant_1_array', 'ant_2_array']
        if self.mode == 'flag':
            array_list += ['flag_array']
        elif self.mode == 'metric':
            array_list += ['metric_array']
        for arr in array_list:
            self_param = getattr(self, arr)
            other_param = getattr(other, arr)
            if not np.all(self_param == other_param):
                return False

        if check_history:
            if self.history != other.history:
                return False

        return True

    def read(self, filename, history=''):
        """
        Read in flag/metric data from a UVH5 file.

        Args:
            filename: The file name to read.
        """

        if isinstance(filename, (tuple, list)):
            self.read(filename[0])
            if len(filename) > 1:
                for f in filename[1:]:
                    f2 = UVFlag(f, history=history)
                    self += f2
                del(f2)

        else:
            if not os.path.exists(filename):
                raise IOError(filename + ' not found.')

            # Open file for reading
            with h5py.File(filename, 'r') as f:
                header = f['/Header']

                self.type = header['type'].value
                self.mode = header['mode'].value
                self.time_array = header['time_array'].value
                self.lst_array = header['lst_array'].value
                self.freq_array = header['freq_array'].value
                self.history = header['history'].value + ' Read by ' + hera_qm_version_str
                self.history += history
                self.polarization_array = header['polarization_array'].value
                if self.type == 'baseline':
                    self.baseline_array = header['baseline_array'].value
                    self.ant_1_array = header['ant_1_array'].value
                    self.ant_2_array = header['ant_2_array'].value
                elif self.type == 'antenna':
                    self.ant_array = header['ant_array'].value

                dgrp = f['/Data']
                if self.mode == 'metric':
                    self.metric_array = dgrp['metric_array'].value
                elif self.mode == 'flag':
                    self.flag_array = dgrp['flag_array'].value

                self.weights_array = dgrp['weights_array'].value

            self.clear_unused_attributes()

    def write(self, filename, clobber=False, data_compression='lzf'):
        """
        Write a UVFlag object to a hdf5 file.

        Args:
            filename: The file to write to.
            clobber: Option to overwrite the file if it already exists. Default is False.
            data_compression: HDF5 filter to apply when writing the data_array. Default is
                 LZF. If no compression is wanted, set to None.
        """

        if os.path.exists(filename):
            if clobber:
                print('File ' + filename + ' exists; clobbering')
            else:
                raise ValueError('File ' + filename + ' exists; skipping')

        with h5py.File(filename, 'w') as f:
            header = f.create_group('Header')

            # write out metadata
            header['type'] = self.type
            header['mode'] = self.mode
            header['time_array'] = self.time_array
            header['lst_array'] = self.lst_array
            header['freq_array'] = self.freq_array
            header['polarization_array'] = self.polarization_array
            header['history'] = self.history + 'Written by ' + hera_qm_version_str

            if self.type == 'baseline':
                header['baseline_array'] = self.baseline_array
                header['ant_1_array'] = self.ant_1_array
                header['ant_2_array'] = self.ant_2_array
            elif self.type == 'antenna':
                header['ant_array'] = self.ant_array

            dgrp = f.create_group("Data")
            if data_compression is not None:
                wtsdata = dgrp.create_dataset('weights_array', chunks=True,
                                              data=self.weights_array,
                                              compression=data_compression)
                if self.mode == 'metric':
                    data = dgrp.create_dataset('metric_array', chunks=True,
                                               data=self.metric_array,
                                               compression=data_compression)
                elif self.mode == 'flag':
                    data = dgrp.create_dataset('flag_array', chunks=True,
                                               data=self.flag_array,
                                               compression=data_compression)
            else:
                wtsdata = dgrp.create_dataset('weights_array', chunks=True,
                                              data=self.weights_array)
                if self.mode == 'metric':
                    data = dgrp.create_dataset('metric_array', chunks=True,
                                               data=self.metric_array)
                elif self.mode == 'flag':
                    data = dgrp.create_dataset('flag_array', chunks=True,
                                               data=self.flag_array)

    def __add__(self, other, inplace=False, axis='time'):
        # TODO: make less general that UVData, but more efficient - have user specify axis
        # TODO: docstring

        # Handle in place
        if inplace:
            this = self
        else:
            this = copy.deepcopy(self)

        # Check that objects are compatible
        if not isinstance(other, this.__class__):
            raise ValueError('Only UVFlag objects can be added to a UVFlag object')
        if this.type != other.type:
            raise ValueError('UVFlag object of type ' + other.type + ' cannot be ' +
                             'added to object of type ' + this.type + '.')
        if this.mode != other.mode:
            raise ValueError('UVFlag object of mode ' + other.mode + ' cannot be ' +
                             'added to object of mode ' + this.type + '.')
        # TODO: Check for overlapping data (see pyuvdata.UVData.__add__)

        # Simplify axis referencing
        axis = axis.lower()
        type_nums = {'wf': 0, 'baseline': 1, 'antenna': 2}
        axis_nums = {'time': [0, 0, 3], 'baseline': [None, 0, None],
                     'antenna': [None, None, 0], 'frequency': [1, 2, 2],
                     'polarization': [2, 3, 4], 'pol': [2, 3, 4],
                     'jones': [2, 3, 4]}
        ax = axis_nums[axis][type_nums[self.type]]
        if axis == 'time':
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            if this.type == 'baseline':
                this.baseline_array = np.concatenate([this.baseline_array, other.baseline_array])
                this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
                this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
        elif axis == 'baseline':
            if self.type != 'baseline':
                raise ValueError('Flag object of type ' + self.type + ' cannot be '
                                 'concatenated along baseline axis.')
            this.time_array = np.concatenate([this.time_array, other.time_array])
            this.lst_array = np.concatenate([this.lst_array, other.lst_array])
            this.baseline_array = np.concatenate([this.baseline_array, other.baseline_array])
            this.ant_1_array = np.concatenate([this.ant_1_array, other.ant_1_array])
            this.ant_2_array = np.concatenate([this.ant_2_array, other.ant_2_array])
        elif axis == 'antenna':
            if self.type != 'antenna':
                raise ValueError('Flag object of type ' + self.type + ' cannot be '
                                 'concatenated along antenna axis.')
            this.ant_array = np.concatenate([this.ant_array, other.ant_array])
        elif axis == 'frequency':
            this.freq_array = np.concatenate([this.freq_array, other.freq_array])
        elif axis in ['polarization', 'pol', 'jones']:
            this.polarization_array = np.concatenate([this.polarization_array,
                                                      other.polarization_array])

        if this.mode == 'flag':
            this.flag_array = np.concatenate([this.flag_array, other.flag_array],
                                             axis=ax)
        elif this.mode == 'metric':
            this.metric_array = np.concatenate([this.metric_array,
                                                other.metric_array], axis=ax)
        this.weights_array = np.concatenate([this.weights_array,
                                             other.weights_array], axis=ax)
        this.history += 'Data combined along ' + axis + ' axis with ' + hera_qm_version_str

        return this

    def __iadd__(self, other):
        """
        In place add.

        Args:
            other: Another UVFlag object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self

    def clear_unused_attributes(self):
        """
        Remove unused attributes. Useful when changing type or mode.
        """
        if hasattr(self, 'baseline_array') and self.type != 'baseline':
            del(self.baseline_array)
        if hasattr(self, 'ant_1_array') and self.type != 'baseline':
            del(self.ant_1_array)
        if hasattr(self, 'ant_2_array') and self.type != 'baseline':
            del(self.ant_2_array)
        if hasattr(self, 'ant_array') and self.type != 'antenna':
            del(self.ant_array)
        if hasattr(self, 'metric_array') and self.mode != 'metric':
            del(self.metric_array)
        if hasattr(self, 'flag_array') and self.mode != 'flag':
            del(self.flag_array)