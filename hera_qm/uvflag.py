# -*- coding: utf-8 -*-
# Copyright (c) 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import
import numpy as np
import os
from pyuvdata import UVData
from pyuvdata import UVCal
from hera_qm.version import hera_qm_version_str
from hera_qm import utils as qm_utils
import warnings
import h5py
import copy


class UVFlag():
    ''' Object to handle flag arrays and waterfalls. Supports reading/writing,
    and stores all relevant information to combine flags and apply to data.
    '''
    def __init__(self, input, mode='metric', copy_flags=False, waterfall=False, history=''):
        '''Initialize UVFlag object.
        Args:
            input: UVData object, UVCal object, or path to previously saved UVFlag object.
                   Can also be a list of any combination of the above options.
            mode: "metric" (default) or "flag" to initialize UVFlag in given mode.
                  The mode determines whether the object has a floating point metric_array
                  or a boolean flag_array.
            copy_flags: Whether to copy flags from input to new UVFlag object. Default is False.
            waterfall: Whether to immediately initialize as a waterfall object,
                       with flag/metric axes: time, frequency, polarization. Default is False.
            history: History string to attach to object.
        '''

        self.mode = mode.lower()  # Gets overwritten if reading file
        self.history = history
        if isinstance(input, (list, tuple)):
            self.__init__(input[0], mode=mode, copy_flags=copy_flags, waterfall=waterfall, history=history)
            if len(input) > 1:
                for i in input[1:]:
                    fobj = UVFlag(i, mode=mode, copy_flags=copy_flags, waterfall=waterfall, history=history)
                    self += fobj
                del(fobj)
        elif isinstance(input, str):
            # Given a path, read input
            self.read(input, history)
        elif waterfall and isinstance(input, (UVData, UVCal)):
            self.type = 'waterfall'
            self.history += 'Flag object with type "waterfall" created by ' + hera_qm_version_str
            self.time_array, ri = np.unique(input.time_array, return_index=True)
            self.freq_array = input.freq_array[0, :]
            if isinstance(input, UVData):
                self.polarization_array = input.polarization_array
                self.lst_array = input.lst_array[ri]
            else:
                self.polarization_array = input.jones_array
                self.lst_array = qm_utils.lst_from_uv(input)[ri]
            if copy_flags:
                self.metric_array = qm_utils.flags2waterfall(input, keep_pol=True)
                self.history += ' Waterfall generated from ' + str(input.__class__) + ' object.'
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
            self.lst_array = qm_utils.lst_from_uv(input)
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
                self.weights_array = np.ones(self.flag_array.shape)
            else:
                self.weights_array = np.ones(self.metric_array.shape)

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
        '''Add two UVFlag objects together along a given axis.
        Args:
            other: second UVFlag object to concatenate with self.
            inplace: Whether to concatenate to self, or create a new UVFlag object. Default is False.
            axis: Axis along which to combine UVFlag objects. Default is time.
        '''

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()

        # Check that objects are compatible
        if not isinstance(other, this.__class__):
            raise ValueError('Only UVFlag objects can be added to a UVFlag object')
        if this.type != other.type:
            raise ValueError('UVFlag object of type ' + other.type + ' cannot be '
                             'added to object of type ' + this.type + '.')
        if this.mode != other.mode:
            raise ValueError('UVFlag object of mode ' + other.mode + ' cannot be '
                             'added to object of mode ' + this.type + '.')

        # Simplify axis referencing
        axis = axis.lower()
        type_nums = {'waterfall': 0, 'baseline': 1, 'antenna': 2}
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

        if not inplace:
            return this

    def __iadd__(self, other):
        """
        In place add.

        Args:
            other: Another UVFlag object which will be added to self.
        """
        self.__add__(other, inplace=True)
        return self

    def __or__(self, other, inplace=False):
        '''Combine two UVFlag objects in "flag" mode by "OR"-ing their flags.
        Args:
            other: second UVFlag object to combine with self.
            inplace: Whether to combine to self, or create a new UVFlag object. Default is False.
        '''
        if (self.mode != 'flag') or (other.mode != 'flag'):
            raise ValueError('UVFlag object must be in "flag" mode to use "or" function.')

        # Handle in place
        if inplace:
            this = self
        else:
            this = self.copy()
        this.flag_array += other.flag_array
        if other.history not in this.history:
            this.history += "Flags OR'd with: " + other.history

        if not inplace:
            return this

    def __ior__(self, other):
        '''In place or
        Args:
            other: second UVFlag object to combine with self.
        '''
        self.__or__(other, inplace=True)
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

    def copy(self):
        ''' Simply return a copy of this object '''
        return copy.deepcopy(self)

    def to_waterfall(self, method='quadmean', keep_pol=True):
        """
        Convert an 'antenna' or 'baseline' type object to waterfall using a given method.
        Args:
            method: How to collapse the dimension(s)
            keep_pol: Whether to also collapse the polarization dimension
        """
        method = method.lower()
        avg_f = qm_utils.averaging_dict[method]
        if self.type == 'waterfall' and (keep_pol or (len(self.polarization_array) == 1)):
            warnings.warn('This object is already a waterfall. Nothing to change.')
            return
        if self.mode == 'flag':
            darr = self.flag_array
        else:
            darr = self.metric_array
        if (not keep_pol) and (len(self.polarization_array) > 1):
            # Collapse pol dimension. But note we retain a polarization axis.
            d, w = avg_f(darr, axis=-1, weights=self.weights_array, returned=True)
            darr = np.expand_dims(d, axis=d.ndim)
            self.weights_array = np.expand_dims(w, axis=w.ndim)
            self.polarization_array = np.array([','.join(map(str, self.polarization_array))])

        if self.type == 'antenna':
            d, w = avg_f(darr, axis=(0, 1), weights=self.weights_array,
                         returned=True)
            darr = np.swapaxes(d, 0, 1)
            self.weights_array = np.swapaxes(w, 0, 1)
        elif self.type == 'baseline':
            Nt = len(np.unique(self.time_array))
            Nf = len(self.freq_array[0, :])
            Np = len(self.polarization_array)
            d = np.zeros((Nt, Nf, Np))
            w = np.zeros((Nt, Nf, Np))
            for i, t in enumerate(np.unique(self.time_array)):
                ind = self.time_array == t
                d[i, :, :], w[i, :, :] = avg_f(darr[ind, :, :], axis=0,
                                               weights=self.weights_array[ind, :, :],
                                               returned=True)
            darr = d
            self.weights_array = w
            self.time_array = np.unique(self.time_array)
        self.metric_array = darr
        self.freq_array = self.freq_array.flatten()
        self.mode = 'metric'
        self.type = 'waterfall'
        self.history += 'Collapsed to type "waterfall" with ' + hera_qm_version_str
        self.clear_unused_attributes()

    def to_baseline(self, uv, force_pol=False):
        '''Convert a UVFlag object of type "waterfall" to type "baseline".
        Broadcasts the flag array to all baselines.
        This function does NOT apply flags to uv.
        Args:
            uv: UVData or UVFlag object of type baseline to match.
            force_pol: If True, will use 1 pol to broadcast to any other pol.
                       Otherwise, will require polarizations match.
        '''
        if self.type == 'baseline':
            return
        if not (isinstance(uv, UVData) or (isinstance(uv, UVFlag) and uv.type == 'baseline')):
            raise ValueError('Must pass in UVData object or UVFlag object of type '
                             '"baseline" to match.')
        if self.type != 'waterfall':
            raise ValueError('Cannot convert from type "' + self.type + '" to "baseline".')
        # Deal with polarization
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = uv.polarization_array
            # Broadcast arrays
            if self.mode == 'flag':
                self.flag_array = self.flag_array.repeat(self.polarization_array.size, axis=-1)
            else:
                self.metric_array = self.metric_array.repeat(self.polarization_array.size, axis=-1)
            self.weights_array = self.weights_array.repeat(self.polarization_array.size, axis=-1)
        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(uv.polarization_array, self.polarization_array):
            raise ValueError('Polarizations could not be made to match.')
        # Populate arrays
        warr = np.zeros_like(uv.flag_array)
        if self.mode == 'flag':
            arr = np.zeros_like(uv.flag_array)
            sarr = self.flag_array
        elif self.mode == 'metric':
            arr = np.zeros_like(uv.flag_array, dtype=float)
            sarr = self.metric_array
        for i, t in enumerate(np.unique(uv.time_array)):
            ti = np.where(uv.time_array == t)
            arr[ti, :, :, :] = sarr[i, :, :][np.newaxis, np.newaxis, :, :]
            warr[ti, :, :, :] = self.weights_array[i, :, :][np.newaxis, np.newaxis, :, :]
        if self.mode == 'flag':
            self.flag_array = arr
        elif self.mode == 'metric':
            self.metric_array = arr
        self.weights_array = warr

        self.baseline_array = uv.baseline_array
        self.ant_1_array = uv.ant_1_array
        self.ant_2_array = uv.ant_2_array
        self.time_array = uv.time_array
        self.lst_array = uv.lst_array
        self.history += 'Broadcast to type "baseline" with ' + hera_qm_version_str

    def to_antenna(self, uv, force_pol=False):
        '''Convert a UVFlag object of type "waterfall" to type "antenna".
        Broadcasts the flag array to all antennas.
        This function does NOT apply flags to uv.
        Args:
            uv: UVCal or UVFlag object of type antenna to match.
            force_pol: If True, will use 1 pol to broadcast to any other pol.
                       Otherwise, will require polarizations match.
        '''
        if self.type == 'antenna':
            return
        if not (isinstance(uv, UVCal) or (isinstance(uv, UVFlag) and uv.type == 'antenna')):
            raise ValueError('Must pass in UVCal object or UVFlag object of type '
                             '"antenna" to match.')
        if self.type != 'waterfall':
            raise ValueError('Cannot convert from type "' + self.type + '" to "antenna".')
        # Deal with polarization
        if isinstance(uv, UVCal):
            polarr = uv.jones_array
        else:
            polarr = uv.polarization_array
        if force_pol and self.polarization_array.size == 1:
            # Use single pol for all pols, regardless
            self.polarization_array = polarr
            # Broadcast arrays
            if self.mode == 'flag':
                self.flag_array = self.flag_array.repeat(self.polarization_array.size, axis=-1)
            else:
                self.metric_array = self.metric_array.repeat(self.polarization_array.size, axis=-1)
            self.weights_array = self.weights_array.repeat(self.polarization_array.size, axis=-1)
        # Now the pol axes should match regardless of force_pol.
        if not np.array_equal(polarr, self.polarization_array):
            raise ValueError('Polarizations could not be made to match.')
        # Populate arrays
        if self.mode == 'flag':
            self.flag_array = np.swapaxes(self.flag_array, 0, 1)[np.newaxis, np.newaxis,
                                                                 :, :, :]
            self.flag_array = self.flag_array.repeat(len(uv.ant_array), axis=0)
        elif self.mode == 'metric':
            self.metric_array = np.swapaxes(self.metric_array, 0, 1)[np.newaxis, np.newaxis,
                                                                     :, :, :]
            self.metric_array = self.metric_array.repeat(len(uv.ant_array), axis=0)
        self.weights_array = np.swapaxes(self.weights_array, 0, 1)[np.newaxis, np.newaxis,
                                                                   :, :, :]
        self.weights_array = self.weights_array.repeat(len(uv.ant_array), axis=0)
        self.ant_array = uv.ant_array
        self.history += 'Broadcast to type "antenna" with ' + hera_qm_version_str

    def to_flag(self):
        '''Convert to flag mode. NOT SMART. Simply removes metric_array and initializes
        flag_array with Falses.
        '''
        if self.mode == 'flag':
            return
        elif self.mode == 'metric':
            self.flag_array = np.zeros_like(self.metric_array, dtype=np.bool)
            self.mode = 'flag'
        else:
            raise ValueError('Unknown UVFlag mode: ' + self.mode + '. Cannot convert to flag.')
        self.history += 'Converted to mode "flag" with ' + hera_qm_version_str
        self.clear_unused_attributes()

    def to_metric(self):
        '''Convert to metric mode. NOT SMART. Simply removes flag_array and initializes
        metric_array with zeros.
        '''
        if self.mode == 'metric':
            return
        elif self.mode == 'flag':
            self.metric_array = np.zeros_like(self.flag_array, dtype=np.float)
            self.mode = 'metric'
        else:
            raise ValueError('Unknown UVFlag mode: ' + self.mode + '. Cannot convert to metric.')
        self.history += 'Converted to mode "metric" with ' + hera_qm_version_str
        self.clear_unused_attributes()

    def antpair2ind(self, ant1, ant2):
        """
        Get blt indices for given (ordered) antenna pair.
        """
        if self.type != 'baseline':
            raise ValueError('UVFlag object of type ' + self.type + ' does not '
                             'contain antenna pairs to index.')
        return np.where((self.ant_1_array == ant1) & (self.ant_2_array == ant2))[0]
