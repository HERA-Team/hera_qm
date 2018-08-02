from __future__ import absolute_import, division, print_function

import numpy as np
import six
import collections
from pyuvdata import utils as uvutils

class UVDWrapper():
    """
    """

    def __init__(self, uvdata):

        self.data = uvdata
        self.data_array = self.data.data_array
        self.polarr = self.data.polarization_array

        self.bl_dict_enum = {}
        for k in range(self.data.ant_1_array.shape[0]):
            i, j = self.data.ant_1_array[k],self.data.ant_2_array[k]
            if (i,j) in self.bl_dict_enum:
                self.bl_dict_enum[(i,j)] = np.append(self.bl_dict_enum[(i,j)],k)
            else:
                self.bl_dict_enum[(i,j)] = [k]

        self.bl_dict_spaced = {}
        for (i,j),k in self.bl_dict_enum.iteritems():
            if self._is_regular_spaced(k):
                kstart = k[0]
                kstop = k[-1]+1
                kstride = 1 if len(k)==1 else k[1]-k[0]
                self.bl_dict_spaced[(i,j)]=(kstart,kstop,kstride)

    def _is_regular_spaced(self,arr):
        arr = np.asarray(arr)
        diffs = arr[1:]-arr[:-1]
        return len(diffs)<2 or all(diffs==diffs[0])

#    def get_data(self,*args,**kwargs):
#
#        key = uvutils.get_iterable(args)
#
#        if len(key)==3:
#            # E.g. get_data(0,1,'x')
#            ant1 = key[0]
#            ant2 = key[1]
#            pol_num = uvutils.polstr2num(key[2])
#
#            if (ant1,ant2) in self.bl_dict_spaced:
#                pol_ind1 = np.where(self.data.polarization_array == pol_num)[0]
#                blt_ind1 = (self.bl_dict_spaced[(ant1,ant2)])
#            elif (ant1,ant2) in self.bl_dict_enum:
#                pol_ind1 = np.where(self.data.polarization_array == pol_num)[0]
#                blt_ind1 = self.bl_dict_enum[(ant1,ant2)]
#            else:
#                blt_ind1 = np.array([], dtype=np.int64)
#                pol_ind1 = np.array([], dtype=np.int64)
#
#            if (ant2,ant1) in self.bl_dict_spaced:
#                blt_ind2 = (self.bl_dict_spaced[(ant2,ant1)])
#                pol_ind2 = np.where(self.data.polarization_array == pol_num)[0]
#            elif (ant2,ant1) in self.bl_dict_enum:
#                blt_ind2 = self.bl_dict_enum[(ant2,ant1)]
#                pol_ind2 = np.where(self.data.polarization_array == pol_num)[0]
#            else:
#                blt_ind2 = np.array([], dtype=np.int64)
#                pol_ind2 = np.array([], dtype=np.int64)
#
#            if len(blt_ind1) + len(blt_ind2) == 0:
#                raise KeyError('Antenna pair {pair} not found in '
#                               'data'.format(pair=(key[0], key[1])))
#
#            pol_ind = (pol_ind1, pol_ind2)
#            if len(blt_ind1) * len(pol_ind[0]) + len(blt_ind2) * len(pol_ind[1]) == 0:
#                raise KeyError('Polarization {pol} not found in data.'.format(pol=key[2]))
#
#        else:
#            raise NotImplementedError()
#
#        # Catch autos
#        if type(blt_ind1) == type(blt_ind2) and np.array_equal(blt_ind1, blt_ind2):
#            blt_ind2 = np.array([])
#
#        force_copy = kwargs.pop('force_copy', False)
#        squeeze = kwargs.pop('squeeze', 'default')
#
#        p_reg_spaced = [False, False]
#        p_start = [0, 0]
#        p_stop = [0, 0]
#        dp = [1, 1]
#        for i, pi in enumerate(pol_ind):
#            if len(pi) == 0:
#                continue
#            if self._is_regular_spaced(pi):
#                p_reg_spaced[i] = True
#                p_start[i] = pi[0]
#                p_stop[i] = pi[-1] + 1
#                if len(pi) != 1:
#                    dp[i] = pi[1] - pi[0]
#
#        if len(blt_ind2) == 0:
#            # only unconjugated baselines
#            if (ant1,ant2) in self.bl_dict_spaced:
#                if p_reg_spaced[0]:
#                    blt_start, blt_stop, dblt = self.bl_dict_spaced[(ant1,ant2)]
#                    out = self.data_array[blt_start:blt_stop:dblt, :, :, p_start[0]:p_stop[0]:dp[0]]
#                else:
#                    out = self.data_array[blt_start:blt_stop:dblt, :, :, pol_ind[0]]
#            else:
#                out = self.data_array[blt_ind1, :, :, :]
#                if p_reg_spaced[0]:
#                    out = out[:, :, :, p_start[0]:p_stop[0]:dp[0]]
#                else:
#                    out = out[:, :, :, pol_ind[0]]
#        elif len(blt_ind1) == 0:
#            # only conjugated baselines
#            if (ant2,ant1) in self.bl_dict_spaced:
#                blt_start, blt_stop, dblt = self.bl_dict_spaced[(ant2,ant1)]
#                if p_reg_spaced[1]:
#                    out = self.data_array[blt_start:blt_stop:dblt, :, :, p_start[1]:p_stop[1]:dp[1]]
#                else:
#                    out = self.data_array[blt_start:blt_stop:dblt, :, :, pol_ind[1]]
#            else:
#                out = self.data_array[blt_ind2, :, :, :]
#                if p_reg_spaced[1]:
#                    out = out[:, :, :, p_start[1]:p_stop[1]:dp[1]]
#                else:
#                    out = out[:, :, :, pol_ind[1]]
#            out = np.conj(out)
#        else:
#            # both conjugated and unconjugated baselines
#            out = (self.data_array[blt_ind1, :, :, :], np.conj(self.data_array[blt_ind2, :, :, :]))
#            if p_reg_spaced[0] and p_reg_spaced[1]:
#                out = np.append(out[0][:, :, :, p_start[0]:p_stop[0]:dp[0]],
#                                out[1][:, :, :, p_start[1]:p_stop[1]:dp[1]], axis=0)
#            else:
#                out = np.append(out[0][:, :, :, pol_ind[0]],
#                                out[1][:, :, :, pol_ind[1]], axis=0)
#
#        if squeeze == 'full':
#            out = np.squeeze(out)
#        elif squeeze == 'default':
#            if out.shape[3] is 1:
#                # one polarization dimension
#                out = np.squeeze(out, axis=3)
#            if out.shape[1] is 1:
#                # one spw dimension
#                out = np.squeeze(out, axis=1)
#        elif squeeze != 'none':
#            raise ValueError('"' + str(squeeze) + '" is not a valid option for squeeze.'
#                             'Only "default", "none", or "full" are allowed.')
#
#        if force_copy:
#            out = np.array(out)
#        elif out.base is not None:
#            # if out is a view rather than a copy, make it read-only
#            out.flags.writeable = False
#
#        return out


###############################################################################################
    def antpair2ind(self, ant1, ant2):
        """
        Get blt indices for given (ordered) antenna pair.
        """
        if (ant1,ant2) in self.bl_dict_enum:
            return self.bl_dict_enum[(ant1,ant2)]
        else:
            return []

    def _key2inds(self, key):
        """
        Interpret user specified key as a combination of antenna pair and/or polarization.
        Args:
            key: Identifier of data. Key can be 1, 2, or 3 numbers:
                if len(key) == 1:
                    if (key < 5) or (type(key) is str):  interpreted as a
                                 polarization number/name, return all blts for that pol.
                    else: interpreted as a baseline number. Return all times and
                          polarizations for that baseline.
                if len(key) == 2: interpreted as an antenna pair. Return all
                    times and pols for that baseline.
                if len(key) == 3: interpreted as antenna pair and pol (ant1, ant2, pol).
                    Return all times for that baseline, pol. pol may be a string.
        Returns:
            blt_ind1: numpy array with blt indices for antenna pair.
            blt_ind2: numpy array with blt indices for conjugate antenna pair.
                      Note if a cross-pol baseline is requested, the polarization will
                      also be reversed so the appropriate correlations are returned.
                      e.g. asking for (1, 2, 'xy') may return conj(2, 1, 'yx'), which
                      is equivalent to the requesting baseline. See utils.conj_pol() for
                      complete conjugation mapping.
            pol_ind: tuple of numpy arrays with polarization indices for blt_ind1 and blt_ind2
        """
        key = uvutils.get_iterable(key)
        if type(key) is str:
            # Single string given, assume it is polarization
            pol_ind1 = np.where(self.polarr == uvutils.polstr2num(key))[0]
            if len(pol_ind1) > 0:
                blt_ind1 = np.arange(self.Nblts)
                blt_ind2 = np.array([], dtype=np.int64)
                pol_ind2 = np.array([], dtype=np.int64)
                pol_ind = (pol_ind1, pol_ind2)
            else:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
        elif len(key) == 1:
            key = key[0]  # For simplicity
            if isinstance(key, collections.Iterable):
                # Nested tuple. Call function again.
                blt_ind1, blt_ind2, pol_ind = self._key2inds(key)
            elif key < 5:
                # Small number, assume it is a polarization number a la AIPS memo
                pol_ind1 = np.where(self.polarr == key)[0]
                if len(pol_ind1) > 0:
                    blt_ind1 = np.arange(self.Nblts)
                    blt_ind2 = np.array([], dtype=np.int64)
                    pol_ind2 = np.array([], dtype=np.int64)
                    pol_ind = (pol_ind1, pol_ind2)
                else:
                    raise KeyError('Polarization {pol} not found in data.'.format(pol=key))
            else:
                # Larger number, assume it is a baseline number
                inv_bl = self.antnums_to_baseline(self.baseline_to_antnums(key)[1],
                                                  self.baseline_to_antnums(key)[0])
                blt_ind1 = np.where(self.baseline_array == key)[0]
                blt_ind2 = np.where(self.baseline_array == inv_bl)[0]
                if len(blt_ind1) + len(blt_ind2) == 0:
                    raise KeyError('Baseline {bl} not found in data.'.format(bl=key))
                pol_ind = (np.arange(self.Npols), np.arange(self.Npols))
        elif len(key) == 2:
            # Key is an antenna pair
            blt_ind1 = self.antpair2ind(key[0], key[1])
            blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in data'.format(pair=key))
            pol_ind = (np.arange(self.Npols), np.arange(self.Npols))
        elif len(key) == 3:
            # Key is an antenna pair + pol
            if (key[0], key[1]) in self.bl_dict_spaced:
                blt_ind1 = self.bl_dict_spaced[(key[0], key[1])]
            else:
                blt_ind1 = self.antpair2ind(key[0], key[1])
            if (key[1], key[0]) in self.bl_dict_spaced:
                blt_ind2 = self.bl_dict_spaced[(key[1], key[0])]
            else:
                blt_ind2 = self.antpair2ind(key[1], key[0])
            if len(blt_ind1) + len(blt_ind2) == 0:
                raise KeyError('Antenna pair {pair} not found in '
                               'data'.format(pair=(key[0], key[1])))
            if type(key[2]) is str:
                # pol is str
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(self.polarr == uvutils.polstr2num(key[2]))[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(self.polarr
                                        == uvutils.polstr2num(uvutils.conj_pol(key[2])))[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            else:
                # polarization number a la AIPS memo
                if len(blt_ind1) > 0:
                    pol_ind1 = np.where(self.polarr == key[2])[0]
                else:
                    pol_ind1 = np.array([], dtype=np.int64)
                if len(blt_ind2) > 0:
                    pol_ind2 = np.where(self.polarr == uvutils.conj_pol(key[2]))[0]
                else:
                    pol_ind2 = np.array([], dtype=np.int64)
            pol_ind = (pol_ind1, pol_ind2)
            if len(blt_ind1) * len(pol_ind[0]) + len(blt_ind2) * len(pol_ind[1]) == 0:
                raise KeyError('Polarization {pol} not found in data.'.format(pol=key[2]))

        # Catch autos
        if type(blt_ind1)==type(blt_ind2) and np.array_equal(blt_ind1, blt_ind2):
            blt_ind2 = np.array([])
        return (blt_ind1, blt_ind2, pol_ind)

    def _smart_slicing(self, data, ind1, ind2, indp, **kwargs):
        """
        Method for quickly picking out the relevant section of data for get_data or get_flags
        Args:
            data: 4-dimensional array in the format of self.data_array
            ind1: list with blt indices for antenna pair (e.g. from self._key2inds)
            ind2: list with blt indices for conjugate antenna pair. (e.g. from self._key2inds)
            indp: tuple of lists with polarization indices for ind1 and ind2 (e.g. from self._key2inds)
        kwargs:
            squeeze: 'default': squeeze pol and spw dimensions if possible (default)
                     'none': no squeezing of resulting numpy array
                     'full': squeeze all length 1 dimensions
            force_copy: Option to explicitly make a copy of the data. Default is False.
        Returns:
            out: numpy array copy (or if possible, a read-only view) of relevant section of data
        """
        force_copy = kwargs.pop('force_copy', False)
        squeeze = kwargs.pop('squeeze', 'default')

        p_reg_spaced = [False, False]
        p_start = [0, 0]
        p_stop = [0, 0]
        dp = [1, 1]
        for i, pi in enumerate(indp):
            if len(pi) == 0:
                continue
            if len(set(np.ediff1d(pi))) <= 1:
                p_reg_spaced[i] = True
                p_start[i] = pi[0]
                p_stop[i] = pi[-1] + 1
                if len(pi) != 1:
                    dp[i] = pi[1] - pi[0]

        if len(ind2) == 0:
            assert isinstance(ind2,list)
            # only unconjugated baselines
            if isinstance(ind1,tuple):
                if p_reg_spaced[0]:
                    out = data[ind1[0]:ind1[1]:ind1[2], :, :, p_start[0]:p_stop[0]:dp[0]]
                else:
                    out = data[ind1[0]:ind1[1]:ind1[2], :, :, indp[0]]
            else:
                out = data[ind1, :, :, :]
                if p_reg_spaced[0]:
                    out = out[:, :, :, p_start[0]:p_stop[0]:dp[0]]
                else:
                    out = out[:, :, :, indp[0]]
        elif len(ind1) == 0:
            assert isinstance(ind1,list)
            # only conjugated baselines
            if isinstance(ind2,tuple):
                if p_reg_spaced[1]:
                    out = data[ind2[0]:ind2[1]:ind2[2], :, :, p_start[1]:p_stop[1]:dp[1]]
                else:
                    out = data[ind2[0]:ind2[1]:ind2[2], :, :, indp[1]]
            else:
                out = data[ind2, :, :, :]
                if p_reg_spaced[1]:
                    out = out[:, :, :, p_start[1]:p_stop[1]:dp[1]]
                else:
                    out = out[:, :, :, indp[1]]
            out = np.conj(out)
        else:
            # both conjugated and unconjugated baselines
            if isinstance(ind1,tuple):
                out1 = data[ind1[0]:ind1[1]:ind1[2],:,:,:]
            else:
                out1 = data[ind1,:,:,:]
            if isinstance(ind1,tuple):
                out2 = data[ind2[0]:ind2[1]:ind2[2],:,:,:]
            else:
                out2 = data[ind2,:,:,:]
            out = (out1, np.conj(out2))
            if p_reg_spaced[0] and p_reg_spaced[1]:
                out = np.append(out[0][:, :, :, p_start[0]:p_stop[0]:dp[0]],
                                out[1][:, :, :, p_start[1]:p_stop[1]:dp[1]], axis=0)
            else:
                out = np.append(out[0][:, :, :, indp[0]],
                                out[1][:, :, :, indp[1]], axis=0)

        if squeeze == 'full':
            out = np.squeeze(out)
        elif squeeze == 'default':
            if out.shape[3] is 1:
                # one polarization dimension
                out = np.squeeze(out, axis=3)
            if out.shape[1] is 1:
                # one spw dimension
                out = np.squeeze(out, axis=1)
        elif squeeze != 'none':
            raise ValueError('"' + str(squeeze) + '" is not a valid option for squeeze.'
                             'Only "default", "none", or "full" are allowed.')

        if force_copy:
            out = np.array(out)
        elif out.base is not None:
            # if out is a view rather than a copy, make it read-only
            out.flags.writeable = False

        return out

    def get_data(self, *args, **kwargs):
        """
        Function for quick access to numpy array with data corresponding to
        a baseline and/or polarization. Returns a read-only view if possible, otherwise a copy.
        Args:
            *args: parameters or tuple of parameters defining the key to identify
                   desired data. See _key2inds for formatting.
            **kwargs: Keyword arguments:
                squeeze: 'default': squeeze pol and spw dimensions if possible
                         'none': no squeezing of resulting numpy array
                         'full': squeeze all length 1 dimensions
                force_copy: Option to explicitly make a copy of the data.
                             Default is False.
        Returns:
            Numpy array of data corresponding to key.
            If data exists conjugate to requested antenna pair, it will be conjugated
            before returning.
        """
        ind1, ind2, indp = self._key2inds(args)
        out = self._smart_slicing(self.data_array, ind1, ind2, indp, **kwargs)
        return out
