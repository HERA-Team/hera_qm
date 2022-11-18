# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to classify antennas by various data quality metrics."""
import numpy as np
from scipy.ndimage import median_filter
import warnings
from . import xrfi


def _check_antpol(ap):
    '''Verifies that input is a valid ant-pol tuple. Otherwise, raises a ValueError'''
    from hera_cal.utils import join_bl
    try:
        assert np.issubdtype(type(ap[0]), np.integer)
        join_bl(ap, ap)
    except:
        raise ValueError(f"{ap} could not be interpreted as an antpol tuple of the form (1, 'Jee').")


class AntennaClassification():
    '''Object designed for storing a classification of antennas (expressed as ant-pol tuples), generally 
    into good/suspect/bad categories,though it allows flexibility in what those are called and what other 
    categories might be included. Enables easy getting and setting via brackets, as well the ability to 
    combine classifications with the + operator in a way that means that an antenna marked as bad in either 
    classification is bad and and that only an antenna marked as good in both classifications remains good.'''
    
    def __init__(self, **kwargs):
        '''Create an AntennaClassification object using arbitrary classifications as keyword arguments. 
        Antennas belonging to each classification are passed in as lists of ant-pol 2-tuples. For example,
        ant_class = AntennaClassification(good=[(0, 'Jee'), (1, 'Jee')], whatever=[(0, 'Jnn'), (10, 'Jee')]).
        Typical use is to pass in 'good', 'suspect', and/or 'bad' as kwargs, since these are the default
        "quality_classes" which the object knows how to combine. Antennas must be uniquely classified.
        
        Arguments:
            **kwargs: each is a list of ant-pol tuples with a classification given by the keyword argument
        '''
        self.clear()
        for cls, antlist in kwargs.items():
            for ant in antlist:
                _check_antpol(ant)
                if ant in self._classification:
                    raise ValueError(f"Antenna {ant} cannot be clasified as {cls} because it is already classified as {self[ant]}")
                self._classification[ant] = cls
    
    def clear(self):
        '''Empties all classifications and resets default good/suspect/bad labels.'''
        self._classification = {}
        self._GOOD = 'good'
        self._SUSPECT = 'suspect'
        self._BAD = 'bad'
    
    def __getitem__(self, key):
        '''Get classification of a specific ant-pol tuple.'''
        return self._classification[key]
    
    def __setitem__(self, key, value):
        '''Set the classification of a specific ant-pol tuple.'''
        self._classification[key] = value
    
    def __iter__(self):
        '''Iterate over ant-pol tuples.'''
        return iter(self._classification)
    
    @property
    def classes(self):
        '''Set of unique antenna classifications.'''
        return set(self._classification.values())

    @property
    def ants(self):
        '''Set of all classified ant-pol tuples.'''
        return set(self._classification.keys())
    
    def get_all(self, classification):
        '''Return set of all ant-pol tuples with the given classification.'''
        return set([ant for ant, cls in self._classification.items() if cls == classification])
   
    @property
    def good_ants(self):
        '''Set of ant-pol tuples with the current good classification (default "good").'''
        return self.get_all(self._GOOD)
    
    def is_good(self, ant):
        '''Returns True if antenna has the current good classification (default "good"), else False.'''
        return (ant in self._classification) and (self[ant] == self._GOOD)
    
    @property
    def suspect_ants(self):
        '''Set of ant-pol tuples with the current suspect classification (default "suspect").'''
        return self.get_all(self._SUSPECT)  

    def is_suspect(self, ant):
        '''Returns True if antenna has the current suspect classification (default "suspect"), else False.'''
        return (ant in self._classification) and (self[ant] == self._SUSPECT)
    
    @property
    def bad_ants(self):
        '''Set of ant-pol tuples with the current bad classification (default "bad").'''
        return self.get_all(self._BAD)
        
    def is_bad(self, ant):
        '''Returns True if antenna has the current bad classification (default "bad"), else False.'''
        return (ant in self._classification) and (self[ant] == self._BAD)

    def define_quality(self, good='good', suspect='suspect', bad='bad'):
        '''Resets the classifications considered good/suspect/bad. These are used for adding
        together AntennaClassification objects.
        
        Arguments:
            good: string to reset the classification considered good. Default "good".
            suspect: string to reset the classification considered suspect. Default "suspect".
            bad: string to reset the classification considered bad. Default "bad".
        '''
        self._GOOD = good
        self._SUSPECT = suspect
        self._BAD = bad
    
    @property
    def quality_classes(self):
        '''3-tuple of classification names considered good, suspect, and bad.'''
        return self._GOOD, self._SUSPECT, self._BAD
    
    def to_quality(self, good_classes=[], suspect_classes=[], bad_classes=[]):
        '''Function to reassigning classifications to good, suspect, or bad. All classifications
        must have a mapping to one of those three, which are by default "good", "suspect", and "bad",
        and can be queried by self.quality_classes and set by self.define_quality().
        
        Arguments:
            good_classes: list of string classfications to be renamed to the current good classification
            suspect_classes: list of string classfications to be renamed to the current suspect classification
            bad_classes: list of string classfications to be renamed to the current bad classification            
        '''        
        # check to make sure all classes have been sorted into good, suspect, or bad
        for cls in self.classes:
            if cls not in self.quality_classes:
                if (cls not in good_classes) and (cls not in suspect_classes) and (cls not in bad_classes):
                    raise ValueError(f'Unable to convert "{cls}" to one of {self.quality_classes}.')
        for new_class, old_classes in zip(self.quality_classes, [good_classes, suspect_classes, bad_classes]):
            for ant, cls in self._classification.items():
                if cls in old_classes:
                    self._classification[ant] = new_class

    def __add__(self, other):
        '''Combines together two AntennaClassification objects, returning a new one. Both objects
        must have the same quality_classes and all ant-pols in both must belong to those quality classes.
        Ant-pols that are bad in either object are bad in the result. Ant-pols that are good in both objects
        are good in the result. All other ant-pols are suspect. Antennas that are classified in one object 
        but absent from the other are included in the result with their classifications preserved.
        '''
        # make sure both obects are of type AntennaClassification
        if not issubclass(type(other), AntennaClassification):
            raise TypeError(f'Cannot add {type(other)} to AntennaClassification object.')
        # make sure both objects have the same names for good, suspect, and bad
        if not set(self.quality_classes) == set(other.quality_classes):
            raise ValueError(f'To combine AntennaClassification objects, their quality classes must be the same. \
                             {self.quality_classes} is not the same as {other.quality_classes}')
        # make sure all antennas in both objects are either good, suspect, or bad
        for o in [self, other]:
            if any([cls not in o.quality_classes for cls in o.classes]):
                raise ValueError(f'To add together two AntennaClassification objects, all classes must be one of \
                                 {o.quality_classes}, but one of {o.classes} is not.')
        
        # Figure out which antenna gets which classification in the combined object
        ants_here = set(self.ants)
        ants_there = set(other.ants)
        new_class = {}
        for ant in (ants_here | ants_there):
            if ant not in ants_here:
                new_class[ant] = other[ant]
            elif ant not in ants_there:
                new_class[ant] = self[ant]
            else:
                if (self[ant] == self._BAD) | (other[ant] == other._BAD):
                    new_class[ant] = self._BAD
                elif (self[ant] == self._GOOD) & (other[ant] == other._GOOD):
                    new_class[ant] = self._GOOD
                else:
                    new_class[ant] = self._SUSPECT
        
        # Build and return combined object, preserving quality classes
        ac = AntennaClassification(**{qual: [ant for ant in new_class if new_class[ant] == qual] for qual in self.quality_classes})
        ac.define_quality(*self.quality_classes)
        return ac

    def __str__(self):
        outstr = ''
        to_show = [cls for cls in self.quality_classes if cls in self.classes]
        to_show += [cls for cls in self.classes if cls not in self.quality_classes]
        pols = sorted(set([ant[1] for ant in self.ants]))
        for pol in pols:
            outstr += f'{pol}:\n----------\n'
            for cls in to_show:
                ants = sorted([ant for ant in self.get_all(cls) if ant[1] == pol])
                outstr += f'{cls} ({len(ants)} antpols):\n' + ', '.join([str(ant[0]) for ant in ants]) + '\n\n'
            outstr += '\n'

        return outstr.rstrip('\n')


def _is_bound(bound):
    '''Returns True if input is a length-2 iterable of numbers, the second >= the first, False otherwise.'''
    try:
        assert len(bound) == 2
        assert float(bound[0]) <= float(bound[1])
    except:
        return False
    return True


def antenna_bounds_checker(data, **kwargs):
    '''Converts scalar data about antennas to a classification using (potentially disjoint) bounds.
    Example usage: antenna_bounds_checker(data, good=[(.5, 1)], suspect=[(0, .5), (1, 2)], bad=(-np.inf, np.inf))
    
    Arguments:
        data: dictionary mapping ant-pol tuples (or autocorrelation keys) to scalar values
        kwargs: named antenna classifications and the range or ranges of data values which map to those 
            categories. Classification bounds are checked in order and the first accepted bound is used, 
            though it is possible for antenna to not get classified. All ranges are inclusive (i.e. >= or <=).
        
    Returns:
        AntennaClassification object using data and bounds to classify antennas in data
    '''
    
    classifiction_dict = {cls: set([]) for cls in kwargs}
    _data = {}
    for ant, val in data.items():
        # check that key is either a valid ant-pol tuple of an autocorrelation tuple
        try:
            _check_antpol(ant)
        except:
            try:  # try to convert autocorrelation 3-tuple into ant-pol
                from hera_cal.utils import split_bl
                ant1, ant2 = split_bl(ant)
                if ant1 == ant2:
                    ant = ant1
                else:
                    raise ValueError
            except:
                raise ValueError(f'{ant} is not a valid ant-pol tuple nor a valid autocorrelation key.')
        _data[ant] = val

        # classify antenna
        for cls, bounds in kwargs.items():
            if _is_bound(bounds):
                bounds = [bounds]
            # iterate over (potentially disjoint) bounts in this 
            for bound in bounds:
                if not _is_bound(bound):
                    raise ValueError(f'Count not convert {bounds} into a valid range or ranges for {cls} antennas.')
                if (val >= bound[0]) and (val <= bound[1]):
                    classifiction_dict[cls].add(ant)
                    break
            if ant in classifiction_dict[cls]:
                break
    
    ac = AntennaClassification(**classifiction_dict)
    ac._data = _data
    return ac


def auto_power_checker(data, good=(5, 30), suspect=(1, 80), int_count=None):
    '''Classifies ant-pols as good, suspect, or bad based on their median autocorrelations
    in units of int_count. Ant-pols not in the good or suspect ranges will be labeled "bad".

    Arguments:
        data: DataContainer containing antenna autocorrelations (other baselines ignored)
        good: 2-tuple or list of 2-tuple bounds for ranges considered good
        suspect: 2-tuple or list of 2-tuple bounds for ranges considered suspect. Ant-pols
            in both the good and suspect ranges will be labeled good.
        int_count: Number of samples per integration in correlator. If default None, use 
            data.times and data.freqs to infer int_count = channel resolution * integration time.
    
    Returns:
        AntennaClassification with "good", "suspect", and "bad" ant-pols based on median power
    '''
    # infer int_count if not provided from data's metadata
    if int_count is None:
        int_time = 24 * 3600 * np.median(np.diff(data.times))
        chan_res = np.median(np.diff(data.freqs))
        int_count = int(int_time * chan_res)

    # get autocorelation keys
    from hera_cal.utils import split_pol
    auto_bls = [bl for bl in data if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1])]
    
    # compute median powers and classify antennas
    auto_med_powers = {bl: np.median(np.abs(data[bl])) / int_count for bl in auto_bls}
    return antenna_bounds_checker(auto_med_powers, good=good, suspect=suspect, bad=(-np.inf, np.inf))


def auto_slope_checker(data, good=(-.2, .2), suspect=(-.4, .4), edge_cut=100, filt_size=17):
    '''Classifies ant-pols as good, suspect, or bad based on their relative rise of fall over
    the band after linear fitting median-filtered autocorrelations. Computes slope relative to
    offset in the linear fit, so a slope of 1 corresponds roughly to a doubling over the band.
    Ant-pols not in the good or suspect ranges will be labeled "bad".

    Arguments:
        data: DataContainer containing antenna autocorrelations (other baselines ignored)
        good: 2-tuple or list of 2-tuple bounds for ranges considered good. In units of relative
            slope over the band.
        suspect: 2-tuple or list of 2-tuple bounds for ranges considered suspect. Ant-pols
            in both the good and suspect ranges will be labeled good.
        edge_cut: integer number of frequency channels to ignore on either side of the band
            when computing slopes.
        filt_size: size of the median filter designed to mitigate RFI before linear filtering
    
    Returns:
        AntennaClassification with "good", "suspect", and "bad" ant-pols based on relative slope
    '''
    # get autocorelation keys
    from hera_cal.utils import split_pol
    auto_bls = [bl for bl in data if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1])]

    # compute relative slope over the band
    relative_slopes = {bl: 0 for bl in auto_bls}
    for bl in auto_bls:
        mean_data = np.mean(data[bl], axis=0)
        med_filt = median_filter(mean_data, size=filt_size)[edge_cut:-edge_cut]
        fit = np.polyfit(np.linspace(-.5, .5, len(mean_data))[edge_cut:-edge_cut], med_filt, 1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            if not np.all(med_filt == 0):
                relative_slopes[bl] = (fit[0] / fit[1] if np.isfinite(fit[1]) else np.sign(fit[0]) * np.inf)

    return antenna_bounds_checker(relative_slopes, good=good, suspect=suspect, bad=(-np.inf, np.inf))


def auto_shape_checker(data, good=(0, 0.0625), suspect=(0.0625, 0.125), flag_spectrum=None, antenna_class=None):
    """
    Classifies ant-pols as good, bad, or suspect based on their dissimilarity to the mean unflagged autocorrelation. 
    
    Arguments:
        data: DataContainer containing antenna autocorrelations (other baselines ignored)
        good: 2-tuple or list of 2-tuple, default=(0, 0.0625)
            2-tuple or list of 2-tuple bounds for ranges considered good.
        suspect: 2-tuple or list of 2-tuple, default=(0.0625, 0.125)
            Bounds for ranges considered suspect.
        flag_spectrum: optional numpy array of shape (Nfreqs,) where True is antenna contaminated by RFI
        antenna_class: AntennaClassification, default=None
            Optional AntennaClassification object. If provided, antennas marked "bad" will be excluded from the median auto.
        
    Returns:
        AntennaClassification with "good", "suspect", and "bad" ant-pols based on their bandpass shape
    """
    # figure out baselines, pols, etc.
    from hera_cal.utils import split_bl
    auto_bls = set([bl for bl in data if split_bl(bl)[0] == split_bl(bl)[1]])
    auto_pols = set([bl[2] for bl in auto_bls])
    ex_bls = set([bl for bl in auto_bls if (antenna_class[split_bl(bl)[0]] == 'bad' if antenna_class is not None else False)])
        
    # compute normalized reference bandpass of good antennas for each polarization
    template_bandpasses = {pol: np.where((flag_spectrum if flag_spectrum is not None else False), np.nan, 
                                         np.nanmean([data[bl] for bl in auto_bls if bl[2] == pol and bl not in ex_bls], 
                                                      axis=(0, 1))) for pol in auto_pols}
    template_bandpasses = {pol: template_bandpasses[pol] / np.nanmean(template_bandpasses[pol]) for pol in auto_pols}

    # compute per-auto distance from reference bandpass
    distance_metrics = {}
    for i, bl in enumerate(auto_bls):
        bandpass = np.where((flag_spectrum if flag_spectrum is not None else False), np.nan, np.mean(data[bl], axis=0))
        bandpass /= np.nanmean(bandpass)
        distance = (np.nanmean(np.abs(bandpass - template_bandpasses[bl[2]])**2))**.5
        distance_metrics[bl] = distance if np.isfinite(distance) else np.inf

    # classify based on distances
    return antenna_bounds_checker(distance_metrics, good=good, suspect=suspect, bad=(-np.inf, np.inf))


def auto_rfi_checker(data, good=(0, 0.01), suspect=(0.01, 0.02), nsig=6, antenna_class=None, flag_broadcast_thresh=0.5, 
                     kernel_widths=[3, 4, 5], mode='dpss_matrix', filter_centers=[0], filter_half_widths=[200e-9], 
                     eigenval_cutoff=[1e-9], cache={}):
    """
    Classifies ant-pols as good, suspect, or bad based on the fraction of channels flagged in that are not among the 
    array-broadcast flags (i.e. channels flagged for >50% of antennas). Flagging takes place in two steps: 
    (1) "channel_diff_flagger" is used to get an initial set of flags and
    (2) "dpss_flagger" is used with the array averaged flags to refine initial per-antenna flags
 
    Arguments:
        data: DataContainer containing antenna autocorrelations (other baselines ignored)
        good: 2-tuple or list of 2-tuple, default=(0, 0.01)
            2-tuple or list of 2-tuple bounds for ranges considered good.
        suspect: 2-tuple or list of 2-tuple, default=(0.01, 0.02)
            Bounds for ranges considered suspect.
        nsig: float, default=6
            The number of sigma in the metric above which to flag pixels. Used in both steps.
        antenna_class: AntennaClassification, default=None
            Optional AntennaClassification object. If provided, the flagging method chosen will skip antennas marked "bad".
            Used in both steps
        flag_broadcast_thresh: float, default=0.5
            The fraction of flags required to trigger a broadcast across all auto-correlations for
            a given (time, frequency) pixel in the combined flag array. Used in both steps.
        kernel_widths: list
            Half-width of the convolution kernels used to produce model. True kernel width is (2 * kernel_width + 1)
            Only used in the "channel_diff_flagger" step
        mode: str, default='dpss_matrix'
            Method used to solve for DPSS model components. Options are 'dpss_matrix', 'dpss_solve', and 'dpss_leastsq'.
            Only used in "dpss_flagger" step
        filter_centers: array-like, default=[0]
            list of floats of centers of delay filter windows in nanosec. Only used in "dpss_flagger"
        filter_half_widths: array-like, default=[200e-9]
            list of floats of half-widths of delay filter windows in nanosec. Only used in "dpss_flagger"
        cache: dictionary, default=None
            Dictionary for caching fitting matrices. By default this value is None to prevent the size of the cached
            matrices from getting too large. By passing in a cache dictionary, this function could be much faster, but
            the memory requirement will also increase.
        
    Returns:
        AntennaClassification with "good", "suspect", and "bad" ant-pols based on the absolute fraction of 
        the band that is flagged
    """
    # Flag using convolution kernels
    antenna_flags, array_flags = xrfi.flag_autos(data, flag_method="channel_diff_flagger", nsig=nsig, antenna_class=antenna_class,
                                                 flag_broadcast_thresh=flag_broadcast_thresh, kernel_widths=kernel_widths)

    # Override antenna flags with array-wide flags for next step
    for key in antenna_flags.keys():
        antenna_flags[key] = array_flags

    # Flag using DPSS filters
    antenna_flags, array_flags = xrfi.flag_autos(data, freqs=data.freqs, flag_method="dpss_flagger", nsig=nsig, antenna_class=antenna_class,
                                                 filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                                 eigenval_cutoff=eigenval_cutoff, flags=antenna_flags, mode=mode, cache=cache)
    
    
    # Calculate the excess fraction of the band that is flagged
    flagged_fraction = {bls: np.mean(flags | array_flags) - np.mean(array_flags) for bls, flags in antenna_flags.items()}

    return antenna_bounds_checker(flagged_fraction, good=good, suspect=suspect, bad=(-np.inf, np.inf))

def even_odd_zeros_checker(sum_data, diff_data, good=(0, 2), suspect=(2, 8)):
    '''Classifies ant-pols as good, suspect, or bad based on the maximum number of zeros
    that appear in single-time even or odd visibility spectra. That maximum is assigned
    to antennas in order of the total number of zeros in baselines that antenna participates
    in, so antennas with more zeros overall are blamed for a particular baseline having zeros.
    
    Arguments:
        sum_data: DataContainer containing full visibility data set
        diff_data: DataContainer containing time-interleaved difference visibility data
        good: 2-tuple or list of 2-tuples of ranges of the maximum number of zeros in an 
            even or odd spectrum attributed to a given antenna. Default is to allow at most
            2 zeros in a spectrum which cannot be attributed to the other antenna involved.
        suspect: 2-tuple or list of 2-tuples of ranges for the maximum number of zeros
            attributable to given antenna considered suspect. Default is 8. Ant-pols
            in both the good and suspect ranges are good, all others are bad.
            
    Returns:
        AntennaClassification with "good", "suspect", and "bad" ant-pols based on number of zeros
    '''
    from hera_cal.utils import split_bl
    ants = sorted(set([ant for bl in sum_data for ant in split_bl(bl)]))
    zero_count_by_ant = {ant: 0 for ant in ants}
    max_zeros_per_spectrum = {}
    
    # calculate the maximum number of zeros per spectrum for each baseline in the odd or even data,
    # as well as the total number of zeros in baselines each antenna is involved in
    for bl in sum_data:
        even_zeros  = np.sum((sum_data[bl] + diff_data[bl]) == 0, axis=1)
        odd_zeros = np.sum((sum_data[bl] - diff_data[bl]) == 0, axis=1)
        max_zeros_per_spectrum[bl] = np.max([even_zeros, odd_zeros])
        for ant in split_bl(bl):
            zero_count_by_ant[ant] += max_zeros_per_spectrum[bl]
    
    # sort dictionary of antennas by number of even/odd visibility zeros it participates in
    zero_count_by_ant = {k: v for k, v in sorted(zero_count_by_ant.items(), key=lambda item: item[1], reverse=True)}
    
    # Loop over antennas in order of zero_count_by_ant, calculating the maximum number of zeros in a spectrum
    # attributable to that antenna. After calculating this for each antenna, all baselines involving that
    # antenna are removed from bls_with_zeros. In this way, a baseline's zeros are only attributed to one antenna
    most_zeros = {}
    bls_with_zeros = set([bl for bl in sum_data if max_zeros_per_spectrum[bl] > 0])
    for ant in zero_count_by_ant:
        remaining_max_zeros = [max_zeros_per_spectrum[bl] for bl in bls_with_zeros if ant in split_bl(bl)]
        if len(remaining_max_zeros) > 0:
            most_zeros[ant] = np.max(remaining_max_zeros)
            bls_with_zeros = set([bl for bl in bls_with_zeros if ant not in split_bl(bl)])
        else:
            most_zeros[ant] = 0
    
    # run and return classifier
    return antenna_bounds_checker(most_zeros, good=good, suspect=suspect, bad=(0, np.inf))
