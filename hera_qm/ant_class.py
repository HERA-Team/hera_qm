# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to classify antennas by various data quality metrics."""
import numpy as np


def _check_antpol(ap):
    '''Verifies that input is a valid ant-pol tuple. Otherwies, raises a ValueError'''
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
        '''Sorted list of unique antenna classifications.'''
        return sorted(set(self._classification.values()))

    @property
    def ants(self):
        '''Sorted list of all classified ant-pol tuples.'''
        return sorted(self._classification.keys())
    
    def get_all(self, classification):
        '''Return all ant-pol tuples with the given classification.'''
        return sorted([ant for ant, cls in self._classification.items() if cls == classification])
   
    @property
    def good_ants(self):
        '''Sorted list of ant-pol tuples with the current good classification (default "good").'''
        return self.get_all(self._GOOD)     
    
    def is_good(self, ant):
        '''Returns True if antenna has the current good classification (default "good").'''
        return self[ant] == self._GOOD
    
    @property
    def suspect_ants(self):
        '''Sorted list of ant-pol tuples with the current suspect classification (default "suspect").'''
        return self.get_all(self._SUSPECT)  

    def is_suspect(self, ant):
        '''Returns True if antenna has the current suspect classification (default "suspect").'''
        return self[ant] == self._SUSPECT    
    
    @property
    def bad_ants(self):
        '''Sorted list of ant-pol tuples with the current bad classification (default "bad").'''
        return self.get_all(self._BAD)
        
    def is_bad(self, ant):
        '''Returns True if antenna has the current bad classification (default "bad").'''
        return self[ant] == self._BAD
        
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
        data: dictionary mapping ant-pol tuples (or autocorrelation keys) to 
        kwargs: named antenna classifications and the range or ranges of data values which map to those 
            categories. Classification bounds are checked in order and the first accepted bound is used. 
            All ranges are inclusive (i.e. >= or <=).
        
    Returns:
        AntennaClassification object using data and bounds to classify antennas in data
    '''
    
    classifiction_dict = {cls: set([]) for cls in kwargs}
    for ant in data:
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
        
        # classify antenna
        for cls, bounds in kwargs.items():
            if _is_bound(bounds):
                bounds = [bounds]
            # iterate over (potentially disjoint) bounts in this 
            for bound in bounds:
                if not _is_bound(bound):
                    raise ValueError(f'Count not convert {bound} into a valid range for {cls} antennas.')
                if (data[ant] >= bound[0]) and (data[ant] <= bound[1]):
                    classifiction_dict[cls].add(ant)
                    break
            if ant in classifiction_dict[cls]:
                break
    
    return AntennaClassification(**classifiction_dict)
