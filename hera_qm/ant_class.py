# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License

"""Class and algorithms to classify antennas by various data quality metrics."""
import numpy as np


def _check_antpol(ap):
    '''Verifies that input is a valid ant-pol tuple. Otherwies, raises a ValueError'''
    from hera_cal.utils import join_bl
    try:
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
        
