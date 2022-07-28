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
    
