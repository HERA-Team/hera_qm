# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
from hera_qm import ant_class


def test_check_antpol():
    ant_class._check_antpol((1, 'Jnn'))
    ant_class._check_antpol((1, 'ee'))
    ant_class._check_antpol((1, 'x'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1.0, 'Jee'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1, 'not a pol'))
    with pytest.raises(ValueError):
        ant_class._check_antpol((1, 1, 'Jee'))


def test_AntennaClassification():
    # test that that doubled antenna raises error
    with pytest.raises(ValueError):
        ant_class.AntennaClassification(good=[(1, 'Jee')], bad=[(1, 'Jee')])
            
    ac = ant_class.AntennaClassification(good=[(0, 'Jnn'), (0, 'Jee')],
                                         bad=[(1,'Jee'), (1, 'Jnn')], 
                                         suspect=[(2, 'Jee')],
                                         weird=[(2, 'Jnn')])  

    # test getter
    assert ac[(0, 'Jnn')] == 'good'
    assert ac[(2, 'Jnn')] == 'weird'

    # test setter
    ac[(2, 'Jee')] = 'strange'
    assert ac[(2, 'Jee')] == 'strange'
    ac[(2, 'Jee')] = 'suspect'

    # test iter
    assert (0, 'Jnn') in ac
    assert len(list(ac.__iter__())) == 6

    # test classes
    assert ac.classes == set(['good', 'bad', 'suspect', 'weird'])

    # test ants
    assert set(ac.ants) == set([(0, 'Jee'), (0, 'Jnn'),
                                (1, 'Jee'), (1, 'Jnn'),
                                (2, 'Jee'), (2, 'Jnn')])

    # test get_all
    assert ac.get_all('weird') == set([(2, 'Jnn')])

    # test good_ants, suspect_ants, bad_ants
    assert ac.good_ants == set([(0, 'Jee'), (0, 'Jnn')])
    assert ac.suspect_ants == set([(2, 'Jee')])
    assert ac.bad_ants == set([(1, 'Jee'), (1, 'Jnn')])

    # test is_good, is_bad, is_suspect
    assert ac.is_good((0, 'Jee'))
    assert not ac.is_good((1, 'Jee'))
    assert ac.is_bad((1, 'Jee'))
    assert not ac.is_bad((2, 'Jee'))
    assert ac.is_suspect((2, 'Jee'))
    assert not ac.is_suspect((3, 'Jee'))

    # test quality_classes, define_quality, to_quality
    assert ac.quality_classes == ('good', 'suspect', 'bad')
    ac.define_quality(suspect='weird')
    assert ac.is_suspect((2, 'Jnn'))
    assert not ac.is_suspect((2, 'Jee'))

    with pytest.raises(ValueError):
        ac.to_quality()
    ac.to_quality(suspect_classes=['suspect', 'weird'])
                
    # test clear
    ac.define_quality(good='lep', suspect='korf', bad='pillot')
    ac.clear()
    assert ac.quality_classes == ('good', 'suspect', 'bad')
    assert len(ac.classes) == 0
    assert len(ac.ants) == 0


