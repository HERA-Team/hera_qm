# -*- coding: utf-8 -*-
# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
from hera_qm import ant_class
from hera_cal import io
from hera_qm.data import DATA_PATH


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

    # test string
    assert 'good' in str(ac)
    assert 'suspect' in str(ac).split('good')[1]
    assert 'bad' in str(ac).split('suspect')[1]
    assert 'weird' in str(ac).split('bad')[1]

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


def test_AntennaClassification_add():
    ac1 = ant_class.AntennaClassification(good=[(0, 'Jnn'), (0, 'Jee'), (1, 'Jnn')],
                                      bad=[(1,'Jee')], 
                                      weird=[(2, 'Jee'), (2, 'Jnn')])
    ac2 = ant_class.AntennaClassification(good=[(0, 'Jnn')],
                                          bad=[(1,'Jee'), (0, 'Jee'), (2, 'Jee')], 
                                          weird=[(1, 'Jnn'), (2, 'Jnn')])
    # test wrong type error
    with pytest.raises(TypeError):
        ac1 += 1

    # test non-quality class error
    with pytest.raises(ValueError):
        ac = ac1 + ac2

    ac1.define_quality(suspect='weird')

    # test quality class mismatch error
    with pytest.raises(ValueError):
        ac = ac1 + ac2

    ac2.define_quality(suspect='weird')
    ac = ac1 + ac2

    # test good + good
    assert ac.is_good((0, 'Jnn'))

    # test good + bad
    assert ac.is_bad((0, 'Jee'))

    # test good + suspect
    assert ac.is_suspect((1, 'Jnn'))

    # test suspect + suspect
    assert ac.is_suspect((2, 'Jnn'))

    # test suspect + bad
    assert ac.is_bad((2, 'Jee'))

    # test bad + bad
    assert ac.is_bad((1,'Jee'))


def test_is_bound():
    assert ant_class._is_bound([0, 1])
    assert ant_class._is_bound(np.array([0, 1]))
    assert ant_class._is_bound((0, 1))
    assert ant_class._is_bound([0, 1.0])

    assert not ant_class._is_bound([2, 1, 3])
    assert not ant_class._is_bound([2, 1])
    assert not ant_class._is_bound([1, 'stuff'])
    assert not ant_class._is_bound([1, 1 + 1j])


def test_antenna_bounds_checker():
    data = {(1, 'Jee'): 1, (2, 'Jee'): 3, (3, 'Jee'): 4, (4, 4, 'ee'): 10}

    ac = ant_class.antenna_bounds_checker(data, good=[(0, 2), (3.5, 4)], weird=(4, np.inf))
    ac.define_quality(bad='weird')
    assert (1, 'Jee') in ac.good_ants
    assert (3, 'Jee') in ac.good_ants
    assert (3, 'Jee') not in ac.bad_ants
    assert (4, 'Jee') in ac.bad_ants
    assert (4, 4, 'ee') not in ac
    assert (2, 'Jee') not in ac

    with pytest.raises(ValueError):
        ac = ant_class.antenna_bounds_checker(data, bad_bound=[(0, -1)])
        ac = ant_class.antenna_bounds_checker(data, bad_bound=(0, -1))
        ac = ant_class.antenna_bounds_checker({(1, 2, 'ee'): 1.0}, bad_bound=[(0, -1)])
        ac = ant_class.antenna_bounds_checker({(1, 2, 'ee'): 1.0}, bound=[(0, 1)])


def test_auto_power_checker():
    hd = io.HERADataFastReader(DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5')
    data, _, _ = hd.read(read_flags=False, read_nsamples=False)
    auto_power_class = ant_class.auto_power_checker(data, good=(2,30), suspect=(1,80))

    for ant in {(36, 'Jee'), (36, 'Jnn'), (51, 'Jnn'), (83, 'Jee'), (83, 'Jnn'), (87, 'Jee'), (98, 'Jee'), (98, 'Jnn'), (117, 'Jnn'), (135, 'Jnn'), (160, 'Jee')}:
        assert ant in auto_power_class.good_ants
    
    for ant in {(51, 'Jee'), (53, 'Jee'), (53, 'Jnn'), (85, 'Jee'), (85, 'Jnn'), (87, 'Jnn'), (117, 'Jee'), (157, 'Jee'), (157, 'Jnn'), (160, 'Jnn')}:
        assert ant in auto_power_class.suspect_ants
    
    for ant in {(65, 'Jee'), (65, 'Jnn'), (68, 'Jee'), (68, 'Jnn'), (93, 'Jee'), (93, 'Jnn'), (116, 'Jee'), (116, 'Jnn'), (135, 'Jee')}:
        assert ant in auto_power_class.bad_ants


def test_auto_slope_checker():
    hd = io.HERADataFastReader(DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5')
    data, _, _ = hd.read(read_flags=False, read_nsamples=False)
    auto_slope_class = ant_class.auto_slope_checker(data, good=(-.2, .2), suspect=(-.4, .4), edge_cut=20)  # smaller edge cut due to downsampling

    for ant in {(83, 'Jee'), (160, 'Jee'), (85, 'Jee'), (98, 'Jee'), (83, 'Jnn'), (160, 'Jnn'), (85, 'Jnn'), (98, 'Jnn'), (36, 'Jee'), (135, 'Jee'), (157, 'Jee'),
                (51, 'Jee'), (87, 'Jnn'), (36, 'Jnn'), (135, 'Jnn'), (157, 'Jnn'), (117, 'Jee'), (53, 'Jee'), (51, 'Jnn'), (117, 'Jnn'), (53, 'Jnn')}:
        assert ant in auto_slope_class.good_ants
    
    for ant in {(68, 'Jee'), (87, 'Jee'), (68, 'Jnn')}:
        assert ant in auto_slope_class.suspect_ants
    
    for ant in {(65, 'Jnn'), (116, 'Jee'), (93, 'Jnn'), (65, 'Jee'), (93, 'Jee'), (116, 'Jnn')}:
        assert ant in auto_slope_class.bad_ants


def test_auto_shape_checker():
    from hera_qm.data import DATA_PATH
    hd = io.HERADataFastReader(DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5')
    hd = io.HERADataFastReader(DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5')
    data, _, _ = hd.read(read_flags=False, read_nsamples=False)
    auto_slope_class = ant_class.auto_slope_checker(data, good=(-.2, .2), suspect=(-.4, .4), edge_cut=20) 
    auto_power_class = ant_class.auto_power_checker(data, good=(2, 30), suspect=(1, 80))
    flag_spectrum = np.mean(data[36,36,'nn'], axis=0) > 1.5e7
    auto_shape_class = ant_class.auto_shape_checker(data, good=(0, 0.0625), suspect=(0.0625, 0.125), 
                                                    flag_spectrum=flag_spectrum,
                                                    antenna_class=(auto_slope_class + auto_power_class))
    for ant in {(160, 'Jee'), (83, 'Jee'), (68, 'Jnn'), (85, 'Jee'), (98, 'Jee'), (135, 'Jee'), (157, 'Jee'), (160, 'Jnn'), (36, 'Jee'), (85, 'Jnn'), 
                (83, 'Jnn'), (98, 'Jnn'), (117, 'Jee'), (87, 'Jnn'), (53, 'Jee'), (135, 'Jnn'), (157, 'Jnn'), (36, 'Jnn'), (117, 'Jnn'), (53, 'Jnn')}:
        assert ant in auto_shape_class.good_ants
    for ant in {(68, 'Jee'), (87, 'Jee'), (51, 'Jee'), (51, 'Jnn')}:
        assert ant in auto_shape_class.suspect_ants
    for ant in {(116, 'Jee'), (116, 'Jnn'), (93, 'Jee'), (65, 'Jee'), (65, 'Jnn'), (93, 'Jnn')}:
        assert ant in auto_shape_class.bad_ants


def test_auto_rfi_checker():
    hd = io.HERADataFastReader(DATA_PATH + '/zen.2459122.49827.sum.downselected.uvh5')
    data, _, _ = hd.read(read_flags=False, read_nsamples=False)

    # Get bad antennas
    auto_power_class = ant_class.auto_power_checker(data, good=(2, 30), suspect=(1, 80))
    auto_slope_class = ant_class.auto_slope_checker(data, good=(-.2, .2), suspect=(-.4, .4), edge_cut=20)  # smaller edge cut due to downsampling
    auto_class = auto_power_class + auto_slope_class

    # Modify metadata to compensate for down-selection
    data.times /= 5

    # Artificially add RFI to autos
    idx = np.arange(0, data.freqs.shape[0], 10)
    data[(36, 36, 'ee')][:, idx] *= 1.2 # Bad auto
    idx = np.arange(0, data.freqs.shape[0], 30)
    data[(83, 83, 'ee')][:, idx] *= 1.2 # Suspect auto

    # Run RFI checker
    auto_rfi_class = ant_class.auto_rfi_checker(data, antenna_class=auto_class, good=(0, 0.1), suspect=(0.1, 0.2),
                                                kernel_widths=[1, 2], filter_centers=[0, 2700e-9, -2700e-9],
                                                filter_half_widths=[200e-9, 200e-9, 200e-9])
    assert (36, 'Jee') in auto_rfi_class.bad_ants
    assert (83, 'Jee') in auto_rfi_class.suspect_ants

    # Make sure antennas that were previously marked bad are still marked bad
    for ant in auto_class.bad_ants:
        assert ant in auto_rfi_class.bad_ants
    
    # Show that all other antennas are marked "good"
    for ant in auto_class.ants:
        if ant not in [(36, 'Jee'), (83, 'Jee')] and ant not in auto_class.bad_ants:
            assert ant in auto_rfi_class.good_ants

def test_even_odd_zeros_checker():
    even, odd = {}, {}
    for bl in [(0, 1, 'ee'), (0, 2, 'ee'), (0, 3, 'ee'), (1, 2, 'ee'), (1, 3, 'ee'), (2, 3, 'ee')]:
        even[bl] = np.ones((2, 1024))
        odd[bl] = np.ones((2, 1024))

    for bl in [(0, 3, 'ee'), (1, 3, 'ee'), (2, 3, 'ee')]:
        even[bl][:, 0:512] = 0

    for bl in [(0, 1, 'ee'), (0, 2, 'ee'), (0, 3, 'ee')]:
        odd[bl][:, 100:105] = 0

    sums, diff = {}, {}
    for bl in even:
        sums[bl] = even[bl] + odd[bl]
        diff[bl] = even[bl] - odd[bl]

    zeros_class = ant_class.even_odd_zeros_checker(sums, diff, good=(0, 2), suspect=(2, 8))
    assert zeros_class[0, 'Jee'] == 'suspect'
    assert zeros_class[1, 'Jee'] == 'good'
    assert zeros_class[2, 'Jee'] == 'good'
    assert zeros_class[3, 'Jee'] == 'bad'
