# Copyright (c) 2022 the HERA Project
# Licensed under the MIT License
"""Tests for the antenna_metrics module."""

import pytest
import numpy as np
from hera_qm import ant_class
from hera_cal import io
from hera_cal.datacontainer import DataContainer
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
    assert ac.classes == {'good', 'bad', 'suspect', 'weird'}

    # test ants
    assert set(ac.ants) == {(0, 'Jee'), (0, 'Jnn'),
                                (1, 'Jee'), (1, 'Jnn'),
                                (2, 'Jee'), (2, 'Jnn')}

    # test get_all
    assert ac.get_all('weird') == {(2, 'Jnn')}

    # test good_ants, suspect_ants, bad_ants
    assert ac.good_ants == {(0, 'Jee'), (0, 'Jnn')}
    assert ac.suspect_ants == {(2, 'Jee')}
    assert ac.bad_ants == {(1, 'Jee'), (1, 'Jnn')}

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

    # test with all Jee antennas previously marked as bad
    input_class = auto_slope_class + auto_power_class
    for ant in input_class.ants:
        if ant[1] == 'Jee':
            input_class[ant] = 'bad'
    auto_shape_class = ant_class.auto_shape_checker(data, good=(0, 0.0625), suspect=(0.0625, 0.125),
                                                    flag_spectrum=flag_spectrum, antenna_class=input_class)
    for ant in {(68, 'Jnn'), (160, 'Jnn'), (85, 'Jnn'), (83, 'Jnn'), (98, 'Jnn'), (87, 'Jnn'), (135, 'Jnn'), (157, 'Jnn'), (36, 'Jnn'), (117, 'Jnn'), (53, 'Jnn')}:
        assert ant in auto_shape_class.good_ants
    for ant in {(51, 'Jnn')}:
        assert ant in auto_shape_class.suspect_ants
    for ant in {(85, 'Jee'), (98, 'Jee'), (36, 'Jee'), (117, 'Jee'), (53, 'Jee'), (135, 'Jee'), (157, 'Jee'), (160, 'Jee'), (83, 'Jee'), (116, 'Jee'),
               (68, 'Jee'), (87, 'Jee'), (51, 'Jee'), (116, 'Jnn'), (93, 'Jee'), (65, 'Jee'), (65, 'Jnn'), (93, 'Jnn')}:
        assert ant in auto_shape_class.bad_ants


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
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
                                                kernel_widths=[1, 2], filter_centers=[0],
                                                filter_half_widths=[200e-9])
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


def test_non_noiselike_diff_by_xengine_checker():
    rng = np.random.default_rng(21)
    sums = DataContainer({(ant1, ant2, 'ee'): (np.ones((2, 1536), dtype=complex) if ant1 != ant2 else np.ones((2, 1536)) * 100)
                          for ant1 in range(10) for ant2 in range(ant1, 10)})
    diffs = DataContainer({})

    sums.freqs = np.linspace(50e6, 225e6, 1536)
    sums.times = np.array([2459866.32713241, 2459866.32724426])
    sums.times_by_bl = {bl[0:2]: sums.times for bl in sums}

    for bl in sums:
        sigma = np.sqrt(sums[bl[0], bl[0], 'ee'] * sums[bl[1], bl[1], 'ee'] / np.median(np.diff(sums.freqs)) / (np.median(np.diff(sums.times)) * 24 * 3600))
        if bl[0] != bl[1]:
            sums[bl] += sigma / 2**.5 * rng.standard_normal((2, 1536)) + 1.0j * sigma / 2**.5 * rng.standard_normal((2, 1536))
            diffs[bl] = sigma / 2**.5 * rng.standard_normal((2, 1536)) + 1.0j * sigma / 2**.5 * rng.standard_normal((2, 1536))
            if (3 in bl) or (7 in bl) or (8 in bl):
                diffs[bl][:, 96:192] = sums[bl][:, 96:192]

        else:
            diffs[bl] = np.zeros_like(sums[bl])

    ac = ant_class.non_noiselike_diff_by_xengine_checker(sums, diffs)
    for ant in [0, 1, 2, 4, 5, 6, 9]:
        assert ac[(ant, 'Jee')] == 'good'
    for ant in [3, 7, 8]:
        assert ac[(ant, 'Jee')] == 'bad'

    ac = ant_class.non_noiselike_diff_by_xengine_checker(sums, diffs, antenna_class=ant_class.AntennaClassification(bad=[(3, 'Jee')]))
    for ant in [0, 1, 2, 4, 5, 6, 9]:
        assert ac[(ant, 'Jee')] == 'good'
    for ant in [7, 8]:
        assert ac[(ant, 'Jee')] == 'bad'


def _build_identity_sim(nants=8, nfreqs=600, relabels=None, noise=0.02, seed=0):
    '''Build synthetic (data, model, bls) for identity-audit tests, with data and model as
    DataContainers: per-baseline smooth models,
    data = model x cable-delay phases + noise. If relabels is given (dict mapping true antenna
    to the label its visibilities receive), the data keys are permuted accordingly, mimicking
    a cabling/M&C mislabeling.'''
    rng = np.random.default_rng(seed)
    freqs = np.linspace(100e6, 200e6, nfreqs)
    dlys = {antnum: rng.uniform(-200e-9, 200e-9) for antnum in range(nants)}
    data, model = {}, {}
    for pol in ['ee', 'nn']:
        for i in range(nants):
            for j in range(i + 1, nants):
                x = np.linspace(0, 1, nfreqs)
                amp = 10 * (1 + 0.5 * np.cos(2 * np.pi * x * rng.integers(1, 4)
                                             + rng.uniform(0, 2 * np.pi)))
                # phases need smooth CURVATURE, not just ramps: a delay search forgives
                # any purely linear phase difference between candidate models
                phs = sum(rng.uniform(2, 6) * np.cos((k + 1) * np.pi * x + rng.uniform(0, 2 * np.pi))
                          for k in range(4))
                model[(i, j, pol)] = (amp * np.exp(1j * phs))[None, :]
                vis = model[(i, j, pol)] * np.exp(2j * np.pi * freqs * (dlys[i] - dlys[j]))[None, :]
                vis = vis + noise * np.mean(amp) * (rng.normal(size=vis.shape)
                                                    + 1j * rng.normal(size=vis.shape)) / np.sqrt(2)
                data[(i, j, pol)] = vis
    if relabels is not None:
        relabeled = {}
        for (i, j, pol), vis in data.items():
            new_i, new_j = relabels.get(i, i), relabels.get(j, j)
            if new_i < new_j:
                relabeled[(new_i, new_j, pol)] = vis
            else:
                relabeled[(new_j, new_i, pol)] = np.conj(vis)
        data = relabeled
    bls = sorted(model.keys())
    return DataContainer(data), DataContainer(model), bls


def test_vis_vs_model_coherence():
    data, model, bls = _build_identity_sim(nants=4)
    # matched data and model: coherent after a single delay
    assert ant_class.vis_vs_model_coherence(data[bls[0]], model[bls[0]]) > 0.9
    # data tested against an unrelated baseline's model: incoherent
    assert ant_class.vis_vs_model_coherence(data[bls[0]], model[bls[1]]) < 0.5
    # 2 unflagged channels make the single-delay fit interpolatory: nan, not a perfect score
    flags = np.ones((1, len(model[bls[0]][0])), dtype=bool)
    assert np.isnan(ant_class.vis_vs_model_coherence(data[bls[0]], model[bls[0]],
                                                     flag_waterfall=flags))
    flags[0, :2] = False
    assert np.isnan(ant_class.vis_vs_model_coherence(data[bls[0]], model[bls[0]],
                                                     flag_waterfall=flags))
    flags[0, 2:100] = False
    assert np.isfinite(ant_class.vis_vs_model_coherence(data[bls[0]], model[bls[0]],
                                                        flag_waterfall=flags))
    # all integrations are used: with one integration fully flagged and one clean, the
    # clean one carries the statistic (and interpolatory integrations are excluded per-time)
    data2 = np.vstack([data[bls[0]], data[bls[0]]])
    model2 = np.vstack([model[bls[0]], model[bls[0]]])
    flags2 = np.zeros_like(data2, dtype=bool)
    flags2[0, :] = True
    assert ant_class.vis_vs_model_coherence(data2, model2, flag_waterfall=flags2) > 0.9
    assert ant_class.vis_vs_model_coherence(data2, model2) > 0.9
    # non-finite data samples are excluded like flags rather than NaNing the whole statistic
    nan_data = np.array(data[bls[0]])
    nan_data[0, 200:210] = np.nan
    assert ant_class.vis_vs_model_coherence(nan_data, model[bls[0]]) > 0.9


def test_antenna_identity_checker_clean():
    data, model, bls = _build_identity_sim()
    groups = {antnum: list(range(8)) for antnum in range(8)}
    identity_class, labeled_to_true, self_coherence = ant_class.antenna_identity_checker(
        data, model, bls, groups)
    assert labeled_to_true == {}
    assert len(identity_class.bad_ants) == 0
    assert len(identity_class.suspect_ants) == 0
    for ant, coh in self_coherence.items():
        assert coh > 0.8
        assert identity_class._data[ant] == coh


def test_antenna_identity_checker_finds_swap():
    # antennas 2 and 3 have their labels exchanged in the data
    data, model, bls = _build_identity_sim(relabels={2: 3, 3: 2})
    groups = {antnum: list(range(8)) for antnum in range(8)}
    # healthy antennas' mean self-coherence includes their baselines TO the swapped pair,
    # which drags it down in a small array (2 of 7 partners); loosen the good bound accordingly
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1))
    assert labeled_to_true == {2: 3, 3: 2}
    for antnum in [2, 3]:
        for antpol in ['Jee', 'Jnn']:
            assert identity_class[(antnum, antpol)] == 'suspect'
    for antnum in [0, 1, 4, 5, 6, 7]:
        for antpol in ['Jee', 'Jnn']:
            assert identity_class[(antnum, antpol)] == 'good'


def test_antenna_identity_checker_repair_to_merely_suspect():
    # a winner that clears only the suspect bound still earns a relabeling: any identity
    # beats a known-wrong one, and relabeled antennas are classified suspect regardless
    data, model, bls = _build_identity_sim(relabels={2: 3, 3: 2})
    groups = {antnum: list(range(8)) for antnum in range(8)}
    identity_class, labeled_to_true, self_coherence = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.999, 1), suspect=(0.5, 1), verbose=False)
    assert labeled_to_true == {2: 3, 3: 2}
    for antpol in ['Jee', 'Jnn']:
        assert identity_class[(2, antpol)] == 'suspect'

    # with the suspect bound above every candidate's coherence, no repair is decisive
    _, labeled_to_true_strict, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.999, 1), suspect=(0.999, 1), verbose=False)
    assert labeled_to_true_strict == {}


def test_antenna_identity_checker_closes_broken_cycles():
    # antennas 2 and 3 are swapped, but labeled-3's scan cannot see its true identity
    # (candidate group restricted), so only 2 -> 3 is measured. The checker must close the
    # cycle anyway: labeled-3's number is claimed and only 2 is left vacant, so it is
    # inferred there -- otherwise antenna 2 would vanish and two streams would collide on
    # 3 -- and classified bad, since the placement is forced rather than measured
    data, model, bls = _build_identity_sim(relabels={2: 3, 3: 2})
    groups = {antnum: list(range(8)) for antnum in range(8)}
    groups[3] = [3]
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1), verbose=False)
    assert labeled_to_true == {2: 3, 3: 2}
    for antpol in ['Jee', 'Jnn']:
        assert identity_class[(2, antpol)] == 'suspect'
        assert identity_class[(3, antpol)] == 'bad'


def test_antenna_identity_checker_closes_5cycle_missing_middle():
    # a 5-cycle with one scan blinded mid-cycle: the closure must walk the whole remaining
    # path (2 -> 1 -> 5 -> 4 -> 3) to find that only labeled-3 is unmoved and only 2 is
    # vacant, recovering the TRUE cycle exactly, with the inferred stream classified bad
    data, model, bls = _build_identity_sim(nants=12, relabels={1: 2, 2: 3, 3: 4, 4: 5, 5: 1})
    groups = {antnum: list(range(12)) for antnum in range(12)}
    groups[3] = [3]
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1), verbose=False)
    assert labeled_to_true == {2: 1, 3: 2, 4: 3, 5: 4, 1: 5}
    for antpol in ['Jee', 'Jnn']:
        assert identity_class[(3, antpol)] == 'bad'
        for antnum in [1, 2, 4, 5]:
            assert identity_class[(antnum, antpol)] == 'suspect'


def test_antenna_identity_checker_closes_two_broken_cycles():
    # two independent swaps, each with one scan blinded: each component closes on its own,
    # correctly, and only the two inferred streams classify bad
    data, model, bls = _build_identity_sim(nants=10, relabels={1: 2, 2: 1, 4: 5, 5: 4})
    groups = {antnum: list(range(10)) for antnum in range(10)}
    groups[2] = [2]
    groups[5] = [5]
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1), verbose=False)
    assert labeled_to_true == {1: 2, 2: 1, 4: 5, 5: 4}
    for antpol in ['Jee', 'Jnn']:
        for antnum in [2, 5]:
            assert identity_class[(antnum, antpol)] == 'bad'
        for antnum in [1, 4]:
            assert identity_class[(antnum, antpol)] == 'suspect'


def test_antenna_identity_checker_fragmented_cycle_infers_bad():
    # a 4-cycle with TWO opposite scans blinded fragments into two accepted paths, and each
    # closes on itself: {2->1, 1->2} and {4->3, 3->4}. Both inferences are factually WRONG
    # (the truth is 1->4 and 3->2), which is exactly why inferred placements classify bad:
    # the bookkeeping stays a closed permutation, but the misplaced streams are never trusted
    data, model, bls = _build_identity_sim(nants=10, relabels={1: 2, 2: 3, 3: 4, 4: 1})
    groups = {antnum: list(range(10)) for antnum in range(10)}
    groups[1] = [1]
    groups[3] = [3]
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1), verbose=False)
    assert labeled_to_true == {2: 1, 1: 2, 4: 3, 3: 4}
    for antpol in ['Jee', 'Jnn']:
        for antnum in [1, 3]:
            assert identity_class[(antnum, antpol)] == 'bad'
        for antnum in [2, 4]:
            assert identity_class[(antnum, antpol)] == 'suspect'


def test_antenna_identity_checker_undecidable():
    # antenna 4's visibilities are pure noise: low coherence, no decisive identity
    data, model, bls = _build_identity_sim()
    rng = np.random.default_rng(1)
    for bl in bls:
        if 4 in bl[:2]:
            data[bl] = rng.normal(size=data[bl].shape) + 1j * rng.normal(size=data[bl].shape)
    groups = {antnum: list(range(8)) for antnum in range(8)}
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups)
    assert labeled_to_true == {}
    for antpol in ['Jee', 'Jnn']:
        assert identity_class[(4, antpol)] == 'bad'


def test_antenna_identity_checker_conflict():
    # antenna 5's visibilities carry antenna 6's signal, but 6's own visibilities are
    # healthy: the claim on identity 6 must be refused and 5 classified as bad
    data, model, bls = _build_identity_sim()
    rng = np.random.default_rng(2)
    freqs = np.linspace(100e6, 200e6, 600)
    for pol in ['ee', 'nn']:
        for j in range(8):
            if j in (5, 6):
                continue
            model_bl = (min(6, j), max(6, j), pol)
            mvis = model[model_bl] if j > 6 else np.conj(model[model_bl])
            vis = mvis * np.exp(2j * np.pi * freqs * 100e-9)[None, :]
            vis = vis + 0.02 * np.mean(np.abs(mvis)) * (rng.normal(size=vis.shape)
                                                        + 1j * rng.normal(size=vis.shape))
            data[(min(5, j), max(5, j), pol)] = (vis if j > 5 else np.conj(vis))
    groups = {antnum: list(range(8)) for antnum in range(8)}
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups)
    assert 5 not in labeled_to_true
    for antpol in ['Jee', 'Jnn']:
        assert identity_class[(5, antpol)] == 'bad'
        assert identity_class[(6, antpol)] == 'good'


def test_antenna_identity_checker_suspect_tier():
    # with the good bound raised above healthy scores, everything lands in the suspect
    # band (and scans of "suspicious" antennas find themselves as winners, so no relabels)
    data, model, bls = _build_identity_sim(nants=4)
    groups = {antnum: list(range(4)) for antnum in range(4)}
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.995, 1), suspect=(0.5, 1))
    assert labeled_to_true == {}
    assert len(identity_class.good_ants) == 0
    assert len(identity_class.bad_ants) == 0
    assert len(identity_class.suspect_ants) == 8


def test_antenna_identity_checker_single_pol_mislabel():
    # only the ee visibilities of antennas 2 and 3 are exchanged: the nn polarization
    # contradicts the proposed relabeling, so nothing is repaired and the ee entries go bad
    data, model, bls = _build_identity_sim()
    remap = {2: 3, 3: 2}
    new_data = {}
    for bl in bls:
        vis = data[bl]
        if bl[2] == 'ee':
            i, j = remap.get(bl[0], bl[0]), remap.get(bl[1], bl[1])
            new_data[(i, j, 'ee') if i < j else (j, i, 'ee')] = (vis if i < j else np.conj(vis))
        else:
            new_data[bl] = vis
    data = DataContainer(new_data)
    groups = {antnum: list(range(8)) for antnum in range(8)}
    identity_class, labeled_to_true, _ = ant_class.antenna_identity_checker(
        data, model, bls, groups, good=(0.7, 1))
    assert labeled_to_true == {}
    for antnum in [2, 3]:
        assert identity_class[(antnum, 'Jee')] == 'bad'
        assert identity_class[(antnum, 'Jnn')] == 'good'


def test_antenna_identity_checker_unauditable():
    # antenna 9 appears in the data and bls but has no model anywhere: unauditable,
    # so its self-coherence is nan and it is left out of the classification entirely
    data, model, bls = _build_identity_sim(nants=4)
    rng = np.random.default_rng(3)
    extra = {}
    for pol in ['ee', 'nn']:
        for j in range(4):
            extra[(j, 9, pol)] = rng.normal(size=(1, 600)) + 1j * rng.normal(size=(1, 600))
    data = DataContainer({**{bl: data[bl] for bl in bls}, **extra})
    bls = bls + sorted(extra.keys())
    groups = {antnum: list(range(4)) + [9] for antnum in list(range(4)) + [9]}
    identity_class, labeled_to_true, self_coherence = ant_class.antenna_identity_checker(
        data, model, bls, groups)
    assert labeled_to_true == {}
    assert np.isnan(self_coherence[(9, 'Jee')])
    assert (9, 'Jee') not in identity_class.ants
