"""Stim repetition cat code (dem/repetition.py)."""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from bosonoise.dem.repetition import (BitFlipNoise, PhaseFlipNoise, bit_flip_circuit, bit_flip_logical_per_round,
                                      decode, herald_detectors, logical_failures, phase_flip_circuit)


def quiet(d, **kw):
    base = dict(p_z=0.0, p_meas=0.0, p_erase=0.0)
    base.update(kw)
    return PhaseFlipNoise(**base)


def test_noiseless_memory_has_no_detections_and_counts_detectors():
    d, r = 5, 4
    c = phase_flip_circuit(d, r, quiet(d, p_erase=1e-9))
    assert c.num_detectors == (d - 1) * (r + 1) + (d - 1) * r  # syndrome changes + heralds
    det, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert not det.any() and not obs.any()
    assert herald_detectors(c).sum() == (d - 1) * r


def test_single_phase_flip_patterns():
    d, r = 4, 3
    for f_mid in (0.0, 1.0):
        c = phase_flip_circuit(d, r, quiet(d, p_z=[0, 0.3, 0, 0], f_mid=f_mid))
        dem = c.detector_error_model()
        n_err = 0
        # every error mechanism flips two syndrome detectors (a cat in the bulk) of stabilizers 0 and 1
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            n_err += 1
            dets = sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
            coords = [c.get_detector_coordinates()[k] for k in dets]
            assert len(dets) == 2 and {x[0] for x in coords} == {0.5, 1.5}
            if f_mid == 0.0:  # between rounds: both neighbours see it in the same round
                assert coords[0][1] == coords[1][1]
            else:  # between the cat's two CX gates: one neighbour this round, the other in the next
                assert abs(coords[0][1] - coords[1][1]) == 1
        assert n_err == (r + 1 if f_mid == 0.0 else r)


def test_distance_helps_below_threshold_and_heralds_help():
    shots = 20000
    f3 = logical_failures(phase_flip_circuit(3, 6, PhaseFlipNoise(p_z=0.02, p_meas=0.01, p_erase=0.0)), shots, seed=2)[0]
    f5 = logical_failures(phase_flip_circuit(5, 6, PhaseFlipNoise(p_z=0.02, p_meas=0.01, p_erase=0.0)), shots, seed=3)[0]
    assert f5 < f3
    c = phase_flip_circuit(5, 6, PhaseFlipNoise(p_z=0.03, p_meas=0.005, p_erase=0.15))
    with_h = logical_failures(c, shots, use_heralds=True, seed=4)[0]
    without = logical_failures(c, shots, use_heralds=False, seed=4)[0]
    assert with_h < without


def test_bit_flip_logical_is_the_odd_parity_probability():
    d = 4
    noise = BitFlipNoise(p_x_idle=[0.01, 0.02, 0.0, 0.03], p_x_stab=[[0.01, 0.02, 0.05], [0.0, 0.0, 0.1], [0.02, 0.0, 0.0]])
    c = bit_flip_circuit(d, 1, noise)
    _, obs = c.compile_detector_sampler(seed=5).sample(400000, separate_observables=True)
    p = bit_flip_logical_per_round(noise, d)
    assert abs(obs.mean() - p) < 4 * np.sqrt(p * (1 - p) / 400000)
    # correlated pairs on one ancilla's two cats do not flip Z_L
    only_pairs = BitFlipNoise(p_x_idle=0.0, p_x_stab=[[0, 0, 0.3]] * (d - 1))
    assert bit_flip_logical_per_round(only_pairs, d) == 0.0
    det, obs = bit_flip_circuit(d, 3, only_pairs).compile_detector_sampler(seed=6).sample(2000, separate_observables=True)
    assert not obs.any() and not det.any()
