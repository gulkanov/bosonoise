"""Reference noise models of repetition cat codes: circuits and the leakage sampler (dem/noise_models.py)."""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from bosonoise.dem.noise_models import Leakage, RoundNoise, decode_memory, memory_circuit, sample_with_leakage


def test_noiseless_memories_are_silent():
    noise = RoundNoise(p_data=[0, 0, 0])
    for basis in ("X", "Z"):
        c = memory_circuit(5, 4, noise, basis)
        det, obs = c.compile_detector_sampler(seed=1).sample(300, separate_observables=True)
        assert not det.any() and not obs.any()
        out = sample_with_leakage(5, 4, noise, 300, basis=basis, seed=2)
        assert not out["detectors"].any() and not out["observables"].any() and not out["leakage"].any()


def test_leakage_sampler_without_leakage_matches_stim():
    noise = RoundNoise(p_data=[0.004, 0.0, 0.03], p_stab_x=[0.002, 0.003, 0.004], cx_channel={"ZI": 0.01, "IZ": 0.005,
                       "ZZ": 0.005}, p_meas=0.02, p_herald=0.05, p_prep=0.005, p_data_meas=0.01)
    d, r, n = 5, 6, 40000
    c = memory_circuit(d, r, noise, "X")
    det_ref, obs_ref = c.compile_detector_sampler(seed=3).sample(n, separate_observables=True)
    out = sample_with_leakage(d, r, noise, n, basis="X", seed=4)
    rate_ref, rate = det_ref.mean(axis=0), out["detectors"].mean(axis=0)
    assert np.all(np.abs(rate - rate_ref) < 5 * np.sqrt(rate_ref * (1 - rate_ref) / n) + 1e-3)
    f_ref = np.mean(decode_memory(c, det_ref)[:, 0] != obs_ref[:, 0])
    f = np.mean(decode_memory(c, out["detectors"])[:, 0] != out["observables"][:, 0])
    assert abs(f - f_ref) < 5 * np.sqrt(f_ref * (1 - f_ref) / n) + 1e-3


def test_ancilla_leakage_pairs_and_flags():
    """An ancilla stuck in e flips both of its cats (chi-matching): harmless for Z_L; heralded; stationary fraction."""
    leak = Leakage(p_leak=0.02, p_return=0.5, pair={"XX": 1.0}, heralded=True)
    noise = RoundNoise(p_data=[0, 0, 0], leak_anc=leak)
    d, r, n = 4, 30, 4000
    out = sample_with_leakage(d, r, noise, n, basis="Z", seed=5)
    assert not out["observables"].any()  # X X pairs never flip Z_1 ... Z_d
    lk = out["leakage"].reshape(n, r, 2 * d - 1)
    frac = lk[:, 10:, d:].mean()  # after the transient
    assert abs(frac - 0.02 / (0.02 + 0.5 * (1 - 0.02))) < 0.01
    her = out["heralds"].reshape(n, r, d - 1)
    assert np.all(her[lk[:, :, d:]])  # every leaked ancilla is heralded
    assert not lk[:, :, :d].any()


def test_data_leakage_randomizes_neighbouring_syndromes():
    leak = Leakage(p_leak=0.01, p_return=0.2, partner={"Z": 0.5}, dephase_self=True, randomize_measurement=False)
    noise = RoundNoise(p_data=[0, 0, 0], leak_data=leak)
    d, r, n = 5, 12, 5000
    out = sample_with_leakage(d, r, noise, n, basis="X", seed=6)
    assert out["detectors"].mean() > 0.005 and out["leakage"][:, :].reshape(n, r, 2 * d - 1)[:, :, d:].sum() == 0


def test_ocelot_model_parameters_and_bit_flip_rate():
    from bosonoise.dem.models import ocelot_model

    lo, hi = ocelot_model(1.5, 5), ocelot_model(6.0, 5)
    assert np.all(hi.p_data[:, 2] > lo.p_data[:, 2])       # phase flips grow with nbar
    assert np.all(hi.p_data[:, 0] < lo.p_data[:, 0])       # idle bit flips fall
    assert set(lo.provenance) >= {"p_data[X]", "p_data[Z]", "p_stab_x", "p_meas", "p_herald", "leak_anc"}
    zf = ocelot_model(4.0, 5, variant="zero-fit")
    assert zf.p_stab_x[:, :2].sum() < ocelot_model(4.0, 5).p_stab_x[:, :2].sum()   # lambda = 3 above zero-fit
    # Z-basis memory, one round: odd-parity probability of the model's X errors (pairs harmless)
    m = ocelot_model(3.0, 4, ancilla_leakage=False)
    m.p_data_meas = 0.0
    prod = np.prod(1 - 2 * m.p_data[:, 0]) * np.prod(1 - 2 * (m.p_stab_x[:, 0] + m.p_stab_x[:, 1]))
    n = 300000
    out = sample_with_leakage(4, 1, m, n, basis="Z", seed=9)
    p = 0.5 * (1 - prod)
    assert abs(out["observables"].mean() - p) < 5 * np.sqrt(p / n)


def test_heralded_bit_flips_are_corrected_in_the_z_memory():
    """Heralded cat bit flips (buffer-emission heralds) become herald -> observable edges of the DEM: decodable."""
    noise = RoundNoise(p_data=[0.0, 0, 0], p_data_herald_x=0.02)
    d, r, n = 5, 5, 20000
    out = sample_with_leakage(d, r, noise, n, basis="Z", seed=12)
    raw = out["observables"][:, 0].mean()
    assert raw > 0.3  # many unprotected flips
    pred = decode_memory(out["circuit"], out["detectors"], use_heralds=True)
    assert np.mean(pred[:, 0] != out["observables"][:, 0]) == 0.0
    assert out["heralds"].shape[1] == r * (d - 1 + d)


def test_allcat_model_reproduces_the_phenomenological_rates_of_le_regent():
    """Per cycle: data phase flip 5 nbar eta and ancilla (measurement) error 0.318/nbar + 6 nbar eta (LR23, p. 8)."""
    from bosonoise.dem.models import allcat_model

    nb, eta = 8.0, 1e-3
    m = allcat_model(nb, 5, eta=eta)
    cx = m.cx_channel
    data = m.p_data[1, 2] + 2 * (cx["IZ"] + cx["ZZ"])
    edge = m.p_data[0, 2] + (cx["IZ"] + cx["ZZ"])            # a boundary cat idles during the other CNOT layer
    anc = 2 * (cx["ZI"] + cx["ZZ"]) + m.p_prep + m.p_meas
    assert abs(data - 5 * nb * eta) < 1e-12 and abs(edge - 5 * nb * eta) < 1e-12
    assert abs(anc - (0.318 / nb + 6 * nb * eta)) < 1e-9
    assert abs(sum(v for k, v in cx.items() if "X" in k or "Y" in k) - 0.5 * np.exp(-2 * nb)) < 1e-15


def test_kerrcat_model_matches_darmawan_and_leakage_preserves_the_bias():
    from bosonoise.dem.models import kerrcat_model

    for kK, infid_da21, bias_da21, prep, meas in [(1.67e-4, 0.047, 319, 0.003, 0.008), (4.16e-4, 0.104, 375, 0.007, 0.012)]:
        m = kerrcat_model(6.25, 5, kappa_over_K=kK)           # DA21 Table II
        cx = m.cx_channel
        infid = sum(cx.values())
        z = cx["ZI"] + cx["IZ"] + cx["ZZ"]
        assert abs(infid - infid_da21) < 1e-9 and abs(z / (infid - z) - bias_da21) < 1e-6
        assert abs(m.p_prep - prep) < 1e-9 and abs(m.p_meas - meas) < 1e-9
    m1 = kerrcat_model(1.0, 5)                                # DA21 Table I, kappa/K = 1.67e-4
    cx = m1.cx_channel
    z = cx["ZI"] + cx["IZ"] + cx["ZZ"]
    assert abs(sum(cx.values()) - 0.046) < 1e-9 and abs(z / (sum(cx.values()) - z) - 16.8) < 1e-6
    # the loss-and-gain estimate 2 kappa (1 + 2 n_th) nbar T_CX is within 5% of the DA21 CX Z part at nbar = 6.25
    assert abs(2 * 1.67e-4 * 1.16 * 115 / (0.047 * 319 / 320) - 1) < 0.05
    # leakage alone (no other noise) never flips Z_L: return and readout of a leaked Kerr cat are phase-only
    quiet = kerrcat_model(6.25, 5, lru=False)
    quiet.cx_channel, quiet.p_data = {}, np.zeros((5, 3))
    quiet.p_meas = quiet.p_prep = quiet.p_data_meas = quiet.p_data_meas_z = 0.0
    quiet.leak_data.p_leak = 0.05
    out = sample_with_leakage(5, 10, quiet, 3000, basis="Z", seed=21)
    assert out["leakage"].any() and not out["observables"].any()
