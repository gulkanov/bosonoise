"""Synthetic-data checks of the raw-data analyses of the public Ocelot dataset (examples/zenodo_*_ocelot.py)."""
import importlib.util
import pathlib

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_logical_correlator_and_tz_fit(tmp_path, monkeypatch):
    """Independent cat flips at p per cycle: <Z_L> decays as (1 - 2p)^(d n); cats named S3..S5 as in the
    second d=3 file; the all-|-alpha> initial state has Z_L = -1 for odd d."""
    z = _load("zenodo_bitflip_ocelot")
    rng = np.random.default_rng(7)
    d, p, shots = 3, 0.004, 3000
    ns = np.array([1, 5, 10, 20, 40, 80, 160])
    cols = {k: [] for k in ["S3_final_state", "S4_final_state", "S5_final_state", "logical_state", "nbar",
                            "S3_initial_state", "S4_initial_state", "S5_initial_state", "num_cycles", "shot"]}
    for init, bit in (("+Z", 0), ("-Z", 1)):
        for n in ns:
            q = 0.5 * (1 - (1 - 2 * p) ** n)
            flips = rng.random((d, shots)) < q
            final = (flips ^ bit).astype(np.int64)
            for i, c in enumerate(("S3", "S4", "S5")):
                cols[f"{c}_final_state"] += list(final[i])
                cols[f"{c}_initial_state"] += [init] * shots
            cols["logical_state"] += list(final.sum(axis=0) % 2)
            cols["nbar"] += [2.0] * shots
            cols["num_cycles"] += [int(n)] * shots
            cols["shot"] += list(range(shots))
    pq.write_table(pa.table(cols), tmp_path / "synthetic_d3")
    monkeypatch.setattr(z, "ZEN", tmp_path)
    data = z.logical_correlator("synthetic_d3", d)
    expected_T = -z.T_CYCLE_US / np.log((1 - 2 * p) ** d)
    for s in ("+Z", "-Z"):
        n, y, e, cnt = data[(2.0, s)]
        assert y[0] > 0.9  # the initial logical sign is removed for both initial states
        A, T, dT, chi2 = z.fit_tz(n, y, e)
        assert abs(T - expected_T) < 4 * dT and abs(A - 1) < 0.05


def test_erasure_conditioning(tmp_path):
    """E_even / E_odd recover the erasure probabilities after an even / odd true syndrome."""
    z = _load("zenodo_erasure_ocelot")
    rng = np.random.default_rng(3)
    shots, N, e_even, e_odd = 4000, 12, 0.05, 0.08
    true = rng.integers(0, 2, shots) * 2  # 0 even, 2 odd, fixed per shot
    s = np.repeat(true[:, None], N, axis=1).astype(np.uint8)
    er = rng.random((shots, N)) < np.where(true[:, None] == 2, e_odd, e_even)
    s[er] = 1
    cols = {f"A1_syndrome_{k}": s[:, k] for k in range(N)}
    cols.update({f"A2_syndrome_{k}": s[:, k] for k in range(N)})  # a copy: fully correlated erasures
    pq.write_table(pa.table(cols), tmp_path / f"num_cycles_{N}")
    acc, pair = z.erasure_stats(tmp_path, ("A1", "A2"))
    c = acc["A1"]
    # conditioning on an erased previous round is excluded from both classes
    assert abs(c["e_even"] / c["n_even"] - e_even) < 0.006 and abs(c["e_odd"] / c["n_odd"] - e_odd) < 0.006
    q = pair[("A1", "A2")]
    assert q["eab"] == q["ea"] == q["eb"]


def test_phase_flip_detection_events_match_stim(monkeypatch):
    """The raw-data -> detector mapping of zenodo_phase_decode_ocelot reproduces Stim's own detectors and observable
    for records written in the Zenodo layout (random initial parities, syndromes 0 even / 2 odd)."""
    stim = pytest.importorskip("stim")
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    z = _load("zenodo_phase_decode_ocelot")
    from bosonoise.dem.repetition import PhaseFlipNoise, phase_flip_circuit

    d, N, shots = 4, 6, 3000
    noise = PhaseFlipNoise(p_z=0.05, p_meas=0.03, p_erase=1e-9, f_mid=0.3, p_init=0.02, p_final=0.02)
    circ = phase_flip_circuit(d, N, noise)
    rng = np.random.default_rng(11)
    # random initial parities: flip the prepared |+> cats with X_L-preserving frames by relabelling records
    meas = circ.compile_sampler(seed=5).sample(shots)
    det_ref, obs_ref = circ.compile_m2d_converter().convert(measurements=meas, separate_observables=True)
    init = meas[:, :d]
    k = d
    cols = {}
    for t in range(N):
        k += d - 1  # herald records (never fire)
        for j in range(d - 1):
            cols[f"A{j + 1}_syndrome_{t}"] = np.where(meas[:, k + j], 2, 0).astype(np.uint8)
        k += d - 1
    fin = meas[:, k:k + d]
    frame = rng.integers(0, 2, size=(shots, d)).astype(bool)  # the data's random initial product state
    for i in range(d):
        cols[f"S{i + 1}_initial_state"] = (init[:, i] ^ frame[:, i]).astype(np.uint8)
        cols[f"S{i + 1}_final_state"] = (fin[:, i] ^ frame[:, i]).astype(np.uint8)
    for t in range(N):
        for j in range(d - 1):
            cols[f"A{j + 1}_syndrome_{t}"] = np.where((cols[f"A{j + 1}_syndrome_{t}"] == 2) ^ frame[:, j] ^ frame[:, j + 1], 2, 0).astype(np.uint8)
    table = pa.table(cols)
    dets, obs, nh = z.detection_events(table, [f"S{i + 1}" for i in range(d)], [f"A{j + 1}" for j in range(d - 1)], N)
    assert nh == 0
    assert np.array_equal(dets, det_ref) and np.array_equal(obs[:, 0], obs_ref[:, 0])
