"""Stim circuits of the repetition cat code (Ocelot layout) and their decoding.

Cats (data) are qubits in the cat computational basis |0> = |alpha>, |1> = |-alpha>. A bit flip X swaps the wells
(exponentially suppressed); a phase flip Z is single-photon loss (Z = a/alpha on the code space). The stabilizers
X_j X_{j+1} (products of photon-number parities) are measured by ancilla j: prepare |+> ((g + f)/sqrt2), CX(j -> cat j)
in layer 1, CX(j -> cat j+1) in layer 2, measure in the X basis (Ocelot App. G5; which cat comes first is a parameter).

The two logical failure modes decouple:
  - X basis memory (logical phase flip, X_L = X_1): cat Z errors, syndrome measurement errors and heralded syndrome
    erasures (ancilla read in e). Decoded with PyMatching; the heralds are detectors of the DEM, so correlated
    matching (enable_correlations) turns a herald into an erased (weight ~0) time-like edge.
  - Z basis memory (logical bit flip, Z_L = Z_1 ... Z_d): cat X errors, independent and correlated X_a X_b pairs from
    one ancilla. Z_L commutes with the X-type stabilizers, so nothing is decoded: the logical error is the probability
    of an odd number of X errors (the circuit is kept for the DEM/dataset format and as a check).

Phase-flip noise per round: Z on each cat in two windows, between its two CX gates (mid, which the neighbouring
stabilizers see in different rounds: space-time edges) and the rest of the cycle; a heralded erasure (outcome
randomized) and a measurement flip on each ancilla; parity readout errors of the initial and final cat parities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _arr(x, n):
    a = np.broadcast_to(np.asarray(x, dtype=float), (n,)).copy()
    if np.any(a < 0) or np.any(a > 0.5 + 1e-12):
        raise ValueError(f"probabilities must lie in [0, 0.5]: {a}")
    return a


@dataclass
class PhaseFlipNoise:
    """Per-round noise of the X basis memory. Arrays: p_z (d,), p_meas and p_erase (d - 1,)."""

    p_z: np.ndarray                 # phase-flip probability of each cat per round
    p_meas: np.ndarray              # non-heralded syndrome flip per ancilla per round
    p_erase: np.ndarray             # heralded erasure per ancilla per round (outcome randomized)
    f_mid: float = 0.15             # share of p_z between the cat's two CX gates
    p_init: float | np.ndarray = 0.0   # initial parity readout error per cat
    p_final: float | np.ndarray = 0.0  # final parity readout error per cat
    left_first: bool = True         # layer 1 couples ancilla j to cat j (else to cat j + 1)


@dataclass
class BitFlipNoise:
    """Per-round noise of the Z basis memory: idle X per cat (d,), and per ancilla (d - 1, 3) the probabilities of
    (X on its first cat only, X on its second cat only, X on both) from one stabilizer measurement."""

    p_x_idle: np.ndarray
    p_x_stab: np.ndarray
    p_init: float | np.ndarray = 0.0
    p_final: float | np.ndarray = 0.0
    left_first: bool = True


def _pairs(d, left_first):
    """(first, second) cat of each ancilla in the CX order."""
    return [((j, j + 1) if left_first else (j + 1, j)) for j in range(d - 1)]


def phase_flip_circuit(d: int, rounds: int, noise: PhaseFlipNoise):
    """X basis memory: cats 0 .. d-1, ancillas d .. 2d-2. Detectors: syndrome changes (round 0 against the measured
    initial parities, the last against the final parities) and one herald detector per ancilla and round.
    Observable 0: the parity change of cat 0 (X_L = X_1)."""
    import stim

    if d < 2 or rounds < 1:
        raise ValueError("need d >= 2 and rounds >= 1")
    pz = _arr(noise.p_z, d)
    pm, pe = _arr(noise.p_meas, d - 1), _arr(noise.p_erase, d - 1)
    pi, pf = _arr(noise.p_init, d), _arr(noise.p_final, d)
    cats, anc = list(range(d)), list(range(d, 2 * d - 1))
    pairs = _pairs(d, noise.left_first)
    # per-window Z probabilities: mid share and the rest (split into equal halves before and after the round)
    pz_mid = pz * noise.f_mid
    pz_rest = 0.5 * (1 - np.sqrt(np.clip(1 - 2 * pz, 0, None) / np.clip(1 - 2 * pz_mid, 1e-300, None)))
    c = stim.Circuit()
    c.append("RX", cats)
    for i in cats:
        c.append("MX", [i], pi[i])
    # measurement record bookkeeping: rec index of each measurement
    n_meas = d
    init_rec = list(range(d))
    prev = None
    for t in range(rounds):
        for i in cats:
            if pz_rest[i] > 0:
                c.append("Z_ERROR", [i], pz_rest[i] if t == 0 else 2 * pz_rest[i] * (1 - pz_rest[i]))
        c.append("RX", anc)
        c.append("CX", [x for j, (a, _) in enumerate(pairs) for x in (anc[j], a)])
        for i in cats:
            if pz_mid[i] > 0:
                c.append("Z_ERROR", [i], pz_mid[i])
        c.append("CX", [x for j, (_, b) in enumerate(pairs) for x in (anc[j], b)])
        herald_rec = []
        for j in range(d - 1):
            if pe[j] > 0:
                c.append("HERALDED_PAULI_CHANNEL_1", [anc[j]], [0.5 * pe[j], 0.0, 0.0, 0.5 * pe[j]])
                herald_rec.append(n_meas)
                n_meas += 1
            else:
                herald_rec.append(None)
        meas_rec = []
        for j in range(d - 1):
            c.append("MX", [anc[j]], pm[j])
            meas_rec.append(n_meas)
            n_meas += 1
        for j in range(d - 1):
            a, b = pairs[j]
            if prev is None:
                recs = [meas_rec[j], init_rec[a], init_rec[b]]
            else:
                recs = [meas_rec[j], prev[j]]
            c.append("DETECTOR", [stim.target_rec(r - n_meas) for r in recs], [j + 0.5, t, 0])
        for j in range(d - 1):
            if herald_rec[j] is not None:
                c.append("DETECTOR", [stim.target_rec(herald_rec[j] - n_meas)], [j + 0.5, t, 1])
        prev = meas_rec
    for i in cats:
        if pz_rest[i] > 0:
            c.append("Z_ERROR", [i], pz_rest[i])
    final_rec = []
    for i in cats:
        c.append("MX", [i], pf[i])
        final_rec.append(n_meas)
        n_meas += 1
    for j in range(d - 1):
        a, b = pairs[j]
        recs = [prev[j], final_rec[a], final_rec[b]]
        c.append("DETECTOR", [stim.target_rec(r - n_meas) for r in recs], [j + 0.5, rounds, 0])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(final_rec[0] - n_meas), stim.target_rec(init_rec[0] - n_meas)], 0)
    return c


def bit_flip_circuit(d: int, rounds: int, noise: BitFlipNoise):
    """Z basis memory: all cats start in |alpha>; X errors per round (idle, then the stabilizer's joint distribution
    after its two CX gates); syndromes are random at round 0, so detectors compare consecutive rounds only.
    Observable 0: Z_L = Z_1 ... Z_d (final Z readout of all cats)."""
    import stim

    px = _arr(noise.p_x_idle, d)
    ps = np.asarray(noise.p_x_stab, dtype=float).reshape(d - 1, 3)
    pf = _arr(noise.p_final, d)
    cats, anc = list(range(d)), list(range(d, 2 * d - 1))
    pairs = _pairs(d, noise.left_first)
    c = stim.Circuit()
    c.append("R", cats)
    n_meas, prev = 0, None
    for t in range(rounds):
        for i in cats:
            if px[i] > 0:
                c.append("X_ERROR", [i], px[i])
        c.append("RX", anc)
        c.append("CX", [x for j, (a, _) in enumerate(pairs) for x in (anc[j], a)])
        c.append("CX", [x for j, (_, b) in enumerate(pairs) for x in (anc[j], b)])
        for j, (a, b) in enumerate(pairs):
            p_a, p_b, p_ab = ps[j]
            if p_ab + p_a + p_b <= 0:
                continue
            c.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], p_ab)
            rest = 1 - p_ab
            c.append("ELSE_CORRELATED_ERROR", [stim.target_x(a)], min(p_a / rest, 1.0) if rest > 0 else 0.0)
            rest -= p_a
            c.append("ELSE_CORRELATED_ERROR", [stim.target_x(b)], min(p_b / rest, 1.0) if rest > 0 else 0.0)
        meas_rec = []
        for j in range(d - 1):
            c.append("MX", [anc[j]])
            meas_rec.append(n_meas)
            n_meas += 1
        if prev is not None:
            for j in range(d - 1):
                c.append("DETECTOR", [stim.target_rec(meas_rec[j] - n_meas), stim.target_rec(prev[j] - n_meas)],
                         [j + 0.5, t, 0])
        prev = meas_rec
    final_rec = []
    for i in cats:
        c.append("M", [i], pf[i])
        final_rec.append(n_meas)
        n_meas += 1
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - n_meas) for r in final_rec], 0)
    return c


def bit_flip_logical_per_round(noise: BitFlipNoise, d: int) -> float:
    """Exact P(odd number of X flips) per round: prod over independent mechanisms of (1 - 2 p_odd)."""
    px = _arr(noise.p_x_idle, d)
    ps = np.asarray(noise.p_x_stab, dtype=float).reshape(d - 1, 3)
    prod = np.prod(1 - 2 * px) * np.prod(1 - 2 * (ps[:, 0] + ps[:, 1]))
    return float(0.5 * (1 - prod))


def matching_for(circuit, use_heralds: bool = True):
    """PyMatching decoder from the circuit's DEM (heralds as detectors; correlated matching uses them)."""
    import pymatching

    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    return pymatching.Matching.from_detector_error_model(dem, enable_correlations=use_heralds)


def herald_detectors(circuit) -> np.ndarray:
    """Boolean mask of the herald detectors (coordinate 2 == 1)."""
    coords = circuit.get_detector_coordinates()
    return np.array([len(coords[k]) > 2 and coords[k][2] == 1 for k in range(circuit.num_detectors)])


def decode(circuit, dets: np.ndarray, *, use_heralds: bool = True, matching=None) -> np.ndarray:
    """Predicted observable flips. use_heralds=False: herald bits are zeroed, so erasures only enter through the
    marginal (averaged) edge weights -- decoding 'without erasure information'."""
    m = matching or matching_for(circuit, use_heralds)
    shots = np.asarray(dets, dtype=np.uint8)
    if not use_heralds:
        shots = shots.copy()
        shots[:, herald_detectors(circuit)] = 0
    return m.decode_batch(shots, enable_correlations=use_heralds)


def logical_failures(circuit, shots: int, *, use_heralds: bool = True, seed: int | None = None, batch: int = 100_000):
    """(failures, shots) of the X basis memory by sampling and decoding."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    m = matching_for(circuit, use_heralds)
    fails, done = 0, 0
    while done < shots:
        n = min(batch, shots - done)
        det, obs = sampler.sample(n, separate_observables=True)
        pred = decode(circuit, det, use_heralds=use_heralds, matching=m)
        fails += int(np.sum(np.any(pred != obs, axis=1)))
        done += n
    return fails, done
