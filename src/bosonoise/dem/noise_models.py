"""Reference circuit-level noise models of repetition cat codes and their syndrome sampling with leakage.

A model produces a RoundNoise: every noise location of one round of the repetition cat code (d data cats, d - 1
ancillas; stabilizers X_j X_{j+1} measured with CX(ancilla j -> cat j) in layer 1 and CX(ancilla j -> cat j+1) in layer
2, as in Ocelot), each parameter with its provenance (our calculation, a dataset, or a literature reference):
  data cats        Pauli (X, Y, Z) per round, split between the window between a cat's two CX gates (share f_mid,
                   space-time edges) and the rest of the round;
  CX gates         a two-qubit Pauli channel on (ancilla, cat) after each gate, as {"XI": p, "IZ": p, "ZZ": p, ...}
                   (control first);
  stabilizer       correlated X errors on the two cats of one stabilizer, (X on the first cat only, on the second only,
                   on both), applied after the second CX (our ULE fault tables for the hybrid cell);
  ancilla          measurement flip, heralded erasure (outcome randomized; the herald is recorded), preparation flip;
  data heralds     heralded bit flips of the cats (a buffer-emission burst flags the flip; Ferrari et al.,
                   arXiv:2607.01375), recorded as herald detectors of type 2;
  leakage          per qubit type: probability to leak per round, to return per round, and what a leaked qubit does
                   (incoherent approximation, Marshall & Kafri, PR Applied 23, 054025 (2025)): Pauli errors on its CX
                   partners, a randomized own measurement, an optional herald; leakage flags are recorded.
Circuits: memory_circuit(d, rounds, noise, basis) gives the full Stim circuit without leakage (for the DEM and
decoding); sample_with_leakage() runs the same rounds in stim.FlipSimulator and injects the leakage effects between the
Stim segments, returning detectors, observables, heralds and leakage flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Leakage:
    """Leakage of one qubit type, per round. While leaked (incoherent approximation): each CX partner receives the
    Pauli channel `partner` ({'X': p, 'Z': p, ...}), the leaked qubit's own measurement is randomized if
    `randomize_measurement` (ancillas) or its Pauli frame is depolarized in Z if `dephase_self` (data), and the herald
    of the ancilla fires if `heralded`; `pair` is a correlated channel on the two cats of a leaked ancilla (e.g. an
    Ocelot ancilla stuck in e flips both cats through chi-matching: {"XX": 1}). On return the qubit is
    re-randomized (uniform Pauli)."""

    p_leak: float = 0.0
    p_return: float = 1.0
    partner: dict = field(default_factory=dict)     # independent single-qubit channel on each CX partner
    pair: dict = field(default_factory=dict)        # ancilla leakage: correlated channel on its two cats, {"XX": p}
    randomize_measurement: bool = True
    dephase_self: bool = False
    heralded: bool = False
    return_paulis: str = "XZ"  # re-randomization on return: "XZ" (maximally mixed) or "Z" (phase only, e.g. a Kerr
                               # cat returning by two-photon loss keeps its well: Puri et al., Sci. Adv. 6, eaay5901)


@dataclass
class RoundNoise:
    """All noise of one round of the repetition cat code (arrays broadcast over cats (d) and ancillas (d - 1))."""

    p_data: np.ndarray                      # (d, 3) per round: X, Y, Z of each cat
    f_mid: float = 0.2                      # share of p_data between a cat's two CX gates
    p_stab_x: np.ndarray | None = None      # (d - 1, 3): X first only, X second only, X both
    cx_channel: dict = field(default_factory=dict)   # two-qubit Pauli channel after each CX, control = ancilla
    p_meas: np.ndarray | float = 0.0        # (d - 1,) syndrome flip
    p_herald: np.ndarray | float = 0.0      # (d - 1,) heralded erasure of the syndrome
    p_prep: np.ndarray | float = 0.0        # (d - 1,) ancilla preparation flip (flips the outcome)
    p_data_meas: float = 0.0                # initial/final data readout flip (parity, X basis)
    p_data_meas_z: float | None = None      # the same for the well (Z-basis) readout; None: as p_data_meas
    p_data_herald_x: np.ndarray | float = 0.0  # (d,) heralded bit flip of each cat per round (buffer-emission herald)
    leak_data: Leakage = field(default_factory=Leakage)
    leak_anc: Leakage = field(default_factory=Leakage)
    provenance: dict = field(default_factory=dict)
    name: str = ""

    def broadcast(self, d):
        pd = np.broadcast_to(np.asarray(self.p_data, float), (d, 3)).copy()
        ps = (np.zeros((d - 1, 3)) if self.p_stab_x is None
              else np.broadcast_to(np.asarray(self.p_stab_x, float), (d - 1, 3)).copy())
        vec = lambda x: np.broadcast_to(np.asarray(x, float), (d - 1,)).copy()
        return pd, ps, vec(self.p_meas), vec(self.p_herald), vec(self.p_prep)


def _pauli_split(p, f):
    """Per-window Pauli probabilities (X, Y, Z) such that the two windows compose to p (independent X and Z
    approximation for the split)."""
    return np.clip(np.asarray(p) * f, 0, 0.5), np.clip(np.asarray(p) * (1 - f), 0, 0.5)


def _append_pauli1(c, q, pxyz):
    if np.sum(pxyz) > 0:
        c.append("PAULI_CHANNEL_1", [q], [float(x) for x in pxyz])


def _append_cx_channel(c, pairs, channel):
    if not channel:
        return
    order = ["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]
    probs = [float(channel.get(k, 0.0)) for k in order]
    if sum(probs) <= 0:
        return
    for ctrl, tgt in pairs:
        c.append("PAULI_CHANNEL_2", [ctrl, tgt], probs)


def _round_segments(d, noise: RoundNoise, basis: str, first: bool, t: int, prev_rec, n_meas):
    """One round as two Stim segments: pre (reset, gates, gate errors) and meas (heralds, ancilla readout,
    detectors). Returns (pre, meas, meas_rec, herald_rec, n_meas)."""
    import stim

    pd, ps, pm, ph, pp = noise.broadcast(d)
    cats, anc = list(range(d)), list(range(d, 2 * d - 1))
    pairs = [(j, j + 1) for j in range(d - 1)]
    mid, rest = _pauli_split(pd, noise.f_mid)
    phx = np.broadcast_to(np.asarray(noise.p_data_herald_x, float), (d,))
    c = stim.Circuit()
    for i in cats:
        _append_pauli1(c, i, rest[i])
    data_herald_rec = []
    for i in cats:
        if phx[i] > 0:
            c.append("HERALDED_PAULI_CHANNEL_1", [i], [0.0, float(phx[i]), 0.0, 0.0])
            data_herald_rec.append((i, n_meas))
            n_meas += 1
    c.append("RX", anc)
    for j in range(d - 1):
        if pp[j] > 0:
            c.append("Z_ERROR", [anc[j]], pp[j])
    c.append("CX", [x for j, (a, _) in enumerate(pairs) for x in (anc[j], a)])
    _append_cx_channel(c, [(anc[j], a) for j, (a, _) in enumerate(pairs)], noise.cx_channel)
    for i in cats:
        _append_pauli1(c, i, mid[i])
    c.append("CX", [x for j, (_, b) in enumerate(pairs) for x in (anc[j], b)])
    _append_cx_channel(c, [(anc[j], b) for j, (_, b) in enumerate(pairs)], noise.cx_channel)
    for j, (a, b) in enumerate(pairs):
        p_a, p_b, p_ab = ps[j]
        if p_ab + p_a + p_b > 0:
            c.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], p_ab)
            rest_p = 1 - p_ab
            c.append("ELSE_CORRELATED_ERROR", [stim.target_x(a)], min(p_a / rest_p, 1.0) if rest_p > 0 else 0.0)
            rest_p -= p_a
            c.append("ELSE_CORRELATED_ERROR", [stim.target_x(b)], min(p_b / rest_p, 1.0) if rest_p > 0 else 0.0)
    pre, c = c, stim.Circuit()
    herald_rec = []
    for j in range(d - 1):
        c.append("HERALDED_PAULI_CHANNEL_1", [anc[j]], [0.5 * ph[j], 0.0, 0.0, 0.5 * ph[j]])
        herald_rec.append(n_meas)
        n_meas += 1
    meas_rec = []
    for j in range(d - 1):
        c.append("MX", [anc[j]], pm[j])
        meas_rec.append(n_meas)
        n_meas += 1
    for j in range(d - 1):
        a, b = pairs[j]
        if prev_rec is None:
            if basis == "X":
                recs = [meas_rec[j], a, b]  # against the initial data readout (records 0 .. d-1)
            else:
                recs = None                 # random at round 0 in the Z basis
        else:
            recs = [meas_rec[j], prev_rec[j]]
        if recs is not None:
            c.append("DETECTOR", [stim.target_rec(r - n_meas) for r in recs], [j + 0.5, t, 0])
    for j in range(d - 1):
        c.append("DETECTOR", [stim.target_rec(herald_rec[j] - n_meas)], [j + 0.5, t, 1])
    for i, r in data_herald_rec:
        c.append("DETECTOR", [stim.target_rec(r - n_meas)], [i, t, 2])
    return pre, c, meas_rec, herald_rec, n_meas


def memory_segments(d: int, rounds: int, noise: RoundNoise, basis: str = "X"):
    """[preparation, (pre, meas) of round 0, ..., of round R-1, final] Stim segments of a memory experiment;
    concatenated they form the full circuit. Basis 'X': logical phase flip (observable: parity change of cat 0); basis 'Z': logical bit flip
    (observable: Z_1 ... Z_d, all cats start in |alpha>)."""
    import stim

    if basis not in ("X", "Z"):
        raise ValueError("basis must be 'X' or 'Z'")
    cats = list(range(d))
    pd, ps, pm, ph, pp = noise.broadcast(d)
    segs = []
    prep = stim.Circuit()
    prep.append("RX" if basis == "X" else "R", cats)
    pm_data = noise.p_data_meas if basis == "X" or noise.p_data_meas_z is None else noise.p_data_meas_z
    for i in cats:  # initial readout (parity in X, well in Z)
        prep.append("MX" if basis == "X" else "M", [i], pm_data)
    segs.append(prep)
    n_meas, prev = d, None
    for t in range(rounds):
        pre, meas, meas_rec, _, n_meas = _round_segments(d, noise, basis, prev is None, t, prev, n_meas)
        segs.append((pre, meas))
        prev = meas_rec
    fin = stim.Circuit()  # the round's idle window precedes its gates, so each round carries exactly p_data
    final_rec = []
    for i in cats:
        fin.append("MX" if basis == "X" else "M", [i], pm_data)
        final_rec.append(n_meas)
        n_meas += 1
    if basis == "X":
        for j in range(d - 1):
            recs = [prev[j], final_rec[j], final_rec[j + 1]]
            fin.append("DETECTOR", [stim.target_rec(r - n_meas) for r in recs], [j + 0.5, rounds, 0])
        fin.append("OBSERVABLE_INCLUDE", [stim.target_rec(final_rec[0] - n_meas), stim.target_rec(0 - n_meas)], 0)
    else:
        fin.append("OBSERVABLE_INCLUDE", [stim.target_rec(r - n_meas) for r in final_rec]
                   + [stim.target_rec(r - n_meas) for r in range(d)], 0)
    segs.append(fin)
    return segs


def memory_circuit(d: int, rounds: int, noise: RoundNoise, basis: str = "X"):
    import stim

    c = stim.Circuit()
    for seg in memory_segments(d, rounds, noise, basis):
        for part in (seg if isinstance(seg, tuple) else (seg,)):
            c += part
    return c


def herald_mask(circuit, kinds=(1, 2)) -> np.ndarray:
    """Herald detectors: kind 1 = syndrome erasure (ancilla), kind 2 = heralded bit flip of a cat."""
    coords = circuit.get_detector_coordinates()
    return np.array([len(coords[k]) > 2 and coords[k][2] in kinds for k in range(circuit.num_detectors)])


def _apply_channel(sim, rng, qubits_mask, channel):
    """Apply a single-qubit Pauli channel {'X': p, 'Y': p, 'Z': p} to the masked (qubit, shot) entries."""
    if not channel or not qubits_mask.any():
        return
    u = rng.random(qubits_mask.shape)
    acc = 0.0
    for pauli in ("X", "Y", "Z"):
        p = channel.get(pauli, 0.0)
        if p <= 0:
            continue
        sel = qubits_mask & (u >= acc) & (u < acc + p)
        if sel.any():
            sim.broadcast_pauli_errors(pauli=pauli, mask=sel)
        acc += p


def _pair_channel(sim, rng, anc_mask, d, pair):
    """Correlated two-qubit channel {"XX": p, "ZZ": p, ...} on the two cats of each masked ancilla (anc_mask: (d-1, S))."""
    if not pair or not anc_mask.any():
        return
    u = rng.random(anc_mask.shape)
    acc = 0.0
    for key, p in pair.items():
        sel = anc_mask & (u >= acc) & (u < acc + p)
        acc += p
        if not sel.any():
            continue
        for pos in (0, 1):
            if key[pos] == "I":
                continue
            m = np.zeros((2 * d - 1, anc_mask.shape[1]), dtype=bool)
            for j in range(d - 1):
                m[j + pos] |= sel[j]
            sim.broadcast_pauli_errors(pauli=key[pos], mask=m)


def sample_with_leakage(d: int, rounds: int, noise: RoundNoise, shots: int, basis: str = "X", seed: int | None = None):
    """Sample the memory with leakage in the incoherent approximation (stim.FlipSimulator between segments).

    Order per round: leakage transitions and the effects on cats (partners of leaked ancillas, leaked cats dephased);
    the gates ('pre'); effects on ancillas (a leaked ancilla's own outcome randomized; the ancillas next to a leaked cat
    get the data 'partner' channel); the readout ('meas'). Heralds of leaked ancillas (if heralded) are OR-ed into the
    herald detectors. Returns dict(detectors (S, D), observables (S, 1), heralds (S, H): per round the ancilla
    (syndrome-erasure) heralds, then the heralded bit flips of the cats if any, leakage (S, (2d-1) R) with the qubit
    order cats then ancillas within each round, circuit)."""
    import stim

    rng = np.random.default_rng(seed)
    segs = memory_segments(d, rounds, noise, basis)
    nq = 2 * d - 1
    sim = stim.FlipSimulator(batch_size=shots, num_qubits=nq, disable_stabilizer_randomization=True,
                             seed=None if seed is None else int(seed) % (2 ** 31))
    leaked = np.zeros((nq, shots), dtype=bool)
    is_anc = np.zeros(nq, dtype=bool)
    is_anc[d:] = True
    ld, la = noise.leak_data, noise.leak_anc
    p_leak = np.where(is_anc, la.p_leak, ld.p_leak)[:, None]
    p_ret = np.where(is_anc, la.p_return, ld.p_return)[:, None]
    flags = []
    sim.do(segs[0])
    for t in range(rounds):
        u = rng.random((nq, shots))
        returning = leaked & (u < p_ret)
        new = (~leaked) & (u < p_leak)
        for pauli in ("X", "Z"):  # re-randomized on return (per qubit type)
            allowed = np.where(is_anc, pauli in la.return_paulis, pauli in ld.return_paulis)[:, None]
            m = returning & allowed & (rng.random((nq, shots)) < 0.5)
            if m.any():
                sim.broadcast_pauli_errors(pauli=pauli, mask=m)
        leaked = (leaked & ~returning) | new
        flags.append(leaked.copy())
        anc_leaked, cat_leaked = leaked[d:], leaked[:d]
        if anc_leaked.any():  # effects of leaked ancillas on their two cats
            cats_mask = np.zeros((nq, shots), dtype=bool)
            for j in range(d - 1):
                cats_mask[j] |= anc_leaked[j]
                cats_mask[j + 1] |= anc_leaked[j]
            _apply_channel(sim, rng, cats_mask, la.partner)
            _pair_channel(sim, rng, anc_leaked, d, la.pair)
        if ld.dephase_self and cat_leaked.any():
            m = np.zeros((nq, shots), dtype=bool)
            m[:d] = cat_leaked & (rng.random((d, shots)) < 0.5)
            sim.broadcast_pauli_errors(pauli="Z", mask=m)
        pre, meas = segs[1 + t]
        sim.do(pre)
        if la.randomize_measurement and anc_leaked.any():
            m = np.zeros((nq, shots), dtype=bool)
            m[d:] = anc_leaked & (rng.random(anc_leaked.shape) < 0.5)
            if m.any():
                sim.broadcast_pauli_errors(pauli="Z", mask=m)
        if cat_leaked.any() and ld.partner:
            near = np.zeros((nq, shots), dtype=bool)
            for j in range(d - 1):
                near[d + j] = cat_leaked[j] | cat_leaked[j + 1]
            _apply_channel(sim, rng, near, ld.partner)
        sim.do(meas)
    for pauli in ("X", "Z"):  # a cat still leaked at the end is read out at random (in the axes it loses, see Leakage)
        if pauli not in ld.return_paulis:
            continue
        m = leaked & (rng.random((nq, shots)) < 0.5)
        m[d:] = False
        if m.any():
            sim.broadcast_pauli_errors(pauli=pauli, mask=m)
    sim.do(segs[-1])
    det = sim.get_detector_flips().T.copy()
    obs = sim.get_observable_flips().T.copy()
    circ = memory_circuit(d, rounds, noise, basis)
    coords = circ.get_detector_coordinates()
    lookup = {(c[1], c[0]): k for k, c in coords.items() if len(c) > 2 and c[2] == 1}
    herald_idx = np.array([[lookup[(float(t), j + 0.5)] for j in range(d - 1)] for t in range(rounds)])
    if la.heralded:
        for t in range(rounds):
            det[:, herald_idx[t]] |= flags[t][d:].T
    lk = np.concatenate([f.T for f in flags], axis=1)
    all_h = sorted((c[1], c[2], c[0], k) for k, c in coords.items() if len(c) > 2 and c[2] in (1, 2))
    return dict(detectors=det, observables=obs, heralds=det[:, [k for *_, k in all_h]], leakage=lk, circuit=circ)


def decode_memory(circuit, dets: np.ndarray, use_heralds: bool = True):
    """PyMatching (correlated matching with the herald detectors) on the circuit's DEM."""
    import pymatching

    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=use_heralds)
    shots = np.asarray(dets, dtype=np.uint8)
    if not use_heralds:
        shots = shots.copy()
        shots[:, herald_mask(circuit)] = 0
    return m.decode_batch(shots, enable_correlations=use_heralds)
