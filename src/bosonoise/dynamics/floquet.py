"""Floquet analysis of a transmon driven through its charge: the semiclassical limit of a displaced oscillator.

A cavity displaced to nbar photons and coupled to the transmon through g n (a + a^dag) acts on the transmon, in the
displaced frame, as a classical charge drive at the cavity frequency (Dumas et al., PRX 14, 041023 (2024), Eq. (15)):

    H(t) = sum_i E_i |i><i| + eps cos(2 pi f_d t) n,        eps = 2 g sqrt(nbar),   f_d = cavity frequency.

The residual coupling g n (a + a^dag) to the displaced vacuum is neglected (relative order 1/sqrt(nbar)); the
low-photon dispersive physics is checked against the exact transmon + cavity model in tests/test_floquet.py.

Floquet states |psi_a(t)> = exp(-2 pi i q_a t) sum_m exp(2 pi i m f_d t) |phi_a,m> solve, in the extended (Sambe) space
with Fourier cutoff K,

    sum_m' ([H0 + m f_d] delta_mm' + (eps/2) n [delta_m',m-1 + delta_m',m+1]) |phi_a,m'> = q_a |phi_a,m>.

A branch is followed from eps = 0, where it is the bare state |i> in the block m = 0, by maximal overlap with the
previous step, so it stays on the replica that continues the bare state.

Floquet-Markov rates for a charge bath n (x) B (golden rule in the Floquet basis):

    Gamma_{a->b} = sum_k |X^k_ba|^2 Gamma(q_a - q_b - k f_d),      X^k_ba = sum_m <phi_b,m| n |phi_a,m+k>,

with a rate function Gamma(w) of dynamics/ule.py (w > 0: energy released to the bath). The time average of
<phi_b(t)|n|phi_a(t)> is X^0_ba; the sidebands k != 0 release (or take) k drive quanta.
Units: GHz (cyclic), ns, rates in 1/ns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sambe_hamiltonian(E: np.ndarray, n: np.ndarray, eps: float, f_d: float, K: int) -> np.ndarray:
    """Extended-space Floquet Hamiltonian, blocks m = -K..K (block index m + K)."""
    N = len(E)
    M = 2 * K + 1
    H = np.zeros((N * M, N * M))
    H0 = np.diag(E)
    for i, m in enumerate(range(-K, K + 1)):
        H[i * N:(i + 1) * N, i * N:(i + 1) * N] = H0 + m * f_d * np.eye(N)
        if i + 1 < M:
            H[i * N:(i + 1) * N, (i + 1) * N:(i + 2) * N] = 0.5 * eps * n
            H[(i + 1) * N:(i + 2) * N, i * N:(i + 1) * N] = 0.5 * eps * n
    return H


@dataclass
class FloquetBranches:
    """Tracked Floquet branches over a grid of drive amplitudes."""

    eps: np.ndarray            # (S,) drive amplitudes (GHz)
    states: tuple              # bare-state labels followed
    q: np.ndarray              # (S, len(states)) quasienergies (GHz)
    bare_weight: np.ndarray    # (S, len(states)) time-averaged population of the bare state it started from
    modes: list                # per step: (len(states), M, N) Fourier components |phi_a,m>
    rest: list                 # per step: dict(q=(R,), modes=(R, M, N)) of the other replicas in the central zone


def _modes_from_vectors(V: np.ndarray, N: int, M: int) -> np.ndarray:
    """(N M, R) eigenvectors -> (R, M, N) Fourier components."""
    return V.T.reshape(V.shape[1], M, N)


def track_branches(E: np.ndarray, n: np.ndarray, f_d: float, eps_grid, K: int, states=(0, 1, 2),
                   keep_rest: bool = True) -> FloquetBranches:
    """Follow the Floquet states that continue the bare states `states` as the drive amplitude grows.

    At every step the Sambe Hamiltonian is diagonalized; each branch moves to the eigenvector of largest overlap with
    its previous vector. With keep_rest, one replica of every other physical state is kept as a possible final state
    of transitions: the replica with the largest weight in the blocks |m| <= 1, labelled by its dominant bare state.
    Rates do not depend on the replica chosen (a shift by one replica maps X^k to X^(k+1) and the released energy
    accordingly), provided K and kmax are large enough.
    """
    E = np.asarray(E, float)
    N, M = len(E), 2 * K + 1
    eps_grid = np.asarray(eps_grid, float)
    prev = np.zeros((N * M, len(states)))
    for j, s in enumerate(states):
        prev[K * N + s, j] = 1.0
    qs, bw, modes, rests = [], [], [], []
    for eps in eps_grid:
        w, V = np.linalg.eigh(sambe_hamiltonian(E, n, eps, f_d, K))
        ov = np.abs(prev.T @ V)
        pick = []
        for j in range(len(states)):
            order = np.argsort(-ov[j])
            pick.append(next(c for c in order if c not in pick))
        Vp = V[:, pick]
        Vp = Vp * np.sign(np.sum(Vp * prev, axis=0))  # continuous sign
        prev = Vp
        md = _modes_from_vectors(Vp, N, M)
        qs.append(w[pick])
        bw.append([np.sum(np.abs(md[j, :, s]) ** 2) for j, s in enumerate(states)])
        modes.append(md)
        if keep_rest:
            # physical representatives of all other states: the replica with the largest weight in the blocks
            # |m| <= 1, one per bare state label (by maximal time-averaged bare population)
            allm = _modes_from_vectors(V, N, M)
            central = np.sum(np.abs(allm[:, K - 1:K + 2, :]) ** 2, axis=(1, 2))
            label = np.argmax(np.sum(np.abs(allm) ** 2, axis=1), axis=1)
            chosen = {}
            for c in np.argsort(-central):
                if c in pick:
                    continue
                if label[c] not in chosen and label[c] not in states:
                    chosen[label[c]] = c
            idx = np.array(sorted(chosen.values()), dtype=int)
            rests.append(dict(q=w[idx], modes=allm[idx], label=label[idx]))
        else:
            rests.append(None)
    return FloquetBranches(eps_grid, tuple(states), np.array(qs), np.array(bw), modes, rests)


def charge_fourier(mb: np.ndarray, ma: np.ndarray, n: np.ndarray, kmax: int) -> np.ndarray:
    """X^k_ba = sum_m <phi_b,m| n |phi_a,m+k> for k = -kmax..kmax (modes as (M, N) arrays)."""
    M = ma.shape[0]
    nma = ma @ n.T  # (M, N): n |phi_a,m> as rows
    out = np.zeros(2 * kmax + 1, dtype=complex)
    for i, k in enumerate(range(-kmax, kmax + 1)):
        lo, hi = max(0, -k), min(M, M - k)
        if lo < hi:
            out[i] = np.sum(np.conj(mb[lo:hi]) * nma[lo + k:hi + k])
    return out


def rate(q_a: float, ma: np.ndarray, q_b: float, mb: np.ndarray, n: np.ndarray, f_d: float, gamma, kmax: int) -> float:
    """Floquet-Markov rate a -> b (1/ns) for a charge bath with rate function gamma(w)."""
    X = charge_fourier(mb, ma, n, kmax)
    k = np.arange(-kmax, kmax + 1)
    return float(np.sum(np.abs(X) ** 2 * gamma(q_a - q_b - k * f_d)))


def branch_rates(fb: FloquetBranches, step: int, n: np.ndarray, f_d: float, gamma, kmax: int) -> dict:
    """Rates out of every tracked branch at grid point `step`: to the other tracked branches and to every kept state.

    Returns {(a_label, b_label): rate}; b_label is ('branch', s) for a tracked branch, ('state', i) for a kept replica.
    """
    out = {}
    md, q = fb.modes[step], fb.q[step]
    rest = fb.rest[step]
    for ja, sa in enumerate(fb.states):
        for jb, sb in enumerate(fb.states):
            if ja != jb:
                out[(sa, ("branch", sb))] = rate(q[ja], md[ja], q[jb], md[jb], n, f_d, gamma, kmax)
        if rest is not None:
            for qr, mr, lab in zip(rest["q"], rest["modes"], rest["label"]):
                out[(sa, ("state", int(lab)))] = rate(q[ja], md[ja], qr, mr, n, f_d, gamma, kmax)
    return out


def one_photon_element(E: np.ndarray, n: np.ndarray, g: float, f_c: float, i: int, j: int, nbar: float) -> float:
    """First-order dressing amplitude of the one-photon-assisted transition |i, n> -> |j, n + 1> through the bath
    operator n (charge), for a mode at f_c coupled by g n (a + a^dag):

        M_ij(n) = g sqrt(n + 1) sum_k n_jk n_ki [1/(E_i - E_k - f_c) + 1/(E_j - E_k + f_c)].

    Total parity: with n odd, M_ij vanishes unless i and j have the same parity (e.g. f -> g); then it is the only
    first-order channel, and the rate is |M_ij|^2 Gamma(E_i - E_j - f_c), linear in n + 1."""
    Ei, Ej = E[i], E[j]
    s = 0.0
    for k in range(len(E)):
        if k in (i, j):
            continue
        s += n[j, k] * n[k, i] * (1.0 / (Ei - E[k] - f_c) + 1.0 / (Ej - E[k] + f_c))
    return float(g * np.sqrt(nbar + 1.0) * s)
