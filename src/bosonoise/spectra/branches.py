"""Branch analysis of the dressed spectrum versus storage photon number.

Static (time-independent Hamiltonian) version of the branch analysis of Dumas et al.,
PRX 14, 041023 (2024), arXiv:2402.06615, with the storage mode in the role of the
readout resonator:

    B_i(0)   = dressed state with maximal weight on the bare state |0, c_i, q_i>
    B_i(n+1) = argmax_k |<k| a^dag |B_i(n)>|^2

Along every branch we record the ancilla/coupler/storage populations. A resonance
(multiphoton hybridization of the branch with other ancilla/coupler states) shows up
as a jump of <N_q> or <N_c> and as a drop of the branch overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.cell import StorageCouplerAncilla


@dataclass
class Branches:
    flux: float
    starts: list  # [(c, q), ...] bare coupler/ancilla labels at n = 0
    energy: np.ndarray  # (B, nmax+1) dressed energies (GHz, relative to ground)
    index: np.ndarray  # (B, nmax+1) dressed-state indices
    overlap: np.ndarray  # (B, nmax+1) normalized |<B(n)|a^dag|B(n-1)>|^2 (1 at n=0)
    mean_s: np.ndarray  # (B, nmax+1)
    mean_c: np.ndarray
    mean_q: np.ndarray
    pop_q: np.ndarray  # (B, nmax+1, n_q)
    pop_c: np.ndarray  # (B, nmax+1, n_c)
    top: list  # per branch, per n: list of (weight, (s, c, q)) dominant bare components

    @property
    def n(self) -> np.ndarray:
        return np.arange(self.energy.shape[1])


def branch_analysis(
    model: StorageCouplerAncilla,
    flux: float,
    starts=((0, 0), (0, 1), (0, 2)),
    nmax: int = 20,
    E: np.ndarray | None = None,
    V: np.ndarray | None = None,
    ntop: int = 3,
) -> Branches:
    if nmax > model.t.n_s - 1:
        raise ValueError("nmax must be below the storage Fock cutoff")
    if E is None or V is None:
        E, V = model.eigh(flux)
    lab = model.basis.labels
    adag = model.op("adag")
    P = V ** 2  # weights of dressed states on bare states (real eigenvectors)
    B, N1 = len(starts), nmax + 1
    energy = np.zeros((B, N1))
    index = np.zeros((B, N1), dtype=int)
    overlap = np.ones((B, N1))
    mean = {k: np.zeros((B, N1)) for k in "scq"}
    pop_q = np.zeros((B, N1, model.t.n_q))
    pop_c = np.zeros((B, N1, model.t.n_c))
    top: list = []
    for b, (c0, q0) in enumerate(starts):
        k = model.dressed_index(V, 0, c0, q0, min_weight=0.0)
        tops = []
        for n in range(N1):
            if n > 0:
                phi = adag @ V[:, k]
                norm2 = float(phi @ phi)
                ov = (V.T @ phi) ** 2 / norm2
                k = int(np.argmax(ov))
                overlap[b, n] = ov[k]
            index[b, n] = k
            energy[b, n] = E[k]
            w = P[:, k]
            mean["s"][b, n] = w @ lab[:, 0]
            mean["c"][b, n] = w @ lab[:, 1]
            mean["q"][b, n] = w @ lab[:, 2]
            pop_q[b, n] = np.bincount(lab[:, 2], weights=w, minlength=model.t.n_q)
            pop_c[b, n] = np.bincount(lab[:, 1], weights=w, minlength=model.t.n_c)
            order = np.argsort(w)[::-1][:ntop]
            tops.append([(float(w[j]), tuple(int(v) for v in lab[j])) for j in order])
        top.append(tops)
    return Branches(
        flux=flux, starts=list(starts), energy=energy, index=index, overlap=overlap,
        mean_s=mean["s"], mean_c=mean["c"], mean_q=mean["q"], pop_q=pop_q, pop_c=pop_c, top=top,
    )


def find_resonances(br: Branches, dq: float = 0.02, dc: float = 0.02, min_overlap: float = 0.98) -> list[dict]:
    """List (branch, n) points where the branch character changes appreciably.

    Criteria: |<N_q>(n) - <N_q>(n-1)| > dq, or |<N_c>(n) - <N_c>(n-1)| > dc, or the
    normalized ladder overlap drops below `min_overlap`.
    """
    out = []
    for b, start in enumerate(br.starts):
        for n in range(1, br.energy.shape[1]):
            jq = br.mean_q[b, n] - br.mean_q[b, n - 1]
            jc = br.mean_c[b, n] - br.mean_c[b, n - 1]
            if abs(jq) > dq or abs(jc) > dc or br.overlap[b, n] < min_overlap:
                out.append(dict(branch=start, n=n, dNq=float(jq), dNc=float(jc),
                                overlap=float(br.overlap[b, n]), top=br.top[b][n]))
    return out
