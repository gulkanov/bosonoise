"""Dressed ancilla transitions versus storage photon number.

For the dressed branch states |i~, n> (ancilla i in {g, e, f}, coupler ground, storage
Fock n; from the branch analysis) we tabulate every downward ancilla transition
|i~, n> -> |j~, m> (i > j, m = n + dm) with its frequency f = E(i~,n) - E(j~,m) (GHz;
positive = energy released to the bath) and the dressed charge matrix elements of the
ancilla (n_q) and coupler (n_c) charge operators, which are what a charge-coupled bath
(Ohmic background, TLS) couples to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.cell import StorageCouplerAncilla
from .branches import branch_analysis

LEVEL = {"g": 0, "e": 1, "f": 2}


@dataclass
class TransitionTable:
    flux: float
    nmax: int
    pair: np.ndarray  # (N,) strings like "f->g"
    n: np.ndarray  # initial Fock number
    m: np.ndarray  # final Fock number
    freq: np.ndarray  # GHz, E_i(n) - E_j(m)
    Mq: np.ndarray  # <j~,m| n_q |i~,n>
    Mc: np.ndarray  # <j~,m| n_c |i~,n>
    n01_bare: float  # |<0|n_q|1>| of the isolated ancilla
    n12_bare: float  # |<1|n_q|2>| of the isolated ancilla
    energy: dict  # branch -> (nmax+1,) dressed energies (GHz)

    def select(self, pair: str, dm: int | None = None) -> "np.ndarray":
        mask = self.pair == pair
        if dm is not None:
            mask &= (self.m - self.n) == dm
        return np.nonzero(mask)[0]

    def dominant(self, pair: str, op: str = "q") -> dict:
        """Per n, the m-channel with the largest |M|^2 (only emission channels, f > 0)."""
        M = self.Mq if op == "q" else self.Mc
        out = {"n": [], "m": [], "freq": [], "M2": []}
        for n in range(self.nmax + 1):
            idx = np.nonzero((self.pair == pair) & (self.n == n) & (self.freq > 0))[0]
            if len(idx) == 0:
                continue
            k = idx[np.argmax(np.abs(M[idx]) ** 2)]
            for key, val in (("n", n), ("m", self.m[k]), ("freq", self.freq[k]), ("M2", abs(M[k]) ** 2)):
                out[key].append(val)
        return {k: np.array(v) for k, v in out.items()}


def dressed_transitions(model: StorageCouplerAncilla, flux: float, nmax: int = 20,
                        pairs=("f->e", "f->g", "e->g"), dms=(-1, 0, 1, 2), E=None, V=None) -> TransitionTable:
    top = nmax + max(dms)
    if top > model.t.n_s - 1:
        raise ValueError("storage cutoff too small for nmax + max(dm)")
    if E is None or V is None:
        E, V = model.eigh(flux)
    br = branch_analysis(model, flux, starts=((0, 0), (0, 1), (0, 2)), nmax=top, E=E, V=V)
    idx = {"g": br.index[0], "e": br.index[1], "f": br.index[2]}
    nc, nq = model.charge_ops(flux)
    rows = []
    for pair in pairs:
        a, b = pair.split("->")
        for n in range(nmax + 1):
            va = V[:, idx[a][n]]
            wq, wc = nq @ va, nc @ va
            for dm in dms:
                m = n + dm
                if m < 0 or m > top:
                    continue
                vb = V[:, idx[b][m]]
                rows.append((pair, n, m, float(E[idx[a][n]] - E[idx[b][m]]), float(vb @ wq), float(vb @ wc)))
    anc = model._ancilla.spectrum(3)[1]
    energy = {k: np.array([E[idx[k][n]] for n in range(nmax + 1)]) for k in idx}
    return TransitionTable(
        flux=flux, nmax=nmax,
        pair=np.array([r[0] for r in rows]), n=np.array([r[1] for r in rows]), m=np.array([r[2] for r in rows]),
        freq=np.array([r[3] for r in rows]), Mq=np.array([r[4] for r in rows]), Mc=np.array([r[5] for r in rows]),
        n01_bare=float(abs(anc[0, 1])), n12_bare=float(abs(anc[1, 2])), energy=energy,
    )
