"""Diabatic-crossing search along a coupler-flux trajectory (skeleton of the resonance atlas).

At every flux point all dressed states are labelled by their dominant bare state
|s, c, q> (greedy unique assignment by weight). For every "branch" state A = |n, 0, q_i>
(q_i in {g, e, f}, n <= nmax) and every other labelled state B we look for sign changes
of E_B - E_A between neighbouring flux points: these are the flux positions where A and B
are degenerate, i.e. where an avoided crossing (multiphoton resonance) is traversed by a
flux pulse. Gaps are NOT computed here (see `refine_gap`): narrow crossings are invisible
to eigenvector tracking, so detection must rely on diabatic energies.

The heuristic coupling order counts the minimal number of applications of the three
coupling terms  X n_c (ds, dc = +-1), X n_q (ds, dq = +-1), n_c n_q (dc, dq = +-1)
needed to connect A and B (nearest-neighbour charge matrix elements only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from ..model.cell import StorageCouplerAncilla


def label_all(V: np.ndarray, ntry: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Greedy unique labelling dressed -> bare. Returns (bare_of_dressed, weight)."""
    W = V ** 2  # rows: bare, cols: dressed
    N = W.shape[1]
    best = W.max(axis=0)
    order = np.argsort(-best)
    taken = np.zeros(W.shape[0], bool)
    lab = -np.ones(N, int)
    wt = np.zeros(N)
    top = np.argpartition(-W, kth=min(ntry, W.shape[0] - 1), axis=0)[:ntry + 1]
    for k in order:
        cand = top[:, k][np.argsort(-W[top[:, k], k])]
        for b in cand:
            if not taken[b]:
                lab[k], wt[k] = b, W[b, k]
                taken[b] = True
                break
    return lab, wt


def coupling_order(ds: int, dc: int, dq: int) -> int:
    a, b, c = abs(ds), abs(dc), abs(dq)
    if (a + b + c) % 2:
        return 99  # not connected by nearest-neighbour charge matrix elements
    return int(max(a, b, c, (a + b + c) // 2))


@dataclass
class DiabaticSweep:
    fluxes: np.ndarray
    E: np.ndarray  # (nflux, nbare) energy of the dressed state labelled by each bare state
    weight: np.ndarray  # (nflux, nbare) weight of that bare state in its dressed state
    labels: np.ndarray  # (nbare, 3)


def diabatic_sweep(model: StorageCouplerAncilla, fluxes, verbose: bool = False) -> DiabaticSweep:
    nb = model.basis.dim
    E_lab = np.full((len(fluxes), nb), np.nan)
    W_lab = np.zeros((len(fluxes), nb))
    for i, fl in enumerate(fluxes):
        H = model.hamiltonian(float(fl))
        E, V = sla.eigh(H, driver="evd")
        E = E - E[0]
        lab, wt = label_all(V)
        ok = lab >= 0
        E_lab[i, lab[ok]] = E[ok]
        W_lab[i, lab[ok]] = wt[ok]
        if verbose:
            print(f"  flux {fl:.4f}: dim {nb}, min label weight {wt[ok].min():.3f}")
    return DiabaticSweep(np.asarray(fluxes, float), E_lab, W_lab, model.basis.labels.copy())


def find_crossings(sw: DiabaticSweep, model: StorageCouplerAncilla, nmax: int = 20,
                   branches=(0, 1, 2), window: float = 1.0, edge_margin: float = 20.0,
                   max_order: int = 8, min_weight: float = 0.3) -> list[dict]:
    """Sign changes of E_B - E_A on the flux grid (A = branch states, B = all others)."""
    lab = sw.labels
    p = model.p
    # flux-independent proxy energy used for the basis cutoff; drop states near the edge
    E_c_ref, _ = model._coupler.at(model.t.ref_flux).spectrum(model.t.n_c)
    proxy = lab[:, 0] * p.f_s + E_c_ref[lab[:, 1]] + model._E_q[lab[:, 2]]
    ecut = model.t.e_cut if model.t.e_cut is not None else np.inf
    far_from_edge = proxy <= ecut - edge_margin
    out = []
    for q0 in branches:
        for n in range(nmax + 1):
            a = model.basis.index[(n, 0, q0)]
            D = sw.E - sw.E[:, [a]]  # (nflux, nbare)
            valid = (np.isfinite(D) & (sw.weight >= min_weight) & (sw.weight[:, [a]] >= min_weight))
            s = np.sign(D)
            change = (s[1:] * s[:-1] < 0) & valid[1:] & valid[:-1] & (np.abs(D[1:]) < window) & (np.abs(D[:-1]) < window)
            for i, b in zip(*np.nonzero(change)):
                if b == a or not far_from_edge[b]:
                    continue
                ds, dc, dq = (int(v) for v in lab[b] - lab[a])
                order = coupling_order(ds, dc, dq)
                if order > max_order:
                    continue
                x0, x1 = sw.fluxes[i], sw.fluxes[i + 1]
                d0, d1 = D[i, b], D[i + 1, b]
                xc = x0 + (x1 - x0) * d0 / (d0 - d1)
                out.append(dict(branch=q0, n=n, partner=tuple(int(v) for v in lab[b]),
                                dsdcdq=(ds, dc, dq), order=order, flux=float(xc),
                                energy=float(np.interp(xc, [x0, x1], [sw.E[i, a], sw.E[i + 1, a]])),
                                slope_GHz_per_Phi0=float((d1 - d0) / (x1 - x0))))
    return out


def refine_gap(model: StorageCouplerAncilla, a_label, b_label, flux_lo: float, flux_hi: float,
               tol: float = 1e-7, max_iter: int = 60) -> dict:
    """Minimal splitting of the two dressed states that carry the most weight on bare A and B
    within [flux_lo, flux_hi] (golden-section search on the splitting)."""
    ia, ib = model.basis.index[tuple(a_label)], model.basis.index[tuple(b_label)]

    def split(fl):
        H = model.hamiltonian(float(fl))
        E, V = sla.eigh(H, driver="evd")
        w = V[ia] ** 2 + V[ib] ** 2
        k = np.argsort(-w)[:2]
        return abs(E[k[0]] - E[k[1]]), E[k], V[[ia, ib]][:, k] ** 2

    gr = (np.sqrt(5) - 1) / 2
    lo, hi = flux_lo, flux_hi
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = split(c)[0], split(d)[0]
    it = 0
    while hi - lo > tol and it < max_iter:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = split(c)[0]
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = split(d)[0]
        it += 1
    xm = 0.5 * (lo + hi)
    gap, Es, weights = split(xm)
    return dict(flux=xm, gap_GHz=float(gap), weights=weights.tolist(), iterations=it)
