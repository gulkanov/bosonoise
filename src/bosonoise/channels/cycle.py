"""One repetition-code stabilizer measurement X_a X_b of a cat-transmon device: fault table.

Ocelot protocol (App. G2/G5): the ancilla is prepared in (|g> + |f>)/sqrt(2), CX gates with cat
a and then cat b are applied sequentially (each a half turn of the storage in the f branch),
the ancilla is unprepared (inverse preparation), read out (g/e/f) and reset. Two-photon
stabilization is off during the gates.

The two ancilla branches B = g, f (weight 1/2 each; the storage states of the branches are
orthogonal after tracing the ancilla) are propagated through CX1 (cell A-a) with the dressed-
ladder ULE of channels/cx2.py. The ancilla-resolved blocks give P(error on cat a, ancilla level
k after CX1). CX2 (cell A-b) is started from level k. Ideal targets: no rotation in the g branch,
a flip of each cat in the f branch. Errors of cat a and cat b are conditionally independent
given the ancilla level between the gates, so
    P(e_a, e_b, k_f | B) = sum_k P(e_a, k | B) P(e_b, k_f | k, B).
Readout after unpreparation (inverse of ge pi/2 then ef pi): a trajectory ending in e reads
out as f -> heralded (erasure flag); ending in g in the f branch or in f in the g branch gives
a random syndrome bit -> silent; otherwise the syndrome is as ideal.
Errors between the gates (cat a idle during CX2) and outside the gates (stabilized idle, storage
phase flips) are not part of this table.

Fault injection (paper App. D.2): fault_tables() splits the table into 'no fault of class F'
and 'exactly one fault of class F in CX(a)' / '... in CX(b)' (first-order Dyson term, all other
processes at full order), which gives the conversion eta_F of each fault class into cat X errors,
correlated X pairs, heralds and silent syndrome faults.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conversion import AsymptoticProjector
from .cx2 import BathModel, LadderGrid, Pulse, initial_blocks, simulate_cx2

G, E, F = 0, 1, 2
FAULT_CLASSES = ("f->e", "e->g", "f->g", "photon-assisted", "heating", "coupler")


def family_class(key) -> str:
    """Physical class of a jump family (op, i, j, dm) of the dressed-ladder bath."""
    op, i, j, dm = key
    if op == "c":
        return "coupler"
    if op != "q":
        return "other"
    if i == F and j == G:
        return "f->g"  # direct, dressing-opened (any photon change)
    if dm != 0:
        return "photon-assisted"
    if j > i:
        return "heating"
    return {(F, E): "f->e", (E, G): "e->g"}.get((i, j), "other")


@dataclass
class CXCell:
    """A calibrated CX (half turn) between the ancilla and one storage."""

    grid: LadderGrid
    pulse: Pulse
    corr: tuple  # (delta_g, delta_f) phase corrections from calibrate_phases(..., f_target_flipped=True)
    bath: BathModel
    projectors: dict  # nbar -> AsymptoticProjector for this storage (its buffer asymmetry)
    nmax: int = 15
    _cache: dict = field(default_factory=dict, repr=False)

    def step(self, nbar: float, level: int):
        """(P(flip, k), P(k)) for k = g, e, f after a CX started from ancilla `level`, storage in |0_L>.

        'flip' = population of |1_L> after re-stabilization (not yet compared with a target)."""
        key = (nbar, level)
        if key not in self._cache:
            blk = simulate_cx2(self.grid, self.pulse, self.bath, initial_blocks(nbar, level, self.nmax), self.nmax,
                               blocks=True, phase_corr=self.corr)
            P = self.projectors[nbar]
            flip, pop = np.zeros(3), np.zeros(3)
            for k in range(3):
                tr = float(np.trace(blk[k]).real)
                if tr > 1e-14:
                    flip[k] = tr * P.p_flip(blk[k] / tr)
                    pop[k] = tr
            norm = pop.sum()  # tiny losses through the photon cutoff
            self._cache[key] = (flip / norm, pop / norm)
        return self._cache[key]

    def component_blocks(self, nbar: float, level: int, classes: tuple) -> dict:
        """Raw storage blocks {c: (n_levels, N, N)} of the multi-class Dyson split (cached)."""
        key = ("blocks", nbar, level, tuple(classes))
        if key not in self._cache:
            preds = [(lambda k, fc=fc: family_class(k) == fc) for fc in classes]
            self._cache[key] = simulate_cx2(self.grid, self.pulse, self.bath, initial_blocks(nbar, level, self.nmax),
                                            self.nmax, blocks=True, phase_corr=self.corr, fault=preds)
        return self._cache[key]

    def herald_projector(self, nbar: float, eta: float) -> AsymptoticProjector:
        """No-click projector for a buffer-output detector of efficiency eta (same map parameters)."""
        key = ("hproj", nbar, eta)
        if key not in self._cache:
            P = self.projectors[nbar]
            self._cache[key] = AsymptoticProjector(nbar, N=P.N, kerr_ratio=P.kerr_ratio, eta_detect=eta)
        return self._cache[key]

    def step_components_herald(self, nbar: float, level: int, classes: tuple, eta: float, flipped: bool,
                               kmin: int = 1) -> dict:
        """{c: Q[k, e, h]}: final ancilla level k, cat error e (relative to the target: flipped in the f branch),
        herald h = at least kmin detected buffer photons (detector efficiency eta) during the re-stabilization,
        normalized together over all components."""
        key = ("hcomp", nbar, level, tuple(classes), eta, flipped, kmin)
        if key not in self._cache:
            P, Ph = self.projectors[nbar], self.herald_projector(nbar, eta)
            res, tot = {}, 0.0
            for c, blk in self.component_blocks(nbar, level, classes).items():
                Q = np.zeros((3, 2, 2))
                for k in range(3):
                    tr = float(np.trace(blk[k]).real)
                    if tr <= 1e-16:
                        continue
                    f1 = P.p_flip(blk[k] / tr)
                    if kmin == 1:
                        s0, s1 = Ph.silent_split(blk[k] / tr)
                    else:  # below threshold: fewer than kmin clicks
                        s0, s1 = Ph.click_distribution(blk[k] / tr, kmin).sum(axis=0)
                    out = {0: (1 - f1, s0), 1: (f1, s1)}  # outcome |0_L>, |1_L>: (total, silent)
                    for outcome, (tot_o, sil_o) in out.items():
                        e = int(outcome == 0) if flipped else int(outcome == 1)
                        Q[k, e, 0] += tr * sil_o
                        Q[k, e, 1] += tr * max(tot_o - sil_o, 0.0)
                res[c] = Q
                tot += Q.sum()
            self._cache[key] = {c: Q / tot for c, Q in res.items()}
        return self._cache[key]

    def step_components(self, nbar: float, level: int, classes: tuple):
        """{c: (flip, pop)} per final level for the multi-class Dyson split (at most one jump per class);
        normalized together."""
        key = ("comp", nbar, level, tuple(classes))
        if key not in self._cache:
            comps = self.component_blocks(nbar, level, classes)
            P = self.projectors[nbar]
            res, tot = {}, 0.0
            for c, blk in comps.items():
                flip, pop = np.zeros(3), np.zeros(3)
                for k in range(3):
                    tr = float(np.trace(blk[k]).real)
                    if tr > 1e-16:
                        flip[k] = tr * P.p_flip(blk[k] / tr)
                        pop[k] = tr
                res[c] = (flip, pop)
                tot += pop.sum()
            self._cache[key] = {c: (f / tot, p / tot) for c, (f, p) in res.items()}
        return self._cache[key]

    def step_fault(self, nbar: float, level: int, fclass: str):
        """((flip, pop) without a fault of class fclass, (flip, pop) with exactly one), per final level;
        normalized together, so that pop_0.sum() + pop_1.sum() = 1 and pop_1.sum() = P(fault)."""
        key = ("fault", nbar, level, fclass)
        if key not in self._cache:
            b0, b1 = simulate_cx2(self.grid, self.pulse, self.bath, initial_blocks(nbar, level, self.nmax), self.nmax,
                                  blocks=True, phase_corr=self.corr, fault=lambda k: family_class(k) == fclass)
            P = self.projectors[nbar]
            parts, tot = [], 0.0
            for blk in (b0, b1):
                flip, pop = np.zeros(3), np.zeros(3)
                for k in range(3):
                    tr = float(np.trace(blk[k]).real)
                    if tr > 1e-16:
                        flip[k] = tr * P.p_flip(blk[k] / tr)
                        pop[k] = tr
                parts.append((flip, pop))
                tot += pop.sum()
            self._cache[key] = tuple((f / tot, p / tot) for f, p in parts)
        return self._cache[key]


def stabilizer_table(cell_a: CXCell, cell_b: CXCell, nbar: float) -> dict:
    """Joint fault table per ancilla branch: T[B][e_a, e_b, k_f] (B = 'g', 'f')."""
    out = {}
    for B, level in (("g", G), ("f", F)):
        flipped = level == F
        fa, pa = cell_a.step(nbar, level)
        err_a = (pa - fa) if flipped else fa  # P(e_a = 1, k)
        T = np.zeros((2, 2, 3))
        for k in range(3):
            if pa[k] < 1e-14:
                continue
            fb, pb = cell_b.step(nbar, k)
            err_b = (pb - fb) if flipped else fb  # P(e_b = 1, k_f | k)
            for ea, wa in ((1, err_a[k]), (0, pa[k] - err_a[k])):
                T[ea, 1] += wa * err_b
                T[ea, 0] += wa * (pb - err_b)
        out[B] = T
    return out


def component_tables(cell_a: CXCell, cell_b: CXCell, nbar: float, classes: tuple) -> dict:
    """{c: {B: T}} for c in {0,1}^K over the given fault classes: at most one fault of each class in the whole
    measurement (CX(a) and CX(b) together), so fault sequences are separated -- e.g. with classes
    (f->e, e->g, f->g): (1,0,0) f->e alone, (1,1,0) the double decay f->e then e->g (within one gate or across
    the two), (0,0,1) the direct f->g. The sum over c is the stabilizer table up to terms second order in one
    class; c = (0,...,0) still contains the unselected classes at full order."""
    out = {}
    for B, level in (("g", G), ("f", F)):
        flipped = level == F
        A = cell_a.step_components(nbar, level, classes)
        for ca, a_res in A.items():
            for cb in A:  # same component labels for CX(b)
                if any(x and y for x, y in zip(ca, cb)):
                    continue  # the same class twice: second order
                c = tuple(x + y for x, y in zip(ca, cb))
                T = _combine(a_res, lambda k, cb=cb: cell_b.step_components(nbar, k, classes)[cb], flipped)
                out.setdefault(c, {}).setdefault(B, np.zeros((2, 2, 3)))
                out[c][B] += T
    return out


def herald_component_tables(cell_a: CXCell, cell_b: CXCell, nbar: float, classes: tuple, eta: float,
                            kmin: int = 1) -> dict:
    """{c: {B: T[e_a, h_a, e_b, h_b, k_f]}}: component_tables resolved by the herald h of each cat's own buffer
    detector (efficiency eta, at least kmin clicks) during its re-stabilization."""
    out = {}
    for B, level in (("g", G), ("f", F)):
        flipped = level == F
        A = cell_a.step_components_herald(nbar, level, classes, eta, flipped, kmin)
        for ca, Qa in A.items():
            for cb in A:
                if any(x and y for x, y in zip(ca, cb)):
                    continue
                c = tuple(x + y for x, y in zip(ca, cb))
                T = np.zeros((2, 2, 2, 2, 3))
                for k in range(3):
                    wa = Qa[k]
                    if wa.sum() < 1e-16:
                        continue
                    Qb = cell_b.step_components_herald(nbar, k, classes, eta, flipped, kmin)[cb]
                    T += wa[:, :, None, None, None] * Qb.transpose(1, 2, 0)[None, None, :, :, :]
                out.setdefault(c, {}).setdefault(B, np.zeros((2, 2, 2, 2, 3)))
                out[c][B] += T
    return out


def summarize_herald(part: dict) -> dict:
    """Heralded vs silent cat errors for one component (branches weighted 1/2), conditional on the component."""
    T = 0.5 * (part["g"] + part["f"])
    P = float(T.sum())
    if P <= 0:
        return dict(p_fault=0.0)
    xa_s, xa_h = T[1, 0].sum(), T[1, 1].sum()
    xb_s, xb_h = T[:, :, 1, 0].sum(), T[:, :, 1, 1].sum()
    odd = T[1, :, 0].sum() + T[0, :, 1].sum()
    odd_silent = T[1, 0, 0, 0].sum() + T[0, 0, 1, 0].sum()  # odd number of X and no click on either cat
    return dict(p_fault=P, x_a=float((xa_s + xa_h) / P), x_a_heralded=float(xa_h / P), x_b=float((xb_s + xb_h) / P),
                x_b_heralded=float(xb_h / P), odd=float(odd / P), odd_silent=float(odd_silent / P),
                false_click_a=float(T[0, 1].sum() / P), false_click_b=float(T[:, :, 0, 1].sum() / P))


def component_name(c: tuple, classes: tuple) -> str:
    active = [classes[m] for m, bit in enumerate(c) if bit]
    if not active:
        return "none of " + "/".join(classes)
    if active == ["f->e", "e->g"]:
        return "double decay f->e->g"
    return " + ".join(active) + (" only" if len(active) == 1 else "")


def _combine(a, b_from, flipped: bool) -> np.ndarray:
    """T[e_a, e_b, k_f] from CX(a) results a = (flip[k], pop[k]) and CX(b) results b_from(k) started from k."""
    fa, pa = a
    err_a = (pa - fa) if flipped else fa
    T = np.zeros((2, 2, 3))
    for k in range(3):
        if pa[k] < 1e-16:
            continue
        fb, pb = b_from(k)
        err_b = (pb - fb) if flipped else fb
        for ea, wa in ((1, err_a[k]), (0, pa[k] - err_a[k])):
            T[ea, 1] += wa * err_b
            T[ea, 0] += wa * (pb - err_b)
    return T


def fault_tables(cell_a: CXCell, cell_b: CXCell, nbar: float, fclass: str) -> dict:
    """{'none': {B: T}, 'in_a': {B: T}, 'in_b': {B: T}}: no fault of class fclass, exactly one in CX(a),
    exactly one in CX(b); their sum is the stabilizer table up to O(P(fault)^2)."""
    out = {"none": {}, "in_a": {}, "in_b": {}}
    for B, level in (("g", G), ("f", F)):
        flipped = level == F
        a0, a1 = cell_a.step_fault(nbar, level, fclass)
        b0 = lambda k: cell_b.step_fault(nbar, k, fclass)[0]
        b1 = lambda k: cell_b.step_fault(nbar, k, fclass)[1]
        out["none"][B] = _combine(a0, b0, flipped)
        out["in_a"][B] = _combine(a1, b0, flipped)
        out["in_b"][B] = _combine(a0, b1, flipped)
    return out


def summarize_fault(part: dict) -> dict:
    """Probability of the fault (per stabilizer measurement, branches weighted 1/2) and its conversion."""
    T = 0.5 * (part["g"] + part["f"])
    P = float(T.sum())
    if P <= 0:
        return dict(p_fault=0.0)
    return dict(p_fault=P, x_a=float(T[1].sum() / P), x_b=float(T[:, 1].sum() / P), xx=float(T[1, 1].sum() / P),
                odd=float((T[1, 0].sum() + T[0, 1].sum()) / P), herald=float(T[:, :, E].sum() / P),
                silent=float(0.5 * (part["f"][:, :, G].sum() + part["g"][:, :, F].sum()) / P))


def summarize(table: dict) -> dict:
    """Per-stabilizer probabilities (branches weighted 1/2)."""
    T = 0.5 * (table["g"] + table["f"])
    herald = T[:, :, E].sum()
    silent = 0.5 * (table["f"][:, :, G].sum() + table["g"][:, :, F].sum())
    return dict(
        p_x_a=float(T[1].sum()), p_x_b=float(T[:, 1].sum()), p_xx=float(T[1, 1].sum()),
        p_herald=float(herald), p_silent_syndrome=float(silent),
        p_x_a_given_herald=float(T[1, :, E].sum() / herald) if herald > 0 else np.nan,
        p_x_b_given_herald=float(T[:, 1, E].sum() / herald) if herald > 0 else np.nan,
        p_x_a_unheralded=float(T[1, :, G].sum() + T[1, :, F].sum()),
        p_x_b_unheralded=float(T[:, 1, G].sum() + T[:, 1, F].sum()),
        p_xx_unheralded=float(T[1, 1, G] + T[1, 1, F]),
        p_x_a_branch_f=float(table["f"][1].sum()), p_x_b_branch_f=float(table["f"][:, 1].sum()),
        p_x_a_branch_g=float(table["g"][1].sum()), p_x_b_branch_g=float(table["g"][:, 1].sum()),
        p_xx_branch_f=float(table["f"][1, 1].sum()), p_xx_branch_g=float(table["g"][1, 1].sum()),
    )


def odd_parity(table: dict, branch: str | None = None) -> float:
    """P(odd number of X errors on the two cats) = the stabilizer's contribution to a logical bit flip
    of Z_1 ... Z_d (a correlated X_a X_b acts trivially). branch=None: ancilla in (g+f)/sqrt2."""
    T = 0.5 * (table["g"] + table["f"]) if branch is None else table[branch]
    return float(T[1, 0].sum() + T[0, 1].sum())
