"""CX^2 cycle of a storage-coupler-ancilla cell: dressed-ladder ULE simulation.

The coupler flux pulse (flat-top Gaussian from the idle point to the 'on' point) is
followed adiabatically: the state is expanded in the instantaneous dressed branch states
|c k~, n; Phi(t)> (coupler level c, ancilla k in g, e, f; storage label n) obtained by the
branch analysis of the full cell Hamiltonian. Default branches: coupler ground (c = 0);
WITH_COUPLER adds the coupler-excited branches (c = 1). The first three branches are always
(c, k) = (0, g), (0, e), (0, f). Non-adiabatic (ramp) transitions are not included here (see
dynamics/apt.py).

Phase convention (needed for the coherence between photon numbers): <0,c,k|ck~,0> > 0 and
<ck~,n+1| a^dag |ck~,n> > 0, so that at the idle point the dressed labels coincide with bare
Fock states and a coherent state keeps its form.

Frame: the storage rotates at the g-branch frequency E(g~,1) - E(g~,0) at every flux (the
experiment calibrates this 'storage interaction frequency' and phase with the ancilla in g).
The flat-top length is set so that the f-branch makes exactly one full turn (CX^2 = identity).

Dissipation: ULE jump operators (dynamics/ule.py) for each bath operator O (ancilla charge
n_q, coupler charge n_c, ancilla excitation number N_q, storage number N_s), grouped into
families (i -> j, dm): L = sum_n sqrt(Gamma_O(w_n)) <j~,n+dm|O|i~,n> |j~,n+dm><i~,n|.
Coherence between photon numbers within a family is kept (ULE); cross terms between
different families (transition frequencies differing by >~ 100 MHz, or different final
ancilla states) are dropped ('family-secular'). The state then stays block-diagonal in the
ancilla level: rho = sum_k sigma_k (x) |k><k|.

After the pulse the ancilla is traced out and the storage state is mapped through the
asymptotic projector of two-photon stabilization (channels/conversion.py): the output is
the bit-flip probability of the cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dynamics.flux import flat_top_gaussian
from ..model.cell import StorageCouplerAncilla
from ..spectra.branches import branch_analysis
from .conversion import AsymptoticProjector, logical_states

LEVELS = ("g", "e", "f")
OPS = ("q", "c", "Nq", "Ns")
DEFAULT_BRANCHES = ((0, 0), (0, 1), (0, 2))  # (coupler level, ancilla level) at n = 0
WITH_COUPLER = DEFAULT_BRANCHES + ((1, 0), (1, 1), (1, 2))


@dataclass
class Ladder:
    """Dressed ladder at one flux: energies E[b, n] (GHz) and X[op][j, m, i, n] = <j,m|op|i,n>
    for the branches b of `branches` ((coupler, ancilla) labels at n = 0)."""

    flux: float
    E: np.ndarray
    X: dict
    nc01: float  # |<0|n_c|1>| of the bare coupler at this flux
    min_overlap: float  # smallest branch overlap (labeling quality)
    branches: tuple = DEFAULT_BRANCHES


def ladder_at(model: StorageCouplerAncilla, flux: float, nmax: int, branches=DEFAULT_BRANCHES) -> Ladder:
    E, V = model.eigh(flux)
    br = branch_analysis(model, flux, starts=branches, nmax=nmax, E=E, V=V)
    adag = model.op("adag")
    N1, nb = nmax + 1, len(branches)
    B = np.zeros((V.shape[0], nb, N1))
    for k, (c0, q0) in enumerate(branches):
        for n in range(N1):
            v = V[:, br.index[k, n]].copy()
            if n == 0:
                s = np.sign(v[model.row(0, c0, q0)])
            else:
                s = np.sign(v @ (adag @ B[:, k, n - 1]))
            B[:, k, n] = v * (s if s != 0 else 1.0)
    Bf = B.reshape(V.shape[0], nb * N1)
    nc, nq = model.charge_ops(flux)
    ops = {"q": nq, "c": nc, "Nq": model.op("N_q"), "Ns": model.op("N_s")}
    X = {k: (Bf.T @ (op @ Bf)).reshape(nb, N1, nb, N1) for k, op in ops.items()}
    En = np.array([[E[br.index[k, n]] for n in range(N1)] for k in range(nb)])
    nc01 = float(abs(model.elements(flux).n_c[0, 1]))
    return Ladder(flux=flux, E=En, X=X, nc01=nc01, min_overlap=float(br.overlap.min()), branches=tuple(branches))


@dataclass
class LadderGrid:
    """Ladders on a flux grid, linearly interpolated in flux."""

    ladders: list
    fluxes: np.ndarray = field(init=False)

    def __post_init__(self):
        self.fluxes = np.array([l.flux for l in self.ladders])
        order = np.argsort(self.fluxes)
        self.ladders = [self.ladders[i] for i in order]
        self.fluxes = self.fluxes[order]
        self._E = np.stack([l.E for l in self.ladders])
        self._X = {op: np.stack([l.X[op] for l in self.ladders]) for op in OPS}

    def weights(self, flux: float):
        i = int(np.clip(np.searchsorted(self.fluxes, flux) - 1, 0, len(self.fluxes) - 2))
        w = (flux - self.fluxes[i]) / (self.fluxes[i + 1] - self.fluxes[i])
        return i, float(np.clip(w, 0.0, 1.0))

    def E(self, flux: float) -> np.ndarray:
        i, w = self.weights(flux)
        return (1 - w) * self._E[i] + w * self._E[i + 1]

    def X(self, op: str, flux: float) -> np.ndarray:
        i, w = self.weights(flux)
        return (1 - w) * self._X[op][i] + w * self._X[op][i + 1]

    @property
    def n_branches(self) -> int:
        return self._E.shape[1]

    def chi_gf(self, flux: float) -> float:
        E = self.E(flux)
        return float((E[2, 1] - E[2, 0]) - (E[0, 1] - E[0, 0]))

    def chi_rel(self, flux: float, b: int) -> float:
        """Storage frequency shift of branch b relative to the (coupler ground, g) branch (GHz)."""
        E = self.E(flux)
        return float((E[b, 1] - E[b, 0]) - (E[0, 1] - E[0, 0]))


@dataclass
class Pulse:
    """Flat-top Gaussian flux pulse from 0 to amp; ramp = n_sigma * sigma (ns)."""

    ramp: float
    amp: float = 0.5
    n_sigma: float = 3.0
    t_flat: float = 0.0

    @property
    def sigma(self) -> float:
        return self.ramp / self.n_sigma

    @property
    def duration(self) -> float:
        return 2 * self.ramp + self.t_flat

    def flux(self, t):
        return flat_top_gaussian(t, 0.0, self.t_flat, self.sigma, self.amp, self.n_sigma)


def calibrate_cx2(grid: LadderGrid, ramp: float, amp: float = 0.5, n_sigma: float = 3.0, dt: float = 0.05,
                  turns: float = 1.0) -> Pulse:
    """Flat-top length such that the f-branch rotation integral over the pulse is `turns` turns
    (1 for CX^2, 0.5 for CX)."""
    p = Pulse(ramp=ramp, amp=amp, n_sigma=n_sigma, t_flat=0.0)
    t = np.arange(0.0, 2 * ramp + dt / 2, dt)
    ramps = np.trapezoid([abs(grid.chi_gf(f)) for f in p.flux(t)], t)
    p.t_flat = (turns - ramps) / abs(grid.chi_gf(amp))
    if p.t_flat < 0:
        raise ValueError("ramps alone exceed the target rotation")
    return p


def calibrate_cx2_to_length(grid: LadderGrid, length: float, amp: float = 0.5, n_sigma: float = 3.0,
                            dt: float = 0.05, ramp_bounds=(1.0, 150.0), turns: float = 1.0) -> Pulse:
    """Pulse with a given total length (ns) whose calibrated flat top gives `turns` f-branch
    turns: the ramp time is solved for (turns = 1: CX^2; 0.5: CX)."""
    from scipy.optimize import brentq

    f = lambda x: calibrate_cx2(grid, x, amp, n_sigma, dt, turns).duration - length  # noqa: E731
    hi = ramp_bounds[1]
    while hi > ramp_bounds[0] and 2 * hi * 0.9 > length:  # keep the flat top non-negative
        hi *= 0.8
    r = brentq(f, ramp_bounds[0], hi, xtol=0.01)
    return calibrate_cx2(grid, r, amp, n_sigma, dt, turns)


@dataclass
class BathModel:
    """Two-sided rate functions per bath operator (None = bath off); see dynamics/ule.py."""

    rates: dict
    standard: bool = False  # True: undressed, n-independent f->e and e->g decay only (Hann-type)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def families(self, grid: LadderGrid, nmax: int, dms=(-2, -1, 0, 1, 2)):
        """Jump coefficients C[(op, i, j, dm)] of shape (n_flux, nmax+1) on the flux grid."""
        key = (id(grid), nmax, tuple(dms))
        if key in self._cache:
            return self._cache[key]
        out = {}
        for op, rate in self.rates.items():
            if rate is None:
                continue
            for i in range(grid.n_branches):
                for j in range(grid.n_branches):
                    if i == j:
                        continue
                    for dm in dms:
                        if self.standard and (op != "q" or dm != 0 or abs(i - j) != 1 or max(i, j) > 2):
                            continue
                        C = np.zeros((len(grid.fluxes), nmax + 1))
                        for a, lad in enumerate(grid.ladders):
                            src = lad if not self.standard else grid.ladders[0]
                            for n in range(nmax + 1):
                                m = n + dm
                                if not 0 <= m <= nmax:
                                    continue
                                nn, mm = (n, m) if not self.standard else (0, 0)
                                x = src.X[op][j, mm, i, nn]
                                w = src.E[i, nn] - src.E[j, mm]
                                C[a, n] = np.sqrt(max(float(rate(np.array(w))), 0.0)) * x
                        if np.max(np.abs(C)) > 1e-9:
                            out[(op, i, j, dm)] = C
        self._cache[key] = out
        return out


def simulate_cx2(grid: LadderGrid, pulse: Pulse, bath: BathModel, sigma0: dict, nmax: int,
                 dt: float = 0.5, dms=(-2, -1, 0, 1, 2), blocks: bool = False,
                 phase_corr: tuple = (0.0, 0.0), fault=None):
    """Storage density matrix (Fock labels 0..nmax, idle frame) after the CX^2 pulse.

    sigma0: {k: storage density matrix} initial ancilla-diagonal blocks (k = 0, 1, 2 for g, e, f).
    blocks: return the (n_branches, nmax+1, nmax+1) branch-resolved blocks instead of their sum.
    phase_corr: (delta_g, delta_f) storage phase correction applied to all blocks and extra
      f-branch rotation (gate-length correction) applied to the f block; see calibrate_phases.
    fault: optional predicate on the jump-family key (op, i, j, dm). Then the first-order Dyson
      split is returned, (rho_0, rho_1): rho_0 = no jump of the selected families, rho_1 = exactly
      one such jump (at any time); both evolve with all other families at full order and with the
      no-jump decay of the selected ones. Tr rho_1 = P(fault); rho_0 + rho_1 = the full result up
      to O(P(fault)^2).
      A list of K predicates (disjoint classes) gives {c: rho_c} for c in {0,1}^K: at most one jump
      of each class, so sequences are separated (e.g. f->e alone, e->g alone, f->e then e->g);
      a second jump of the same class is dropped (second order in that class).
    Split-step integration: exact dressed phases (Strang), first-order dissipative step
    (all rates x dt < 1e-4 here).
    """
    fam = bath.families(grid, nmax, dms)
    N1 = nmax + 1
    sig = np.zeros((grid.n_branches, N1, N1), dtype=complex)
    for k, s in sigma0.items():
        sig[k] = s[:N1, :N1]
    multi = isinstance(fault, (list, tuple))
    preds = list(fault) if multi else ([fault] if fault is not None else [])
    K = len(preds)
    comps = [tuple(int(b) for b in np.binary_repr(v, K)) if K else () for v in range(2 ** K)]
    state = {c: np.zeros_like(sig) for c in comps}
    state[comps[0]] = sig  # all zeros
    cls = {key: next((m for m, pr in enumerate(preds) if pr(key)), None) for key in fam}
    nsteps = int(np.ceil(pulse.duration / dt))
    h = pulse.duration / nsteps
    n = np.arange(N1)

    def phase(fl, tau):
        E = grid.E(fl)
        ws = E[0, 1] - E[0, 0]
        hk = E[:, :N1] - E[:, :1] - ws * n[None, :]
        ph = np.exp(-2j * np.pi * hk * tau)
        return ph[:, :, None] * ph[:, None, :].conj()

    for s in range(nsteps):
        fl = float(pulse.flux((s + 0.5) * h))
        P = phase(fl, h / 2)
        for c in comps:
            state[c] *= P
        i0, w = grid.weights(fl)
        new = {c: x.copy() for c, x in state.items()}
        for key, C in fam.items():
            op, i, j, dm = key
            c = (1 - w) * C[i0] + w * C[i0 + 1]
            if dm >= 0:
                src, dst = slice(0, N1 - dm), slice(dm, N1)
            else:
                src, dst = slice(-dm, N1), slice(0, N1 + dm)
            cs = c[src]
            g = np.abs(c) ** 2
            decay = h * 0.5 * (g[:, None] + g[None, :])
            m = cls[key]
            for comp in comps:
                x = state[comp]
                new[comp][i] -= decay * x[i]
                if m is None:  # not selected: full dissipator within the component
                    target = comp
                elif comp[m] == 0:  # first jump of class m: move to the component with that bit set
                    target = comp[:m] + (1,) + comp[m + 1:]
                else:  # a second jump of the same class is dropped
                    continue
                new[target][j][dst, dst] += h * cs[:, None] * x[i][src, src] * cs[None, :].conj()
        state = {c: x * P for c, x in new.items()}
    out = {}
    for c, x in state.items():
        x[2] = apply_phase(x[2], phase_corr[1])
        x = apply_phase(x, phase_corr[0])
        out[c] = x if blocks else x.sum(axis=0)
    if multi:
        return out
    if K:
        return out[(0,)], out[(1,)]
    return out[()]


def apply_phase(rho: np.ndarray, delta: float) -> np.ndarray:
    """exp(i delta a^dag a) rho exp(-i delta a^dag a) on the last two (Fock) axes."""
    ph = np.exp(1j * delta * np.arange(rho.shape[-1]))
    return rho * ph[:, None] * ph[None, :].conj()


def calibrate_phases(grid: LadderGrid, pulse: Pulse, nmax: int, nbar: float = 1.0,
                     proj: AsymptoticProjector | None = None, f_target_flipped: bool = False) -> tuple[float, float]:
    """Emulates the experimental CX^2 calibration (Ocelot App. F7, Fig. 15b,c) without noise:
    the storage phase correction delta_g minimizes the bit-flip probability with the ancilla
    in g, then the f-branch correction delta_f (a gate-length fine tuning) with the ancilla
    in f; both at |alpha|^2 = nbar (the experiment uses 1). Removes the linear part of the
    deterministic Kerr distortion; the rest is physical and kept. For a CX (half turn) the
    f-branch target is the flipped state (f_target_flipped=True)."""
    from scipy.optimize import minimize_scalar

    proj = proj or AsymptoticProjector(nbar)
    nob = BathModel({})
    sg = simulate_cx2(grid, pulse, nob, initial_blocks(nbar, 0, nmax), nmax)
    sf = simulate_cx2(grid, pulse, nob, initial_blocks(nbar, 2, nmax), nmax)
    opt = dict(bounds=(-0.5, 0.5), method="bounded", options={"xatol": 1e-6})
    dg = minimize_scalar(lambda d: proj.p_flip(apply_phase(sg, d)), **opt).x
    sign = -1.0 if f_target_flipped else 1.0
    df = minimize_scalar(lambda d: sign * proj.p_flip(apply_phase(sf, d + dg)), **opt).x
    return float(dg), float(df)


def initial_blocks(nbar: float, level: int, nmax: int) -> dict:
    """Ancilla in `level`, storage in |0_L> (truncated to nmax, renormalized)."""
    z, _ = logical_states(nbar, nmax + 1)
    z = z / np.linalg.norm(z)
    return {level: np.outer(z, z.conj())}


def cx2_flip(grid: LadderGrid, pulse: Pulse, bath: BathModel, nbar: float, level: int, nmax: int,
             proj: AsymptoticProjector | None = None, dt: float = 0.5, phase_corr=(0.0, 0.0)) -> float:
    proj = proj or AsymptoticProjector(nbar)
    rho = simulate_cx2(grid, pulse, bath, initial_blocks(nbar, level, nmax), nmax, dt=dt, phase_corr=phase_corr)
    return proj.p_flip(rho / np.trace(rho).real)
