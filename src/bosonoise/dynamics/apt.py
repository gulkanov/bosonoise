"""First-order adiabatic perturbation theory (APT) for flux pulses on the coupler.

For H(Phi) (GHz) and a flux pulse Phi(t), the amplitude to leave the adiabatically followed
state |0(t)> into |k(t)> is, to first order,

    c_k(T) = - int_0^T dt  Phidot(t) * m_k(Phi(t)) * exp(i 2 pi int_0^t [E_k - E_0] dt'),
    m_k(Phi) = <k| dH/dPhi |0> / (E_0 - E_k),

with dH/dPhi = -A'(Phi) cos(phi_c) - B'(Phi) sin(phi_c) in the irrotational gauge
(TimeDependentCell). Eigenstates are computed on a flux grid, followed by maximal overlap
between neighbouring grid points (which follows the *diabatic* state through avoided
crossings narrower than the grid resolution; their Landau-Zener probabilities are treated
separately), and phases are fixed by parallel transport (<v(Phi_j)|v(Phi_{j+1})> > 0).
Valid when all |c_k|^2 << 1 and no crossing of state 0 with a strongly coupled state is
resolved by the grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from .cell_td import TimeDependentCell


@dataclass
class AdiabaticPath:
    fluxes: np.ndarray
    E0: np.ndarray  # (nflux,) energy of the followed state (GHz)
    Ek: np.ndarray  # (nflux, K) energies of followed targets
    mk: np.ndarray  # (nflux, K) complex, <k|dH/dPhi|0>/(E0-Ek)  (1/Phi0)
    targets0: np.ndarray  # (K,) eigen-indices of the targets at fluxes[0]
    overlap_min: float  # smallest step-to-step overlap of state 0 (diagnostic)


def _dH_dPhi(td: TimeDependentCell, flux: float, h: float = 1e-6):
    A1, B1 = td.sq.AB(flux - h)
    A2, B2 = td.sq.AB(flux + h)
    dA, dB = (A2 - A1) / (2 * h), (B2 - B1) / (2 * h)
    return (-dA * td.C_op - dB * td.S_op).tocsr()


def adiabatic_path(td: TimeDependentCell, fluxes: np.ndarray, index0: int, n_targets: int = 10) -> AdiabaticPath:
    fluxes = np.asarray(fluxes, float)
    Es, Vs = [], []
    for fl in fluxes:
        E, V = np.linalg.eigh(td.H(fl).toarray())
        Es.append(E)
        Vs.append(V)
    N = len(Es[0])
    # follow all states by greedy maximal overlap, with parallel-transport phases
    perm = [np.arange(N)]
    Vf = [Vs[0]]
    min_ov0 = 1.0
    for j in range(1, len(fluxes)):
        O = np.abs(Vf[-1].conj().T @ Vs[j])  # rows: previous followed states, cols: new eigenstates
        p = -np.ones(N, int)
        taken = np.zeros(N, bool)
        for r in np.argsort(-O.max(axis=1)):
            for c in np.argsort(-O[r]):
                if not taken[c]:
                    p[r], taken[c] = c, True
                    break
        Vn = Vs[j][:, p]
        ph = np.einsum("ij,ij->j", Vf[-1].conj(), Vn)
        Vn = Vn * (np.conj(ph) / np.maximum(np.abs(ph), 1e-300))
        min_ov0 = min(min_ov0, float(abs(ph[index0])))
        perm.append(p)
        Vf.append(Vn)
    Ef = [Es[j][perm[j]] for j in range(len(fluxes))]
    # matrix elements at every grid point
    m_all = np.zeros((len(fluxes), N), complex)
    for j, fl in enumerate(fluxes):
        dH = _dH_dPhi(td, fl)
        v0 = Vf[j][:, index0]
        num = Vf[j].conj().T @ (dH @ v0)
        den = Ef[j][index0] - Ef[j]
        den[index0] = np.inf
        m_all[j] = num / den
    score = np.max(np.abs(m_all), axis=0)
    score[index0] = -1
    targets = np.argsort(-score)[:n_targets]
    return AdiabaticPath(fluxes=fluxes, E0=np.array([e[index0] for e in Ef]), Ek=np.array([e[targets] for e in Ef]),
                         mk=m_all[:, targets], targets0=targets, overlap_min=min_ov0)


def apt_amplitudes(path: AdiabaticPath, flux_of_t, t: np.ndarray) -> np.ndarray:
    """First-order amplitudes c_k(T) for the followed targets along the pulse Phi(t) (t in ns)."""
    fl_t = np.asarray(flux_of_t(t), float)
    order = np.argsort(path.fluxes)
    x = path.fluxes[order]
    phidot = np.gradient(fl_t, t)
    out = np.zeros(path.mk.shape[1], complex)
    E0 = CubicSpline(x, path.E0[order])(fl_t)
    for k in range(path.mk.shape[1]):
        Ek = CubicSpline(x, path.Ek[order, k])(fl_t)
        mre = CubicSpline(x, path.mk[order, k].real)(fl_t)
        mim = CubicSpline(x, path.mk[order, k].imag)(fl_t)
        w = 2 * np.pi * (Ek - E0)
        theta = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
        integrand = phidot * (mre + 1j * mim) * np.exp(1j * theta)
        out[k] = -np.trapezoid(integrand, t)
    return out
