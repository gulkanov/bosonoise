"""Universal Lindblad equation (ULE) jump operators and bath rate functions.

Nathan & Rudner, PRB 102, 115109 (2020): for a system H_S = sum_n E_n |n><n| coupled to a
bath through H_int = X (x) B, the ULE jump operator is

    L = sum_{m,n} sqrt(Gamma(E_n - E_m)) X_mn |m><n|,       Gamma(w) = 2 pi J(w),

with J the bath spectral function; w > 0 is energy released to the bath, and the golden-rule
rate of n -> m is Gamma(w_nm) |X_mn|^2. Unlike the secular (Pauli) master equation, L keeps
the coherence between transitions whose frequencies differ by less than the scale on which
Gamma varies (for a cat: the same ancilla transition at different storage photon numbers),
and the equation is of Lindblad form. The Lamb shift is neglected.

Units: frequencies in GHz (cyclic), time in ns, rates in 1/ns. Temperatures in mK.
Rate functions are callables Gamma(w) acting elementwise on arrays of w (GHz).
"""
from __future__ import annotations

import numpy as np

H_OVER_KB_MK = 47.992430  # h * 1 GHz / k_B in mK


def jump_operator(E: np.ndarray, X: np.ndarray, rate) -> np.ndarray:
    """ULE jump operator in the eigenbasis of H_S (energies E, GHz; X in the same basis)."""
    W = E[None, :] - E[:, None]  # W[m, n] = E_n - E_m
    return np.sqrt(np.clip(rate(W), 0.0, None)) * X


def lindblad_rhs(H: np.ndarray, Ls, rho: np.ndarray) -> np.ndarray:
    """d rho / dt (1/ns) for H in GHz."""
    out = -2j * np.pi * (H @ rho - rho @ H)
    for L in Ls:
        LdL = L.conj().T @ L
        out += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return out


# ---------------------------------------------------------------- rate functions
def detailed_balance(emission, T_mK: float | None):
    """Two-sided rate: Gamma(w) = emission(|w|) (w >= 0), exp(-h|w|/k_B T) emission(|w|) (w < 0).

    T_mK = None or 0: no absorption.
    """

    def rate(w):
        w = np.asarray(w, dtype=float)
        g = emission(np.abs(w))
        if not T_mK:
            return np.where(w >= 0, g, 0.0)
        return np.where(w >= 0, g, g * np.exp(-np.abs(w) * H_OVER_KB_MK / T_mK))

    return rate


def flat(gamma: float):
    """Frequency-independent emission rate function (use with detailed_balance)."""
    return lambda w: np.full(np.shape(w), float(gamma))


def piecewise_flat(edges, gammas):
    """Emission rate gammas[i] on [edges[i-1], edges[i]) in |w| (edges ascending, len = len(gammas) - 1)."""
    edges, gammas = np.asarray(edges, float), np.asarray(gammas, float)
    return lambda w: gammas[np.searchsorted(edges, np.abs(np.asarray(w, float)), side="right")]


def power_law(gamma_ref: float, f_ref: float, s: float):
    """Gamma(|w|) = gamma_ref (|w| / f_ref)^s (s = 1: Ohmic; s = 0: flat; s = -1: 1/f)."""
    return lambda w: gamma_ref * (np.maximum(np.abs(np.asarray(w, float)), 1e-12) / f_ref) ** s


def classical_dephasing(gamma_phi: float, shape: str = "white", f_ref: float = 1e-3, f_cut: float = 20.0):
    """Symmetric (classical) frequency-noise rate function for an operator normalized so that
    <e|O|e> - <g|O|g> = 1 (e.g. the ancilla excitation number).

    Pure dephasing of the g-e coherence: Gamma_phi = Gamma(0) / 2.
      white: Gamma(w) = 2 gamma_phi for |w| < f_cut (upper bound for a noise that sets T_phi);
      1/f:   Gamma(w) = 2 gamma_phi * f_ref / |w| (normalized at f_ref, GHz).
    """
    if shape == "white":
        return lambda w: np.where(np.abs(np.asarray(w, float)) < f_cut, 2.0 * gamma_phi, 0.0)
    if shape == "1/f":
        return lambda w: 2.0 * gamma_phi * f_ref / np.maximum(np.abs(np.asarray(w, float)), f_ref)
    raise ValueError(shape)
