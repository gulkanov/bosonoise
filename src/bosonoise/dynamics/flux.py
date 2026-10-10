"""Time-dependent coupler flux: pulse shapes and the irrotational-gauge SQUID Hamiltonian.

For an asymmetric dc SQUID with junction energies E_Jl, E_Jr and junction capacitances
C_l, C_r (C = C_l + C_r), the irrotational-gauge Hamiltonian is (You, Sauls, Koch,
PRB 99, 174512 (2019), Eq. 29)

    H_irr = 4 E_C n^2 - E_Jl cos(phi - a phi_e) - E_Jr cos(phi + b phi_e),
    a = C_r / C,  b = C_l / C,  phi_e = 2 pi Phi_e(t) / Phi_0,

which we write as  4 E_C n^2 - A(t) cos(phi) - B(t) sin(phi)  with

    A = E_Jl cos(a phi_e) + E_Jr cos(b phi_e),   B = E_Jl sin(a phi_e) - E_Jr sin(b phi_e).

Statically sqrt(A^2 + B^2) = E_Jmax sqrt(cos^2(phi_e/2) + d^2 sin^2(phi_e/2)) (our static
TunableTransmon), and the sin(phi) term can be gauged away; for time-dependent flux it
cannot, and the irrotational choice removes the spurious n * dPhi/dt coupling.
Default capacitance ratio: C_l/C_r = E_Jl/E_Jr (junction capacitance proportional to area);
this is an assumption to be scanned (not published for Ocelot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def flat_top_gaussian(t, t_start: float, t_flat: float, sigma: float, amp: float, n_sigma: float = 3.0):
    """Flux Phi(t)/Phi_0 of a flat-top pulse with Gaussian edges.

    Rise: amp * exp(-(t - t1)^2 / 2 sigma^2) for t_start <= t < t1 = t_start + n_sigma*sigma,
    flat amp for t1 <= t <= t2 = t1 + t_flat, fall mirrored; 0 outside. The Gaussian edges
    are truncated at n_sigma*sigma and shifted so that the pulse starts exactly at 0.
    """
    t = np.asarray(t, dtype=float)
    t1 = t_start + n_sigma * sigma
    t2 = t1 + t_flat
    t_end = t2 + n_sigma * sigma
    g0 = np.exp(-0.5 * n_sigma ** 2)
    edge = lambda dt: amp * (np.exp(-0.5 * (dt / sigma) ** 2) - g0) / (1 - g0)  # noqa: E731
    out = np.zeros_like(t)
    rise = (t >= t_start) & (t < t1)
    flat = (t >= t1) & (t <= t2)
    fall = (t > t2) & (t <= t_end)
    out[rise] = edge(t[rise] - t1)
    out[flat] = amp
    out[fall] = edge(t[fall] - t2)
    return out


@dataclass(frozen=True)
class IrrotationalSquid:
    EJl: float
    EJr: float
    b: float  # C_l / C ; a = 1 - b

    @classmethod
    def from_tunable(cls, EJmax: float, d: float, cap_ratio: str | float = "area") -> "IrrotationalSquid":
        EJl, EJr = EJmax * (1 + d) / 2, EJmax * (1 - d) / 2
        b = EJl / EJmax if cap_ratio == "area" else float(cap_ratio)
        return cls(EJl, EJr, b)

    def AB(self, flux):
        phi_e = 2 * np.pi * np.asarray(flux, dtype=float)
        a = 1.0 - self.b
        A = self.EJl * np.cos(a * phi_e) + self.EJr * np.cos(self.b * phi_e)
        B = self.EJl * np.sin(a * phi_e) - self.EJr * np.sin(self.b * phi_e)
        return A, B


def charge_basis_ops(ncut: int, ng: float = 0.0):
    """n, cos(phi), sin(phi) in the charge basis n = -ncut..ncut (sin(phi) is imaginary)."""
    n = np.arange(-ncut, ncut + 1, dtype=float)
    up = np.diag(np.ones(2 * ncut), -1)  # e^{i phi}|n> = |n+1>
    cos = 0.5 * (up + up.T)
    sin = (up - up.T) / 2j
    return np.diag(n - ng), cos, sin


def squid_hamiltonian(EC: float, sq: IrrotationalSquid, flux: float, ncut: int = 25, ng: float = 0.0):
    nop, cos, sin = charge_basis_ops(ncut, ng)
    A, B = sq.AB(flux)
    return 4 * EC * nop @ nop - A * cos - B * sin
