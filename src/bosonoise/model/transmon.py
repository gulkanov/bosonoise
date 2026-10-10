"""Charge-basis transmon models with the full cosine potential.

H = 4 E_C (n - n_g)^2 - E_J cos(phi),   n in {-ncut, ..., ncut}

In the charge basis cos(phi) = (|n><n+1| + h.c.)/2, so H is real tridiagonal and is
diagonalized exactly (no Duffing / phi^4 truncation, no few-level truncation of the
bare element: the number of kept eigenstates is a convergence parameter).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import eigh_tridiagonal


def _fix_signs(vecs: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Deterministic eigenvector sign convention (continuity of matrix elements in flux).

    The sign of every eigenvector is chosen such that its overlap with a fixed, generic,
    non-parity-symmetric weight vector w(n) = exp(0.1 n) is positive.
    """
    w = np.exp(0.1 * charges)
    s = np.sign(w @ vecs)
    s[s == 0] = 1.0
    return vecs * s


@dataclass(frozen=True)
class Transmon:
    """Fixed-frequency transmon (or a tunable one at fixed flux)."""

    EJ: float  # Josephson energy (GHz)
    EC: float  # charging energy e^2/2C (GHz)
    ng: float = 0.0  # offset charge (units of 2e)
    ncut: int = 20  # charge-basis cutoff: n in [-ncut, ncut]

    @property
    def charges(self) -> np.ndarray:
        return np.arange(-self.ncut, self.ncut + 1, dtype=float)

    def eigensys(self, nlev: int) -> tuple[np.ndarray, np.ndarray]:
        """Lowest `nlev` eigenvalues (GHz) and charge-basis eigenvectors (columns)."""
        if nlev > 2 * self.ncut + 1:
            raise ValueError("nlev exceeds charge-basis dimension")
        n = self.charges
        diag = 4.0 * self.EC * (n - self.ng) ** 2
        off = np.full(len(n) - 1, -0.5 * self.EJ)
        E, V = eigh_tridiagonal(diag, off, select="i", select_range=(0, nlev - 1))
        return E, _fix_signs(V, n)

    def spectrum(self, nlev: int) -> tuple[np.ndarray, np.ndarray]:
        """Energies relative to the ground state and the charge operator <i|n|j>."""
        E, V = self.eigensys(nlev)
        nmat = V.T @ (self.charges[:, None] * V)
        return E - E[0], nmat

    def f01(self) -> float:
        E, _ = self.eigensys(2)
        return float(E[1] - E[0])

    def anharmonicity(self) -> float:
        E, _ = self.eigensys(3)
        return float((E[2] - E[1]) - (E[1] - E[0]))


@dataclass(frozen=True)
class TunableTransmon:
    """Flux-tunable transmon with an asymmetric SQUID.

    E_J(Phi) = EJmax * sqrt(cos^2(pi Phi/Phi0) + d^2 sin^2(pi Phi/Phi0)),  d = (EJ1-EJ2)/(EJ1+EJ2).

    For *static* flux the SQUID phase offset is gauged away. For time-dependent flux the
    irrotational-gauge treatment (You, Sauls, Koch, PRB 99, 174512 (2019)) must be used;
    that is implemented in the dynamics module, not here.
    """

    EJmax: float
    EC: float
    d: float
    ng: float = 0.0
    ncut: int = 20

    def EJ(self, flux: float) -> float:
        x = np.pi * flux
        return float(self.EJmax * np.sqrt(np.cos(x) ** 2 + self.d ** 2 * np.sin(x) ** 2))

    def at(self, flux: float) -> Transmon:
        return Transmon(EJ=self.EJ(flux), EC=self.EC, ng=self.ng, ncut=self.ncut)

    def with_(self, **kw) -> "TunableTransmon":
        return replace(self, **kw)
