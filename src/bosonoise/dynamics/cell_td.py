"""Storage-coupler-ancilla cell with a time-dependent coupler flux (irrotational gauge).

H(t) = H_fixed - A(Phi(t)) * cos(phi_c) - B(Phi(t)) * sin(phi_c)

The coupler is represented in a *fixed* eigenbasis (the eigenstates of its irrotational
Hamiltonian at a reference flux), truncated to n_c levels; n_c, cos(phi_c), sin(phi_c) and
n_c^2 are projected on that basis. The truncation must be converged over the whole flux
trajectory (see tests). Storage and ancilla are as in model.cell (lab frame, no RWA).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from ..model.cell import CellParams, StorageCouplerAncilla, Truncation
from .flux import IrrotationalSquid, charge_basis_ops


class TimeDependentCell:
    def __init__(self, params: CellParams, trunc: Truncation, flux_ref: float = 0.5, cap_ratio="area"):
        self.p, self.t = params, trunc
        self.sq = IrrotationalSquid.from_tunable(params.EJmax_c, params.d_c, cap_ratio)
        # reuse the static builder for the kept basis and the ancilla/storage operators
        self.static = StorageCouplerAncilla(params, Truncation(**{**vars(trunc), "ref_flux": flux_ref}))
        nop, cos, sin = charge_basis_ops(trunc.ncut, params.ng_c)
        A, B = self.sq.AB(flux_ref)
        Hc = 4 * params.EC_c * nop @ nop - A * cos - B * sin
        E, V = np.linalg.eigh(Hc)
        V = V[:, : trunc.n_c]
        self.Ec_ref = E[: trunc.n_c] - E[0]
        proj = lambda M: V.conj().T @ M @ V  # noqa: E731
        self.kin_c = proj(4 * params.EC_c * nop @ nop)
        self.cos_c, self.sin_c, self.n_c = proj(cos), proj(sin), proj(nop)
        self._build(E[0])

    def _kept(self, full):
        keep = self.static._full_to_kept()
        return full[keep][:, keep]

    def _build(self, E0c):
        t, p = self.t, self.p
        Is, Iq = sp.identity(t.n_s, format="csr"), sp.identity(t.n_q, format="csr")
        adag = sp.diags(np.sqrt(np.arange(1, t.n_s)), -1, shape=(t.n_s, t.n_s), format="csr")
        X = (adag + adag.T).tocsr()
        s = np.arange(t.n_s, dtype=float)
        Hs = sp.diags(p.f_s * s + 0.5 * p.K_s_extra * s * (s - 1))
        E_q, n_q = self.static._E_q, self.static._n_q
        Ic = sp.identity(t.n_c, format="csr")
        kin = sp.csr_matrix(self.kin_c - E0c * np.eye(t.n_c))
        nc, nq = sp.csr_matrix(self.n_c), sp.csr_matrix(n_q)
        H = (sp.kron(sp.kron(Hs, Ic), Iq) + sp.kron(sp.kron(Is, kin), Iq) + sp.kron(sp.kron(Is, Ic), sp.diags(E_q))
             + p.g_sc * sp.kron(sp.kron(X, nc), Iq) + p.g_sq * sp.kron(sp.kron(X, Ic), nq)
             + p.g_cq * sp.kron(sp.kron(Is, nc), nq))
        self.H_fixed = self._kept(H.tocsr()).tocsr()
        self.C_op = self._kept(sp.kron(sp.kron(Is, sp.csr_matrix(self.cos_c)), Iq).tocsr()).tocsr()
        self.S_op = self._kept(sp.kron(sp.kron(Is, sp.csr_matrix(self.sin_c)), Iq).tocsr()).tocsr()

    @property
    def dim(self) -> int:
        return self.H_fixed.shape[0]

    def H(self, flux: float) -> sp.csr_matrix:
        A, B = self.sq.AB(flux)
        return (self.H_fixed - A * self.C_op - B * self.S_op).tocsr()

    def eigh(self, flux: float):
        E, V = np.linalg.eigh(self.H(flux).toarray())
        return E - E[0], V

    def _commutators(self):
        if not hasattr(self, "_K"):
            Hf, C, S = self.H_fixed, self.C_op, self.S_op
            self._K = ((Hf @ C - C @ Hf).tocsr(), (Hf @ S - S @ Hf).tocsr(), (C @ S - S @ C).tocsr())
        return self._K

    def propagate(self, psi0: np.ndarray, flux_of_t, t_grid: np.ndarray, E_shift: float = 0.0,
                  method: str = "midpoint") -> np.ndarray:
        """Propagate psi along t_grid (ns); H in GHz, phase 2*pi*H*t.

        method="magnus4": 4th-order Magnus step with two Gauss-Legendre nodes,
            Omega = -i w (dt/2)(H1 + H2) - (sqrt(3)/12) w^2 dt^2 [H2, H1],  w = 2 pi,
            where [H2, H1] = (A2-A1)[Hf,C] + (B2-B1)[Hf,S] + (A2 B1 - A1 B2)[C,S] uses
            commutators precomputed once (H = Hf - A C - B S).
        method="midpoint": piecewise-constant H at the step midpoint (2nd order).
        WARNING (tested 2026-09-29): in the lab frame ||H|| dt >> 1, so neither scheme is
        controlled unless dt resolves the fastest transition coupled by the flux (coupler
        0->1 ~ 9 GHz, 0->2 ~ 18 GHz). dt = 0.05-0.1 ns gave a spurious, method-independent
        1.7e-5 excitation (stroboscopic aliasing); dt = 0.0125 ns gave 7e-12. Use
        dt <~ 0.0125 ns, or adiabatic-frame / adiabatic-perturbation methods for production.
        E_shift (GHz) is subtracted from H (global phase only) to reduce the Krylov cost.
        Returns psi at every grid point, shape (len(t_grid), dim).
        """
        psi = np.asarray(psi0, dtype=complex)
        out = np.empty((len(t_grid), self.dim), dtype=complex)
        out[0] = psi
        I = sp.identity(self.dim, format="csr")
        w = 2 * np.pi
        c = np.sqrt(3) / 6
        if method == "magnus4":
            K1, K2, K3 = self._commutators()
        for k in range(1, len(t_grid)):
            t0, dt = t_grid[k - 1], t_grid[k] - t_grid[k - 1]
            if method == "midpoint":
                G = (-1j * w * dt) * (self.H(flux_of_t(t0 + 0.5 * dt)) - E_shift * I)
            elif method == "magnus4":
                A1, B1 = self.sq.AB(flux_of_t(t0 + (0.5 - c) * dt))
                A2, B2 = self.sq.AB(flux_of_t(t0 + (0.5 + c) * dt))
                Hsum = 2 * (self.H_fixed - E_shift * I) - (A1 + A2) * self.C_op - (B1 + B2) * self.S_op
                comm = (A2 - A1) * K1 + (B2 - B1) * K2 + (A2 * B1 - A1 * B2) * K3
                G = (-1j * w * 0.5 * dt) * Hsum - (np.sqrt(3) / 12) * (w * dt) ** 2 * comm
            else:
                raise ValueError(method)
            psi = expm_multiply(G.tocsr(), psi)
            out[k] = psi
        return out
