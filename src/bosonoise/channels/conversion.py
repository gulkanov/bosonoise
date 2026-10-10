"""Conversion of ancilla faults into cat bit flips (fault injection through the dissipative map).

A fault during a (chi-matched) CX^2 changes the angle by which the storage state is rotated.
After the gate, two-photon stabilization drives the state back into the cat manifold. The
bit-flip probability of the cycle is the population of the logical state |1_L> after the
stabilization, for a cycle that starts in |0_L>.

The asymptotic map is computed with the asymptotic projector of the Lindbladian
(Albert & Jiang, PRA 89, 022118 (2014); Albert et al., PRX 6, 041031 (2016)):
P_inf(rho) = sum_kl R_k (M^-1)_kl Tr(J_l^dag rho), with R_k the right and J_l the left null
vectors of the Liouvillian (the steady-state manifold of D[a^2 - alpha^2] is 4-dimensional:
the cat qubit), M_kl = Tr(J_k^dag R_l). This avoids integrating the stiff dynamics.
Model: two-photon dissipation with an optional commensurate Kerr-cat Hamiltonian from a
detuned buffer (Ocelot App. F8, Eqs. F5-F6):
    d rho/dt = -i[H_eff, rho] + kappa_2 D[a^2 - alpha^2] rho,
    H_eff = K (a^dag^2 - alpha^2)(a^2 - alpha^2),   K / kappa_2 = -Delta_b / kappa_b,
which leaves the steady states unchanged but makes the dissipative map asymmetric in the
sign of a phase shift. No single-photon loss (with it the steady state is unique and the
flip probability of one stabilization step is not defined by the asymptotic map).

Logical states: |0_L>, |1_L> = (|C+> +- |C->)/sqrt(2) with the normalized even/odd cats
|C+-> ~ |alpha> +- |-alpha>; they are orthonormal and -> |+-alpha> for large |alpha|.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from scipy.special import gammaln


def coherent(alpha: complex, N: int) -> np.ndarray:
    n = np.arange(N)
    amp = np.exp(-0.5 * abs(alpha) ** 2 + n * np.log(abs(alpha) + 1e-300) - 0.5 * gammaln(n + 1)) \
        * np.exp(1j * n * np.angle(alpha))
    return amp / np.linalg.norm(amp)


def logical_states(nbar: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """(|0_L>, |1_L>) in a Fock space of dimension N (alpha = sqrt(nbar) > 0)."""
    c = coherent(np.sqrt(nbar), N)
    even, odd = c.copy(), c.copy()
    even[1::2] = 0.0
    odd[0::2] = 0.0
    even, odd = even / np.linalg.norm(even), odd / np.linalg.norm(odd)
    return (even + odd) / np.sqrt(2), (even - odd) / np.sqrt(2)


def _dissipator(L: sp.csr_matrix, N: int) -> sp.csr_matrix:
    I = sp.identity(N, format="csr")
    LdL = (L.conj().T @ L).tocsr()
    return (sp.kron(L.conj(), L) - 0.5 * sp.kron(I, LdL) - 0.5 * sp.kron(LdL.T, I)).tocsr()


class AsymptoticProjector:
    """Asymptotic projector of D[a^2 - nbar] - i kerr_ratio [(a^dag^2 - nbar)(a^2 - nbar), .]
    (column-stacking vectorization). kerr_ratio = K / kappa_2 = -Delta_b / kappa_b (0: symmetric map).

    eta_detect > 0: the no-click map instead. Each two-photon jump L = a^2 - nbar is one photon leaving the buffer
    (adiabatic buffer); a detector of efficiency eta_detect on the buffer output clicks on a fraction of them
    (Ferrari et al., arXiv:2607.01375). The generator L - eta J, J rho = L rho L^dag, has the same dark code space,
    and its asymptotic map sends rho to the part that relaxed WITHOUT any click: silent_split(rho) gives
    (P(no click, |0_L>), P(no click, |1_L>)), linear in rho, exact (no Poisson assumption). The unconditional map
    (eta_detect = 0) is unchanged by monitoring."""

    def __init__(self, nbar: float, N: int | None = None, kerr_ratio: float = 0.0, eta_detect: float = 0.0):
        self.nbar = nbar
        self.kerr_ratio = kerr_ratio
        self.eta_detect = eta_detect
        self.N = N or int(2 * nbar + 8 * np.sqrt(nbar) + 14)
        N = self.N
        a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
        L = (a @ a - nbar * sp.identity(N, format="csr")).tocsr()
        S = _dissipator(L, N)
        if kerr_ratio:
            H = (L.conj().T @ L).tocsr()
            I = sp.identity(N, format="csr")
            S = S - 1j * kerr_ratio * (sp.kron(I, H) - sp.kron(H.T, I))
        if eta_detect:
            S = S - eta_detect * sp.kron(L.conj(), L)  # remove the detected jumps: no-click evolution
        S = S.tocsc()
        vr, R = eigs(S, k=4, sigma=-1e-7)
        vl, J = eigs(S.conj().T.tocsc(), k=4, sigma=-1e-7)
        self.gap_check = float(np.max(np.abs(np.concatenate([vr, vl]))))
        M = J.conj().T @ R
        self.R = R
        self.W = np.linalg.solve(M, J.conj().T)  # P = R W
        self.zero, self.one = logical_states(nbar, N)

    def asymptotic(self, rho: np.ndarray) -> np.ndarray:
        v = rho.reshape(-1, order="F")
        return (self.R @ (self.W @ v)).reshape(self.N, self.N, order="F")

    def _fit(self, rho: np.ndarray, tail_tol: float = 1e-12) -> np.ndarray:
        """Pad rho to N, or truncate it if its population above N is below tail_tol (e.g. a small cat
        simulated with a large common cutoff); otherwise this is an error."""
        d = rho.shape[0]
        if d < self.N:
            full = np.zeros((self.N, self.N), dtype=complex)
            full[:d, :d] = rho
            rho = full
        elif d > self.N:
            tail = float(np.real(np.trace(rho[self.N:, self.N:])))
            if tail > tail_tol * float(np.real(np.trace(rho))):
                raise ValueError(f"rho larger than the projector's Fock space (population {tail:.1e} above N = {self.N})")
            rho = rho[:self.N, :self.N]
        return rho

    def p_flip(self, rho: np.ndarray, tail_tol: float = 1e-12) -> float:
        """Population of |1_L> after stabilization of the storage state rho (normalized to the code space)."""
        r = self.asymptotic(self._fit(rho, tail_tol))
        p1 = np.real(self.one.conj() @ r @ self.one)
        p0 = np.real(self.zero.conj() @ r @ self.zero)
        return float(p1 / (p0 + p1))

    def click_distribution(self, rho: np.ndarray, kmax: int, tail_tol: float = 1e-12) -> np.ndarray:
        """P(exactly n detected clicks, final |0_L> / |1_L>) for n = 0 .. kmax-1, shape (kmax, 2), relative to Tr rho.

        Requires eta_detect > 0. v_0 = rho, v_{n+1} = eta J (-L_eta)^{-1} (1 - P_eta) v_n (a detected jump during the
        transient; the dark code-space part never clicks), P(n clicks) = <i_L| P_eta v_n |i_L>. Exact and linear in rho."""
        if not self.eta_detect:
            raise ValueError("click_distribution needs eta_detect > 0")
        if not hasattr(self, "_lu"):
            from scipy.sparse.linalg import splu

            N = self.N
            a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
            L = (a @ a - self.nbar * sp.identity(N, format="csr")).tocsr()
            S = _dissipator(L, N)
            if self.kerr_ratio:
                H = (L.conj().T @ L).tocsr()
                I = sp.identity(N, format="csr")
                S = S - 1j * self.kerr_ratio * (sp.kron(I, H) - sp.kron(H.T, I))
            S = S - self.eta_detect * sp.kron(L.conj(), L)
            self._jump = (self.eta_detect * sp.kron(L.conj(), L)).tocsr()
            self._lu = splu((1e-10 * sp.identity(N * N) - S).astype(complex).tocsc())  # (-L_eta)^{-1} off its null space
        v = self._fit(rho, tail_tol).reshape(-1, order="F").astype(complex)
        out = np.zeros((kmax, 2))
        proj = lambda x: self.R @ (self.W @ x)
        for n in range(kmax):
            r = proj(v).reshape(self.N, self.N, order="F")
            out[n] = (np.real(self.zero.conj() @ r @ self.zero), np.real(self.one.conj() @ r @ self.one))
            x = self._lu.solve(v - proj(v))
            v = self._jump @ (x - proj(x))
        return out

    def silent_split(self, rho: np.ndarray, tail_tol: float = 1e-12) -> tuple[float, float]:
        """(P(no click and |0_L>), P(no click and |1_L>)) for the no-click projector (eta_detect > 0), relative to
        Tr rho; for eta_detect = 0 this is the unconditional split."""
        r = self.asymptotic(self._fit(rho, tail_tol))
        return (float(np.real(self.zero.conj() @ r @ self.zero)), float(np.real(self.one.conj() @ r @ self.one)))

    def flip_probability(self, thetas) -> np.ndarray:
        """P(flip) for |0_L> rotated by exp(i theta a^dag a)."""
        n = np.arange(self.N)
        out = []
        for th in np.atleast_1d(thetas):
            psi = np.exp(1j * th * n) * self.zero
            out.append(self.p_flip(np.outer(psi, psi.conj())))
        return np.array(out)

    def eta_uniform(self, theta_max: float, n_theta: int = 181) -> float:
        """Mean P(flip) for theta uniform on [0, theta_max]."""
        if abs(theta_max) < 1e-12:
            return 0.0
        th = np.linspace(0.0, theta_max, n_theta)
        return float(np.trapezoid(self.flip_probability(th), th) / theta_max)
