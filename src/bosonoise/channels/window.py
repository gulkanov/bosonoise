"""Stabilization windows of a dissipative cat qubit: pulsed two-photon stabilization between gates (paper App. J).

The storage alone, buffer adiabatically eliminated. While stabilization is on,
    L_on  = kappa2 ( D[a^2 - nbar] - i kerr_ratio [(a^dag^2 - nbar)(a^2 - nbar), .] ) + L_idle,
    L_idle = kappa1 (1 + nth) D[a] + kappa1 nth D[a^dag] + kphi D[a^dag a] - i [(K/2) a^dag^2 a^2, .],
and while it is off (gates) only L_idle plus the deterministic phases of the gate. Rates in 1/us, K in rad/us.
The pump degrades the storage's own loss and dephasing (Carde et al., PR Applied 23, 024073 (2025); Ocelot Fig. 9e-h:
T1, T2 with and without two-photon dissipation), so the off-period idle rates can be set separately (kappa1_off,
kphi_off; default: the same as with the stabilization on).
Optionally (kappa_b given) the stabilization is the exact adiabatic elimination of a buffer with detuning delta_b and
cross-Kerr chi_sb a^dag a b^dag b (stabilization_superop): with A = a^2 - nbar and G = diag 1/(delta_b + chi_sb n -
i kappa_b/2) over the storage photon number n while the buffer photon exists,
    d rho/dt = -i (H_NH rho - rho H_NH^dag) + J rho J^dag,  H_NH = -g2^2 A^dag G A,  J = sqrt(kappa_b) g2 G A,
g2^2 = kappa2 kappa_b / 4. It is trace preserving, reduces to kappa2 D[A] at delta_b = chi_sb = 0 and to the
kerr_ratio = -delta_b/kappa_b form to first order in delta_b/kappa_b; chi_sb distorts the jump into
(1 - 2i chi_sb n/kappa_b)(a^2 - nbar) to first order (the cross-Kerr dephasing of Putterman et al., PRX 15, 011070
(2025), App. G, where the same non-perturbative G is used).
All terms commute with the rotation by pi (a -> -a), so the Liouvillian is block diagonal in the parity of m - n of
the matrix elements rho_mn. The cat's Z mode, |alpha><alpha| - |-alpha><-alpha|, lives in the odd sector (m - n odd):
the leading eigenvalue mu of the ODD-sector cycle map is the per-cycle decay of <Z>, so the Pauli bit-flip
probability per cycle is (1 - mu)/2 exactly, including any incomplete re-stabilization carried over to the next
cycle (which an asymptotic projector applied after each gate would miss).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import minimize_scalar


def _dissipator(L: sp.csr_matrix, N: int) -> sp.csr_matrix:
    I = sp.identity(N, format="csr")
    LdL = (L.conj().T @ L).tocsr()
    return (sp.kron(L.conj(), L) - 0.5 * sp.kron(I, LdL) - 0.5 * sp.kron(LdL.T, I)).tocsr()


def _commutator(H: sp.csr_matrix, N: int) -> sp.csr_matrix:
    """Superoperator of -i[H, .] (column stacking)."""
    I = sp.identity(N, format="csr")
    return (-1j * (sp.kron(I, H) - sp.kron(H.T, I))).tocsr()


def stabilization_superop(N: int, nbar: float, kappa2: float, kerr_ratio: float = 0.0, kappa_b: float | None = None,
                          chi_sb: float = 0.0, delta_b: float = 0.0) -> sp.csr_matrix:
    """Full (all sectors) superoperator of the two-photon stabilization, rates included (column stacking).
    kappa_b None: kappa2 (D[A] - i kerr_ratio [A^dag A, .]); otherwise the exact buffer elimination (module docstring),
    kappa_b, chi_sb, delta_b in rad/us like kappa2."""
    a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
    A = (a @ a - nbar * sp.identity(N, format="csr")).tocsr()
    if kappa_b is None:
        S = _dissipator(A, N)
        if kerr_ratio:
            S = S + kerr_ratio * _commutator((A.conj().T @ A).tocsr(), N)
        return (kappa2 * S).tocsr()
    G = sp.diags(1.0 / (delta_b + chi_sb * np.arange(N) - 0.5j * kappa_b), format="csr")
    g2sq = kappa2 * kappa_b / 4.0
    H = (-g2sq * (A.conj().T @ G @ A)).tocsr()
    J = (np.sqrt(kappa_b * g2sq) * (G @ A)).tocsr()
    I = sp.identity(N, format="csr")
    return (-1j * (sp.kron(I, H) - sp.kron(H.conj(), I)) + sp.kron(J.conj(), J)).tocsr()


def odd_indices(N: int) -> np.ndarray:
    """Column-stacked indices k = m + N n of the elements rho_mn with m - n odd."""
    m, n = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.flatnonzero(((m - n) % 2 != 0).reshape(-1, order="F"))


class CatWindow:
    """Liouvillians of one storage mode (Fock cutoff N) restricted to the odd sector."""

    def __init__(self, nbar: float, N: int | None = None, kappa2: float = 0.17, kerr_ratio: float = 0.0,
                 kappa1: float = 1 / 60, nth: float = 0.0, kphi: float = 0.0, K: float = 0.0,
                 kappa1_off: float | None = None, kphi_off: float | None = None, kappa_b: float | None = None,
                 chi_sb: float = 0.0, delta_b: float = 0.0):
        self.nbar, self.N = nbar, N or int(2 * nbar + 8 * np.sqrt(nbar) + 14)
        N = self.N
        a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
        ad = a.T.tocsr()
        num = (ad @ a).tocsr()
        stab = stabilization_superop(N, nbar, kappa2, kerr_ratio, kappa_b, chi_sb, delta_b)

        def idle(k1, kp):
            L = k1 * (1 + nth) * _dissipator(a, N) + k1 * nth * _dissipator(ad, N) + kp * _dissipator(num, N)
            return L + _commutator(0.5 * K * (ad @ ad @ a @ a).tocsr(), N) if K else L

        self.idx = odd_indices(N)
        sel = lambda S: S.tocsr()[self.idx][:, self.idx].toarray()
        self.L_stab = sel(stab)
        self.L_idle = sel(idle(kappa1, kphi))
        if kappa1_off is None and kphi_off is None:
            self.L_idle_off = self.L_idle
        else:
            self.L_idle_off = sel(idle(kappa1 if kappa1_off is None else kappa1_off, kphi if kphi_off is None else kphi_off))
        m, n = np.divmod(self.idx, N)[::-1]  # k = m + N n  ->  m = k % N, n = k // N
        self.m, self.n = m, n

    def on(self, T: float, s: float = 1.0) -> np.ndarray:
        """Odd-sector propagator for time T (us) with the stabilization at relative strength s^2 (kappa2 ~ g2^2)."""
        return la.expm((s * s * self.L_stab + self.L_idle) * T)

    def off(self, T: float) -> np.ndarray:
        """Odd-sector propagator for time T (us) with the stabilization off (off-period idle rates)."""
        return la.expm(self.L_idle_off * T)

    def phases(self, phi: np.ndarray) -> np.ndarray:
        """Diagonal map rho_mn -> exp(-i (phi_m - phi_n)) rho_mn for a phase vector phi over Fock states (len >= N)."""
        return np.diag(np.exp(-1j * (phi[self.m] - phi[self.n])))

    @staticmethod
    def bitflip_per_cycle(M: np.ndarray) -> float:
        """Pauli bit-flip probability per cycle from the odd-sector cycle map M: (1 - mu_max)/2."""
        ev = la.eigvals(M)
        mu = ev[np.argmax(np.abs(ev))]
        return float(0.5 * (1.0 - mu.real))

    def static_TX(self) -> float:
        """Continuous stabilization: T_X (us) as AWS define it, the inverse Pauli rate, from the slowest odd mode."""
        ev = la.eigvals(self.L_stab + self.L_idle)
        lam = ev[np.argmax(ev.real)].real  # <Z> ~ exp(lam t) = exp(-2 Gamma t)
        scale = float(np.max(np.abs(ev)))
        return float(-2.0 / lam) if lam < -1e-12 * scale else float("inf")  # noise-free: Z_L conserved

    def pulsed_bitflip(self, T_on: float, T_off: float, kerr_rotation: float = 0.0, track: str = "mean"):
        """Per-cycle Pauli bit flip of the repeated cycle 'off T_off, then on T_on', the state rotated back by
        f * kerr_rotation (rad; the mean Kerr rotation of the coherent state during T_off, K nbar T_off) just before the
        stabilization returns. track: 'mean' (f = 1, aligned with the coherent state), 'none' (f = 0) or 'optimal'
        (f in [-0.5, 1.5] minimizing the flip: for small rotations f ~ 0.1, because the flip is set by phase errors of
        the lowest Fock coherences, which the Kerr term barely rotates). Returns (p, f)."""
        M_on, M_off = self.on(T_on), self.off(T_off)
        dmn = (self.m - self.n).astype(float)

        def p_of(f):
            d = np.exp(1j * f * kerr_rotation * dmn)
            op = sla.LinearOperator(M_on.shape, matvec=lambda v: M_on @ (d * (M_off @ v)), dtype=complex)
            ev = sla.eigs(op, k=3, which="LM", return_eigenvectors=False, tol=1e-13)
            return float(0.5 * (1.0 - ev[np.argmax(np.abs(ev))].real))

        if track == "optimal":
            f = minimize_scalar(lambda f: np.log(max(p_of(f), 1e-300)), bounds=(-0.5, 1.5), method="bounded",
                                options={"xatol": 0.01}).x
        else:
            f = {"mean": 1.0, "none": 0.0}[track]
        return p_of(f), f
