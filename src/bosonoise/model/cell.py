"""Storage (x) tunable coupler (x) ancilla-transmon cell, lab frame, no RWA.

H(Phi_c) = f_s a^dag a + (K_s_extra/2) a^dag a^dag a a
         + H_c[Phi_c] + H_q
         + (g_sc n_c + g_sq n_q)(a + a^dag) + g_cq n_c n_q

* H_c, H_q: charge-basis transmons with the full cosine (see model.transmon), represented
  in their own eigenbases ("hierarchical diagonalization"); the numbers of kept levels
  n_c, n_q are convergence parameters.
* g_sc, g_sq, g_cq are the coefficients of the *charge* operators (capacitive couplings,
  flux independent). The familiar "g_01" couplings are g * <0|n|1> (see `g01`).
* K_s_extra: flux-independent storage self-Kerr from spectator modes that are not in this
  reduction (buffer/ATS and its series-inductor modes). It is a spectator term, not a
  property of the storage-coupler-ancilla subsystem.
* Counter-rotating terms are kept: the storage enters through (a + a^dag).

Basis: product states |s, c, q> (storage Fock s, coupler level c, ancilla level q),
optionally truncated by a bare-energy cutoff `e_cut` evaluated with a *flux-independent*
proxy (coupler levels at `ref_flux`), so that the kept basis does not change along a
flux sweep.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from .transmon import Transmon, TunableTransmon


@dataclass(frozen=True)
class CellParams:
    f_s: float  # bare storage frequency (GHz)
    EJ_q: float  # ancilla Josephson energy (GHz)
    EC_q: float  # ancilla charging energy (GHz)
    EJmax_c: float  # coupler EJ1 + EJ2 (GHz)
    EC_c: float  # coupler charging energy (GHz)
    d_c: float  # coupler SQUID asymmetry
    g_sc: float  # storage-coupler charge coupling (GHz)
    g_sq: float  # storage-ancilla direct charge coupling (GHz)
    g_cq: float  # coupler-ancilla charge coupling (GHz)
    ng_q: float = 0.0
    ng_c: float = 0.0
    K_s_extra: float = 0.0  # spectator storage self-Kerr (GHz)

    def ancilla(self, ncut: int = 20) -> Transmon:
        return Transmon(self.EJ_q, self.EC_q, self.ng_q, ncut)

    def coupler(self, ncut: int = 20) -> TunableTransmon:
        return TunableTransmon(self.EJmax_c, self.EC_c, self.d_c, self.ng_c, ncut)

    def with_(self, **kw) -> "CellParams":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def load_cell_params(path, cell: str | None = None) -> CellParams:
    """Load CellParams from a calibration YAML.

    Single-cell files store them under ``params``; multi-cell files (e.g. the joint A4
    calibration) under ``cells[<cell>]``.
    """
    import yaml

    with open(path, encoding="utf-8") as fh:
        rec = yaml.safe_load(fh)
    if "params" in rec:
        return CellParams(**rec["params"])
    if cell is None:
        raise ValueError(f"{path} holds several cells {list(rec['cells'])}; pass cell=")
    return CellParams(**rec["cells"][cell])


@dataclass(frozen=True)
class Truncation:
    n_s: int = 8  # storage Fock states 0 .. n_s-1
    n_c: int = 6  # coupler eigenstates kept
    n_q: int = 8  # ancilla eigenstates kept
    ncut: int = 20  # charge-basis cutoff of each transmon
    e_cut: float | None = None  # bare-energy cutoff (GHz) of the product basis
    ref_flux: float = 0.5  # coupler flux used for the flux-independent cutoff proxy


@dataclass
class ElementSpectra:
    E_c: np.ndarray
    n_c: np.ndarray
    E_q: np.ndarray
    n_q: np.ndarray


@dataclass
class CellBasis:
    labels: np.ndarray  # (N, 3) integer array of (s, c, q)
    index: dict = field(repr=False)  # (s, c, q) -> row

    @property
    def dim(self) -> int:
        return len(self.labels)


class StorageCouplerAncilla:
    """Builder of the static cell Hamiltonian and of basis operators."""

    def __init__(self, params: CellParams, trunc: Truncation = Truncation()):
        self.p = params
        self.t = trunc
        self._ancilla = params.ancilla(trunc.ncut)
        self._E_q, self._n_q = self._ancilla.spectrum(trunc.n_q)
        self._coupler = params.coupler(trunc.ncut)
        self.basis = self._make_basis()
        self._ops_cache: dict = {}

    # ------------------------------------------------------------------ elements
    def elements(self, flux: float) -> ElementSpectra:
        E_c, n_c = self._coupler.at(flux).spectrum(self.t.n_c)
        return ElementSpectra(E_c, n_c, self._E_q, self._n_q)

    def g01(self, flux: float) -> dict:
        """Effective '0-1' couplings (GHz) at a given coupler flux."""
        el = self.elements(flux)
        return dict(
            g_sc01=self.p.g_sc * abs(el.n_c[0, 1]),
            g_sq01=self.p.g_sq * abs(el.n_q[0, 1]),
            g_cq01=self.p.g_cq * abs(el.n_c[0, 1] * el.n_q[0, 1]),
        )

    # ------------------------------------------------------------------ basis
    def _make_basis(self) -> CellBasis:
        t = self.t
        s, c, q = np.meshgrid(np.arange(t.n_s), np.arange(t.n_c), np.arange(t.n_q), indexing="ij")
        labels = np.stack([s.ravel(), c.ravel(), q.ravel()], axis=1)
        if t.e_cut is not None:
            E_c_ref, _ = self._coupler.at(t.ref_flux).spectrum(t.n_c)
            proxy = labels[:, 0] * self.p.f_s + E_c_ref[labels[:, 1]] + self._E_q[labels[:, 2]]
            labels = labels[proxy <= t.e_cut]
        index = {tuple(int(v) for v in row): i for i, row in enumerate(labels)}
        return CellBasis(labels=labels, index=index)

    def _full_to_kept(self) -> np.ndarray:
        t = self.t
        lab = self.basis.labels
        return (lab[:, 0] * t.n_c + lab[:, 1]) * t.n_q + lab[:, 2]

    # ------------------------------------------------------------------ operators
    def _storage_ops(self):
        n = self.t.n_s
        adag = sp.diags(np.sqrt(np.arange(1, n)), -1, shape=(n, n), format="csr")
        return adag

    def op(self, name: str) -> sp.csr_matrix:
        """Operators in the kept basis: 'adag', 'a', 'X_s', 'N_s', 'N_c', 'N_q'."""
        if name in self._ops_cache:
            return self._ops_cache[name]
        lab = self.basis.labels
        if name in ("N_s", "N_c", "N_q"):
            col = {"N_s": 0, "N_c": 1, "N_q": 2}[name]
            M = sp.diags(lab[:, col].astype(float), format="csr")
        else:
            t = self.t
            adag = self._storage_ops()
            Ic, Iq = sp.identity(t.n_c, format="csr"), sp.identity(t.n_q, format="csr")
            full = {"adag": adag, "a": adag.T.tocsr(), "X_s": (adag + adag.T).tocsr()}[name]
            full = sp.kron(sp.kron(full, Ic), Iq, format="csr")
            keep = self._full_to_kept()
            M = full[keep][:, keep].tocsr()
        self._ops_cache[name] = M
        return M

    def charge_ops(self, flux: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Charge operators n_c (flux dependent) and n_q in the kept basis."""
        t = self.t
        el = self.elements(flux)
        Is = sp.identity(t.n_s, format="csr")
        Ic, Iq = sp.identity(t.n_c, format="csr"), sp.identity(t.n_q, format="csr")
        keep = self._full_to_kept()
        nc = sp.kron(sp.kron(Is, sp.csr_matrix(el.n_c)), Iq, format="csr")[keep][:, keep]
        nq = sp.kron(sp.kron(Is, Ic), sp.csr_matrix(el.n_q), format="csr")[keep][:, keep]
        return nc.tocsr(), nq.tocsr()

    # ------------------------------------------------------------------ Hamiltonian
    def hamiltonian(self, flux: float, dense: bool = True):
        t, p = self.t, self.p
        el = self.elements(flux)
        lab = self.basis.labels
        s = lab[:, 0].astype(float)
        diag = (
            p.f_s * s
            + 0.5 * p.K_s_extra * s * (s - 1.0)
            + el.E_c[lab[:, 1]]
            + el.E_q[lab[:, 2]]
        )
        adag = self._storage_ops()
        X = (adag + adag.T).tocsr()
        Is = sp.identity(t.n_s, format="csr")
        Ic, Iq = sp.identity(t.n_c, format="csr"), sp.identity(t.n_q, format="csr")
        nc = sp.csr_matrix(el.n_c)
        nq = sp.csr_matrix(el.n_q)
        V = (
            p.g_sc * sp.kron(sp.kron(X, nc), Iq)
            + p.g_sq * sp.kron(sp.kron(X, Ic), nq)
            + p.g_cq * sp.kron(sp.kron(Is, nc), nq)
        ).tocsr()
        keep = self._full_to_kept()
        V = V[keep][:, keep]
        H = sp.diags(diag) + V
        if dense:
            H = H.toarray()
            H = 0.5 * (H + H.T)
        return H

    def eigh(self, flux: float, subset_by_index=None):
        """Dressed energies (relative to the dressed ground state) and eigenvectors."""
        H = self.hamiltonian(flux, dense=True)
        E, V = sla.eigh(H, subset_by_index=subset_by_index, driver="evr" if subset_by_index else "evd")
        return E - E[0], V

    # ------------------------------------------------------------------ labelling
    def row(self, s: int, c: int, q: int) -> int:
        return self.basis.index[(s, c, q)]

    def dressed_index(self, V: np.ndarray, s: int, c: int, q: int, min_weight: float = 0.5) -> int:
        """Index of the dressed state with maximal weight on the bare state |s,c,q>."""
        w = V[self.row(s, c, q), :] ** 2
        k = int(np.argmax(w))
        if w[k] < min_weight:
            raise RuntimeError(
                f"bare state |s={s},c={c},q={q}> is strongly hybridized (max weight {w[k]:.3f})"
            )
        return k
