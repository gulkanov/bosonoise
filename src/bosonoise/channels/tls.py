"""TLS ensembles and bath-induced ancilla transitions in the dressed basis.

Physics
-------
The ancilla couples to its environment through its charge n_q. The environment is
  * a smooth background, taken flat over the ~0.5 GHz window around the ancilla transitions
    and referenced per transition family to the measured coupler-idle T1 values
    (T1_ge for e->g; T1_fe for f->e and for the dressing-opened f->g channel);
  * discrete two-level defects (TLS) j with frequency f_j, coupling g01_j (defined as the
    coupling to the *bare* ancilla 0<->1 charge element, so that a dressed channel with
    element M couples with V = g01_j |M| / |<0|n_q|1>|) and linewidth gamma_j (FWHM,
    angular units: the TLS state decays at gamma_j).

For each dressed channel |i~,n> -> |j~,m> (frequency f_c) and each TLS the transfer is
computed exactly in the two-state model {|i~,n; 0_T>, |j~,m; 1_T>} with the second state
decaying at gamma_j (weak coupling -> golden rule with a Lorentzian; strong coupling ->
damped vacuum-Rabi exchange). Channels and TLS are combined as independent decay
processes (multiplicative survival, proportional branching). Absorption channels (f_c < 0)
are neglected (T -> 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson

from ..spectra.transitions import TransitionTable

TWO_PI = 2 * np.pi


@dataclass(frozen=True)
class TLSEnsemble:
    f: np.ndarray  # GHz
    g01: np.ndarray  # GHz
    gamma: np.ndarray  # 1/ns (angular FWHM)

    @property
    def size(self) -> int:
        return len(self.f)


def sample_tls(rng: np.random.Generator, density_per_GHz: float, f_window=(4.8, 5.6),
               g01_range=(2e-5, 2e-3), gamma_range=(1e-4, 1e-2)) -> TLSEnsemble:
    """Poisson number of TLS uniform in f_window; g01 and gamma log-uniform in their ranges (GHz, 1/ns)."""
    N = rng.poisson(density_per_GHz * (f_window[1] - f_window[0]))
    f = rng.uniform(*f_window, N)
    g = np.exp(rng.uniform(np.log(g01_range[0]), np.log(g01_range[1]), N))
    gam = np.exp(rng.uniform(np.log(gamma_range[0]), np.log(gamma_range[1]), N))
    return TLSEnsemble(f, g, gam)


def sample_tls_literature(rng: np.random.Generator, f_window=(4.8, 5.6), *,
                          surf_Gamma1_per_us: float = 1e-2, surf_alpha: float = 10.0,
                          surf_g_min_MHz: float = 0.03, surf_g_max_MHz: float = 0.4,
                          surf_Gamma2_per_us=(2.0, 30.0),
                          junc_sigma_per_GHz_um2: float = 3.0, junc_area_um2: float = 0.05,
                          junc_g_min_MHz: float = 0.03, junc_g_max_MHz: float = 30.0,
                          junc_T2_ns=(30.0, 150.0)) -> TLSEnsemble:
    """Two-population TLS ensemble with the literature defaults (paper App. E.4).

    Surface TLS: N(g >= g0) per Hz = Gamma1 / (alpha (2 pi g0)^2), truncated at g_max
    (Weeden et al. 2025, Eqs. 4-6), Lorentzian half-width Gamma2 log-uniform.
    Junction TLS: dN/(df dg) = sigma_J A_J sqrt(1 - g^2/g_max^2)/g (Martinis et al. 2005, Eq. 6
    with S = 2 hbar g), linewidth from a Ramsey T2* log-uniform (dephasing-dominated).
    g is the coupling to the bare ancilla 0-1 transition (H > g(sigma+ tau- + h.c.)), in GHz here;
    returned gamma = 2 * half-width (our FWHM convention, 1/ns).
    """
    W = f_window[1] - f_window[0]
    # surface population: cumulative N(>=g) per GHz = C (1/g^2 - 1/gmax^2), g in Hz
    C = surf_Gamma1_per_us * 1e6 / (surf_alpha * (2 * np.pi) ** 2) * 1e9  # per GHz * Hz^2
    gmin, gmax = surf_g_min_MHz * 1e6, surf_g_max_MHz * 1e6
    Ns = rng.poisson(C * (1 / gmin ** 2 - 1 / gmax ** 2) * W)
    u = rng.uniform(0, 1, Ns)
    g_s = 1.0 / np.sqrt(1 / gmin ** 2 - u * (1 / gmin ** 2 - 1 / gmax ** 2))  # inverse CDF
    G2 = np.exp(rng.uniform(np.log(surf_Gamma2_per_us[0]), np.log(surf_Gamma2_per_us[1]), Ns)) * 1e-3  # 1/ns
    # junction population: density per GHz per ln g ~ sigma A sqrt(1 - g^2/gmax^2)
    gjmin, gjmax = junc_g_min_MHz, junc_g_max_MHz
    lam = junc_sigma_per_GHz_um2 * junc_area_um2 * W * np.log(gjmax / gjmin)  # upper envelope
    Nj_env = rng.poisson(lam)
    g_env = np.exp(rng.uniform(np.log(gjmin), np.log(gjmax), Nj_env))
    keep = rng.uniform(0, 1, Nj_env) < np.sqrt(np.clip(1 - (g_env / gjmax) ** 2, 0, 1))
    g_j = g_env[keep] * 1e6
    T2 = np.exp(rng.uniform(np.log(junc_T2_ns[0]), np.log(junc_T2_ns[1]), len(g_j)))
    f = rng.uniform(*f_window, Ns + len(g_j))
    g = np.concatenate([g_s, g_j]) * 1e-9  # GHz
    gamma = np.concatenate([2 * G2, 2.0 / T2])  # FWHM, 1/ns
    return TLSEnsemble(f, g, gamma)


def surface_mean_rate_per_us(Gamma1_per_us=1e-2, alpha=10.0, g_min_MHz=0.03, g_max_MHz=0.4) -> float:
    """Frequency-averaged golden-rule rate of the explicit surface TLS, 2 Gamma1 ln(gmax/gmin)/alpha."""
    return 2 * Gamma1_per_us * np.log(g_max_MHz / g_min_MHz) / alpha


def two_level_transfer(V, Delta, gamma, t):
    """P(leave |a>) after time t for i d/dt (ca, cb) = [[0, V], [V, Delta - i gamma/2]] (ca, cb), c(0) = (1, 0).

    V, Delta, gamma in rad/ns, t in ns; arrays broadcast.
    """
    V, Delta, gamma = np.broadcast_arrays(np.asarray(V, float), np.asarray(Delta, float), np.asarray(gamma, float))
    D = Delta - 0.5j * gamma
    Om = np.sqrt(0.25 * D ** 2 + V ** 2 + 0j)
    small = np.abs(Om * t) < 1e-8
    with np.errstate(invalid="ignore", divide="ignore"):
        sinc_term = np.where(small, 1j * 0.5 * D * t, 0.5j * D * np.sin(Om * t) / np.where(small, 1.0, Om))
    ca = np.exp(-0.5j * D * t) * (np.cos(Om * t) + sinc_term)
    return np.clip(1.0 - np.abs(ca) ** 2, 0.0, 1.0)


class AncillaBath:
    """Background + TLS transitions out of the dressed branch states at one coupler flux."""

    def __init__(self, table: TransitionTable, table_idle: TransitionTable, T1_ge_us: float, T1_fe_us: float,
                 background_scale: float = 1.0):
        self.tab = table
        self.tab_idle = table_idle
        ref_eg = abs(table_idle.Mq[table_idle.select("e->g", 0)[0]]) ** 2  # <g~0|n_q|e~0> at idle
        ref_fe = abs(table_idle.Mq[table_idle.select("f->e", 0)[0]]) ** 2  # <e~0|n_q|f~0> at idle
        G_eg, G_fe = 1e-3 / T1_ge_us, 1e-3 / T1_fe_us  # 1/ns
        fam_rate = np.where(table.pair == "e->g", G_eg / ref_eg, G_fe / ref_fe)
        emission = table.freq > 0
        self.G_bg = background_scale * np.where(emission, fam_rate * np.abs(table.Mq) ** 2, 0.0)  # 1/ns per channel
        self.emission = emission
        self.n01 = table.n01_bare
        # channels whose TLS transfer can matter (|M| > 1e-3 of the bare 0-1 element; the rest < 1e-6 relative)
        self._active = np.nonzero(emission & (np.abs(table.Mq) > 1e-3 * self.n01))[0]
        n_all = np.arange(table.nmax + 1)
        is_f = (table.pair == "f->e") | (table.pair == "f->g")
        self._rows_f = [np.nonzero(is_f & (table.n == n))[0] for n in n_all]
        self._rows_e = [np.nonzero((table.pair == "e->g") & (table.n == n))[0] for n in n_all]
        self.meta = dict(T1_ge_us=T1_ge_us, T1_fe_us=T1_fe_us, ref_eg=ref_eg, ref_fe=ref_fe,
                         background_scale=background_scale)

    def idle_tls_rates(self, tls: TLSEnsemble, t: float = 1e4) -> tuple[float, float]:
        """Explicit-TLS decay rates (1/ns) of |e~,0> and |f~,0> at the coupler-idle point."""
        ti = self.tab_idle
        out = []
        for pair in ("e->g", "f->e"):
            k = ti.select(pair, 0)[0]
            if tls.size == 0:
                out.append(0.0)
                continue
            V = TWO_PI * abs(ti.Mq[k]) / self.n01 * tls.g01
            P = two_level_transfer(V, TWO_PI * (ti.freq[k] - tls.f), tls.gamma, t)
            out.append(-np.log(max(np.prod(1 - P), 1e-300)) / t)
        return out[0], out[1]

    # ------------------------------------------------------------------ per-channel TLS transfer
    def tls_transfer(self, tls: TLSEnsemble, t: float) -> np.ndarray:
        """(n_channels,) combined TLS-induced leave probability per channel: 1 - prod_j (1 - P_cj)."""
        out = np.zeros(len(self.tab.freq))
        if tls.size == 0:
            return out
        a = self._active
        V = TWO_PI * np.outer(np.abs(self.tab.Mq[a]) / self.n01, tls.g01)
        D = TWO_PI * (self.tab.freq[a, None] - tls.f[None, :])
        P = two_level_transfer(V, D, np.broadcast_to(tls.gamma, V.shape), t)
        out[a] = 1.0 - np.prod(1.0 - P, axis=1)
        return out

    def _leave(self, rows, t, Ptls):
        """Survival and per-row weights for leaving along the given channel rows during t."""
        G = self.G_bg[rows]
        S = np.exp(-G.sum() * t) * np.prod(1.0 - Ptls[rows])
        W = G * t + Ptls[rows]
        return S, W

    def f_state_populations(self, tls: TLSEnsemble, t: float):
        """P_g, P_e, P_f versus n after time t starting from |f~, n> (no ramps)."""
        tab = self.tab
        Pt_full = self.tls_transfer(tls, t)
        Pt_half = self.tls_transfer(tls, 0.5 * t)
        out = np.zeros((tab.nmax + 1, 3))
        for n in range(tab.nmax + 1):
            rows_f = self._rows_f[n]
            S_f, W = self._leave(rows_f, t, Pt_full)
            to_g = tab.pair[rows_f] == "f->g"
            Wt = W.sum()
            P_fg = (1 - S_f) * (W[to_g].sum() / Wt if Wt > 0 else 0.0)
            P_fe = (1 - S_f) - P_fg
            rows_e = self._rows_e[n]  # e~ reached at m ~ n
            S_e, _ = self._leave(rows_e, 0.5 * t, Pt_half)
            P_eg = 1 - S_e
            out[n] = (P_fg + P_fe * P_eg, P_fe * (1 - P_eg), S_f)
        return out  # columns: P_g, P_e, P_f

    def state_rates(self, tls: TLSEnsemble, t: float, state: str = "e") -> np.ndarray:
        """Effective short-time decay rate -ln(S)/t (1/ns) of |state~, n> versus n."""
        tab = self.tab
        Pt = self.tls_transfer(tls, t)
        groups = self._rows_e if state == "e" else self._rows_f
        rates = np.zeros(tab.nmax + 1)
        for n in range(tab.nmax + 1):
            S, _ = self._leave(groups[n], t, Pt)
            rates[n] = -np.log(max(S, 1e-300)) / t
        return rates


def poisson_average(values_by_n: np.ndarray, nbar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average f(n) over Poisson(nbar); returns (average, tail mass beyond the tabulated n)."""
    n = np.arange(values_by_n.shape[0])
    W = poisson.pmf(n[None, :], np.asarray(nbar)[:, None])
    return W @ values_by_n, 1.0 - W.sum(axis=1)
