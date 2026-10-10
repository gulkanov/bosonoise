"""Bayesian one-defect fit of the direct f -> g decay of Ocelot Fig. 18b (F9), against a parity-even alternative.

An f-prepared ancilla A4 is held 800 ns at the CX operating point with |alpha|^2 photons in storage S4; its g population
rises to ~0.08 near |alpha|^2 ~ 10 while e stays at its baseline (paper App. E). Two hypotheses for the defect (TLS) that
produces the peak:

  A  opened channel: a charge-coupled TLS near omega_gf - omega_s ~ 5.0 GHz drives |f~,n> -> |g~,n+1> through the
     dressing-opened charge element M_fg(n), which grows with n (the mechanism of the paper);
  B  critical current: a TLS in the ancilla junction that modulates E_J couples through cos(phi_q), a parity-even
     operator, and drives |f~,n> -> |g~,n> at omega_gf ~ 10.4 GHz, with the dressed element <g~,n|cos phi_q|f~,n>.

Both channels are swept by the dispersive shift at about chi_f - chi_g per photon.

Forward model (both hypotheses): the smooth charge background of channels/tls.AncillaBath (referenced to the measured idle
T1 of A4, 41/25 us) plus one TLS on the hypothesis' channel, with the exact two-state transfer of channels/tls (the TLS
decays at 1/T1_TLS), quasi-static spectral diffusion as a Gaussian average over the TLS frequency (std sigma_sd),
independent decay processes as in AncillaBath, and Poisson photon statistics. The vectorized model is checked against
AncillaBath.f_state_populations in `prepare`.

Parameters and priors (identical for A and B): TLS frequency f_T uniform over the channel's band for n = 0..20 +- 6 MHz;
effective coupling V_c to the channel at n = 10, log-uniform 0.01-1 MHz (A: V_c = g01 |M_fg(10)|/|n_01|;
B: V_c = (dE_J sin theta) |<g~,10|cos phi|f~,10>|); T1_TLS log-uniform 0.1-10 us; sigma_sd log-uniform 0.03-3 MHz.
Data: the 101 vector-extracted points of Fig. 18b (P_g), with a free offset (flat prior) and an unknown Gaussian noise
level (Jeffreys prior), both marginalized analytically: L ~ S_c^(-(N-1)/2), S_c the centred residual sum of squares.
The evidence is a quadrature over a coarse grid of the full prior, then over a fine grid of the region that holds the
posterior (log L within 25 of its maximum on the coarse grid; 0.3% in V_c, 25 kHz in f_T, 13 x 13 in T1_TLS and
sigma_sd), checked by repeating the quadrature on every other grid point. The residuals of a one-defect model are
correlated (misfit, not noise), so the evidences are also computed for every k-th point (k = 2, 3, 4, all offsets).

Usage:
  python examples/f9_bayes_ocelot.py prepare           # cell models (A4-S4, A4-S5; three coupler priors) -> channels CSV
  python examples/f9_bayes_ocelot.py fit --model A     # posterior and evidences of hypothesis A (B likewise)
  python examples/f9_bayes_ocelot.py summary           # Bayes factors, bands of both storages, figure
Output: data/tls/f9_bayes_channels_A4.csv, data/tls/f9_bayes_{A,B}.json, data/tls/f9_bayes_{A,B}_marginals.csv,
        data/tls/f9_bayes_{A,B}_curves.csv, data/tls/f9_bayes_summary.json, figures/f9_bayes.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tls"
CHANNELS = OUT / "f9_bayes_channels_A4.csv"
CALIBS = {"6.569": "data/calibration/ocelot_A4_fcmin6.569.yaml", "6.25": "data/calibration/ocelot_A4_fcmin6.250.yaml",
          "6.75": "data/calibration/ocelot_A4_fcmin6.750.yaml"}
REF_CALIB = "6.569"  # the reference coupler prior (A2-S3 prior)
T1_A4 = (41.0, 25.0)  # T1_ge, T1_fe at the coupler idle point (us), Ocelot Fig. 9c,d
T_PULSE = 800.0  # ns
NMAX = 40
N_REF = 10  # V_c is the coupling to the channel at this photon number
PRIOR = dict(f_margin_MHz=6.0, V_MHz=(0.01, 1.0), T1_us=(0.1, 10.0), sd_MHz=(0.03, 3.0))
COLS = ["calib", "cell", "n", "fA_GHz", "aA", "fB_GHz", "cB", "G_fe", "G_fg", "G_eg"]


# ======================================================================================================== prepare
def _cell(calib, cell, e_cut=250.0):
    """Cell model at 'on' as in examples/tls_single_map_ocelot.build, plus the dressed <g~,n|cos phi_q|f~,n>."""
    import scipy.sparse as sp

    from bosonoise.channels.tls import AncillaBath
    from bosonoise.model import StorageCouplerAncilla, Truncation
    from bosonoise.model.cell import load_cell_params
    from bosonoise.spectra.branches import branch_analysis
    from bosonoise.spectra.transitions import dressed_transitions

    p = load_cell_params(ROOT / CALIBS[calib], cell)
    m = StorageCouplerAncilla(p, Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=8, n_q=12, ncut=25, e_cut=e_cut))
    t0 = time.time()
    E, V = m.eigh(0.5)
    tab = dressed_transitions(m, 0.5, nmax=NMAX, E=E, V=V)
    br = branch_analysis(m, 0.5, starts=((0, 0), (0, 1), (0, 2)), nmax=NMAX + 2, E=E, V=V)
    # cos(phi_q) in the ancilla eigenbasis (charge basis: cos phi = (|n><n+1| + h.c.)/2), embedded in the kept basis
    _, U = m._ancilla.eigensys(m.t.n_q)
    K = U.shape[0]
    Cq = U.T @ (0.5 * (np.eye(K, k=1) + np.eye(K, k=-1))) @ U
    keep = m._full_to_kept()
    C = sp.kron(sp.kron(sp.identity(m.t.n_s), sp.identity(m.t.n_c)), sp.csr_matrix(Cq), format="csr")[keep][:, keep]
    ig, ie, i_f = br.index[0], br.index[1], br.index[2]
    c_fg = np.array([V[:, ig[n]] @ (C @ V[:, i_f[n]]) for n in range(NMAX + 1)])
    c_eg = np.array([V[:, ig[n]] @ (C @ V[:, ie[n]]) for n in range(NMAX + 1)])  # parity-forbidden: regression check
    m_idle = StorageCouplerAncilla(p, Truncation(n_s=10, n_c=8, n_q=10, ncut=25))
    tab_idle = dressed_transitions(m_idle, 0.0, nmax=3)
    bath = AncillaBath(tab, tab_idle, *T1_A4)
    print(f"  {calib} {cell}: dim {m.basis.dim}, {time.time() - t0:.0f} s; bare <g|cos phi|f> = {Cq[0, 2]:+.4f}, "
          f"<g|cos phi|e> = {Cq[0, 1]:.1e}; max |<g~,n|cos phi|e~,n>| = {np.abs(c_eg).max():.1e}")
    return tab, bath, c_fg, float(Cq[0, 2])


def _per_n(tab, bath, c_fg):
    rows = []
    for n in range(NMAX + 1):
        sel = lambda pair: (tab.pair == pair) & (tab.n == n)  # noqa: E731
        kA = np.nonzero(sel("f->g") & (tab.m == n + 1))[0][0]
        rows.append(dict(n=n, fA_GHz=float(tab.freq[kA]), aA=float(abs(tab.Mq[kA]) / tab.n01_bare),
                         fB_GHz=float(tab.energy["f"][n] - tab.energy["g"][n]), cB=float(abs(c_fg[n])),
                         G_fe=float(bath.G_bg[sel("f->e")].sum()), G_fg=float(bath.G_bg[sel("f->g")].sum()),
                         G_eg=float(bath.G_bg[sel("e->g")].sum())))
    return rows


def prepare():
    from bosonoise.channels.tls import TLSEnsemble, poisson_average

    out, checks = [], {}
    for calib in CALIBS:
        for cell in ("S4", "S5"):
            tab, bath, c_fg, c_bare = _cell(calib, cell)
            rows = _per_n(tab, bath, c_fg)
            for r in rows:
                out.append(dict(calib=calib, cell=cell, **r))
            if calib == REF_CALIB and cell == "S4":
                # the vectorized forward model against AncillaBath (no TLS; one TLS on channel A)
                ch = {k: np.array([r[k] for r in rows]) for k in rows[0]}
                nb = np.linspace(0, 20, 41)
                for label, f_T in (("none", None), ("tls", ch["fA_GHz"][10])):
                    if f_T is None:
                        tls = TLSEnsemble(np.array([]), np.array([]), np.array([]))
                        mine = populations(ch, np.zeros((NMAX + 1, 1)))
                    else:
                        tls = TLSEnsemble(np.array([f_T]), np.array([0.5e-3]), np.array([1e-3]))
                        P = two_level(2 * np.pi * 0.5e-3 * ch["aA"][:, None],
                                      2 * np.pi * (ch["fA_GHz"][:, None] - f_T), 1e-3)
                        mine = populations(ch, P)
                    ref, _ = poisson_average(bath.f_state_populations(tls, T_PULSE), nb)
                    mine_avg, _ = poisson_average(mine[:, :1], nb)
                    checks[label] = float(np.abs(mine_avg[:, 0] - ref[:, 0]).max())
                checks["bare_g_cos_f"] = c_bare
                print(f"  forward-model check against AncillaBath: max |dP_g| = {checks}")
    with open(CHANNELS, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# F9 channels of A4 at the CX 'on' point: A = |f~,n> -> |g~,n+1> (charge, freq fA, aA = |M_fg|/|n01|), "
                 "B = |f~,n> -> |g~,n> (cos phi_q, freq fB, cB = |<g~,n|cos phi|f~,n>|); smooth background rates "
                 f"(1/ns) from AncillaBath (T1 {T1_A4} us). Checks: {json.dumps(checks)}. examples/f9_bayes_ocelot.py\n")
        fh.write(",".join(COLS) + "\n")
        for r in out:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.10g}" for k in COLS) + "\n")
    print("saved", CHANNELS)


# ======================================================================================================== forward model
def two_level(V, Delta, gamma, t=T_PULSE):
    from bosonoise.channels.tls import two_level_transfer
    return two_level_transfer(V, Delta, gamma, t)


def populations(ch, P_T, t=T_PULSE):
    """P_g per n (rows) after t from |f~,n>, for TLS transfers P_T (n, k) on the hypothesis' channel (columns k).

    Independent decay processes as in AncillaBath.f_state_populations: survival multiplies, the leave probability is
    shared in proportion to G t (background) and P_T (TLS); e decays over half the time on average.
    """
    G_fe, G_fg, G_eg = ch["G_fe"][:, None], ch["G_fg"][:, None], ch["G_eg"][:, None]
    S_f = np.exp(-(G_fe + G_fg) * t) * (1.0 - P_T)
    W_fe, W_fg = G_fe * t, G_fg * t + P_T
    P_fg = (1.0 - S_f) * W_fg / (W_fe + W_fg)
    P_fe = (1.0 - S_f) - P_fg
    P_eg = 1.0 - np.exp(-G_eg * 0.5 * t)
    return P_fg + P_fe * P_eg


def load_channels(calib=REF_CALIB, cell="S4"):
    rows = [r for r in csv.DictReader(l for l in open(CHANNELS, encoding="utf-8") if not l.startswith("#"))
            if r["calib"] == calib and r["cell"] == cell]
    return {k: np.array([float(r[k]) for r in rows]) for k in COLS[2:]}


def load_data():
    rows = [r for r in csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig18b.csv", encoding="utf-8")
                                      if not l.startswith("#"))]
    a2 = np.array([float(r["alpha2"]) for r in rows])
    Pg = np.array([float(r["P_g"]) for r in rows])
    return a2, Pg


class Hypothesis:
    """Channel of one hypothesis: frequencies f(n) (GHz) and coupling profile a(n)/a(N_REF)."""

    def __init__(self, ch, which):
        self.ch = ch
        self.f = ch["fA_GHz"] if which == "A" else ch["fB_GHz"]
        a = ch["aA"] if which == "A" else ch["cB"]
        self.shape = a / a[N_REF]
        self.a_ref = a[N_REF]
        band = self.f[:21]
        self.f_lo = band.min() - PRIOR["f_margin_MHz"] * 1e-3
        self.f_hi = band.max() + PRIOR["f_margin_MHz"] * 1e-3
        self.which = which


def loglike_block(hyp, W, y, V_MHz, T1_us, sds_MHz, f_prior, df, subsets=None, weights=None):
    """log L (len(sds), len(f_prior)) for one (V_c, T1_TLS) and all sigma_sd, f_T on the prior grid f_prior.

    With `subsets` (a list of index arrays into the data) the result is (len(subsets), len(sds), len(f_prior)): the
    likelihood of each data subset, each with its own offset and noise level marginalized. With `weights` (relative inverse
    variances of the points) the residual sum is weighted (heteroscedastic noise with an unknown overall scale).
    """
    from scipy.ndimage import gaussian_filter1d

    n_pad = int(np.ceil((4.0 * max(sds_MHz) * 1e-3 + 1e-3) / df))  # convolution margin, a whole number of steps
    f_ext = f_prior[0] + df * np.arange(-n_pad, len(f_prior) + n_pad)
    idx = n_pad + np.arange(len(f_prior))
    V = 2 * np.pi * V_MHz * 1e-3 * hyp.shape[:, None]
    P = two_level(V, 2 * np.pi * (hyp.f[:, None] - f_ext[None, :]), 1e-3 / T1_us)
    subs = [np.arange(len(y))] if subsets is None else subsets
    out = np.empty((len(subs), len(sds_MHz), len(f_prior)))
    for j, sd in enumerate(sds_MHz):
        Ps = gaussian_filter1d(P, sd * 1e-3 / df, axis=1, mode="nearest") if sd > 0 else P
        pops = populations(hyp.ch, Ps[:, idx])
        r = y[:, None] - W @ pops
        for s, sub in enumerate(subs):
            rs = r[sub]
            if weights is None:
                Sc = ((rs - rs.mean(axis=0)) ** 2).sum(axis=0)
            else:
                w = weights[sub][:, None]
                Sc = (w * (rs - (w * rs).sum(axis=0) / w.sum()) ** 2).sum(axis=0)
            out[s, j] = -0.5 * (len(sub) - 1) * np.log(Sc)
    return out[0] if subsets is None else out


def model_curve(hyp, W, V_MHz, T1_us, sd_MHz, f_T, df=2.5e-5):
    from scipy.ndimage import gaussian_filter1d

    pad = 4.0 * sd_MHz * 1e-3 + 1e-3
    f_ext = np.arange(f_T - pad, f_T + pad + 0.5 * df, df)
    V = 2 * np.pi * V_MHz * 1e-3 * hyp.shape[:, None]
    P = two_level(V, 2 * np.pi * (hyp.f[:, None] - f_ext[None, :]), 1e-3 / T1_us)
    Ps = gaussian_filter1d(P, sd_MHz * 1e-3 / df, axis=1, mode="nearest")
    k = int(np.argmin(np.abs(f_ext - f_T)))
    return W @ populations(hyp.ch, Ps[:, k:k + 1])[:, 0]


# ======================================================================================================== fit
THINNING = (1, 2, 3, 4)
# Fine grid. The transfer of one defect over the 800-ns pulse varies with the defect frequency on the 1/t ~ 1 MHz scale
# (power-broadened lines ~0.2 MHz); the posterior is ~0.1 MHz wide in f_T and ~1% in V_c; T1_TLS and sigma_sd are weakly
# constrained. Convergence is checked by repeating the quadrature on every other grid point in each direction.
FINE = dict(dlnV=0.003, nV_max=600, nT1=13, nsd=13, df_MHz=0.025)


def _logspace(lo, hi, n):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def _subsets(N):
    """The full data first, then every k-th point for k = 2, 3, 4 (all offsets)."""
    subs, names = [], []
    for k in THINNING:
        for o in range(k):
            subs.append(np.arange(o, N, k))
            names.append(f"k{k}o{o}")
    return subs, names


def fit(which, calib=REF_CALIB):
    """Posterior and evidence of one hypothesis; evidences also for thinned data (correlated residuals)."""
    from scipy.special import logsumexp
    from scipy.stats import poisson

    ch = load_channels(calib)
    hyp = Hypothesis(ch, which)
    a2, y = load_data()
    W = poisson.pmf(np.arange(NMAX + 1)[None, :], a2[:, None])
    N = len(y)
    subs, names = _subsets(N)
    S = len(subs)
    lnV = np.log(PRIOR["V_MHz"][1] / PRIOR["V_MHz"][0])
    lnT = np.log(PRIOR["T1_us"][1] / PRIOR["T1_us"][0])
    lnS = np.log(PRIOR["sd_MHz"][1] / PRIOR["sd_MHz"][0])
    F = hyp.f_hi - hyp.f_lo
    print(f"hypothesis {which}: band n=0..20 {hyp.f[20]:.4f}-{hyp.f[0]:.4f} GHz, prior width {F * 1e3:.1f} MHz, "
          f"a(10) = {hyp.a_ref:.4f}, coupling profile a(n)/a(10) = {hyp.shape[0]:.3f} (n=0) .. {hyp.shape[20]:.3f} (n=20)",
          flush=True)

    # ---- stage 1: coarse grid over the whole prior (equal prior mass per cell), all data subsets
    Vs, T1s, sds = _logspace(*PRIOR["V_MHz"], 61), _logspace(*PRIOR["T1_us"], 9), _logspace(*PRIOR["sd_MHz"], 9)
    df1 = 0.05e-3
    f1 = np.arange(hyp.f_lo + 0.5 * df1, hyp.f_hi, df1)
    t0 = time.time()
    L1 = np.empty((S, len(Vs), len(T1s), len(sds), len(f1)), dtype=np.float32)
    for i, V in enumerate(Vs):
        for k, T1 in enumerate(T1s):
            L1[:, i, k] = loglike_block(hyp, W, y, V, T1, sds, f1, df1, subs)
    logZ1 = [float(logsumexp(L1[s].astype(float)) - np.log(L1[s].size)) for s in range(S)]
    print(f"  coarse: {L1.shape[1:]} x {S} data subsets in {time.time() - t0:.0f} s; max log L (all points) "
          f"{L1[0].max():.2f}", flush=True)
    # ---- stage 2: fine grid over the region with log L > max - 25 for any subset (one coarse step of margin)
    keep = np.zeros(L1.shape[1:], dtype=bool)
    for s in range(S):
        keep |= L1[s] > L1[s].max() - 25.0
    box = {}
    for ax_, (name, g) in enumerate((("V", Vs), ("T1", T1s), ("sd", sds), ("f", f1))):
        hit = np.nonzero(keep.any(axis=tuple(a for a in range(4) if a != ax_)))[0]
        lo, hi = max(hit.min() - 1, 0), min(hit.max() + 1, len(g) - 1)
        box[name] = (g[lo], g[hi])
    del L1, keep
    nV = min(int(np.ceil(np.log(box["V"][1] / box["V"][0]) / FINE["dlnV"])) + 1, FINE["nV_max"])
    Vf = _logspace(*box["V"], max(nV, 3))
    T1f, sdf = _logspace(*box["T1"], FINE["nT1"]), _logspace(*box["sd"], FINE["nsd"])
    df2 = FINE["df_MHz"] * 1e-3
    f2 = np.arange(box["f"][0], box["f"][1] + 0.5 * df2, df2)
    print(f"  posterior box: V {box['V'][0]:.4f}-{box['V'][1]:.4f} MHz, T1 {box['T1'][0]:.2f}-{box['T1'][1]:.2f} us, "
          f"sd {box['sd'][0]:.3f}-{box['sd'][1]:.3f} MHz, f {box['f'][0]:.5f}-{box['f'][1]:.5f} GHz; fine grid "
          f"{len(Vf)} V_c (step {100 * np.log(Vf[1] / Vf[0]):.2f}%) x {len(T1f)} T1 x {len(sdf)} sd x {len(f2)} f_T "
          f"(step {df2 * 1e6:.0f} kHz)", flush=True)
    t0 = time.time()
    L2 = np.empty((len(Vf), len(T1f), len(sdf), len(f2)))
    acc = np.full(S, -np.inf)
    for i, V in enumerate(Vf):
        for k, T1 in enumerate(T1f):
            blk = loglike_block(hyp, W, y, V, T1, sdf, f2, df2, subs)
            L2[i, k] = blk[0]
            acc = np.logaddexp(acc, logsumexp(blk.reshape(S, -1), axis=1))
        if (i + 1) % max(1, len(Vf) // 10) == 0:
            print(f"    fine: {i + 1}/{len(Vf)} V_c values, {time.time() - t0:.0f} s", flush=True)
    cell = (np.log(Vf[1] / Vf[0]) / lnV) * (np.log(T1f[1] / T1f[0]) / lnT) * (np.log(sdf[1] / sdf[0]) / lnS) * df2 / F
    logZ2 = acc + np.log(cell)
    # convergence: the same quadrature on every other grid point in V_c, in (T1, sd) and in f_T
    conv = dict(full=float(logZ2[0]),
                V_every_other=float(logsumexp(L2[::2]) + np.log(2 * cell)),
                T1_sd_every_other=float(logsumexp(L2[:, ::2, ::2]) + np.log(4 * cell)),
                f_every_other=float(logsumexp(L2[..., ::2]) + np.log(2 * cell)))
    print(f"  fine: {L2.shape} in {time.time() - t0:.0f} s; log Z = {logZ2[0]:.3f}; convergence {conv}", flush=True)
    # ---- posterior summaries from the fine grid (all points)
    w = np.exp(L2 - L2.max())
    w /= w.sum()
    marg = {"V_MHz": (Vf, w.sum(axis=(1, 2, 3))), "T1_us": (T1f, w.sum(axis=(0, 2, 3))),
            "sd_MHz": (sdf, w.sum(axis=(0, 1, 3))), "f_GHz": (f2, w.sum(axis=(0, 1, 2)))}

    def q(x, p, qs=(0.16, 0.5, 0.84)):
        c = np.cumsum(p)
        return [float(np.interp(v, c, x)) for v in qs]

    imax = np.unravel_index(np.argmax(L2), L2.shape)
    MAP = dict(V_MHz=float(Vf[imax[0]]), T1_us=float(T1f[imax[1]]), sd_MHz=float(sdf[imax[2]]), f_GHz=float(f2[imax[3]]))
    n_star = float(np.interp(MAP["f_GHz"], hyp.f[::-1], np.arange(NMAX + 1)[::-1]))  # resonant photon number of f_T
    curve = model_curve(hyp, W, MAP["V_MHz"], MAP["T1_us"], MAP["sd_MHz"], MAP["f_GHz"])
    res = y - curve
    res -= res.mean()
    sigma_hat = float(np.sqrt((res ** 2).sum() / (N - 1)))
    rng = np.random.default_rng(20261001)  # posterior predictive band from 200 draws of the fine grid
    draws = rng.choice(w.size, size=200, p=w.ravel())
    curves = np.array([model_curve(hyp, W, Vf[i], T1f[k], sdf[j], f2[m])
                       for i, k, j, m in (np.unravel_index(d, w.shape) for d in draws)])
    summ = dict(
        hypothesis=which, calib=calib, N=N, logZ_fine=float(logZ2[0]), logZ_coarse=logZ1[0], convergence=conv,
        logL_max=float(L2.max()), sigma_hat=sigma_hat, MAP=MAP, n_star_MAP=n_star,
        quantiles={k: q(*v) for k, v in marg.items()},
        coupling_physical=dict(name="g01_MHz" if which == "A" else "dEJ_sin_theta_MHz", a_ref=float(hyp.a_ref),
                               quantiles=[v / hyp.a_ref for v in q(*marg["V_MHz"])], MAP=MAP["V_MHz"] / hyp.a_ref),
        subsets=dict(names=names, n_points=[len(s) for s in subs], logZ_fine=[float(v) for v in logZ2],
                     logZ_coarse=logZ1),
        prior=dict(PRIOR, f_range_GHz=[float(hyp.f_lo), float(hyp.f_hi)]),
        grids=dict(coarse=[len(Vs), len(T1s), len(sds), len(f1)], fine=list(L2.shape), df_coarse_MHz=df1 * 1e3,
                   df_fine_MHz=df2 * 1e3, box={k: [float(a), float(b)] for k, (a, b) in box.items()}),
        band_GHz=[float(hyp.f[20]), float(hyp.f[0])],
    )
    (OUT / f"f9_bayes_{which}.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    with open(OUT / f"f9_bayes_{which}_marginals.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# posterior marginals of hypothesis {which} on the fine grid (examples/f9_bayes_ocelot.py)\n")
        fh.write("parameter,value,posterior_mass\n")
        for k, (x, p) in marg.items():
            for xv, pv in zip(x, p):
                fh.write(f"{k},{xv:.8g},{pv:.6g}\n")
    with open(OUT / f"f9_bayes_{which}_curves.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# hypothesis {which}: data P_g, MAP curve and the 16/50/84% posterior predictive (offset free; "
                 "examples/f9_bayes_ocelot.py)\n")
        fh.write("alpha2,P_g,MAP,p16,p50,p84\n")
        off = float((y - curve).mean())
        band = np.percentile(curves + off, [16, 50, 84], axis=0)
        for i in range(N):
            fh.write(f"{a2[i]:.4f},{y[i]:.5f},{curve[i] + off:.5f},{band[0, i]:.5f},{band[1, i]:.5f},{band[2, i]:.5f}\n")
    print(json.dumps({k: summ[k] for k in ("logZ_coarse", "logZ_fine", "logL_max", "sigma_hat", "MAP", "n_star_MAP")},
                     indent=1))
    print("quantiles:", json.dumps(summ["quantiles"]), "physical coupling:", summ["coupling_physical"])
    print("thinned evidences:", dict(zip(names, [round(v, 2) for v in logZ2])), flush=True)


# ======================================================================================================== sensitivity
# Coarse-grid evidences reproduce the fine ones to 0.2 in log Z and to 0.01 in ln B(A/B) (paper App. E.2), so prior and
# model variants are compared on the coarse grid.
VARIANTS = {
    "baseline": dict(),
    "wide T1_TLS and sigma_sd priors (to 30 us, 10 MHz)": dict(T1_us=(0.1, 30.0), sd_MHz=(0.03, 10.0), nT1=11, nsd=11),
    "no spectral diffusion": dict(sd_fixed=0.0),
    "binomial noise (weights 1/[P(1-P)], smoothed data)": dict(noise="binomial"),
    "coupler prior f_c,min 6.25 GHz": dict(calib="6.25"),
    "coupler prior f_c,min 6.75 GHz": dict(calib="6.75"),
}


def sensitivity(which):
    from scipy.ndimage import uniform_filter1d
    from scipy.special import logsumexp
    from scipy.stats import poisson

    a2, y = load_data()
    W = poisson.pmf(np.arange(NMAX + 1)[None, :], a2[:, None])
    out = {}
    for name, v in VARIANTS.items():
        t0 = time.time()
        calib = v.get("calib", REF_CALIB)
        hyp = Hypothesis(load_channels(calib), which)
        weights = None
        if v.get("noise") == "binomial":
            ys = np.clip(uniform_filter1d(y, 5, mode="nearest"), 0.005, 0.995)
            weights = 1.0 / (ys * (1 - ys))
        Vs = _logspace(*PRIOR["V_MHz"], 61)
        T1s = _logspace(*v.get("T1_us", PRIOR["T1_us"]), v.get("nT1", 9))
        sds = np.array([v["sd_fixed"]]) if "sd_fixed" in v else _logspace(*v.get("sd_MHz", PRIOR["sd_MHz"]),
                                                                          v.get("nsd", 9))
        df1 = 0.05e-3
        f1 = np.arange(hyp.f_lo + 0.5 * df1, hyp.f_hi, df1)
        L = np.empty((len(Vs), len(T1s), len(sds), len(f1)))
        for i, V in enumerate(Vs):
            for k, T1 in enumerate(T1s):
                L[i, k] = loglike_block(hyp, W, y, V, T1, sds, f1, df1, weights=weights)
        w = np.exp(L - L.max())
        w /= w.sum()

        def med(x, p):
            return float(np.interp(0.5, np.cumsum(p), x))

        f_med = med(f1, w.sum(axis=(0, 1, 2)))
        out[name] = dict(
            calib=calib, logZ=float(logsumexp(L) - np.log(L.size)), logL_max=float(L.max()),
            coupling_median_MHz=med(Vs, w.sum(axis=(1, 2, 3))) / hyp.a_ref, T1_median_us=med(T1s, w.sum(axis=(0, 2, 3))),
            sd_median_MHz=(med(sds, w.sum(axis=(0, 1, 3))) if len(sds) > 1 else float(sds[0])),
            n_star_median=float(np.interp(f_med, hyp.f[::-1], np.arange(NMAX + 1)[::-1])))
        print(f"  {which} | {name}: log Z {out[name]['logZ']:.2f}, max log L {out[name]['logL_max']:.2f}, coupling "
              f"{out[name]['coupling_median_MHz']:.2f} MHz, T1 {out[name]['T1_median_us']:.2f} us, sd "
              f"{out[name]['sd_median_MHz']:.2f} MHz ({time.time() - t0:.0f} s)", flush=True)
    (OUT / f"f9_bayes_sensitivity_{which}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ======================================================================================================== smooth channel free
# The fits above fix the smooth opened channel at the idle-referenced bath (s = 1). Its true rate at the operating point
# is not known: the on-gate decay is faster than at idle (measured T1(f) 15-16 us against 25 us), and the bath at
# omega_gf - omega_s ~ 5.00 GHz need not equal the one at the f -> e frequency ~5.08 GHz. Here the smooth f -> g rate is
# scaled by a free factor s (uniform prior 0-30); the defect parameters keep their priors; coarse grid (which reproduces
# the fine-grid evidences, see `fit`). Variants: all points with homoscedastic noise (reference), binomial noise weights,
# and only the points at |alpha|^2 >= 2 (the baseline region carries the offset).
S_SMOOTH = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.5, 15.0, 17.5, 20.0,
                     25.0, 30.0])
SMOOTH_VARIANTS = {"reference": dict(), "binomial noise": dict(noise="binomial"), "|alpha|^2 >= 2 only": dict(a2_min=2.0)}


def _trapz_weights(x):
    w = np.zeros_like(x, dtype=float)
    w[1:] += 0.5 * np.diff(x)
    w[:-1] += 0.5 * np.diff(x)
    return w / (x[-1] - x[0])


def smooth_free(which, calib=REF_CALIB):
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    from scipy.special import logsumexp
    from scipy.stats import poisson

    ch = load_channels(calib)
    hyp = Hypothesis(ch, which)
    a2_all, y_all = load_data()
    Vs, T1s, sds = _logspace(*PRIOR["V_MHz"], 61), _logspace(*PRIOR["T1_us"], 9), _logspace(*PRIOR["sd_MHz"], 9)
    df = 0.05e-3
    f1 = np.arange(hyp.f_lo + 0.5 * df, hyp.f_hi, df)
    n_pad = int(np.ceil((4.0 * sds.max() * 1e-3 + 1e-3) / df))
    f_ext = f1[0] + df * np.arange(-n_pad, len(f1) + n_pad)
    idx = n_pad + np.arange(len(f1))
    chs = [dict(ch, G_fg=s * ch["G_fg"]) for s in S_SMOOTH]
    ws = _trapz_weights(S_SMOOTH)  # uniform prior on [0, 30]
    out = {}
    for vname, v in SMOOTH_VARIANTS.items():
        sel = a2_all >= v.get("a2_min", -1.0)
        a2, y = a2_all[sel], y_all[sel]
        N = len(y)
        W = poisson.pmf(np.arange(NMAX + 1)[None, :], a2[:, None])
        wt = None
        if v.get("noise") == "binomial":
            ys = np.clip(uniform_filter1d(y_all, 5, mode="nearest")[sel], 0.005, 0.995)
            wt = (1.0 / (ys * (1 - ys)))[:, None]
        L = np.empty((len(S_SMOOTH), len(Vs), len(T1s), len(sds), len(f1)), dtype=np.float32)
        t0 = time.time()
        for i, V in enumerate(Vs):
            for k, T1 in enumerate(T1s):
                P = two_level(2 * np.pi * V * 1e-3 * hyp.shape[:, None],
                              2 * np.pi * (hyp.f[:, None] - f_ext[None, :]), 1e-3 / T1)
                for j, sd in enumerate(sds):
                    Ps = gaussian_filter1d(P, sd * 1e-3 / df, axis=1, mode="nearest")[:, idx]
                    for m_, c in enumerate(chs):
                        r = y[:, None] - W @ populations(c, Ps)
                        if wt is None:
                            Sc = ((r - r.mean(axis=0)) ** 2).sum(axis=0)
                        else:
                            Sc = (wt * (r - (wt * r).sum(axis=0) / wt.sum()) ** 2).sum(axis=0)
                        L[m_, i, k, j] = -0.5 * (N - 1) * np.log(Sc)
        logm = np.array([logsumexp(L[m_].astype(float)) - np.log(L[m_].size) for m_ in range(len(S_SMOOTH))])
        logZ = float(logsumexp(logm, b=ws))
        logZ_s1 = float(logm[list(S_SMOOTH).index(1.0)])
        # joint posterior (uniform prior in s), marginals accumulated slice by slice (no float64 copy of L)
        Lmax = float(L.max())
        ps, pV, pT, pD, pF = (np.zeros(n_) for n_ in (len(S_SMOOTH), len(Vs), len(T1s), len(sds), len(f1)))
        for m_ in range(len(S_SMOOTH)):
            wm = np.exp(L[m_].astype(float) - Lmax) * ws[m_]
            ps[m_] = wm.sum()
            pV += wm.sum(axis=(1, 2, 3))
            pT += wm.sum(axis=(0, 2, 3))
            pD += wm.sum(axis=(0, 1, 3))
            pF += wm.sum(axis=(0, 1, 2))
        tot = ps.sum()
        ps, pV, pT, pD, pF = (x_ / tot for x_ in (ps, pV, pT, pD, pF))

        def q(x, p, qs=(0.16, 0.5, 0.84)):
            cc = np.cumsum(p)
            return [float(np.interp(z, cc / cc[-1], x)) for z in qs]

        sf = np.linspace(0, S_SMOOTH[-1], 3001)  # s quantiles on a fine axis (log-linear interpolation of logm)
        lf = np.interp(sf, S_SMOOTH, logm)
        pf = np.exp(lf - lf.max())
        cf = np.cumsum(pf)
        cf /= cf[-1]
        f_q = q(f1, pF)
        n_q = [float(np.interp(z, hyp.f[::-1], np.arange(NMAX + 1)[::-1])) for z in f_q]
        imax = np.unravel_index(np.argmax(L), L.shape)
        MAP = dict(s=float(S_SMOOTH[imax[0]]), V_MHz=float(Vs[imax[1]]), T1_us=float(T1s[imax[2]]),
                   sd_MHz=float(sds[imax[3]]), f_GHz=float(f1[imax[4]]))
        out[vname] = dict(
            N=N, s=S_SMOOTH.tolist(), log_marginal_per_s=logm.tolist(), log_profile_per_s=[float(L[m_].max())
                                                                                          for m_ in range(len(S_SMOOTH))],
            logZ_s_free=logZ, logZ_s1=logZ_s1, lnBF_free_vs_s1=logZ - logZ_s1,
            lnBF_s1_vs_s0=float(logm[2] - logm[0]),
            s_quantiles=[float(np.interp(z, cf, sf)) for z in (0.16, 0.5, 0.84)],
            s_95_upper=float(np.interp(0.95, cf, sf)), s_posterior_mass_per_grid=ps.tolist(),
            coupling_quantiles_MHz=[v_ / hyp.a_ref for v_ in q(Vs, pV)],
            T1_quantiles_us=q(T1s, pT), sd_quantiles_MHz=q(sds, pD),
            n_star_quantiles=sorted(n_q), MAP=MAP, MAP_coupling_MHz=MAP["V_MHz"] / hyp.a_ref)
        print(f"  {which} | {vname}: {time.time() - t0:.0f} s; log Z (s free) {logZ:.2f}, (s = 1) {logZ_s1:.2f}; "
              f"s {out[vname]['s_quantiles']}, s95 {out[vname]['s_95_upper']:.1f}; coupling "
              f"{[round(c_, 2) for c_ in out[vname]['coupling_quantiles_MHz']]} MHz; MAP {MAP}", flush=True)
        if vname == "reference":  # decomposition of the MAP fit: defect + smooth channel, and smooth channel alone
            V, T1, sd, fT = MAP["V_MHz"], MAP["T1_us"], MAP["sd_MHz"], MAP["f_GHz"]
            c = dict(ch, G_fg=MAP["s"] * ch["G_fg"])
            Pfull = two_level(2 * np.pi * V * 1e-3 * hyp.shape[:, None], 2 * np.pi * (hyp.f[:, None] - f_ext[None, :]),
                              1e-3 / T1)
            Pk = gaussian_filter1d(Pfull, sd * 1e-3 / df, axis=1, mode="nearest")[:, idx][:, [imax[4]]]
            full = (W @ populations(c, Pk))[:, 0]
            smooth = (W @ populations(c, np.zeros_like(Pk)))[:, 0]
            s1 = (W @ populations(ch, np.zeros_like(Pk)))[:, 0]
            off = float((y - full).mean())
            with open(OUT / f"f9_bayes_smooth_{which}_curves.csv", "w", encoding="utf-8", newline="\n") as fh:
                fh.write(f"# hypothesis {which}, smooth channel free: data, MAP (defect + smooth, s = {MAP['s']:g}), the "
                         "smooth channel alone at that s and at s = 1 (offset of the MAP fit). examples/f9_bayes_ocelot.py\n")
                fh.write("alpha2,P_g,MAP,smooth_MAP_s,smooth_s1\n")
                for i in range(N):
                    fh.write(f"{a2[i]:.4f},{y[i]:.5f},{full[i] + off:.5f},{smooth[i] + off:.5f},{s1[i] + off:.5f}\n")
        del L
    (OUT / f"f9_bayes_smooth_{which}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ======================================================================================================== summary
def summary():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bosonoise.plotting import SERIES, TEXT_PRIMARY, TEXT_SECONDARY, use_style

    res = {k: json.loads((OUT / f"f9_bayes_{k}.json").read_text(encoding="utf-8")) for k in ("A", "B")}
    lnB = res["A"]["logZ_fine"] - res["B"]["logZ_fine"]
    # bands of both hypotheses on both storages of A4 (n = 0..20), for the three coupler priors
    bands = {}
    for calib in CALIBS:
        for cell in ("S4", "S5"):
            ch = load_channels(calib, cell)
            bands[f"{calib} {cell}"] = dict(A=[float(ch["fA_GHz"][20]), float(ch["fA_GHz"][0])],
                                            B=[float(ch["fB_GHz"][20]), float(ch["fB_GHz"][0])],
                                            sweepA_MHz=float((ch["fA_GHz"][20] - ch["fA_GHz"][0]) / 20 * 1e3),
                                            sweepB_MHz=float((ch["fB_GHz"][20] - ch["fB_GHz"][0]) / 20 * 1e3))
    # where the MAP defect of each hypothesis would show up on the other storage S5. The model shares the coupler
    # parameters of C7 and C8, so the ancilla's on-point frequency is the same for both cells; in the device the two
    # couplers may shift it differently by an unmeasured delta. The defect is met by S5 at some n* in [0, 20] iff the S5
    # band shifted by delta covers f_T: f_T - f5(0) <= delta <= f_T - f5(20).
    other = {}
    for k in ("A", "B"):
        fT = res[k]["MAP"]["f_GHz"]
        ch5 = load_channels(REF_CALIB, "S5")
        f5 = ch5["fA_GHz"] if k == "A" else ch5["fB_GHz"]
        inside = f5[:41].min() <= fT <= f5[:41].max()
        other[k] = dict(f_T_GHz=fT, S5_band_n0_40_GHz=[float(f5[40]), float(f5[0])], resonant_on_S5=bool(inside),
                        n_star_S5=float(np.interp(fT, f5[::-1], np.arange(NMAX + 1)[::-1])) if inside else None,
                        delta_MHz_for_n0_20=[float((fT - f5[0]) * 1e3), float((fT - f5[20]) * 1e3)])
    # robustness against correlated residuals: ln B(A/B) for every k-th point (mean and range over the k offsets)
    rob = {}
    ra, rb = res["A"]["subsets"], res["B"]["subsets"]
    for kk in THINNING:
        v = [za - zb for name, za, zb in zip(ra["names"], ra["logZ_fine"], rb["logZ_fine"]) if name.startswith(f"k{kk}o")]
        rob[f"every {kk}"] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
    # residual autocorrelation of the MAP fits and the AR(1) effective number of independent points
    acf = {}
    for k in ("A", "B"):
        rows = [r for r in csv.DictReader(l for l in open(OUT / f"f9_bayes_{k}_curves.csv", encoding="utf-8")
                                          if not l.startswith("#"))]
        r = np.array([float(x["P_g"]) - float(x["MAP"]) for x in rows])
        r -= r.mean()
        rho1 = float((r[1:] * r[:-1]).sum() / (r * r).sum())
        acf[k] = dict(lag1=rho1, N_eff=float(len(r) * (1 - rho1) / (1 + rho1)), rms=float(np.sqrt((r * r).mean())))
    # prior side of hypothesis B: junction defects of the literature prior in the window that puts the peak at
    # |alpha|^2 = 2..18 (upper bound: every such defect assumed to couple through the critical current strongly enough)
    ch = load_channels()
    window_B = abs(ch["fB_GHz"][2] - ch["fB_GHz"][18])
    dens = dict(literature_prior=3.0 * 0.05 * np.log(30.0 / 0.03), strongly_coupled_Zanuz=8.5 * 0.05 + 0.1)  # per GHz
    prior_B = {k: dict(per_GHz=float(v), P_max=float(1 - np.exp(-v * window_B))) for k, v in dens.items()}
    sens = {}
    if all((OUT / f"f9_bayes_sensitivity_{k}.json").exists() for k in ("A", "B")):
        sa, sb = (json.loads((OUT / f"f9_bayes_sensitivity_{k}.json").read_text(encoding="utf-8")) for k in ("A", "B"))
        for name in sa:
            sens[name] = dict(lnB=sa[name]["logZ"] - sb[name]["logZ"], dlogL_max=sa[name]["logL_max"] - sb[name]["logL_max"],
                              g01_MHz=sa[name]["coupling_median_MHz"], dEJ_sin_theta_MHz=sb[name]["coupling_median_MHz"],
                              n_star=[sa[name]["n_star_median"], sb[name]["n_star_median"]])
    # smooth channel free (smooth_free): its strength s, and the defect comparison with s marginalized
    smooth = {k: json.loads((OUT / f"f9_bayes_smooth_{k}.json").read_text(encoding="utf-8")) for k in ("A", "B")}
    sfree = {}
    for vname in smooth["A"]:
        a_, b_ = smooth["A"][vname], smooth["B"][vname]
        sfree[vname] = dict(lnB_A_over_B_s_free=a_["logZ_s_free"] - b_["logZ_s_free"],
                            lnBF_s_free_vs_s1={"A": a_["lnBF_free_vs_s1"], "B": b_["lnBF_free_vs_s1"]},
                            s_quantiles={"A": a_["s_quantiles"], "B": b_["s_quantiles"]},
                            s_95_upper={"A": a_["s_95_upper"], "B": b_["s_95_upper"]},
                            coupling_quantiles_MHz={"A": a_["coupling_quantiles_MHz"], "B": b_["coupling_quantiles_MHz"]})
    out = dict(smooth_fixed_s1=dict(lnB_A_over_B=float(lnB), B_A_over_B=float(np.exp(lnB)), sensitivity=sens,
                                    logZ={k: res[k]["logZ_fine"] for k in res},
                                    logZ_coarse={k: res[k]["logZ_coarse"] for k in res},
                                    logL_max={k: res[k]["logL_max"] for k in res},
                                    sigma_hat={k: res[k]["sigma_hat"] for k in res}, thinned_lnB=rob,
                                    residual_autocorrelation=acf),
               smooth_free=sfree, window_B_GHz=float(window_B), prior_B=prior_B, bands=bands, MAP_defect_on_S5=other)
    (OUT / "f9_bayes_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    titles = {"A": "(a) A: opened channel, charge defect", "B": "(b) B: critical-current defect"}
    for j, k in enumerate(("A", "B")):
        rows = [r for r in csv.DictReader(l for l in open(OUT / f"f9_bayes_smooth_{k}_curves.csv", encoding="utf-8")
                                          if not l.startswith("#"))]
        a2 = np.array([float(r["alpha2"]) for r in rows])
        get = lambda c: np.array([float(r[c]) for r in rows])  # noqa: E731
        base = get("P_g")[a2 < 2].mean()
        s_map = smooth[k]["reference"]["MAP"]["s"]
        ax = axs[j]
        ax.plot(a2, get("P_g") - base, "o", ms=2.5, color=TEXT_PRIMARY, label="Ocelot Fig. 18b (A4-S4)")
        ax.plot(a2, get("MAP") - base, "-", color=SERIES[j], lw=1.4, label=f"defect + smooth channel (s = {s_map:g})")
        ax.plot(a2, get("smooth_MAP_s") - base, "--", color=SERIES[j], lw=1.1, label="smooth channel alone")
        ax.plot(a2, get("smooth_s1") - base, ":", color=TEXT_SECONDARY, lw=1.1, label="smooth channel, s = 1")
        ax.set_title(titles[k], fontsize=8.2, loc="left")
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_ylabel(r"$\Delta P_g$ after 800 ns")
        ax.set_ylim(-0.012, 0.13)
        ax.legend(fontsize=7, loc="upper left")
    ax = axs[2]
    for j, k in enumerate(("A", "B")):
        for vname, ls in (("reference", "-"), ("binomial noise", "--"), ("|alpha|^2 >= 2 only", ":")):
            sv = smooth[k][vname]
            s_ = np.array(sv["s"])
            sf = np.linspace(0, s_[-1], 601)
            lf = np.interp(sf, s_, sv["log_marginal_per_s"])
            p = np.exp(lf - lf.max())
            ax.plot(sf, p / (p.sum() * (sf[1] - sf[0])), ls, color=SERIES[j], lw=1.2,
                    label=f"{k}, {vname.replace('|alpha|^2', '|α|²')}")
    ax.axvline(1.0, color=TEXT_SECONDARY, lw=0.8, ls="-.")
    ax.text(1.2, ax.get_ylim()[1] * 0.92, "idle-referenced (s = 1)", fontsize=6.5, color=TEXT_SECONDARY, va="top")
    ax.set_xlabel("strength of the smooth photon-stimulated f→g channel, s")
    ax.set_ylabel("posterior density")
    ax.set_xlim(0, 20)
    ax.legend(fontsize=6.3, loc="upper right")
    ax.set_title("(c) the smooth channel in F9", fontsize=8.2, loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"f9_bayes.{ext}", dpi=160)
    print("saved figures/f9_bayes.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "fit", "sensitivity", "smooth", "summary"))
    ap.add_argument("--model", choices=("A", "B"))
    a = ap.parse_args()
    if a.step == "prepare":
        prepare()
    elif a.step == "fit":
        fit(a.model)
    elif a.step == "sensitivity":
        sensitivity(a.model)
    elif a.step == "smooth":
        smooth_free(a.model)
    else:
        summary()


if __name__ == "__main__":
    main()
