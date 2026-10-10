"""Could a second defect, instead of the stronger smooth channel, make the tail of F9? (paper App. E.2)

The one-defect fits of Ocelot Fig. 18b (examples/f9_bayes_ocelot.py, paper App. E.2) favour a smooth photon-stimulated
f -> g channel s = 5-11 times its idle-referenced strength, which supplies the slow rise and the tail while the defect
makes the peak. The alternative compared here keeps the smooth channel at s = 1 and adds a second defect of the same kind
(hypothesis A: both on the opened channel; B: both coupled through the critical current), with the same priors as the
first.

Model: the forward model of f9_bayes_ocelot.py with two independent defects (survival probabilities multiply; the leave
probability is shared in proportion to G t and the two transfers). Offset (flat prior) and noise level (Jeffreys prior)
are marginalized analytically, as there, so the evidences are directly comparable.

Grid G (both defects): V_c 61 log steps over 0.01-1 MHz, T1_TLS and sigma_sd 5 log steps each over 0.1-10 us and
0.03-3 MHz, f_T in 0.05-MHz steps over the band +- 6 MHz. The one-defect evidences are recomputed on G, so that all
models are compared on the same grid. Against the 9 x 9 grid of f9_bayes_ocelot.smooth_free (`--validate`), ln B of
s free over s = 1 is 8.04 against 8.05 for B, and 3.76 against 3.18 for A, whose posterior sits at the T1_TLS edge of the
prior (edge points carry 1/5 instead of 1/9 of the prior mass).
Two-defect evidence (exchangeable defects, product prior):
  Z2 = |G|^-2 sum_{t1 in R} [2 sum_{t2 in G} L(t1, t2) - sum_{t2 in R} L(t1, t2)],
with R the region where one defect makes the peak: the grid points within a log-likelihood margin of the one-defect maxima
at s = 1, 4 and 8, chosen so that R holds >= 99.9% of each of these one-defect posteriors. Configurations in which
neither defect alone makes the peak are neglected (their prior volume is small).

Usage: python examples/f9_two_defects_ocelot.py --model A   (B likewise)
Output: data/tls/f9_two_defects_{A,B}.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp
from scipy.stats import poisson

from f9_bayes_ocelot import (NMAX, OUT, PRIOR, REF_CALIB, S_SMOOTH, T_PULSE, Hypothesis, _logspace, _trapz_weights,
                             load_channels, load_data, two_level)  # same directory

GRID = dict(nV=61, nT1=5, nsd=5, df_MHz=0.05)
S_REGION = (1.0, 4.0, 8.0)
COVERAGE = 0.999
K_MAX = 4000
CHUNK = 64


def populations2(ch, P1, P2, t=T_PULSE):
    """P_g per n after t from |f~,n> with two defects (transfers P1, P2 on the hypothesis' channel); P2 = 0 gives the
    one-defect model of f9_bayes_ocelot.populations."""
    G_fe, G_fg, G_eg = ch["G_fe"][:, None], ch["G_fg"][:, None], ch["G_eg"][:, None]
    S_f = np.exp(-(G_fe + G_fg) * t) * (1.0 - P1) * (1.0 - P2)
    W_fe, W_fg = G_fe * t, G_fg * t + P1 + P2
    P_fg = (1.0 - S_f) * W_fg / (W_fe + W_fg)
    P_fe = (1.0 - S_f) - P_fg
    P_eg = 1.0 - np.exp(-G_eg * 0.5 * t)
    return P_fg + P_fe * P_eg


class Data:
    """The centred residual sum S_c = sum_k ((y_k - y_mean) - (m_k - m_mean))^2 with m = W p, as a quadratic form in the
    per-photon-number populations p: S_c = S_yy - 2 a.p + p.G_c.p (offset marginalized as in f9_bayes_ocelot)."""

    def __init__(self, W, y):
        W, y = np.asarray(W, float), np.asarray(y, float)
        yc = y - y.mean()
        self.N = len(y)
        self.Syy = float(yc @ yc)
        self.a = (yc @ W).astype(np.float32)
        s = W.sum(axis=0)
        self.Gc = (W.T @ W - np.outer(s, s) / len(y)).astype(np.float32)


def loglike(ch, P1, Ps, D):
    """log L (chunk, n_f) for defect-1 transfer P1 (n, 1) and defect-2 transfers Ps (chunk, n, n_f)."""
    pops = populations2(ch, P1, Ps)
    lin = np.tensordot(D.a, pops, axes=([0], [1]))
    quad = (pops * np.matmul(D.Gc[None], pops)).sum(axis=1)
    Sc = np.maximum(D.Syy - 2.0 * lin.astype(float) + quad.astype(float), 1e-12)
    return -0.5 * (D.N - 1) * np.log(Sc)


def run(which, grid=None, one_defect_only=False, calib=REF_CALIB, T1_us=None, sd_MHz=None, tag="", dry=False):
    """T1_us, sd_MHz: prior ranges (default those of f9_bayes_ocelot); tag: suffix of the output file; dry: stop after
    selecting the region R (prints its size)."""
    t0 = time.time()
    ch = load_channels(calib)
    hyp = Hypothesis(ch, which)
    a2, y = load_data()
    W = poisson.pmf(np.arange(NMAX + 1)[None, :], a2[:, None]).astype(np.float32)
    y = y.astype(np.float32)
    ch32 = {k: v.astype(np.float32) for k, v in ch.items()}  # single precision throughout the likelihood (|d log L| < 1e-3)
    D = Data(W, y)
    g = dict(GRID, **(grid or {}))
    Vs = _logspace(*PRIOR["V_MHz"], g["nV"])
    T1s = _logspace(*(T1_us or PRIOR["T1_us"]), g["nT1"])
    sds = _logspace(*(sd_MHz or PRIOR["sd_MHz"]), g["nsd"])
    df = g["df_MHz"] * 1e-3
    f1 = np.arange(hyp.f_lo + 0.5 * df, hyp.f_hi, df)
    n_pad = int(np.ceil((4.0 * sds.max() * 1e-3 + 1e-3) / df))
    f_ext = f1[0] + df * np.arange(-n_pad, len(f1) + n_pad)
    idx = n_pad + np.arange(len(f1))
    combos = [(i, k, j) for i in range(len(Vs)) for k in range(len(T1s)) for j in range(len(sds))]
    Ps = np.empty((len(combos), NMAX + 1, len(f1)), dtype=np.float32)
    c = 0
    for i, V in enumerate(Vs):
        for k, T1 in enumerate(T1s):
            P = two_level(2 * np.pi * V * 1e-3 * hyp.shape[:, None], 2 * np.pi * (hyp.f[:, None] - f_ext[None, :]),
                          1e-3 / T1)
            for j, sd in enumerate(sds):
                assert combos[c] == (i, k, j)
                Ps[c] = gaussian_filter1d(P, sd * 1e-3 / df, axis=1, mode="nearest")[:, idx]
                c += 1
    size = Ps.shape[0] * Ps.shape[2]
    print(f"{which}: grid {len(Vs)} x {len(T1s)} x {len(sds)} x {len(f1)} = {size} points ({time.time() - t0:.0f} s)",
          flush=True)

    # one defect on G, smooth channel scaled by s
    chs = {s: dict(ch32, G_fg=np.float32(s) * ch32["G_fg"]) for s in S_SMOOTH}
    zero = np.zeros((NMAX + 1, 1), dtype=np.float32)
    L1 = {}
    for s in S_SMOOTH:
        L1[s] = np.concatenate([loglike(chs[s], zero, Ps[a:a + CHUNK], D) for a in range(0, len(Ps), CHUNK)])
    logm = np.array([logsumexp(L1[s]) - np.log(size) for s in S_SMOOTH])
    logZ1_free = float(logsumexp(logm, b=_trapz_weights(S_SMOOTH)))
    logZ1 = {float(s): float(m) for s, m in zip(S_SMOOTH, logm)}
    print(f"{which}: one defect on G: log Z (s free) {logZ1_free:.2f}, (s = 1) {logZ1[1.0]:.2f}; max log L s=1 "
          f"{L1[1.0].max():.2f}, s=8 {L1[8.0].max():.2f} ({time.time() - t0:.0f} s)", flush=True)
    if one_defect_only:
        return dict(logZ1_s_free=logZ1_free, logZ1_s1=logZ1[1.0])

    # region R where one defect makes the peak
    score = np.max([L1[s] - L1[s].max() for s in S_REGION], axis=0)
    order = np.argsort(score, axis=None)[::-1]
    for K in range(200, K_MAX + 1, 100):
        Rmask = np.zeros(score.size, bool)
        Rmask[order[:K]] = True
        Rmask = Rmask.reshape(score.shape)
        cov = {s: float(np.exp(logsumexp(L1[s][Rmask]) - logsumexp(L1[s]))) for s in S_REGION}
        if min(cov.values()) >= COVERAGE:
            break
    R = np.argwhere(Rmask)
    R = R[np.argsort(-score[Rmask])]
    print(f"{which}: region R {len(R)} points, coverage {cov} ({time.time() - t0:.0f} s)", flush=True)
    if dry:
        return dict(region_points=len(R), size=size)

    # two defects, smooth channel at s = 1
    ch1 = chs[1.0]
    terms, best = [], (-np.inf, None)
    for q, (c1, i1) in enumerate(R):
        P1 = Ps[c1][:, i1:i1 + 1]
        L2 = np.concatenate([loglike(ch1, P1, Ps[a:a + CHUNK], D) for a in range(0, len(Ps), CHUNK)])
        lI = logsumexp(L2)
        lIR = logsumexp(L2[Rmask])
        terms.append(lI + np.log(2.0 - np.exp(lIR - lI)))
        m = np.unravel_index(np.argmax(L2), L2.shape)
        if L2[m] > best[0]:
            best = (float(L2[m]), (int(c1), int(i1), int(m[0]), int(m[1])))
        if q % 200 == 0:
            print(f"  {which}: {q}/{len(R)} defect-1 points, running log Z2 {logsumexp(terms) - 2 * np.log(size):.2f}, "
                  f"max log L {best[0]:.2f} ({time.time() - t0:.0f} s)", flush=True)
    logZ2 = float(logsumexp(terms) - 2 * np.log(size))

    def describe(cidx, iidx):
        i, k, j = combos[cidx]
        n_res = float(np.interp(f1[iidx], hyp.f[::-1], np.arange(NMAX + 1)[::-1]))
        return dict(V_MHz=float(Vs[i]), coupling_MHz=float(Vs[i] / hyp.a_ref), T1_us=float(T1s[k]),
                    sd_MHz=float(sds[j]), f_GHz=float(f1[iidx]), n_resonant=n_res)

    c1, i1, c2, i2 = best[1]
    res = dict(model=which, N=len(y), grid=dict(g, size=size), prior_T1_us=[float(T1s[0]), float(T1s[-1])],
               prior_sd_MHz=[float(sds[0]), float(sds[-1])], region_points=len(R), region_coverage=cov,
               logZ1_s_free=logZ1_free, logZ1_per_s=logZ1, maxlogL1_s1=float(L1[1.0].max()),
               maxlogL1_s8=float(L1[8.0].max()), maxlogL1_s_any=float(max(L1[s].max() for s in S_SMOOTH)),
               logZ2_s1=logZ2, maxlogL2_s1=best[0],
               lnB_one_defect_s_free_vs_two_defects_s1=logZ1_free - logZ2,
               lnB_two_defects_s1_vs_one_defect_s1=logZ2 - logZ1[1.0],
               MAP_two_defects=[describe(c1, i1), describe(c2, i2)], seconds=time.time() - t0)
    (OUT / f"f9_two_defects_{which}{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("A", "B"), required=True)
    ap.add_argument("--validate", action="store_true",
                    help="one-defect evidences on the reference grid of f9_bayes_ocelot.smooth_free and on this grid")
    ap.add_argument("--wide", action="store_true",
                    help="robustness variant: 9 x 9 grid in T1_TLS and sigma_sd (0.1-MHz steps in f_T), priors widened to 0.1-30 us and "
                         "0.03-10 MHz (output f9_two_defects_<model>_wide.json)")
    ap.add_argument("--dry", action="store_true", help="stop after selecting the region R")
    a = ap.parse_args()
    if a.validate:
        for name, gr in (("smooth_free grid", dict(nV=61, nT1=9, nsd=9, df_MHz=0.05)), ("this grid", None)):
            r = run(a.model, grid=gr, one_defect_only=True)
            print(f"{a.model} {name}: log Z1 (s free) {r['logZ1_s_free']:.3f}, (s = 1) {r['logZ1_s1']:.3f}", flush=True)
        return
    if a.wide:
        run(a.model, grid=dict(nT1=9, nsd=9, df_MHz=0.1), T1_us=(0.1, 30.0), sd_MHz=(0.03, 10.0), tag="_wide", dry=a.dry)
        return
    run(a.model, dry=a.dry)


if __name__ == "__main__":
    main()
