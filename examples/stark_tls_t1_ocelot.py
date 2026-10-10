"""Photon-number dependence of the ancilla T1 at the 'on' point from Stark-swept TLS (A4-S5; Fig. 37d test).

At 'on' the e->g transition of the dressed ancilla shifts by ~chi_ge n (A4-S5: -6.7 MHz at
n = 4), while chi-matching makes the f->e transition almost photon-number independent
(+0.18 MHz at n = 4). A frequency-selective bath (a TLS a few MHz below the ancilla e->g
line) therefore changes T1(e) with the storage photon number but not T1(f). Fig. 37d
(A4-S5 coupler, pulsed with and without |alpha|^2 = 4; VALIDATION ONLY) shows exactly this
asymmetry: during the 'on' periods the e decay rate grows by ~1.76x, the f rate by ~1.19x
(duty cycle 0.4 of the 1.6-us cycles; paper App. C.2, App. K).

For the literature-prior TLS ensembles (conditioned on the idle T1s, Fig. 9) we compute, per
realization, the Poisson-averaged dressed decay rates of |e~,n> and |f~,n> at 'on' over one
640-ns on-period, at nbar = 4 and at n = 0, and their ratios r_e, r_f. Reported: the fraction
of realizations that reproduce the Fig. 37d pattern (r_e >= 1.5 and r_f <= 1.3) and related
fractions. No parameter is tuned.

Usage: python examples/stark_tls_t1_ocelot.py [--nreal 4000] [--area-um2 0.05 0.15]
Output: data/tls/stark_tls_summary.json, figures/stark_tls_t1.png/.pdf
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.tls import AncillaBath, poisson_average, sample_tls_literature, surface_mean_rate_per_us
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

from tls_ensemble_ocelot import CELLS  # same directory
from tls_single_map_ocelot import T1, build

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_ON = 640.0  # ns, one CX^2-length on-period (Fig. 37d protocol)
TARGET = dict(r_e=1.76, r_f=1.19)  # from Fig. 37d (paper App. K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="A4S5")
    ap.add_argument("--nreal", type=int, default=4000)
    ap.add_argument("--area-um2", nargs="+", type=float, default=[0.05, 0.15])
    ap.add_argument("--surf-T1-us", type=float, default=100.0)
    ap.add_argument("--idle-tol", type=float, default=1.5)
    ap.add_argument("--window", nargs=2, type=float, default=[4.8, 5.4])
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()
    calib, key, anc = CELLS[a.cell]
    tab_on, tab_idle = build(calib, key)
    scale = max(0.0, 1.0 - surface_mean_rate_per_us(1.0 / a.surf_T1_us) * T1[anc][0])
    bath = AncillaBath(tab_on, tab_idle, *T1[anc], background_scale=scale)
    bg_e, bg_f = scale * 1e-3 / T1[anc][0], scale * 1e-3 / T1[anc][1]
    nbar = np.array([0.0, 4.0])
    summary = {"args": vars(a), "target_fig37d": TARGET, "results": {}}
    use_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    for ai, area in enumerate(a.area_um2):
        rng = np.random.default_rng(a.seed)
        t0 = time.time()
        re, rf, n_try = [], [], 0
        while len(re) < a.nreal:
            n_try += 1
            tls = sample_tls_literature(rng, tuple(a.window), surf_Gamma1_per_us=1.0 / a.surf_T1_us, junc_area_um2=area)
            ie, i_f = bath.idle_tls_rates(tls)
            if ie + bg_e > a.idle_tol * 1e-3 / T1[anc][0] or i_f + bg_f > a.idle_tol * 1e-3 / T1[anc][1]:
                continue
            Re, _ = poisson_average(bath.state_rates(tls, T_ON, "e"), nbar)
            Rf, _ = poisson_average(bath.state_rates(tls, T_ON, "f"), nbar)
            re.append(Re[1] / Re[0])
            rf.append(Rf[1] / Rf[0])
        re, rf = np.array(re), np.array(rf)
        res = dict(acceptance=len(re) / n_try,
                   frac_re_ge_1p5=float(np.mean(re >= 1.5)),
                   frac_pattern=float(np.mean((re >= 1.5) & (rf <= 1.3))),
                   frac_rf_ge_1p5=float(np.mean(rf >= 1.5)),
                   frac_re_ge_1p5_given_rf_le_1p3=float(np.mean(re[rf <= 1.3] >= 1.5)),
                   re_quantiles=[float(v) for v in np.quantile(re, [0.5, 0.9, 0.975, 0.99])],
                   rf_quantiles=[float(v) for v in np.quantile(rf, [0.5, 0.9, 0.975, 0.99])])
        summary["results"][f"A_J={area:g}"] = res
        print(f"A_J = {area:g} um^2: accept {res['acceptance']:.2f}; P(r_e>=1.5) {100 * res['frac_re_ge_1p5']:.2f}%, "
              f"P(r_f>=1.5) {100 * res['frac_rf_ge_1p5']:.2f}%, P(Fig.37d pattern: r_e>=1.5 & r_f<=1.3) "
              f"{100 * res['frac_pattern']:.2f}%; r_e median/90/97.5/99%: {np.round(res['re_quantiles'], 3)}, "
              f"r_f: {np.round(res['rf_quantiles'], 3)} ({time.time() - t0:.0f} s)", flush=True)
        ax.loglog(re, rf, ".", ms=2, alpha=0.35, color=SERIES[ai], label=f"TLS realizations, A_J = {area:g} µm²")
    ax.plot([TARGET["r_e"]], [TARGET["r_f"]], "*", ms=14, color=TEXT_PRIMARY, label="Fig. 37d (on-period estimate)")
    ax.axvline(1.5, color="#52514e", lw=0.6, ls=":")
    ax.axhline(1.3, color="#52514e", lw=0.6, ls=":")
    ax.set_xlabel(r"$r_e$ = e decay rate at $\bar n=4$ / at $n=0$ ('on')")
    ax.set_ylabel(r"$r_f$ = f decay rate at $\bar n=4$ / at $n=0$ ('on')")
    ax.set_title("Stark-swept TLS: photons act on e, not on f (χ-matching)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"stark_tls_t1.{ext}", dpi=160)
    with open(ROOT / "data" / "tls" / "stark_tls_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("saved data/tls/stark_tls_summary.json")


if __name__ == "__main__":
    main()
