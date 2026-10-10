"""TLS spectral diffusion in the literature-prior ensembles (A4-S4: F9 pair; A4-S5: App. J pair).

Every error-rate or population data point averages over minutes to hours, while a TLS
frequency wanders (slow diffusion sigma_f = 2 D_K sqrt(t) with D_K = 0.03 ... 3 MHz h^-1/2, spanning
Carroll et al. 2022 and Klimov et al. 2018, i.e. sigma_f ~ 0.1 ... 6 MHz over 10 min ... 1 h; a fast
Ornstein-Uhlenbeck component with rms ~ 5 MHz and tau_c ~ 30 s for some TLS, Berritta et al. 2026). On the
microsecond scale of one CX the TLS is static, so we use the quasi-static limit: each
observable (including the idle T1 used to condition the realization) is averaged over K
Gaussian frequency offsets of every TLS with standard deviation sigma_sd.

Per realization (same observables as examples/tls_ensemble_ocelot.py):
  * F9 peak (A4-S4 protocol: |f>, 800 ns at 'on'): interior maximum of Delta P_g(|alpha|^2);
  * P(f -> g) within one CX (T_CX), Poisson-averaged at nbar = 4;
  * lambda_e, lambda_f: effective decay rates at 'on' (nbar = 4) over the idle 1/T1.
No parameter is tuned to error data. Output: data/tls/spectral_diffusion_summary.json,
figures/tls_spectral_diffusion.png/.pdf

Usage: python examples/tls_spectral_diffusion_ocelot.py [--sigma-MHz 0 1 5] [--nreal 2000] [--K 8]
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

from bosonoise.channels.tls import AncillaBath, TLSEnsemble, poisson_average, sample_tls_literature, surface_mean_rate_per_us
from bosonoise.plotting import SERIES, use_style

from tls_ensemble_ocelot import CELLS, NBAR  # same directory
from tls_single_map_ocelot import T1, build

ROOT = pathlib.Path(__file__).resolve().parents[1]


def jittered(tls: TLSEnsemble, rng, sigma_GHz: float, K: int):
    if sigma_GHz == 0 or tls.size == 0:
        return [tls]
    return [TLSEnsemble(tls.f + rng.normal(0.0, sigma_GHz, tls.size), tls.g01, tls.gamma) for _ in range(K)]


def averaged_observables(bath: AncillaBath, copies, T_cx: float, t_f9: float = 800.0):
    f9 = np.mean([bath.f_state_populations(c, t_f9) for c in copies], axis=0)
    cx = np.mean([bath.f_state_populations(c, T_cx)[:, 0] for c in copies], axis=0)
    re = np.mean([bath.state_rates(c, T_cx, "e") for c in copies], axis=0)
    rf = np.mean([bath.state_rates(c, T_cx, "f") for c in copies], axis=0)
    avg_f9, _ = poisson_average(f9, NBAR)
    dPg = avg_f9[:, 0] - avg_f9[0, 0]
    k = int(np.argmax(dPg))
    peak = dPg[k] if 2.0 <= NBAR[k] <= 18.0 else 0.0
    p_fg4 = float(poisson_average(cx, np.array([4.0]))[0][0])
    lam_e = float(poisson_average(re, np.array([4.0]))[0][0] / (1e-3 / bath.meta["T1_ge_us"]))
    lam_f = float(poisson_average(rf, np.array([4.0]))[0][0] / (1e-3 / bath.meta["T1_fe_us"]))
    return peak, p_fg4, lam_e, lam_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["A4S4", "A4S5"])
    ap.add_argument("--sigma-MHz", nargs="+", type=float, default=[0.0, 1.0, 5.0])
    ap.add_argument("--nreal", type=int, default=2000)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--window", nargs=2, type=float, default=[4.8, 5.4])
    ap.add_argument("--area-um2", type=float, default=0.05)
    ap.add_argument("--surf-T1-us", type=float, default=100.0)
    ap.add_argument("--idle-tol", type=float, default=1.5)
    ap.add_argument("--Tcx", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()
    summary = {"args": vars(a), "cells": {}}
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ci, cell in enumerate(a.cells):
        calib, key, anc = CELLS[cell]
        tab_on, tab_idle = build(calib, key)
        G_surf = surface_mean_rate_per_us(1.0 / a.surf_T1_us)
        scale = max(0.0, 1.0 - G_surf * T1[anc][0])
        bath = AncillaBath(tab_on, tab_idle, *T1[anc], background_scale=scale)
        bg_e, bg_f = scale * 1e-3 / T1[anc][0], scale * 1e-3 / T1[anc][1]
        summary["cells"][cell] = {}
        for si, sig in enumerate(a.sigma_MHz):
            rng = np.random.default_rng(a.seed)  # same TLS draws for every sigma (paired comparison)
            t0 = time.time()
            obs, n_try = [], 0
            while len(obs) < a.nreal:
                n_try += 1
                tls = sample_tls_literature(rng, tuple(a.window), surf_Gamma1_per_us=1.0 / a.surf_T1_us,
                                            junc_area_um2=a.area_um2)
                copies = jittered(tls, rng, sig * 1e-3, a.K)
                idle = np.mean([bath.idle_tls_rates(c) for c in copies], axis=0)
                if idle[0] + bg_e > a.idle_tol * 1e-3 / T1[anc][0] or idle[1] + bg_f > a.idle_tol * 1e-3 / T1[anc][1]:
                    continue
                obs.append(averaged_observables(bath, copies, a.Tcx))
            obs = np.array(obs)
            peak, pfg, le, lf = obs.T
            res = dict(acceptance=len(obs) / n_try, frac_F9_like_ge_0p05=float(np.mean(peak >= 0.05)),
                       frac_F9_like_ge_0p02=float(np.mean(peak >= 0.02)),
                       p_fg_nbar4_quantiles=[float(v) for v in np.quantile(pfg, [0.5, 0.84, 0.975])],
                       lambda_e_quantiles=[float(v) for v in np.quantile(le, [0.16, 0.5, 0.84, 0.975])],
                       lambda_f_quantiles=[float(v) for v in np.quantile(lf, [0.16, 0.5, 0.84, 0.975])],
                       frac_lambda_f_ge_1p5=float(np.mean(lf >= 1.5)), frac_lambda_f_ge_2=float(np.mean(lf >= 2.0)))
            summary["cells"][cell][f"sigma{sig:g}MHz"] = res
            print(f"{cell} sigma_sd={sig:g} MHz: accept {res['acceptance']:.2f}; F9-like(>=0.05) {100 * res['frac_F9_like_ge_0p05']:.2f}% "
                  f"(>=0.02: {100 * res['frac_F9_like_ge_0p02']:.2f}%); p_fg(4) med/84/97.5 "
                  f"{[f'{v:.1e}' for v in res['p_fg_nbar4_quantiles']]}; lambda_f med {res['lambda_f_quantiles'][1]:.2f} "
                  f"97.5% {res['lambda_f_quantiles'][3]:.2f}; P(lambda_f>=1.5) {100 * res['frac_lambda_f_ge_1p5']:.1f}% "
                  f"({time.time() - t0:.0f} s)", flush=True)
            if cell == "A4S4":
                axs[0].hist(peak[peak > 1e-4], bins=np.logspace(-4, 0, 41), histtype="step", color=SERIES[si],
                            label=f"σ_sd = {sig:g} MHz")
            axs[1 + (cell == "A4S5")].hist(lf, bins=np.linspace(0.8, 3.0, 45), histtype="step", color=SERIES[si],
                                           label=f"σ_sd = {sig:g} MHz")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"F9 peak height $\Delta P_g$ (A4-S4)")
    axs[0].set_ylabel("realizations")
    axs[0].axvline(0.05, color="#52514e", lw=0.8, ls="--")
    for ax, cell in zip(axs[1:], ("A4-S4", "A4-S5")):
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_f$ at $\bar n = 4$ (on-point / idle f$\to$e rate)")
        ax.set_title(cell, fontsize=9)
    for ax in axs:
        ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"tls_spectral_diffusion.{ext}", dpi=160)
    with open(ROOT / "data" / "tls" / "spectral_diffusion_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("saved data/tls/spectral_diffusion_summary.json")


if __name__ == "__main__":
    main()
