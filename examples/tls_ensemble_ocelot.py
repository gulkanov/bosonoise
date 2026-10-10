"""Statistical TLS ensembles for the Ocelot cells: F9-like anomalies, CX ancilla faults, lambda.

For each cell (A4-S4: F9 pair; A4-S5: App. J pair; A2-S3) and each TLS density, sample
N realizations of a TLS ensemble on the ancilla charge (Poisson number in a frequency
window, log-uniform couplings and linewidths; ranges are CLI parameters, see
paper App. E.4 for the literature basis) and compute, per realization:
  * F9 observable (A4-S4 protocol: |f>, 800 ns at 'on'): interior peak height of
    Delta P_g(|alpha|^2) (peak at 2 <= |alpha|^2 <= 18);
  * CX ancilla faults during T_CX at 'on' starting from |f~, n>, Poisson-averaged versus nbar:
    P(f -> g, direct or via e) -- the bit-flip-relevant double-decay-type events;
  * lambda factors at nbar = 4: effective decay rate of |e~>, |f~> at 'on' relative to the
    coupler-idle rates 1/T1 (App. J compares 2.5 (sim) and 1.5 (pulsed measurement)).
No parameter is tuned to the error data; the ensemble statistics are the prediction.

Usage: python examples/tls_ensemble_ocelot.py --prior literature --area-um2 0.02 0.05 0.15 --nreal 2000 --tag literature_prior
       python examples/tls_ensemble_ocelot.py --nreal 2000 --tag sensitivity_broadprior      (broad prior, --density)
Output: data/tls/ensemble_summary_<tag>.json, figures/tls_ensemble_<tag>.png/.pdf
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

from bosonoise.channels.tls import (AncillaBath, TLSEnsemble, poisson_average, sample_tls,
                                   sample_tls_literature, surface_mean_rate_per_us)
from bosonoise.plotting import SERIES, TEXT_SECONDARY, use_style

from tls_single_map_ocelot import T1, build  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = {
    "A4S4": ("data/calibration/ocelot_A4_fcmin6.569.yaml", "S4", "A4"),
    "A4S5": ("data/calibration/ocelot_A4_fcmin6.569.yaml", "S5", "A4"),
    "A2S3": ("data/calibration/ocelot_A2S3_fitA.yaml", None, "A2"),
}
NBAR = np.linspace(0, 20, 81)


def realization_observables(bath, tls, T_cx, t_f9=800.0):
    pops_f9 = bath.f_state_populations(tls, t_f9)
    avg_f9, _ = poisson_average(pops_f9, NBAR)
    dPg = avg_f9[:, 0] - avg_f9[0, 0]
    k = int(np.argmax(dPg))
    f9_peak = dPg[k] if 2.0 <= NBAR[k] <= 18.0 else 0.0
    pops_cx = bath.f_state_populations(tls, T_cx)
    p_fg, _ = poisson_average(pops_cx[:, 0], NBAR)  # f -> g within T_CX (direct + via e)
    re = bath.state_rates(tls, T_cx, "e")
    rf = bath.state_rates(tls, T_cx, "f")
    lam_e = float(poisson_average(re, np.array([4.0]))[0][0] / (1e-3 / bath.meta["T1_ge_us"]))
    lam_f = float(poisson_average(rf, np.array([4.0]))[0][0] / (1e-3 / bath.meta["T1_fe_us"]))
    return f9_peak, NBAR[k], p_fg, lam_e, lam_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--density", nargs="+", type=float, default=[1.0, 5.0, 20.0], help="TLS per GHz in the window")
    ap.add_argument("--window", nargs=2, type=float, default=[4.8, 5.4])
    ap.add_argument("--g01-MHz", nargs=2, type=float, default=[0.02, 2.0])
    ap.add_argument("--T1tls-us", nargs=2, type=float, default=[0.3, 10.0])
    ap.add_argument("--nreal", type=int, default=1000)
    ap.add_argument("--Tcx", type=float, default=400.0, help="CX duration at 'on' (ns); App. J: ~400 ns for A4-S5")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--tag", default="provisional")
    ap.add_argument("--prior", choices=("broad", "literature"), default="broad")
    ap.add_argument("--area-um2", nargs="+", type=float, default=[0.05], help="junction area(s), literature prior")
    ap.add_argument("--surf-T1-us", type=float, default=100.0, help="TLS-limited T1 setting the surface density")
    ap.add_argument("--idle-tol", type=float, default=1.5, help="reject if idle total rate > tol * measured")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = ROOT / "data" / "tls"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(a), "cells": {}}
    use_style()
    fig, axs = plt.subplots(1, len(a.cells), figsize=(4.6 * len(a.cells), 3.8), constrained_layout=True, squeeze=False)
    for ci, cell in enumerate(a.cells):
        calib, key, anc = CELLS[cell]
        print(f"cell {cell}")
        tab_on, tab_idle = build(calib, key)
        if a.prior == "literature":
            G_surf = surface_mean_rate_per_us(1.0 / a.surf_T1_us)  # 1/us, mean explicit surface contribution
            scale = max(0.0, 1.0 - G_surf * T1[anc][0])
            bath = AncillaBath(tab_on, tab_idle, *T1[anc], background_scale=scale)
            print(f"  literature prior: surface T1_TLS={a.surf_T1_us} us -> mean explicit surface rate "
                  f"{G_surf:.4f}/us; background scale {scale:.3f}")
        else:
            bath = AncillaBath(tab_on, tab_idle, *T1[anc])
        none = TLSEnsemble(np.array([]), np.array([]), np.array([]))
        base = realization_observables(bath, none, a.Tcx)
        summary["cells"][cell] = {"no_tls": {"f9_peak": float(base[0]), "p_fg_nbar4": float(np.interp(4, NBAR, base[2])),
                                             "lambda_e": base[3], "lambda_f": base[4]}}
        grid = a.density if a.prior == "broad" else a.area_um2
        for di, rho in enumerate(grid):
            t0 = time.time()
            f9, P, le, lf = [], [], [], []
            n_try = 0
            while len(f9) < a.nreal:
                n_try += 1
                if a.prior == "broad":
                    tls = sample_tls(rng, rho, tuple(a.window), (a.g01_MHz[0] * 1e-3, a.g01_MHz[1] * 1e-3),
                                     (1e-3 / a.T1tls_us[1], 1e-3 / a.T1tls_us[0]))
                else:
                    tls = sample_tls_literature(rng, tuple(a.window), surf_Gamma1_per_us=1.0 / a.surf_T1_us,
                                                junc_area_um2=rho)
                    re_i, rf_i = bath.idle_tls_rates(tls)
                    bg_e = bath.meta["background_scale"] * 1e-3 / T1[anc][0]
                    bg_f = bath.meta["background_scale"] * 1e-3 / T1[anc][1]
                    if re_i + bg_e > a.idle_tol * 1e-3 / T1[anc][0] or rf_i + bg_f > a.idle_tol * 1e-3 / T1[anc][1]:
                        continue  # inconsistent with the measured coupler-idle T1 (static data)
                o = realization_observables(bath, tls, a.Tcx)
                f9.append(o[0]); P.append(o[2]); le.append(o[3]); lf.append(o[4])
            f9, P, le, lf = map(np.array, (f9, P, le, lf))
            q = np.quantile(P, [0.5, 0.84, 0.975], axis=0)
            res = {
                "frac_F9_like_ge_0.05": float(np.mean(f9 >= 0.05)),
                "frac_F9_like_ge_0.02": float(np.mean(f9 >= 0.02)),
                "p_fg_quantiles_at_nbar": {str(nb): [float(np.interp(nb, NBAR, qq)) for qq in q] for nb in (1, 2, 4, 8, 12)},
                "lambda_e_quantiles": [float(v) for v in np.quantile(le, [0.16, 0.5, 0.84, 0.975])],
                "lambda_f_quantiles": [float(v) for v in np.quantile(lf, [0.16, 0.5, 0.84, 0.975])],
            }
            res["acceptance_idle_T1"] = len(f9) / n_try
            summary["cells"][cell][f"{'rho' if a.prior == 'broad' else 'AJ'}{rho:g}"] = res
            print(f"  {'rho' if a.prior == 'broad' else 'A_J'}={rho:g}: accept {res['acceptance_idle_T1']:.2f}; F9-like>=0.05 in {100 * res['frac_F9_like_ge_0.05']:.1f}% ; "
                  f"p_fg(nbar=4) median/84/97.5% = {[f'{v:.1e}' for v in res['p_fg_quantiles_at_nbar']['4']]}; "
                  f"lambda_e med {res['lambda_e_quantiles'][1]:.2f}, lambda_f med {res['lambda_f_quantiles'][1]:.2f} "
                  f"({time.time() - t0:.0f} s)")
            ax = axs[0, ci]
            ax.fill_between(NBAR, q[0], q[2], color=SERIES[di], alpha=0.15, lw=0)
            lab = f"{rho:g} TLS/GHz" if a.prior == "broad" else f"A_J={rho:g} um^2"
            ax.plot(NBAR, q[0], color=SERIES[di], label=f"{lab}: median, band to 97.5%")
        ax = axs[0, ci]
        ax.plot(NBAR, base[2], color=TEXT_SECONDARY, ls="--", label="no TLS (dressed background)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\bar n = |\alpha|^2$")
        ax.set_ylabel(fr"$P(f\to g)$ within one CX ({a.Tcx:g} ns)")
        ax.set_title(f"{cell}: TLS ensembles ({a.tag})", fontsize=9)
        ax.legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"tls_ensemble_{a.tag}.{ext}", dpi=160)
    with open(out / f"ensemble_summary_{a.tag}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("saved", out / f"ensemble_summary_{a.tag}.json")


if __name__ == "__main__":
    main()
