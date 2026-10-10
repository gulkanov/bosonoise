"""Reproduce Fig. 7 of Putterman et al. (arXiv:2409.13025) with our calibrated charge-basis model.

Solid lines: our storage-coupler-ancilla model (calibration A, data/calibration/ocelot_A2S3_fitA.yaml).
Hollow markers: AWS circuit-quantization model (vector-extracted Fig. 7, data/digitized/ocelot_fig7.csv).
Black points: measured chi at the 'on' point (Fig. 12, A2<->S3).
Dotted lines: calibration B (data-anchored on the measured on-point chi).

Usage:  python examples/fig7_ocelot.py
Output: figures/fig7_ocelot_reproduction.png/.pdf, data/validation/fig7_ocelot_model.csv (our curves, fits A and B)
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.spectra import sweep

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRUNC = Truncation(n_s=10, n_c=7, n_q=9, ncut=25)


def load_fit(tag):
    rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{tag}.yaml", encoding="utf-8"))
    return CellParams(**rec["params"])


def load_aws():
    lines = [ln for ln in open(ROOT / "data" / "digitized" / "ocelot_fig7.csv", encoding="utf-8") if not ln.startswith("#")]
    rows = list(csv.reader(lines))
    data = np.array(rows[1:], float)
    return {h: data[:, i] for i, h in enumerate(rows[0])}


def main():
    use_style()
    aws = load_aws()
    fl = np.linspace(0.0, 0.5, 101)
    ours = {k: sweep(StorageCouplerAncilla(load_fit(k), TRUNC), fl) for k in ("A", "B")}
    A, B = ours["A"], ours["B"]
    sub = slice(None, None, 6)  # thin the AWS markers
    xa = aws["flux_Phi0"][sub]
    mk = dict(ls="none", marker="o", mfc="none", mew=1.0, ms=5)

    fig, ax = plt.subplots(5, 1, figsize=(4.6, 10.5), sharex=True, constrained_layout=True)
    # (a) frequencies
    for j, (k, col, lab) in enumerate([("f_s", "omega_s_GHz", r"$f_s$"), ("f_ge", "omega_a_ge_GHz", r"$f_{a,ge}$"),
                                       ("f_ef", "omega_a_ef_GHz", r"$f_{a,ef}$")]):
        ax[0].plot(fl, A[k], color=SERIES[j], label=f"{lab} (ours)")
        ax[0].plot(xa, aws[col][sub], color=SERIES[j], **mk)
    ax[0].set_ylabel("frequency (GHz)")
    ax[0].legend(ncol=3, loc="center left", fontsize=7)
    # (b) chi ratio
    ratio = A["chi_ge"] / A["chi_gf"]
    D = A["f_s"] - A["f_ge"]
    K = A["f_ef"] - A["f_ge"]
    naive = (D - 2 * K) / (2 * D - K)
    m = np.abs(A["chi_gf"]) > 20e-6  # ratio undefined where both chi ~ 0 (effective coupling crosses zero near Phi ~ 0.13)
    ax[1].plot(fl[m], ratio[m], color=SERIES[0], label="ours, exact")
    ax[1].plot(fl[m], naive[m], color=SERIES[1], label="ours, Eq. (D5)")
    ax[1].plot(xa, aws["chi_ratio_numerics"][sub], color=SERIES[0], **mk)
    ax[1].plot(xa, aws["chi_ratio_naive_D5"][sub], color=SERIES[1], **mk)
    mB = np.abs(B["chi_gf"]) > 20e-6
    ax[1].plot(fl[mB], (B["chi_ge"] / B["chi_gf"])[mB], color=SERIES[0], ls=":", label="ours, calib. B")
    ax[1].errorbar([0.5], [1.18 / 1.14], yerr=[1.18 / 1.14 * np.hypot(0.01 / 1.18, 0.01 / 1.14)], color=TEXT_PRIMARY,
                   marker="s", ms=4, ls="none", label="measured (Fig. 12)")
    ax[1].set_ylabel(r"$\chi_{ge}/\chi_{gf}$")
    ax[1].legend(fontsize=7, loc="lower left")
    # (c) chi
    for j, (k, col, lab) in enumerate([("chi_ge", "chi_ge_MHz", r"$\chi_{ge}$"), ("chi_gf", "chi_gf_MHz", r"$\chi_{gf}$")]):
        ax[2].plot(fl, A[k] * 1e3, color=SERIES[j], label=f"{lab} (ours)")
        ax[2].plot(fl, B[k] * 1e3, color=SERIES[j], ls=":")
        ax[2].plot(xa, aws[col][sub], color=SERIES[j], **mk)
    ax[2].errorbar([0.5, 0.5], [-1.18, -1.14], yerr=[0.01, 0.01], color=TEXT_PRIMARY, marker="s", ms=4, ls="none",
                   label="measured (Fig. 12)")
    ax[2].set_ylabel(r"$\chi/2\pi$ (MHz)")
    ax[2].legend(fontsize=7, loc="lower left")
    # (d) conditional Kerr
    for j, (k, col, lab) in enumerate([("K_g", "K_s_g_kHz", r"$K_{s|g}$"), ("K_e", "K_s_e_kHz", r"$K_{s|e}$"),
                                       ("K_f", "K_s_f_kHz", r"$K_{s|f}$")]):
        ax[3].plot(fl, A[k] * 1e6, color=SERIES[j], label=f"{lab} (ours)")
        ax[3].plot(xa, aws[col][sub], color=SERIES[j], **mk)
    ax[3].set_ylabel(r"storage Kerr $K/2\pi$ (kHz)")
    ax[3].legend(fontsize=7, loc="lower left", ncol=3)
    # (e) coupler
    ax[4].plot(fl, A["f_c"], color=SERIES[0], label=r"$f_c$ (ours)")
    ax[4].errorbar([0.0, 0.5], [9.0, 6.6], yerr=[0.05, 0.05], color=TEXT_PRIMARY, marker="s", ms=4, ls="none",
                   label="AWS model (App. D3)")
    ax[4].set_ylabel("coupler (GHz)")
    ax[4].set_xlabel(r"coupler flux $\Phi_{x,c}/\Phi_0$")
    ax[4].legend(fontsize=7, loc="lower left")
    for i, a in enumerate(ax):
        a.text(-0.16, 1.0, f"({'abcde'[i]})", transform=a.transAxes, fontsize=9, va="top")
    ax[0].set_title("Ocelot A2-S3: our model (lines) vs AWS model (o)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig7_ocelot_reproduction.{ext}", dpi=170)
    keys = ("f_s", "f_ge", "f_ef", "f_c", "chi_ge", "chi_gf", "K_g", "K_e", "K_f")
    with open(ROOT / "data" / "validation" / "fig7_ocelot_model.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Our A2-S3 model against coupler flux, fits A and B (all in GHz; chi and K are the dispersive shifts and "
                 "conditional storage Kerr). examples/fig7_ocelot.py\n")
        fh.write("flux_Phi0," + ",".join(f"{k}_{t}" for t in ("A", "B") for k in keys) + "\n")
        for i, x in enumerate(fl):
            fh.write(f"{x:.4f}," + ",".join(f"{ours[t][k][i]:.9g}" for t in ("A", "B") for k in keys) + "\n")
    # numbers for the notebook
    i5 = -1
    print("at Phi=0.5 (ours A / ours B / AWS):")
    for k, col, sc in [("chi_ge", "chi_ge_MHz", 1e3), ("chi_gf", "chi_gf_MHz", 1e3), ("K_g", "K_s_g_kHz", 1e6),
                       ("K_e", "K_s_e_kHz", 1e6), ("K_f", "K_s_f_kHz", 1e6), ("f_ge", "omega_a_ge_GHz", 1), ("f_ef", "omega_a_ef_GHz", 1)]:
        print(f"  {k:6s} {A[k][i5] * sc:10.4f} {B[k][i5] * sc:10.4f} {aws[col][-1]:10.4f}")
    print("  ratio  ", round(ratio[-1], 4), round(B['chi_ge'][-1] / B['chi_gf'][-1], 4), round(aws['chi_ratio_numerics'][-1], 4))
    print("saved figures/fig7_ocelot_reproduction.png")


if __name__ == "__main__":
    main()
