"""Ocelot Fig. 36 (A4-S5, CX^2, ancilla |f>): the T1-scaling factor beta required by the
AWS model grows with |alpha|^2 -- evidence of a photon-number-dependent channel.

Left: measured CX^2 bit-flip probability (vector-extracted) with the AWS simulations for
constant beta = 1, 1.5, 2, 2.5. Right: the beta needed at each |alpha|^2, obtained by
log-linear interpolation between the published beta curves (extrapolated beyond 2.5).
VALIDATION DATA ONLY; nothing here calibrates our model.

Usage: python examples/fig36_beta_ocelot.py
Output: figures/fig36_beta_ocelot.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig36.csv", encoding="utf-8")
                               if not l.startswith("#")))
    a2 = np.array([float(r["alpha2"]) for r in rows])
    data = np.array([float(r["data"]) for r in rows])
    err = np.array([float(r["data_err"]) for r in rows])
    betas = [1.0, 1.5, 2.0, 2.5]
    sims = np.array([[float(r[f"sim_beta{b:g}"]) for r in rows] for b in betas])  # (4, 7)
    need = []
    for i in range(len(a2)):
        y = np.log(sims[:, i])
        p = np.polyfit(betas, y, 1)  # log p nearly linear in beta over this range
        need.append((np.log(data[i]) - p[1]) / p[0])
    need = np.array(need)
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for j, b in enumerate(betas):
        ax[0].semilogy(a2, sims[j], "-", marker=".", color=SERIES[j], label=f"AWS sim, β = {b:g}")
    ax[0].errorbar(a2, data, yerr=err, fmt="o", ms=5, mfc="none", color=TEXT_PRIMARY, label="measured (Fig. 36)")
    ax[0].set_xlabel(r"$|\alpha|^2$")
    ax[0].set_ylabel(r"CX$^2$ bit-flip probability per cycle")
    ax[0].set_title("Ocelot A4-S5, ancilla |f⟩ (vector-extracted)", fontsize=9)
    ax[0].legend(fontsize=7)
    ok = a2 >= 1.5  # at |alpha|^2 = 1 the beta curves bunch together: beta is unconstrained by the error bar
    ax[1].plot(a2[ok], need[ok], "o-", color=SERIES[0], label="constrained")
    ax[1].plot(a2[~ok], need[~ok], "o", mfc="none", color=SERIES[0], label="|α|²=1: unconstrained (curves bunch)")
    ax[1].legend(fontsize=7, loc="lower right")
    ax[1].axhline(2.5, color="#52514e", lw=0.8, ls="--")
    ax[1].text(1.05, 2.58, "β = 2.5 used by AWS (tuned at |α|²=4)", fontsize=7, color="#52514e")
    ax[1].set_xlabel(r"$|\alpha|^2$")
    ax[1].set_ylabel("β required at each point")
    ax[1].set_title("Required T1-scaling grows with photon number", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig36_beta_ocelot.{ext}", dpi=160)
    print("required beta:", dict(zip(a2.round(1), need.round(2))))


if __name__ == "__main__":
    main()
