"""Diagnostic: uniform ancilla-decay scaling lambda that each model would need to reach the
measured CX^2 f-excess (Fig. 16), versus |alpha|^2, for the dressed (this work) and the
undressed (Hann-type) ULE simulations. This is the Fig. 16 analogue of AWS's beta analysis of
Fig. 36 (examples/fig36_beta_ocelot.py). A model that captures the photon-number dependence
needs a lambda that does not drift with |alpha|^2. Nothing is fitted; lambda is read off the
lambda-scan of examples/predict_cx2_ule_ocelot.py.

Usage: python examples/plot_cx2_lambda_needed.py
Output: figures/cx2_lambda_needed.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "validation" / "cx2_ule_excess.csv", encoding="utf-8")
                               if not l.startswith("#")))
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    use_style()
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, pair in zip(axs.flat, pairs):
        sel = [r for r in rows if r["pair"] == pair]
        x = np.array([float(r["nbar"]) for r in sel])
        for key, col, lab in (("lambda_needed", SERIES[0], "dressed (this work)"),
                              ("lambda_needed_std", SERIES[1], "undressed (Hann-type)")):
            y = np.array([float(r[key]) for r in sel])
            ax.plot(x, y, "o-", color=col, ms=4, label=lab)
        ax.axhline(1.0, color="#52514e", lw=0.6)
        ax.set_title(pair, fontsize=9)
        ax.set_ylim(0, 5.2)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"decay scaling $\lambda$ needed")
    axs.flat[0].legend(fontsize=7, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"cx2_lambda_needed.{ext}", dpi=160)
    print("saved figures/cx2_lambda_needed.png")


if __name__ == "__main__":
    main()
