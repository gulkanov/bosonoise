"""Ramp-induced storage->ancilla exchange during a CX flux pulse (first-order APT).

For the dressed |g, n=1> (storage one photon, ancilla ground, coupler ground) we compute,
with first-order adiabatic perturbation theory validated against brute-force propagation
(paper App. C.4), the probability of ending in |e, n=0> after a flat-top Gaussian-edge flux
pulse 0 -> Phi0/2 -> 0, versus the edge width sigma. To first order the amplitude scales
as sqrt(n), so P(n) ~ n * P(1). Ramp shapes are NOT published for most Ocelot pairs
(64 ns ramps for A1-S1 and A4-S4); sigma is scanned.

Usage: python examples/apt_ramp_scan_ocelot.py [--flat 300] [--nsig 4]
Output: data/apt/ramp_scan.csv, figures/apt_ramp_scan.png/.pdf
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.dynamics.apt import adiabatic_path, apt_amplitudes
from bosonoise.dynamics.cell_td import TimeDependentCell
from bosonoise.dynamics.flux import flat_top_gaussian
from bosonoise.model import Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = {
    "A2S3": ("data/calibration/ocelot_A2S3_fitA.yaml", None),
    "A4S4": ("data/calibration/ocelot_A4_fcmin6.569.yaml", "S4"),
    "A4S5": ("data/calibration/ocelot_A4_fcmin6.569.yaml", "S5"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", type=float, default=300.0, help="flat-top duration (ns)")
    ap.add_argument("--nsig", type=float, default=4.0, help="Gaussian edge truncated at nsig*sigma")
    ap.add_argument("--nc", type=int, default=12)
    a = ap.parse_args()
    sigmas = np.array([1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40], float)
    res = {}
    for name, (calib, cell) in CELLS.items():
        p = load_cell_params(ROOT / calib, cell)
        td = TimeDependentCell(p, Truncation(n_s=3, n_c=a.nc, n_q=5, ncut=25), flux_ref=0.25)
        E0, V0 = td.eigh(0.0)
        i0 = int(np.argmax(np.abs(V0[td.static.basis.index[(1, 0, 0)], :]) ** 2))
        ie = int(np.argmax(np.abs(V0[td.static.basis.index[(0, 0, 1)], :]) ** 2))
        t0 = time.time()
        path = adiabatic_path(td, np.linspace(0, 0.5, 201), i0, n_targets=8)
        k = int(np.nonzero(path.targets0 == ie)[0][0])
        P = []
        for s in sigmas:
            pulse = lambda t, s=s: flat_top_gaussian(t, 0.0, a.flat, s, 0.5, n_sigma=a.nsig)  # noqa: E731
            T = a.flat + 2 * a.nsig * s
            t = np.arange(0.0, T + 1e-9, min(0.01, s / 50))
            P.append(abs(apt_amplitudes(path, pulse, t)[k]) ** 2)
        res[name] = np.array(P)
        det = (E0[i0] - E0[ie]) * 1e3
        print(f"{name}: idle detuning (|g,1> - |e,0>) = {det:.1f} MHz; path min overlap {path.overlap_min:.5f} "
              f"({time.time() - t0:.0f} s)")
        print("   P(|g,1> -> |e,0>) vs sigma: " + ", ".join(f"{s:g}ns:{v:.1e}" for s, v in zip(sigmas, res[name])))
    out = ROOT / "data" / "apt"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ramp_scan.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# first-order APT, |g,1> -> |e,0>, flat top {a.flat} ns, Gaussian edges truncated at {a.nsig} sigma; P(n) ~ n P(1)\n")
        fh.write("sigma_ns," + ",".join(res) + "\n")
        for i, s in enumerate(sigmas):
            fh.write(f"{s:g}," + ",".join(f"{res[c][i]:.4e}" for c in res) + "\n")
    use_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    for j, c in enumerate(res):
        ax.semilogy(sigmas, np.maximum(res[c], 1e-16), "o-", ms=4, color=SERIES[j], label=c)
    ax.axvline(16, color="#52514e", lw=0.8, ls=":")
    ax.set_xlabel(r"Gaussian edge width $\sigma$ (ns)")
    ax.set_ylabel(r"$P(|g,1\rangle\to|e,0\rangle)$ per CX pulse")
    ax.set_title(f"Ramp-induced storage→ancilla exchange (APT; flat {a.flat:g} ns, edges cut at {a.nsig:g}σ)", fontsize=8)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"apt_ramp_scan.{ext}", dpi=160)


if __name__ == "__main__":
    main()
