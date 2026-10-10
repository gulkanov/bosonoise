"""Zero-fit comparison: dressing-opened direct f->g in the A4-S4 model vs Ocelot Fig. 18b (F9).

Model (no parameter tuned to error data):
  * dressed charge matrix elements R_fg(n) = sum_m |<g~,m|n_q|f~,n>|^2 / |<e|n_q|f>|^2 at the
    'on' point from the static A4 calibration (data/branches/dressed_charge_elements_*_nmax40.csv);
  * flat charge-bath spectrum between 4.97 and 5.11 GHz, so Gamma_fg(n) = R_fg(n) / T1(f->e);
  * T1(f->e) of A4: 25 us (Fig. 9d, coupler off, rep-code cooldown) and 12.5 us (App. J
    erasure-based estimate for A4-S5) as a bracket; storage in a coherent state, Poisson
    photon statistics, no storage decay during the 800 ns pulse, ramps ignored.
Data: vector-extracted Fig. 18b (A4-S4, 800 ns, DIFFERENT cooldown), Delta P_g = P_g - P_g(0).

Usage: python examples/compare_f9_dressing.py
Output: figures/f9_dressing_vs_data.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from bosonoise.plotting import SERIES, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_PULSE_US = 0.8


def load_csv(path):
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    d = load_csv(ROOT / "data" / "digitized" / "ocelot_fig18b.csv")
    m = load_csv(ROOT / "data" / "branches" / "dressed_charge_elements_ocelot_A4S4_fitA_flux0.500_nmax40.csv")
    n, R = m["n"].astype(int), m["R_fg_q"]
    nbar = np.linspace(0, 20, 201)
    W = poisson.pmf(n[None, :], nbar[:, None])  # (nbar, n)
    tail = 1 - W.sum(axis=1)
    dR = W @ (R - R[0])
    use_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ax.plot(d["alpha2"], d["P_g"] - d["P_g"][0], "o", ms=3.5, color=SERIES[0], label="Ocelot Fig. 18b (A4-S4), data")
    out = {}
    for T1, ls in ((25.0, "-"), (12.5, "--")):
        dP = T_PULSE_US / T1 * dR
        out[T1] = dP
        ax.plot(nbar, dP, ls, color=SERIES[1], label=f"dressing-opened f→g, model, $T_1^{{fe}}$={T1:g} µs")
    ax.set_xlabel(r"storage mean photon number $|\alpha|^2$")
    ax.set_ylabel(r"$\Delta P_g$ after 800 ns at 'on'")
    ax.set_title("Direct f→g in A4-S4: smooth dressing vs data (no fit)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"f9_dressing_vs_data.{ext}", dpi=160)
    i10 = np.argmin(np.abs(nbar - 10.2))
    j10 = np.argmin(np.abs(d["alpha2"] - 10.2))
    data_peak = d["P_g"][j10] - d["P_g"][0]
    print(f"Poisson tail beyond n=40 at nbar=20: {tail[-1]:.1e}")
    print(f"at |alpha|^2=10.2: data dP_g={data_peak:.4f}; model {out[25.0][i10]:.2e} (T1=25us), {out[12.5][i10]:.2e} (T1=12.5us)"
          f" -> data/model = {data_peak / out[25.0][i10]:.0f} .. {data_peak / out[12.5][i10]:.0f}")
    print(f"at |alpha|^2=20: data dP_g={d['P_g'][-1] - d['P_g'][0]:.4f}; model {out[25.0][-1]:.2e} .. {out[12.5][-1]:.2e}")
    print("model is monotonic in |alpha|^2:", bool(np.all(np.diff(out[25.0]) >= -1e-12)))
    print("saved figures/f9_dressing_vs_data.png")


if __name__ == "__main__":
    main()
