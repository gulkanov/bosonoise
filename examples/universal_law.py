"""The dressing-opened f -> g channel in measurable dispersive parameters (paper Sec. 2.2, Table 5).

For a flat charge bath the one-photon decay |f~, n> -> |g~, n+1> of a transmon ancilla has the rate

    Gamma_fg(n) = (n + 1) rho Gamma_fe,   rho = |<g~, n+1| n_q |f~, n>|^2 / ((n + 1) |<e|n_q|f>|^2)    (exact),

and in the dispersive limit (two interfering paths, paper Eq. (2))

    rho_law = |chi_ge| alpha / (2 |Delta (Delta + alpha)|),

with chi_ge the e-g dispersive shift per photon, alpha = f_ge - f_ef > 0 and Delta = f_mode - f_ge (signed); the coupling g
does not appear. With a gate of duration T = k / |chi| (k = 1/2 for a conditional pi rotation) the opened-channel
probability per gate and photon is k Gamma_fe alpha / (2 |Delta (Delta + alpha)|): chi drops out as well.

  Ocelot  exact dressed ladders of the storage-coupler-ancilla cell at the CX^2 'on' point (all 8 pairs); chi_ge, alpha
          and Delta read off the same ladders (dressed energies at n = 0, 1).
  GKP     transmon in the charge basis with the cavity as a mode, devices A (Sivak et al. 2023) and B (Eickbusch et al.
          2022) of paper App. G; rho from the first-order element (dynamics/floquet.one_photon_element),
          chi_ge from second-order perturbation theory with counter-rotating terms.

Usage: python examples/universal_law.py [--plot]
Output: data/validation/universal_law.csv, figures/universal_law.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.dynamics.floquet import one_photon_element
from bosonoise.model.transmon import Transmon
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

from predict_cx2_ule_ocelot import PAIRS, grid_for  # same directory
from universal_map import GKP  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "universal_law.csv"
KEYS = ["system", "kind", "f_ge_GHz", "alpha_MHz", "Delta_MHz", "chi_ge_kHz", "rho_exact", "rho_law", "ratio",
        "rho_exact_n10", "per_gate_per_photon_k_half_per_us"]


def ocelot_rows():
    rows = []
    for pair in PAIRS:
        on = grid_for(pair).ladders[-1]
        E, Xq = on.E, on.X["q"]  # E[b, n]: b = g, e, f (coupler in 0)
        f_ge, f_ef = E[1, 0] - E[0, 0], E[2, 0] - E[1, 0]
        f_s = E[0, 1] - E[0, 0]
        chi = (E[1, 1] - E[1, 0]) - (E[0, 1] - E[0, 0])
        alpha, delta = f_ge - f_ef, f_s - f_ge
        ref = abs(Xq[1, 0, 2, 0]) ** 2
        rho = abs(Xq[0, 1, 2, 0]) ** 2 / ref
        rho10 = abs(Xq[0, 11, 2, 10]) ** 2 / ref / 11
        rows.append(_row(f"Ocelot {pair}", "ocelot", f_ge, alpha, delta, chi, rho, rho10))
    return rows


def gkp_rows():
    rows = []
    for name, d in GKP.items():
        E, n = Transmon(EJ=d["EJ"], EC=d["EC"], ng=0.0, ncut=40).spectrum(20)
        g, w = d["g"], d["fc"]

        def shift(i):  # second-order dispersive shift of level i per photon, with counter-rotating terms
            return g ** 2 * sum(abs(n[i, k]) ** 2 * (1 / (E[i] - E[k] - w) - 1 / (E[k] - E[i] - w))
                                for k in range(len(E)) if k != i)
        chi = shift(1) - shift(0)
        f_ge, f_ef = E[1] - E[0], E[2] - E[1]
        ref = abs(n[1, 2]) ** 2
        rho = one_photon_element(E, n, g, w, 2, 0, 0) ** 2 / ref
        rho10 = one_photon_element(E, n, g, w, 2, 0, 10) ** 2 / ref / 11
        rows.append(_row(name, "gkp", f_ge, f_ge - f_ef, w - f_ge, chi, rho, rho10))
    return rows


def _row(system, kind, f_ge, alpha, delta, chi, rho, rho10):
    law = abs(chi) * alpha / (2 * abs(delta * (delta + alpha)))
    per_gate = 0.5 * alpha / (2 * abs(delta * (delta + alpha))) * 1e-3  # k = 1/2; GHz^-1 = ns -> us (times Gamma_fe in 1/us)
    return dict(system=system, kind=kind, f_ge_GHz=f_ge, alpha_MHz=alpha * 1e3, Delta_MHz=delta * 1e3,
                chi_ge_kHz=chi * 1e6, rho_exact=rho, rho_law=law, ratio=rho / law, rho_exact_n10=rho10,
                per_gate_per_photon_k_half_per_us=per_gate)


def save(rows):
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Dressing-opened f -> g branching per photon: exact (rho_exact at n = 0, rho_exact_n10 at n = 10) against "
                 "the dispersive law |chi_ge| alpha / (2 |Delta (Delta + alpha)|); per_gate_per_photon_k_half_per_us = "
                 "alpha / (4 |Delta (Delta + alpha)|) in us (multiply by Gamma_fe in 1/us). examples/universal_law.py\n")
        fh.write(",".join(KEYS) + "\n")
        for r in rows:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in KEYS) + "\n")
    print("saved", OUT)


def load():
    return [{k: (v if k in ("system", "kind") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(l for l in open(OUT, encoding="utf-8") if not l.startswith("#"))]


def plot(rows):
    use_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    for kind, c, label in (("ocelot", SERIES[0], "Ocelot pairs at the CX² 'on' point"),
                           ("gkp", SERIES[2], "GKP devices A, B (if the ancilla used f)")):
        sel = [r for r in rows if r["kind"] == kind]
        ax.plot([r["rho_law"] for r in sel], [r["rho_exact"] for r in sel], "o", ms=6, color=c, label=label)
    lim = np.array([1e-6, 3e-2])
    ax.plot(lim, lim, "-", color=TEXT_PRIMARY, lw=0.8, label="exact = law")
    ax.fill_between(lim, lim / 1.3, lim * 1.3, color=TEXT_PRIMARY, alpha=0.08, lw=0, label="±30%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"dispersive law $|\chi_{ge}|\,\alpha\,/\,(2|\Delta(\Delta+\alpha)|)$")
    ax.set_ylabel(r"exact one-photon f→g branching per photon, $\rho$")
    ax.set_title("The dressing-opened f→g channel in measurable dispersive parameters", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"universal_law.{ext}", dpi=160)
    print("saved figures/universal_law.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="redraw from the CSV")
    a = ap.parse_args()
    if a.plot:
        return plot(load())
    rows = ocelot_rows() + gkp_rows()
    for r in rows:
        print(f"{r['system']:24s} f_ge {r['f_ge_GHz']:.4f} GHz  alpha {r['alpha_MHz']:6.1f}  Delta {r['Delta_MHz']:7.1f} MHz  "
              f"chi_ge {r['chi_ge_kHz']:8.1f} kHz  rho {r['rho_exact']:.3e}  law {r['rho_law']:.3e}  ratio {r['ratio']:.2f}  "
              f"rho(n=10) {r['rho_exact_n10']:.3e}")
    save(rows)
    plot(rows)


if __name__ == "__main__":
    main()
