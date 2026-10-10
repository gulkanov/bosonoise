"""Ancilla T1(e->g) versus coupler flux: the coupler-participation model against Fig. 18c (A3, A3-S3 coupler).

Model (no fit): the ancilla charge bath is flat and referenced to the measured idle
T1_ge(A3) = 35 us (Fig. 9c; the rep-code cooldown). A coupler charge bath with T1_c at the
'on' point is inherited through the coupler participation of the dressed e->g transition,
    1/T1(Phi) = s |X^q_eg(Phi)|^2 / (|X^q_eg(0)|^2 T1_ge) + |X^c_eg(Phi)|^2 / (nc01(on)^2 T1_c),
where s < 1 removes the idle share of the coupler loss so that T1(0) = T1_ge (measured),
with the dressed elements from the A3-S3 ladders (examples/predict_cx2_ule_ocelot.py cache).
Data: vector-extracted Fig. 18c (tools/extract_ocelot_fig18c.py; different cooldown;
VALIDATION ONLY). TLS appear as narrow dips; the coupler term as a smooth decline to Phi0/2.

Usage: python examples/fig18c_coupler_loss_ocelot.py
Output: figures/fig18c_coupler_loss.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
T1C_US = [2.0, 3.0, 10.0]


def main():
    pre = load_preset("ocelot")
    T1ge = pre["ancilla"]["A3"]["T1_ge_us"]
    grid = grid_for("A3-S3")
    fl = grid.fluxes
    Xq = np.array([l.X["q"][0, 0, 1, 0] for l in grid.ladders])
    Xc = np.array([l.X["c"][0, 0, 1, 0] for l in grid.ladders])
    nc01_on = grid.ladders[-1].nc01
    G_q = Xq ** 2 / (Xq[0] ** 2 * T1ge)  # 1/us
    part = (Xc / nc01_on) ** 2
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig18c.csv", encoding="utf-8")
                               if not l.startswith("#")))
    d_fl = np.array([float(r["flux_Phi0"]) for r in rows])
    d_T1 = np.array([float(r["T1_us"]) for r in rows])
    d_err = np.array([float(r["T1_err_us"]) for r in rows])
    use_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ok = d_err < 0.5 * d_T1
    ax.errorbar(d_fl[ok], d_T1[ok], yerr=d_err[ok], fmt="o", ms=4, mfc="none", color=TEXT_PRIMARY,
                label="Fig. 18c, A3 (other cooldown; validation)")
    ax.plot(fl, 1.0 / G_q, "--", color=SERIES[1], label="ancilla bath only (idle T1 = 35 µs, Fig. 9c)")
    for i, T1c in enumerate(T1C_US):
        G_c = part / T1c  # inherited coupler loss, 1/us
        scale = 1.0 - G_c[0] * T1ge  # idle share of the coupler loss removed from the ancilla background
        T1 = 1.0 / (scale * G_q + G_c)
        ax.plot(fl, T1, "-", color=SERIES[[0, 2, 3][i]], label=f"+ coupler loss, T1_c(on) = {T1c:g} µs")
        print(f"T1_c = {T1c:g} us: model T1_eg at Phi = 0 / 0.25 / 0.4 / 0.5: "
              + " / ".join(f"{np.interp(x, fl, T1):.1f}" for x in (0.0, 0.25, 0.4, 0.5)) + " us")
    print("coupler participation of e->g vs flux:", dict(zip(np.round(fl, 3), np.round(part, 4))))
    smooth_idle = np.median(d_T1[(d_fl <= 0.2) & ok])
    tail = d_T1[(d_fl >= 0.4) & ok & (d_T1 > 15)]
    print(f"data: median T1 for Phi <= 0.2: {smooth_idle:.1f} us; non-TLS points at Phi >= 0.4: {np.round(tail, 1)} "
          f"-> median {np.median(tail):.1f} us, ratio {smooth_idle / np.median(tail):.2f}")
    ax.set_xlabel(r"coupler flux $\Phi/\Phi_0$")
    ax.set_ylabel(r"ancilla $T_1$ (e→g) (µs)")
    ax.set_ylim(0, 60)
    ax.legend(fontsize=7, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fig18c_coupler_loss.{ext}", dpi=160)


if __name__ == "__main__":
    main()
