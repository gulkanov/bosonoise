"""Deterministic Kerr distortion of the CX^2 cycle: calibration at |alpha|^2 = 1 versus per |alpha|^2.

During the unstabilized CX^2 the storage self-Kerr of the dressed g and f branches at 'on'
shears the cat. The experiment calibrates the linear storage phase (and the CX^2 length)
at |alpha|^2 = 1 only (App. F7) and notes that per-|alpha|^2 calibration could improve the
gates (p.29). Here, without any bath, we compare the bit-flip probability of the g and f
protocols for (a) the |alpha|^2 = 1 calibration used in predict_cx2_ule_ocelot.py and
(b) a calibration repeated at every |alpha|^2, and report the optimal phases.
Re-stabilization map with the nominal buffer detuning (App. F8), as in the main prediction.

Usage: python examples/kerr_calibration_ocelot.py
Output: data/validation/cx2_kerr_calibration.csv (model only)
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases, initial_blocks, simulate_cx2
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, NMAX, PAIRS, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBARS = [1.0, 2.0, 3.0, 4.0]


def main():
    pre = load_preset("ocelot")
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    out = []
    print("pair  nbar | p_g: cal@1  cal@nbar (delta_g) | p_f: cal@1  cal@nbar (delta_f)   [x1e-3]")
    for pair in PAIRS:
        anc, st = pair.split("-")
        grid = grid_for(pair)
        chi = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
        pulse = calibrate_cx2_to_length(grid, L[pair] * 1e3 + 0.5 / chi)
        kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
        projs = {nb: AsymptoticProjector(nb, kerr_ratio=kr) for nb in NBARS}
        corr1 = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0])
        for nb in NBARS:
            corr = calibrate_phases(grid, pulse, NMAX, nb, projs[nb])
            row = dict(pair=pair, nbar=nb, delta_g_cal1=corr1[0], delta_f_cal1=corr1[1], delta_g=corr[0], delta_f=corr[1])
            for lev, tag in ((0, "g"), (2, "f")):
                for c, name in ((corr1, "cal1"), (corr, "calnbar")):
                    rho = simulate_cx2(grid, pulse, BathModel({}), initial_blocks(nb, lev, NMAX), NMAX, phase_corr=c)
                    row[f"p{tag}_{name}"] = projs[nb].p_flip(rho / np.trace(rho).real)
            out.append(row)
            print(f"{pair} {nb:.0f} | {row['pg_cal1'] * 1e3:6.3f} {row['pg_calnbar'] * 1e3:6.3f} ({corr[0]:+.3f}) | "
                  f"{row['pf_cal1'] * 1e3:6.3f} {row['pf_calnbar'] * 1e3:6.3f} ({corr[1]:+.3f})", flush=True)
    path = ROOT / "data" / "validation" / "cx2_kerr_calibration.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Deterministic (no-bath) CX^2 bit-flip probability, Pauli per cycle: storage phase / gate length\n"
                 "# calibrated at |alpha|^2 = 1 (cal1) or at each |alpha|^2 (calnbar). examples/kerr_calibration_ocelot.py\n")
        keys = list(out[0])
        fh.write(",".join(keys) + "\n")
        for r in out:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
