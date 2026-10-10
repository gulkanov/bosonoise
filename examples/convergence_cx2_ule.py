"""Convergence checks of the dressed-ladder ULE CX^2 simulation (examples/predict_cx2_ule_ocelot.py).

Varies, one at a time, for one pair: the bare-energy cutoff e_cut (+25 GHz) and the coupler /
ancilla levels kept, the flux grid (13 -> 25 points), the time step (0.5 -> 0.25 ns), the
simulated photon cutoff (15 -> 17) and the Fock space of the asymptotic projector (+10).
Reported: p_f (dressed bath), p_g (no bath: Kerr distortion) and the undressed p_f at |alpha|^2 = 2, 4.

Usage: python examples/convergence_cx2_ule.py [--pair A4-S5]
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import (BathModel, LadderGrid, calibrate_cx2_to_length, calibrate_phases, initial_blocks,
                                    ladder_at, simulate_cx2)
from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, FLUX_GRID, NL, charge_bath  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINE_GRID = sorted(set(FLUX_GRID) | {0.05, 0.15, 0.225, 0.275, 0.325, 0.375, 0.4125, 0.4375, 0.46, 0.4775, 0.49, 0.4975})


def build(p, e_extra=0.0, n_c=8, n_q=10, fluxes=FLUX_GRID, nl=NL):
    e_cut = (nl + 3) * p.f_s + 30.0 + e_extra
    m = StorageCouplerAncilla(p, Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=n_c, n_q=n_q, ncut=25, e_cut=e_cut))
    return LadderGrid([ladder_at(m, f, nl) for f in fluxes])


def evaluate(grid, pre, pair, nmax=15, dt=0.5, dN=0):
    anc, st = pair.split("-")
    L_cx = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}[pair]
    chi = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
    pulse = calibrate_cx2_to_length(grid, L_cx * 1e3 + 0.5 / chi)
    kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, N=int(2 * nb + 8 * np.sqrt(nb) + 14) + dN, kerr_ratio=kr) for nb in (1.0, 2.0, 4.0)}
    corr = calibrate_phases(grid, pulse, nmax, 1.0, projs[1.0])
    q = charge_bath(grid, pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"])
    out = [pulse.duration]
    for nb in (2.0, 4.0):
        for bath, lev in ((BathModel({"q": q}), 2), (BathModel({}), 0), (BathModel({"q": q}, standard=True), 2)):
            rho = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, nmax), nmax, dt=dt, phase_corr=corr)
            out.append(projs[nb].p_flip(rho / np.trace(rho).real))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="A4-S5")
    a = ap.parse_args()
    pre = load_preset("ocelot")
    anc, st = a.pair.split("-")
    p = load_cell_params(ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml", st)
    base_grid = build(p)
    cases = {
        "reference": lambda: evaluate(base_grid, pre, a.pair),
        "e_cut +25 GHz, n_c 10, n_q 12": lambda: evaluate(build(p, 25.0, 10, 12), pre, a.pair),
        "flux grid 25 points": lambda: evaluate(build(p, fluxes=FINE_GRID), pre, a.pair),
        "dt 0.25 ns": lambda: evaluate(base_grid, pre, a.pair, dt=0.25),
        "photon cutoff 17 (ladder 19)": lambda: evaluate(build(p, nl=19), pre, a.pair, nmax=17),
        "projector N +10": lambda: evaluate(base_grid, pre, a.pair, dN=10),
    }
    hdr = "case | gate ns | n=2: p_f  p_g(Kerr)  p_f(std) | n=4: p_f  p_g(Kerr)  p_f(std)"
    print(hdr)
    ref = None
    for name, fn in cases.items():
        v = fn()
        ref = v if ref is None else ref
        rel = np.max(np.abs(v[1:] - ref[1:]) / np.abs(ref[1:])) if name != "reference" else 0.0
        print(f"{name:32s} | {v[0]:6.1f} | " + " ".join(f"{x:.4e}" for x in v[1:4]) + " | "
              + " ".join(f"{x:.4e}" for x in v[4:]) + f" | max rel. change {rel:.1e}", flush=True)


if __name__ == "__main__":
    main()
