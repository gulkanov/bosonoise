"""Screening for multiphoton resonances that involve spectator modes (readout resonator,
buffer, second coupler) at the CX 'on' point, for all 8 Ocelot pairs.

Resonance condition (energy conservation for a process from the branch |i, n> of the
calibrated 3-mode cell, i = g, e, f):
    E_q(i) + n w_s = E_q(k) + E_c(c) + (n + d) w_s + r w_spec,
with ancilla level k <= 8, active-coupler level c <= 2, storage change d in [-4, 4] and
r = 1, 2 spectator quanta. Energies: bare transmon levels of the calibrated ancilla and
coupler (at Phi0/2), measured spectator frequencies (readout Fig. 9b; buffer Fig. 9b/13a).
The second coupler of the ancilla idles at its maximum frequency, which is not published
(> 8.5 GHz, App. F1); it is scanned over 8.5-9.5 GHz.

For every coincidence we report the detuning and the minimal number of elementary
exchanges, which sets the order of the effective coupling. Dressed shifts (chi n up to
~40 MHz at n = 20) can close detunings of that size only. Screening only: a coincidence of
order >= 5 has an effective coupling ~ g (g/Delta)^(order-1) << kHz and is harmless over a
~1 us gate.

Usage: python examples/spectator_screening_ocelot.py
Output: data/validation/spectator_screening.csv (model only)
"""
from __future__ import annotations

import pathlib

import numpy as np

from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.presets import load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIRS = ["A1-S1", "A1-S2", "A2-S2", "A2-S3", "A3-S3", "A3-S4", "A4-S4", "A4-S5"]
WINDOW_MHZ = 60.0


def order(i, k, c, d, r):
    """Minimal number of elementary exchanges: ancilla steps, coupler quanta, storage photons, spectator quanta."""
    return abs(i - k) + c + abs(d) + r


def main():
    pre = load_preset("ocelot")
    rows = []
    for pair in PAIRS:
        anc, st = pair.split("-")
        p = load_cell_params(ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml", st)
        m = StorageCouplerAncilla(p, Truncation(n_s=4, n_c=4, n_q=9, ncut=25))
        Eq = m._E_q - m._E_q[0]
        Ec = m.elements(0.5).E_c[:3] - m.elements(0.5).E_c[0]
        spectators = {"readout": [pre["readout"][f"R{anc[1]}"]["f_GHz"]],
                      "buffer": [pre["buffer"][f"B{st[1]}"]["f_GHz"]],
                      "coupler2": list(np.linspace(8.5, 9.5, 21))}
        for name, freqs in spectators.items():
            for fsp in freqs:
                for i in (0, 1, 2):
                    for k in range(9):
                        for c in range(3):
                            for d in range(-4, 5):
                                for r in (1, 2):
                                    det = (Eq[i] - Eq[k] - Ec[c] - d * p.f_s - r * fsp) * 1e3
                                    if abs(det) <= WINDOW_MHZ:
                                        rows.append((pair, name, fsp, "gef"[i], k, c, d, r, det, order(i, k, c, d, r)))
    rows.sort(key=lambda x: (x[0], x[9], abs(x[8])))
    path = ROOT / "data" / "validation" / "spectator_screening.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Bare-energy coincidences within 60 MHz involving spectator modes at the CX 'on' point. "
                 "examples/spectator_screening_ocelot.py\n")
        fh.write("pair,spectator,f_spectator_GHz,branch,ancilla_level_final,coupler_level_final,storage_change,"
                 "spectator_quanta,detuning_MHz,order\n")
        for r in rows:
            fh.write(",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    for pair in PAIRS:
        sel = [r for r in rows if r[0] == pair]
        low = min((r[9] for r in sel), default=None)
        best = [r for r in sel if r[9] == low][:3]
        print(f"{pair}: {len(sel)} coincidences within {WINDOW_MHZ:g} MHz; lowest order {low}: " + "; ".join(
            f"{r[1]}({r[2]:.2f}) |{r[3]},n> -> |q{r[4]},c{r[5]},n{r[6]:+d},{r[7]}x> {r[8]:+.0f} MHz" for r in best))
    print("saved", path)


if __name__ == "__main__":
    main()
