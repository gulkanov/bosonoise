"""The f -> g strength that F9 favours, tested on the CX^2 gate data of the same ancilla (paper App. E.3).

The one-defect fits of F9 (Fig. 18b, A4-S4; examples/f9_bayes_ocelot.py, paper App. E.2) favour a smooth
photon-stimulated f -> g channel s = 5-11 times its idle-referenced strength (posterior medians 7.7 and 9.0 for the
two defect hypotheses). Here only the opened family |f~,n> -> |g~,n+1> (charge operator, branch 2 -> 0, photon change
+1) of the CX^2 simulation of the A4 pairs is set to s times its idle-referenced rate, and the fault-driven f-excess
is compared with Fig. 16, as in examples/measured_bath_cx2_ocelot.py. s comes from F9; nothing is tuned to Fig. 16.

  zerofit_fg{s}    idle-referenced bath, f -> g family x s
  ongate_fg{s}     measured on-gate decay (Fig. 37d, pulsed with |alpha|^2 = 4) with the f -> g family set to
                   s x idle-referenced (x s / lambda_fe on top of the on-gate scaling of the f family)
  ongate_np_fg{s}  the same with the on-gate decay measured without photons (Fig. 37d, pulsed). The value with
                   photons already contains part of the photon-stimulated f -> g decay; this variant avoids counting
                   it twice.

F9 probes the opened band of A4-S4 (~5.00 GHz); the band of A4-S5 lies 40-97 MHz lower, so using the same s there is
an assumption.

Usage: python examples/f9_strength_cx2_ocelot.py
Output: data/validation/f9_strength_cx2.csv
"""
from __future__ import annotations

import dataclasses
import pathlib
import time

import numpy as np

from bosonoise.channels.cx2 import BathModel, initial_blocks, simulate_cx2
from bosonoise.presets import load_preset

from measured_bath_cx2_ocelot import compare, fig37d_means, setup  # same directory
from predict_cx2_ule_ocelot import NBARS, NMAX, charge_bath  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "f9_strength_cx2.csv"
PAIRS = ("A4-S4", "A4-S5")
S_VALUES = (5.0, 8.0, 11.0)  # the 68% range and centre of the F9 posteriors of s (paper App. E.2)
# Fig. 37d conditions: pulsed with |alpha|^2 = 4 (the closest to the CX^2 conditions) and pulsed without photons
CONDITIONS = {"ongate": "pulsed_alpha2_4", "ongate_np": "pulsed"}


@dataclasses.dataclass
class ScaledFG(BathModel):
    """A bath whose opened family |f~,n> -> |g~,n+1> (branch 2 -> 0, photon change +1) is scaled by s_fg."""
    s_fg: float = 1.0

    def families(self, grid, nmax, dms=(-2, -1, 0, 1, 2)):
        out = super().families(grid, nmax, dms)
        return {k: (np.sqrt(self.s_fg) * C if (k[1], k[2], k[3]) == (2, 0, 1) else C) for k, C in out.items()}


def run_pair(pair, pre, t37):
    anc = pair.split("-")[0]
    T1ge, T1fe = pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"]
    t0 = time.time()
    grid, pulse, corr, projs = setup(pair, pre)
    on, idle = grid.ladders[-1], grid.ladders[0]
    df_eg = (on.X["q"][0, 0, 1, 0] / idle.X["q"][0, 0, 1, 0]) ** 2
    df_fe = (on.X["q"][1, 0, 2, 0] / idle.X["q"][1, 0, 2, 0]) ** 2
    q0 = charge_bath(grid, T1ge, T1fe)
    baths = {"nobath": BathModel({}), "zerofit": BathModel({"q": q0})}
    for s in S_VALUES:
        baths[f"zerofit_fg{s:g}"] = ScaledFG({"q": q0}, s_fg=s)
    lams = {}
    for tag, cond in CONDITIONS.items():
        lam_eg = T1ge / (t37[("e", cond)][0] * df_eg)
        lam_fe = T1fe / (t37[("f", cond)][0] * df_fe)
        lams[tag] = (lam_eg, lam_fe)
        qon = charge_bath(grid, T1ge, T1fe, lam=(lam_eg, lam_fe))
        baths[tag] = BathModel({"q": qon})
        for s in S_VALUES:
            baths[f"{tag}_fg{s:g}"] = ScaledFG({"q": qon}, s_fg=s / lam_fe)
    rows = []
    for nb in NBARS:
        row = dict(pair=pair, nbar=nb)
        for tag, (le, lf) in lams.items():
            row[f"lam_eg_{tag}"], row[f"lam_fe_{tag}"] = le, lf
        for name, bath in baths.items():
            for lev, tag in ((2, "f"), (0, "g")):
                blk = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, NMAX), NMAX, blocks=True, phase_corr=corr)
                rho = blk.sum(axis=0)
                rho = rho / np.trace(rho).real
                row[f"p{tag}_{name}"] = projs[nb].p_flip(rho)
        rows.append(row)
    print(f"{pair}: " + "; ".join(f"{tag} lambda_eg {le:.2f}, lambda_fe {lf:.2f}" for tag, (le, lf) in lams.items())
          + f" ({time.time() - t0:.0f} s)", flush=True)
    return rows


def main():
    pre = load_preset("ocelot")
    t37 = fig37d_means()
    rows = []
    for pair in PAIRS:
        rows += run_pair(pair, pre, t37)
    rows = compare(rows)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# CX^2 fault-driven f-excess of the A4 pairs with the opened f -> g family at s x its idle-referenced "
                 "strength (s from the F9 fits, paper App. E.2), vs Fig. 16 (f - g). Nothing tuned to Fig. 16. "
                 "examples/f9_strength_cx2_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    names = [k[6:] for k in rows[0] if k.startswith("ratio_")]
    print("data/model, median over |alpha|^2 >= 2 (all |alpha|^2 in brackets)")
    for pair in PAIRS:
        sel = [r for r in rows if r["pair"] == pair]
        hi = [r for r in sel if r["nbar"] >= 2]
        print(f"{pair}: " + "; ".join(
            f"{n} {np.nanmedian([r['ratio_' + n] for r in hi]):.2f} ({np.nanmedian([r['ratio_' + n] for r in sel]):.2f})"
            for n in names))
    print("saved", OUT)


if __name__ == "__main__":
    main()
