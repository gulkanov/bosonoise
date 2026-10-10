"""The bit-flip floor of the cats served by ancilla A4 with the f -> g strength that F9 favours (paper App. E.3).

F9 (Fig. 18b, A4-S4) favours a smooth photon-stimulated f -> g channel s = 5-11 times its idle-referenced strength
(examples/f9_bayes_ocelot.py, paper App. E.2). With the measured on-gate decay of A4 (Fig. 37d) and s ~ 8 the model
reproduces the CX^2 f-excess of both A4 pairs (examples/f9_strength_cx2_ocelot.py, paper App. E.3). Here the same baths
enter the stabilizer fault tables of the syndrome cycle (as examples/stabilizer_faults_ocelot.py with ladder cutoff 40
and simulation cutoff 38), and the per-cat floor c0 + c1 n, the bias ceiling kappa_1 T_c / c1 and the bias are
evaluated as in examples/bias_ceiling_ocelot.py.

Baths:
  A4  'ongate'       measured on-gate decay (Fig. 37d, pulsed with |alpha|^2 = 4), f -> g family at lambda_fe x idle
      'ongate_s{s}'  the same with the f -> g family at s x idle-referenced, s = 5, 8, 11
  A3  'coupler18c'   coupler loss inferred from Fig. 18c (as in measured_bath_cx2_ocelot.py)
Cat S5 is served by A4 only; cat S4 by A3 (coupler loss) and A4. Nothing is tuned to error data.

Usage: python examples/f9_floor_ocelot.py
Output: data/validation/f9_floor.csv (fault tables), data/validation/f9_floor_summary.csv (per-cat floor, ceiling, bias)
"""
from __future__ import annotations

import csv
import pathlib
import time

import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases
from bosonoise.channels.cycle import CXCell, odd_parity, stabilizer_table, summarize
from bosonoise.dynamics.ule import detailed_balance, flat
from bosonoise.presets import load_preset

from bias_ceiling_ocelot import T1_EFF_US, T_REP_US, idle_at, idle_model  # same directory
from f9_strength_cx2_ocelot import ScaledFG  # same directory
from measured_bath_cx2_ocelot import coupler_T1_from_18c, fig37d_means  # same directory
from predict_cx2_ule_ocelot import DELTA_B_MHZ, charge_bath, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NL, NMAX = 40, 38
NBARS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
S_VALUES = (5.0, 8.0, 11.0)
CONDITION = "pulsed_alpha2_4"
OUT = ROOT / "data" / "validation"


def lengths():
    return {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}


def cell(pre, anc, st, L_cx_us, bath_of_grid):
    """A calibrated CX of examples/stabilizer_faults_ocelot.make_cell with a bath built from its grid."""
    grid = grid_for(f"{anc}-{st}", NL)
    pulse = calibrate_cx2_to_length(grid, L_cx_us * 1e3, turns=0.5)
    kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kr) for nb in sorted(set(NBARS) | {1.0})}
    corr = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0], f_target_flipped=True)
    return CXCell(grid=grid, pulse=pulse, corr=corr, bath=bath_of_grid(grid), projectors=projs, nmax=NMAX)


def a4_bath(pre, t37, s):
    T1ge, T1fe = pre["ancilla"]["A4"]["T1_ge_us"], pre["ancilla"]["A4"]["T1_ef_us"]

    def make(grid):
        on, idle = grid.ladders[-1], grid.ladders[0]
        df_eg = (on.X["q"][0, 0, 1, 0] / idle.X["q"][0, 0, 1, 0]) ** 2
        df_fe = (on.X["q"][1, 0, 2, 0] / idle.X["q"][1, 0, 2, 0]) ** 2
        lam_eg = T1ge / (t37[("e", CONDITION)][0] * df_eg)
        lam_fe = T1fe / (t37[("f", CONDITION)][0] * df_fe)
        q = charge_bath(grid, T1ge, T1fe, lam=(lam_eg, lam_fe))
        return BathModel({"q": q}) if s is None else ScaledFG({"q": q}, s_fg=s / lam_fe)
    return make


def a3_bath(pre, T1c):
    T1ge, T1fe = pre["ancilla"]["A3"]["T1_ge_us"], pre["ancilla"]["A3"]["T1_ef_us"]

    def make(grid):
        on = grid.ladders[-1]
        qc = charge_bath(grid, T1ge, T1fe, coupler_T1_us=T1c)
        return BathModel({"q": qc, "c": detailed_balance(flat(1.0 / (T1c * 1e3 * on.nc01 ** 2)), None)})
    return make


def tables(pre, anc, sa, sb, name, make_bath, L):
    t0 = time.time()
    ca = cell(pre, anc, sa, L[f"{anc}-{sa}"], make_bath)
    cb = cell(pre, anc, sb, L[f"{anc}-{sb}"], make_bath)
    rows = []
    for nb in NBARS:
        tab = stabilizer_table(ca, cb, nb)
        rows.append(dict(stabilizer=f"{anc}:{sa}{sb}", variant=name, nbar=nb, **summarize(tab), p_odd=odd_parity(tab)))
    print(f"{anc} {name}: {time.time() - t0:.0f} s", flush=True)
    return rows


def main():
    pre = load_preset("ocelot")
    L = lengths()
    t37 = fig37d_means()
    T1c = coupler_T1_from_18c(pre)[0]
    rows = tables(pre, "A3", "S3", "S4", "coupler18c", a3_bath(pre, T1c), L)
    a4_variants = {"ongate": None} | {f"ongate_s{s:g}": s for s in S_VALUES}
    for name, s in a4_variants.items():
        rows += tables(pre, "A4", "S4", "S5", name, a4_bath(pre, t37, s), L)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "f9_floor.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Stabilizer fault tables of A3 (coupler loss from Fig. 18c, T1_c {T1c:.3f} us) and A4 (measured on-gate "
                 "decay, Fig. 37d; f -> g at s x idle-referenced). examples/f9_floor_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    S = {(r["stabilizer"].split(":")[0], r["variant"], r["nbar"]): r for r in rows}
    idle = idle_model()
    summ = []
    for v in a4_variants:
        for cat in ("S4", "S5"):
            x = np.array(NBARS)
            g = np.array([(S[("A3", "coupler18c", n)]["p_x_b"] if cat == "S4" else 0.0)
                          + S[("A4", v, n)]["p_x_a" if cat == "S4" else "p_x_b"] for n in NBARS])
            t = g + np.array([idle_at(idle, cat, n)[0] for n in NBARS])
            big = x >= 8
            c1, c0 = np.polyfit(x[big], g[big], 1)
            kT = T_REP_US / T1_EFF_US[cat]
            summ.append(dict(cat=cat, a4_variant=v, c0=float(c0), c1=float(c1), ceiling=kT / c1,
                             gate_4=float(g[x == 4][0]), gate_10=float(g[x == 10][0]), growth_4_to_10=float(g[x == 10][0] / g[x == 4][0]),
                             bias_4=float(kT * 4 / t[x == 4][0]), bias_16=float(kT * 16 / t[x == 16][0]),
                             nstar_gate=float(x[np.argmin(g)]), nstar_total=float(x[np.argmin(t)])))
    keys = list(summ[0])
    with open(OUT / "f9_floor_summary.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Per-cat floor c0 + c1 n (fit n = 8-16), ceiling kappa_1 T_c / c1, bias (gate + data-derived idle part), "
                 "growth of the gate part from n = 4 to 10, for cats S4 (A3 coupler loss + A4) and S5 (A4). "
                 "examples/f9_floor_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in summ:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    for r in summ:
        print(f"{r['cat']} {r['a4_variant']:12s}: c0 {r['c0']:.2e} c1 {r['c1']:.2e} ceiling {r['ceiling']:.0f}; "
              f"gate(10)/gate(4) {r['growth_4_to_10']:.2f}; bias at 4 {r['bias_4']:.0f}, at 16 {r['bias_16']:.0f}; "
              f"n* gate {r['nstar_gate']:g}, total {r['nstar_total']:g}")


if __name__ == "__main__":
    main()
