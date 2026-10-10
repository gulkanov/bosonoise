"""Fault tables of the four Ocelot repetition-code stabilizers (X_i X_{i+1} measured by A_i).

For each stabilizer (A1: S1,S2; A2: S2,S3; A3: S3,S4; A4: S4,S5) the two CX gates are calibrated
like in the experiment: half-turn pulses of the Fig. 12 CX lengths (purple lines), storage and
f-branch phase corrections optimized at |alpha|^2 = 1 without noise (f target = flipped), and
the detuned-buffer re-stabilization map of each storage (App. F8). The dressed-ladder ULE then
gives the joint table of (error on cat a, error on cat b, final ancilla level) per ancilla
branch (channels/cycle.py), from which we report single-cat X errors, correlated X pairs,
heralded (final e -> reads f) and silent syndrome faults.

Variants (no parameter tuned to error data):
  'zero-fit'  ancilla charge bath referenced to the idle T1 (Fig. 9c,d), T_anc = 0;
  'T50'       the same with ancilla heating at 50 mK (typical transmon value, a prior);
  'lam2', 'lam3'  decay rates x2, x3 during the gates (diagnostic: the CX^2 f-excess needs a median
              lambda = 2.8 in Pauli units, paper Sec. 6; AWS use beta = 2.5 for A4-S5, App. J).

Usage: python examples/stabilizer_faults_ocelot.py                  (default run: |alpha|^2 = 1 ... 4)
       python examples/stabilizer_faults_ocelot.py --nl 40 --nmax 38 --nbars 1,2,3,4,5,6,8,10,12,14,16
              --variants zero-fit,lam3,T50 --tag large_nbar   (all four stabilizers: bias_ceiling_ocelot.py,
              logical_projection_ocelot.py and tools/build_ocelot_noise_table.py read A1-A4 from this file)
  --nl / --nmax: ladder and simulation photon cutoffs (defaults 17 / 15 suffice for |alpha|^2 <= 4).
Output: data/validation/stabilizer_faults[_tag].csv (model only), figures/stabilizer_faults[_tag].png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases
from bosonoise.channels.cycle import CXCell, odd_parity, stabilizer_table, summarize
from bosonoise.plotting import SERIES, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, NL, NMAX, charge_bath, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
STABILIZERS = [("A1", "S1", "S2"), ("A2", "S2", "S3"), ("A3", "S3", "S4"), ("A4", "S4", "S5")]
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VARIANTS = {"zero-fit": dict(T_mK=None, lam=1.0), "T50": dict(T_mK=50.0, lam=1.0), "lam2": dict(T_mK=None, lam=2.0),
            "lam3": dict(T_mK=None, lam=3.0)}


def make_cell(pre, anc, st, L_cx_us, variant, nbars=NBARS, nl=NL, nmax=NMAX):
    pair = f"{anc}-{st}"
    grid = grid_for(pair, nl)
    pulse = calibrate_cx2_to_length(grid, L_cx_us * 1e3, turns=0.5)
    kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kr) for nb in sorted(set(nbars) | {1.0})}
    corr = calibrate_phases(grid, pulse, nmax, 1.0, projs[1.0], f_target_flipped=True)
    bath = BathModel({"q": charge_bath(grid, pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"],
                                       T_mK=VARIANTS[variant]["T_mK"], lam=VARIANTS[variant]["lam"])})
    return CXCell(grid=grid, pulse=pulse, corr=corr, bath=bath, projectors=projs, nmax=nmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=int, default=NL)
    ap.add_argument("--nmax", type=int, default=NMAX)
    ap.add_argument("--nbars", default=",".join(f"{x:g}" for x in NBARS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--stabilizers", default="A1,A2,A3,A4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    nbars = [float(x) for x in a.nbars.split(",")]
    variants = a.variants.split(",")
    stabs = [s for s in STABILIZERS if s[0] in a.stabilizers.split(",")]
    sfx = f"_{a.tag}" if a.tag else ""
    pre = load_preset("ocelot")
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    rows = []
    for anc, sa, sb in stabs:
        t0 = time.time()
        for variant in variants:
            ca = make_cell(pre, anc, sa, L[f"{anc}-{sa}"], variant, nbars, a.nl, a.nmax)
            cb = make_cell(pre, anc, sb, L[f"{anc}-{sb}"], variant, nbars, a.nl, a.nmax)
            for nb in nbars:
                tab = stabilizer_table(ca, cb, nb)
                rows.append(dict(stabilizer=f"{anc}:{sa}{sb}", variant=variant, nbar=nb,
                                 gate_a_ns=ca.pulse.duration, gate_b_ns=cb.pulse.duration, **summarize(tab),
                                 p_odd=odd_parity(tab), p_odd_branch_g=odd_parity(tab, "g")))
        print(f"{anc}: CX lengths {ca.pulse.duration:.0f} / {cb.pulse.duration:.0f} ns (ramps {ca.pulse.ramp:.1f} / "
              f"{cb.pulse.ramp:.1f} ns) ({time.time() - t0:.0f} s)", flush=True)
    path = ROOT / "data" / "validation" / f"stabilizer_faults{sfx}.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Fault tables of the Ocelot repetition-code stabilizers (dressed-ladder ULE, per measurement; "
                 f"ladder cutoff {a.nl}, simulation cutoff {a.nmax}). examples/stabilizer_faults_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("\nstabilizer variant nbar | p_X(a) p_X(b) p_XX | herald  silent | p_X(a|herald) p_X(b|herald)   [x1e-3 except conditionals]")
    for r in rows:
        if r["nbar"] in (2.0, 3.0, 4.0, 8.0, 12.0, 16.0):
            print(f"{r['stabilizer']} {r['variant']:8s} {r['nbar']:.0f} | {r['p_x_a'] * 1e3:6.3f} {r['p_x_b'] * 1e3:6.3f} "
                  f"{r['p_xx'] * 1e3:6.3f} | {r['p_herald'] * 1e3:6.2f} {r['p_silent_syndrome'] * 1e3:6.3f} | "
                  f"{r['p_x_a_given_herald']:.3f} {r['p_x_b_given_herald']:.3f}")
    use_style()
    fig, axs = plt.subplots(1, len(stabs), figsize=(3.75 * len(stabs), 3.8), sharey=True, constrained_layout=True,
                            squeeze=False)
    axs = axs[0]
    for ax, (anc, sa, sb) in zip(axs, stabs):
        sel = [r for r in rows if r["stabilizer"] == f"{anc}:{sa}{sb}" and r["variant"] == variants[0]]
        x = [r["nbar"] for r in sel]
        ax.semilogy(x, [r["p_x_a"] for r in sel], "-o", ms=3, color=SERIES[0], label=f"X on {sa}")
        ax.semilogy(x, [r["p_x_b"] for r in sel], "-o", ms=3, color=SERIES[1], label=f"X on {sb}")
        ax.semilogy(x, [r["p_xx"] for r in sel], "-o", ms=3, color=SERIES[2], label="correlated XX")
        ax.semilogy(x, [r["p_herald"] for r in sel], "--", color=SERIES[3], label="heralded (e → reads f)")
        ax.semilogy(x, [r["p_silent_syndrome"] for r in sel], ":", color=SERIES[4], label="silent syndrome")
        ax.set_title(f"{anc}: X_{sa[1]} X_{sb[1]} ({variants[0]})", fontsize=9)
        ax.set_xlabel(r"$|\alpha|^2$")
    axs[0].set_ylabel("probability per stabilizer measurement")
    axs[0].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"stabilizer_faults{sfx}.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
