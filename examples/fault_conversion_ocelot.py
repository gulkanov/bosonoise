"""Conversion map of the Ocelot syndrome cycle: which ancilla faults become which cat errors (paper Sec. 4, App. D).

For every stabilizer (A1: S1 S2 ... A4: S4 S5), bath variant and |alpha|^2, the jump families of the dressed-ladder
bath are grouped in classes (f->e, e->g, direct f->g, heating; channels/cycle.py) and split into fault SEQUENCES with
at most one jump of each class per measurement (multi-class Dyson split, channels/cx2.py): f->e alone, the double
decay f->e->g (within a gate or across the two), e->g alone, the direct f->g, heating, ... (a single-class split would
count the double decay under both f->e and e->g). For each sequence we report
    p_F     probability per stabilizer measurement,
    eta_F   conversion: P(X on cat a | F), P(X on cat b | F), P(X_a X_b | F), P(odd | F) (the logical-relevant part),
            P(herald | F) (ancilla read in e), P(silent | F) (wrong-branch ancilla: random syndrome bit),
and the deterministic part (no bath: Kerr shear and chi(n) of the unstabilized half turn). 'other' = the no-sequence
component minus the deterministic part (unselected classes, e.g. photon-assisted). Closure: the sum over components
reproduces the full table up to terms second order in one class. Model only; no data.

Usage: python examples/fault_conversion_ocelot.py [--nbars 2,4,8,16 --variants zero-fit,lam3,T50] [--tag TAG]
Output: data/validation/fault_conversion[_TAG].csv, figures/fault_conversion[_TAG].png/.pdf
A variant 'lam<x>' that is not predefined scales the idle-referenced charge bath by x at zero temperature (e.g. lam0.333
for an ancilla three times longer-lived); use it with --tag so that the default data product is not overwritten.
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

from bosonoise.channels.cx2 import BathModel
from bosonoise.channels.cycle import (CXCell, component_name, component_tables, family_class, odd_parity,
                                      stabilizer_table, summarize, summarize_fault)
from bosonoise.plotting import SERIES, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import NMAX  # noqa: F401  (same directory; documents the default cutoffs)
from stabilizer_faults_ocelot import STABILIZERS, VARIANTS, make_cell

ROOT = pathlib.Path(__file__).resolve().parents[1]
NL, NSIM = 40, 38


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbars", default="2,4,8,16")
    ap.add_argument("--variants", default="zero-fit,lam3,T50")
    ap.add_argument("--stabilizers", default="A1,A2,A3,A4")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    nbars = [float(x) for x in a.nbars.split(",")]
    variants = a.variants.split(",")
    for v in variants:  # scaled idle-referenced baths beyond the predefined variants, e.g. lam0.333
        if v not in VARIANTS and v.startswith("lam"):
            VARIANTS[v] = dict(T_mK=None, lam=float(v[3:]))
    sfx = f"_{a.tag}" if a.tag else ""
    stabs = [s for s in STABILIZERS if s[0] in a.stabilizers.split(",")]
    pre = load_preset("ocelot")
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    rows = []
    for anc, sa, sb in stabs:
        t0 = time.time()
        for variant in variants:
            ca = make_cell(pre, anc, sa, L[f"{anc}-{sa}"], variant, nbars, NL, NSIM)
            cb = make_cell(pre, anc, sb, L[f"{anc}-{sb}"], variant, nbars, NL, NSIM)
            present = {family_class(k) for c in (ca, cb) for k in c.bath.families(c.grid, NSIM)}
            classes = tuple(fc for fc in ("f->e", "e->g", "f->g", "heating") if fc in present)
            det_a = CXCell(ca.grid, ca.pulse, ca.corr, BathModel({}), ca.projectors, nmax=NSIM)
            det_b = CXCell(cb.grid, cb.pulse, cb.corr, BathModel({}), cb.projectors, nmax=NSIM)
            for nb in nbars:
                full = stabilizer_table(ca, cb, nb)
                det = stabilizer_table(det_a, det_b, nb)
                base = dict(stabilizer=f"{anc}:{sa}{sb}", variant=variant, nbar=nb)
                s_full = summarize(full)
                rows.append(dict(base, fault="total", where="-", p_fault=1.0, x_a=s_full["p_x_a"], x_b=s_full["p_x_b"],
                                 xx=s_full["p_xx"], odd=odd_parity(full), herald=s_full["p_herald"],
                                 silent=s_full["p_silent_syndrome"]))
                s_det = summarize(det)
                rows.append(dict(base, fault="deterministic", where="-", p_fault=1.0, x_a=s_det["p_x_a"], x_b=s_det["p_x_b"],
                                 xx=s_det["p_xx"], odd=odd_parity(det), herald=s_det["p_herald"],
                                 silent=s_det["p_silent_syndrome"]))
                comps = component_tables(ca, cb, nb, classes)
                for c, part in sorted(comps.items()):
                    sc = summarize_fault(part)
                    if not any(c):  # no selected fault: deterministic + unselected classes at full order
                        other = sc["odd"] * sc["p_fault"] - odd_parity(det)
                        rows.append(dict(base, fault="other", where="-", p_fault=1.0, odd=other))
                        continue
                    if sc["p_fault"] > 1e-15:
                        rows.append(dict(base, fault=component_name(c, classes), where="-", **sc))
        print(f"{anc}: done ({time.time() - t0:.0f} s)", flush=True)
    keys = ["stabilizer", "variant", "nbar", "fault", "where", "p_fault", "x_a", "x_b", "xx", "odd", "herald", "silent"]
    path = ROOT / "data" / "validation" / f"fault_conversion{sfx}.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Conversion map per stabilizer measurement: fault probability p_fault and conditional probabilities (X on cat a, "
                 "on cat b, correlated XX, odd parity = logical-relevant, ancilla herald, silent syndrome) for exactly one fault of "
                 "each class in CX(a) or CX(b); rows 'total' and 'deterministic' are unconditional. Dressed-ladder ULE, cutoffs "
                 f"{NL}/{NSIM}. examples/fault_conversion_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in keys) + "\n")
    # decomposition of the logical-relevant error and closure check
    print("\nstabilizer variant nbar | p_odd total | deterministic + other + sum_seq p P(odd|seq) = closure | largest (x1e-3)")
    groups = {}
    for r in rows:
        groups.setdefault((r["stabilizer"], r["variant"], r["nbar"]), []).append(r)
    for (st, v, nb), rs in groups.items():
        tot = next(r["odd"] for r in rs if r["fault"] == "total")
        det = next(r["odd"] for r in rs if r["fault"] == "deterministic")
        oth = next(r["odd"] for r in rs if r["fault"] == "other")
        contrib = {}
        for r in rs:
            if r["fault"] not in ("total", "deterministic", "other"):
                contrib[r["fault"]] = contrib.get(r["fault"], 0.0) + r["p_fault"] * r["odd"]
        clos = det + oth + sum(contrib.values())
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:4]
        print(f"{st} {v:8s} {nb:4.0f} | {tot * 1e3:7.3f} | {det * 1e3:6.3f} + {oth * 1e3:6.3f} + {sum(contrib.values()) * 1e3:6.3f} = "
              f"{clos * 1e3:7.3f} ({clos / tot:5.3f}) | " + ", ".join(f"{k} {c * 1e3:.3f}" for k, c in top))
    use_style()
    fig, axs = plt.subplots(1, len(stabs), figsize=(3.9 * len(stabs), 3.8), sharey=True, constrained_layout=True, squeeze=False)
    for ax, (anc, sa, sb) in zip(axs[0], stabs):
        st = f"{anc}:{sa}{sb}"
        for v, ls in zip(variants[:2], ("-", "--")):
            sel = [r for r in rows if r["stabilizer"] == st and r["variant"] == v]
            if not sel:
                continue
            x = sorted({r["nbar"] for r in sel})
            ax.semilogy(x, [next(r["odd"] for r in sel if r["nbar"] == nb and r["fault"] == "deterministic") for nb in x],
                        ls, color="#52514e", label="deterministic (Kerr, χ(n))" if ls == "-" else None)
            names = [n for n in dict.fromkeys(r["fault"] for r in sel) if n not in ("total", "deterministic")]
            for i, fc in enumerate(names):
                y = [sum(r["p_fault"] * r["odd"] for r in sel if r["nbar"] == nb and r["fault"] == fc) for nb in x]
                if max(y) > 1e-8:
                    ax.semilogy(x, np.maximum(y, 1e-9), ls, color=SERIES[i % len(SERIES)], label=fc if ls == "-" else None)
        lab = [{"lam3": "λ = 3"}.get(v, v) for v in variants[:2]]
        ax.set_title(f"{anc}: contributions to P(odd X) (solid {lab[0]}" + (f", dashed {lab[1]})" if len(lab) > 1 else ")"),
                     fontsize=8)
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_ylim(1e-7, 1e-2)
    axs[0][0].set_ylabel("per stabilizer measurement")
    axs[0][0].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"fault_conversion{sfx}.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
