"""Heralded versus silent cat errors in the Ocelot syndrome cycle: monitoring the buffer output (paper App. I).

Ferrari et al. (arXiv:2607.01375): a bit flip of a dissipatively stabilized cat comes with a burst of buffer photons, so
monitoring the buffer output can flag bit flips as erasures. Here the idea is applied to the gate-induced errors of the
syndrome cycle. After each CX the storage relaxes under two-photon dissipation; a detector of efficiency eta on that
cat's buffer output clicks on a fraction of the emitted photons (no-click projector of channels/conversion.py: exact,
linear in the state). Each fault sequence of the multi-class Dyson split (channels/cycle.py) is resolved into cat
errors with and without a click, plus clicks without an error (false heralds, e.g. from the deterministic Kerr shear).
Physics to look for: a decay DURING a gate leaves a partial rotation (burst, heralded); a gate skipped entirely
(ancilla already in g) leaves the cat in the code space (silent).

A single detected photon is not a useful herald: the deterministic distortion of every CX (Kerr shear, chi(n)) makes
the normal re-stabilization emit too. The herald is therefore 'at least k detected photons' (exact n-click
distribution, channels/conversion.py), scanned over k.

Usage: python examples/herald_conversion_ocelot.py [--nbars 2,4,8 --etas 0.1,0.3,0.6,1 --kmins 1,2,3,5 --variants ...]
Output: data/validation/herald_conversion.csv, figures/herald_conversion.png/.pdf
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

from bosonoise.channels.cycle import component_name, family_class, herald_component_tables, summarize_herald
from bosonoise.plotting import SERIES, use_style
from bosonoise.presets import load_preset

from stabilizer_faults_ocelot import STABILIZERS, make_cell  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NL, NSIM = 40, 38
KEYS = ["p_fault", "x_a", "x_a_heralded", "x_b", "x_b_heralded", "odd", "odd_silent", "false_click_a", "false_click_b"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbars", default="2,4,8")
    ap.add_argument("--etas", default="0.1,0.3,0.6,1")
    ap.add_argument("--variants", default="zero-fit,lam3")
    ap.add_argument("--kmins", default="1,2,3,5")
    ap.add_argument("--stabilizers", default="A1,A2,A3,A4")
    a = ap.parse_args()
    nbars = [float(x) for x in a.nbars.split(",")]
    etas = [float(x) for x in a.etas.split(",")]
    kmins = [int(x) for x in a.kmins.split(",")]
    stabs = [s for s in STABILIZERS if s[0] in a.stabilizers.split(",")]
    pre = load_preset("ocelot")
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    rows = []
    for anc, sa, sb in stabs:
        t0 = time.time()
        for variant in a.variants.split(","):
            ca = make_cell(pre, anc, sa, L[f"{anc}-{sa}"], variant, nbars, NL, NSIM)
            cb = make_cell(pre, anc, sb, L[f"{anc}-{sb}"], variant, nbars, NL, NSIM)
            present = {family_class(k) for c in (ca, cb) for k in c.bath.families(c.grid, NSIM)}
            classes = tuple(fc for fc in ("f->e", "e->g", "f->g", "heating") if fc in present)
            for nb in nbars:
                for eta, kmin in [(e_, k_) for e_ in etas for k_ in kmins]:
                    comps = herald_component_tables(ca, cb, nb, classes, eta, kmin)
                    total = {k: 0.0 for k in KEYS}
                    for c, part in sorted(comps.items()):
                        s = summarize_herald(part)
                        if s["p_fault"] <= 1e-15:
                            continue
                        name = component_name(c, classes) if any(c) else "no fault (deterministic + other)"
                        rows.append(dict(stabilizer=f"{anc}:{sa}{sb}", variant=variant, nbar=nb, eta=eta, kmin=kmin,
                                         component=name, **s))
                        for k in KEYS[1:]:
                            total[k] += s["p_fault"] * s[k]
                    total["p_fault"] = 1.0
                    rows.append(dict(stabilizer=f"{anc}:{sa}{sb}", variant=variant, nbar=nb, eta=eta, kmin=kmin,
                                     component="total", **total))
        print(f"{anc}: done ({time.time() - t0:.0f} s)", flush=True)
    keys = ["stabilizer", "variant", "nbar", "eta", "kmin", "component"] + KEYS
    path = ROOT / "data" / "validation" / "herald_conversion.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Buffer-emission heralds of gate-induced cat errors per stabilizer measurement: for each fault sequence, "
                 "P(sequence) and conditional P(X on cat a/b), heralded parts (click on that cat's buffer detector, efficiency "
                 "eta), P(odd number of X), its part without any click, and clicks without error; 'total' rows are "
                 "unconditional (probability per measurement). examples/herald_conversion_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in keys) + "\n")
    print("\nstabilizer variant nbar eta kmin | P(X_a) heralded frac | P(X_b) heralded frac | P(odd) silent frac | "
          "false heralds a/b (x1e-3)")
    for r in rows:
        if r["component"] == "total" and r["eta"] in (0.3, 0.6) and r["nbar"] in (4.0, 8.0):
            print(f"{r['stabilizer']} {r['variant']:8s} {r['nbar']:3.0f} {r['eta']:4.2f} {r['kmin']:2d} | {r['x_a'] * 1e3:6.3f} "
                  f"{r['x_a_heralded'] / r['x_a']:5.2f} | {r['x_b'] * 1e3:6.3f} {r['x_b_heralded'] / r['x_b']:5.2f} | "
                  f"{r['odd'] * 1e3:6.3f} {r['odd_silent'] / r['odd']:5.2f} | {r['false_click_a'] * 1e3:6.2f} "
                  f"{r['false_click_b'] * 1e3:6.2f}")
    use_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, (anc, sa, sb) in enumerate(stabs):
        st = f"{anc}:{sa}{sb}"
        for v, ls in (("zero-fit", "-"), ("lam3", "--")):
            for nb, mk in zip(nbars, "os^"):
                sel = [r for r in rows if r["stabilizer"] == st and r["variant"] == v and r["nbar"] == nb
                       and r["component"] == "total" and r["kmin"] == 3]
                if not sel:
                    continue
                x = [r["eta"] for r in sel]
                axs[0].plot(x, [1 - r["odd_silent"] / r["odd"] for r in sel], ls, marker=mk, ms=3, color=SERIES[i],
                            label=f"{anc}, |α|²={nb:g}" if v == "zero-fit" else None)
                axs[1].plot(x, [0.5 * (r["false_click_a"] + r["false_click_b"]) for r in sel], ls, marker=mk, ms=3,
                            color=SERIES[i])
    axs[0].set_xlabel("buffer detector efficiency η")
    axs[0].set_ylabel("fraction of logical-relevant cat errors heralded")
    axs[0].set_title("herald = at least 3 detected photons; solid zero-fit, dashed λ = 3", fontsize=9)
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].set_xlabel("buffer detector efficiency η")
    axs[1].set_ylabel("false clicks per cat and measurement")
    axs[1].set_yscale("log")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"herald_conversion.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
