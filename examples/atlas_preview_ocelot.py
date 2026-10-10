"""Resonance-atlas skeleton: diabatic crossings of the g/e/f branches (n <= 20) along the
coupler-flux trajectory Phi_c = 0 -> Phi0/2 of a CX flux pulse (Ocelot A2-S3 model).

Every crossing found here is traversed twice per CX gate (ramp up and ramp down).
This script only locates crossings and classifies them by (ds, dc, dq) and heuristic
coupling order; `--refine K` additionally computes the minimal gap of the K
lowest-order crossings by a golden-section search (slow: ~40 diagonalizations each).

Usage:  python examples/atlas_preview_ocelot.py [--fit A] [--dflux 0.01] [--refine 0]
Outputs: data/atlas/crossings_*.csv, figures/atlas_preview_*.png/.pdf
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.plotting import ordinal_cmap, use_style
from bosonoise.spectra.crossings import diabatic_sweep, find_crossings, refine_gap

ROOT = pathlib.Path(__file__).resolve().parents[1]
QNAME = {0: "g", 1: "e", 2: "f"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="A")
    ap.add_argument("--dflux", type=float, default=0.01)
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--e-cut", type=float, default=160.0)
    ap.add_argument("--max-order", type=int, default=6)
    ap.add_argument("--refine", type=int, default=0)
    a = ap.parse_args()
    rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{a.fit}.yaml", encoding="utf-8"))
    p = CellParams(**rec["params"])
    tr = Truncation(n_s=int(a.e_cut / p.f_s) + 3, n_c=8, n_q=14, ncut=25, e_cut=a.e_cut)
    m = StorageCouplerAncilla(p, tr)
    fluxes = np.round(np.arange(0.0, 0.5 + 1e-9, a.dflux), 6)
    t0 = time.time()
    sw = diabatic_sweep(m, fluxes)
    print(f"sweep: {len(fluxes)} flux points, dim {m.basis.dim}, {time.time() - t0:.0f} s; "
          f"min label weight {np.nanmin(np.where(sw.weight > 0, sw.weight, np.nan)):.3f}")
    fc = np.array([m.elements(float(x)).E_c[1] for x in fluxes])  # bare coupler 0-1 (GHz)

    xs = find_crossings(sw, m, nmax=a.nmax, max_order=a.max_order)
    xs.sort(key=lambda d: (d["order"], d["branch"], d["n"]))
    print(f"crossings (order <= {a.max_order}): {len(xs)}")
    from collections import Counter
    cnt = Counter((QNAME[d["branch"]], d["dsdcdq"], d["order"]) for d in xs)
    print("most common (branch, (ds,dc,dq), order):")
    for k, v in cnt.most_common(25):
        print(f"   {k}: {v}")

    if a.refine:
        print(f"refining gaps of the {a.refine} lowest-order crossings ...")
        for d in xs[: a.refine]:
            A = (d["n"], 0, d["branch"])
            lo, hi = d["flux"] - a.dflux, d["flux"] + a.dflux
            # smaller local model for speed: energy cutoff just above the crossing energy
            mloc = StorageCouplerAncilla(p, Truncation(n_s=int((d["energy"] + 40) / p.f_s) + 3, n_c=8, n_q=14,
                                                       ncut=25, e_cut=d["energy"] + 40.0))
            t1 = time.time()
            r = refine_gap(mloc, A, d["partner"], max(0.0, lo), min(0.5, hi), tol=2e-6)
            d.update(gap_MHz=r["gap_GHz"] * 1e3, flux_refined=r["flux"])
            print(f"   {QNAME[d['branch']]} n={d['n']:2d} -> {d['partner']} {d['dsdcdq']} order {d['order']}: "
                  f"flux {r['flux']:.5f}, gap {r['gap_GHz'] * 1e3:.4f} MHz ({time.time() - t1:.0f} s)")

    out = ROOT / "data" / "atlas"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ocelot_A2S3_fit{a.fit}_dflux{a.dflux}"
    with open(out / f"crossings_{tag}.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# diabatic crossings of branch states |n,0,q> with |s,c,q'> along Phi_c in [0,0.5]; params fit {a.fit};"
                 f" truncation {vars(tr)}\n")
        fh.write("branch,n,partner_s,partner_c,partner_q,ds,dc,dq,order,flux,f_c01_bare_GHz,energy_GHz,"
                 "slope_GHz_per_Phi0,gap_MHz\n")
        for d in xs:
            fcx = float(np.interp(d["flux"], fluxes, fc))
            fh.write(f"{QNAME[d['branch']]},{d['n']},{d['partner'][0]},{d['partner'][1]},{d['partner'][2]},"
                     f"{d['dsdcdq'][0]},{d['dsdcdq'][1]},{d['dsdcdq'][2]},{d['order']},{d['flux']:.5f},{fcx:.4f},"
                     f"{d['energy']:.6f},{d['slope_GHz_per_Phi0']:.4f},{d.get('gap_MHz', float('nan')):.6g}\n")

    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    cmap = ordinal_cmap(a.max_order)  # darkest = lowest order (strongest coupling)
    for q0, ax in zip((0, 1, 2), axs):
        sel = [d for d in xs if d["branch"] == q0]
        if sel:
            sc = ax.scatter([d["flux"] for d in sel], [d["n"] for d in sel], c=[d["order"] for d in sel],
                            cmap=cmap, vmin=0.5, vmax=a.max_order + 0.5, s=[60 / d["order"] for d in sel], edgecolors="none")
        ax.set_title(f"branch |{QNAME[q0]}, n>: {len(sel)} crossings")
        ax.set_xlabel(r"coupler flux $\Phi_c/\Phi_0$")
        ax.set_xlim(0, 0.5)
        ax.grid(alpha=0.3)
        sec = ax.secondary_xaxis("top", functions=(lambda x: np.interp(x, fluxes, fc),
                                                   lambda y: np.interp(y, fc[::-1], fluxes[::-1])))
        sec.set_xlabel("bare coupler $f_{01}$ (GHz)", fontsize=8)
    axs[0].set_ylabel("storage photon number n")
    if xs:
        fig.colorbar(sc, ax=axs, label="heuristic coupling order", shrink=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"atlas_preview_{tag}.{ext}", dpi=150)
    print("saved figure and", out / f"crossings_{tag}.csv")


if __name__ == "__main__":
    main()
