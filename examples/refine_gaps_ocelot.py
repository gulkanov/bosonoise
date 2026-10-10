"""Minimal gaps of selected diabatic crossings from the atlas skeleton (Ocelot A2-S3 model).

Reads data/atlas/crossings_<tag>.csv, selects crossings by family (ds, dc, dq) and branch,
refines the avoided-crossing gap with a golden-section search on the splitting, and
estimates the single-passage Landau-Zener transfer probability for a Gaussian flux edge:

    P = 1 - exp(-pi * Delta^2 / (2 v)),  Delta = minimal splitting (rad/ns),
    v = |d(E_B - E_A)/dPhi| * |dPhi/dt|   (rad/ns^2)

with |dPhi/dt| evaluated on a Gaussian edge Phi(t) = A exp(-t^2 / 2 sigma^2) at the
crossing flux (A = 0.5 Phi0). sigma is a free input (ramp shapes are NOT reported for
A2-S3; 64 ns ramps are reported for A1-S1 and A4-S4).

Usage: python examples/refine_gaps_ocelot.py --family=-3,1,2 --branch f [--ns 3,6,10,15,20 --sigma 16]
       (write --family=... with '=': argparse reads a bare -3,1,2 as an option)
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.spectra.crossings import refine_gap

ROOT = pathlib.Path(__file__).resolve().parents[1]
QIDX = {"g": 0, "e": 1, "f": 2}


def lz_probability(gap_GHz, slope_GHz_per_Phi0, flux, sigma_ns, amp=0.5):
    x = min(max(flux / amp, 1e-9), 1 - 1e-12)
    t = sigma_ns * np.sqrt(2 * np.log(1 / x))  # time before the top of the edge
    dphidt = amp * x * t / sigma_ns ** 2  # Phi0/ns
    v = 2 * np.pi * abs(slope_GHz_per_Phi0) * dphidt  # rad/ns^2
    delta = 2 * np.pi * gap_GHz
    return 1 - np.exp(-np.pi * delta ** 2 / (2 * v)), dphidt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="A")
    ap.add_argument("--tag", default="ocelot_A2S3_fitA_dflux0.01")
    ap.add_argument("--family", default="-3,1,2")
    ap.add_argument("--branch", default="f")
    ap.add_argument("--ns", default="3,6,10,15,20")
    ap.add_argument("--sigma", type=float, default=16.0)
    ap.add_argument("--window", type=float, default=0.01)
    a = ap.parse_args()
    fam = tuple(int(v) for v in a.family.split(","))
    ns = [int(v) for v in a.ns.split(",")]
    rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{a.fit}.yaml", encoding="utf-8"))
    p = CellParams(**rec["params"])
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "atlas" / f"crossings_{a.tag}.csv", encoding="utf-8")
                               if not l.startswith("#")))
    sel = [r for r in rows if r["branch"] == a.branch and (int(r["ds"]), int(r["dc"]), int(r["dq"])) == fam
           and int(r["n"]) in ns]
    out = ROOT / "data" / "atlas" / f"gaps_{a.tag}_{a.branch}_{'_'.join(map(str, fam))}.csv"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# refined gaps; family {fam}, branch {a.branch}; LZ estimate for Gaussian edge sigma={a.sigma} ns\n")
        fh.write("branch,n,partner,flux,gap_MHz,slope_GHz_per_Phi0,dphidt_Phi0_per_ns,P_LZ_single_passage\n")
        for r in sorted(sel, key=lambda r: int(r["n"])):
            n = int(r["n"])
            A = (n, 0, QIDX[a.branch])
            B = (int(r["partner_s"]), int(r["partner_c"]), int(r["partner_q"]))
            e = float(r["energy_GHz"])
            tr = Truncation(n_s=int((e + 40) / p.f_s) + 3, n_c=8, n_q=14, ncut=25, e_cut=e + 40.0)
            m = StorageCouplerAncilla(p, tr)
            fx = float(r["flux"])
            t0 = time.time()
            res = refine_gap(m, A, B, max(0.0, fx - a.window), min(0.5, fx + a.window), tol=1e-6)
            P, dphidt = lz_probability(res["gap_GHz"], float(r["slope_GHz_per_Phi0"]), res["flux"], a.sigma)
            print(f"{a.branch} n={n:2d} -> {B}: flux {res['flux']:.6f}, gap {res['gap_GHz'] * 1e3:.5f} MHz, "
                  f"weights {np.round(res['weights'], 3).tolist()}, P_LZ(sigma={a.sigma}ns) = {P:.3e} "
                  f"(dim {m.basis.dim}, {time.time() - t0:.0f} s)")
            fh.write(f"{a.branch},{n},\"{B}\",{res['flux']:.7f},{res['gap_GHz'] * 1e3:.6g},{r['slope_GHz_per_Phi0']},"
                     f"{dphidt:.5g},{P:.4e}\n")
    print("saved", out)


if __name__ == "__main__":
    main()
