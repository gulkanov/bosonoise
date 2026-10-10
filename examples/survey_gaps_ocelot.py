"""Gap survey: one representative crossing per family (branch, ds, dc, dq) at fixed n.

The heuristic 'order' of a crossing ignores that the coupling amplitudes differ by more
than an order of magnitude (g_cq01 ~ 330 MHz vs g_sq01 ~ 14 MHz in the A2-S3 model), so
higher-order processes through the coupler can have larger gaps than low-order ones.
This script measures the gap of every family at a fixed photon number n.

Usage: python examples/survey_gaps_ocelot.py --n 10 [--sigma 16]
Output: data/atlas/gap_survey_n<n>_<tag>.csv
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

from refine_gaps_ocelot import QIDX, lz_probability  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="A")
    ap.add_argument("--tag", default="ocelot_A2S3_fitA_dflux0.01")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=16.0)
    ap.add_argument("--window", type=float, default=0.01)
    a = ap.parse_args()
    rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{a.fit}.yaml", encoding="utf-8"))
    p = CellParams(**rec["params"])
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "atlas" / f"crossings_{a.tag}.csv", encoding="utf-8")
                               if not l.startswith("#")))
    sel = [r for r in rows if int(r["n"]) == a.n]
    print(f"{len(sel)} crossings at n={a.n}")
    out = ROOT / "data" / "atlas" / f"gap_survey_n{a.n}_{a.tag}.csv"
    cache = {}
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# gap survey at n={a.n}; LZ single-passage probability for Gaussian edge sigma={a.sigma} ns, amplitude 0.5 Phi0\n")
        fh.write("branch,n,partner,ds,dc,dq,order,flux,f_c01_bare_GHz,gap_kHz,slope_GHz_per_Phi0,P_LZ\n")
        for r in sorted(sel, key=lambda r: (r["branch"], float(r["flux"]))):
            A = (a.n, 0, QIDX[r["branch"]])
            B = (int(r["partner_s"]), int(r["partner_c"]), int(r["partner_q"]))
            e = float(r["energy_GHz"])
            key = round(e + 40.0, 0)
            if key not in cache:
                cache[key] = StorageCouplerAncilla(p, Truncation(n_s=int((e + 40) / p.f_s) + 3, n_c=8, n_q=14,
                                                                 ncut=25, e_cut=e + 40.0))
            m = cache[key]
            fx = float(r["flux"])
            t0 = time.time()
            res = refine_gap(m, A, B, max(0.0, fx - a.window), min(0.5, fx + a.window), tol=1e-6)
            P, _ = lz_probability(res["gap_GHz"], float(r["slope_GHz_per_Phi0"]), res["flux"], a.sigma)
            line = (f"{r['branch']},{a.n},\"{B}\",{r['ds']},{r['dc']},{r['dq']},{r['order']},{res['flux']:.6f},"
                    f"{r['f_c01_bare_GHz']},{res['gap_GHz'] * 1e6:.5g},{r['slope_GHz_per_Phi0']},{P:.3e}")
            fh.write(line + "\n")
            fh.flush()
            print(f"{r['branch']} n={a.n} -> {B} ({r['ds']},{r['dc']},{r['dq']}) ord {r['order']}: flux {res['flux']:.5f}, "
                  f"gap {res['gap_GHz'] * 1e6:10.4f} kHz, P_LZ {P:.2e}  ({time.time() - t0:.0f} s)", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
