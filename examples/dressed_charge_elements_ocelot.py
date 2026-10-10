"""Photon-number dependence of dressed charge matrix elements between branch states.

Preliminary (pre-ULE) estimate of how the storage field opens *direct* ancilla transitions:
in the bare basis <g|n_q|f> = 0 (parity), but between dressed branch states
<g~, m| n_q |f~, n> is non-zero and grows with n. With a charge bath of spectral density
S(omega), Fermi's golden rule gives Gamma(i~n -> j~m) = |<j~m|n|i~n>|^2 S(omega_nm).

We report, versus storage Fock number n at the "on" point:
  R_fg(n) = sum_m |<g~,m|n_q|f~,n>|^2 / |<e|n_q|f>|^2_bare   (direct f->g relative to bare f->e)
  R_fe(n) = sum_m |<e~,m|n_q|f~,n>|^2 / |<e|n_q|f>|^2_bare
  R_eg(n) = sum_m |<g~,m|n_q|e~,n>|^2 / |<g|n_q|e>|^2_bare
and the same through the coupler charge n_c (normalized to the ancilla bare element).
The transition frequencies differ between the m-channels; no bath model is applied here
(flat-S assumption only for orientation). Full ULE treatment: examples/predict_cx2_ule_ocelot.py.

Usage: python examples/dressed_charge_elements_ocelot.py [--fit A] [--flux 0.5]
Output: data/branches/dressed_charge_elements_<tag>.csv, figure
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, use_style
from bosonoise.spectra import branch_analysis

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="A")
    ap.add_argument("--flux", type=float, default=0.5)
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--calib", default=None, help="calibration YAML (default: A2-S3 fit --fit)")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--ecut", type=float, default=160.0, help="bare-energy cutoff (GHz); raise with --nmax")
    ap.add_argument("--nq", type=int, default=14)
    a = ap.parse_args()
    if a.calib is None:
        rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{a.fit}.yaml", encoding="utf-8"))
        p = CellParams(**rec["params"])
        label = "A2S3"
    else:
        p = load_cell_params(a.calib, a.cell)
        label = a.label or f"A4{a.cell}"
    tr = Truncation(n_s=int(a.ecut / p.f_s) + 3, n_c=8, n_q=a.nq, ncut=25, e_cut=a.ecut)
    m = StorageCouplerAncilla(p, tr)
    E, V = m.eigh(a.flux)
    # track two extra photons so that the m = n+1, n+2 channels exist for every reported n
    br = branch_analysis(m, a.flux, starts=((0, 0), (0, 1), (0, 2)), nmax=a.nmax + 2, E=E, V=V)
    nc, nq = m.charge_ops(a.flux)
    el = m.elements(a.flux)
    ref = {"ge": el.n_q[0, 1] ** 2, "ef": el.n_q[1, 2] ** 2}
    idx = {"g": br.index[0], "e": br.index[1], "f": br.index[2]}
    N = a.nmax + 1
    rows = []
    for n in range(N):
        out = {"n": n}
        for (i, j, refkey) in (("f", "g", "ef"), ("f", "e", "ef"), ("e", "g", "ge")):
            vi = V[:, idx[i][n]]
            for opname, op in (("q", nq), ("c", nc)):
                w = op @ vi
                tot, best = 0.0, (0.0, None, None)
                for mm in range(max(0, n - 2), n + 3):
                    amp2 = float((V[:, idx[j][mm]] @ w) ** 2)
                    tot += amp2
                    if amp2 > best[0]:
                        best = (amp2, mm, float(E[idx[i][n]] - E[idx[j][mm]]))
                out[f"R_{i}{j}_{opname}"] = tot / ref[refkey]
                out[f"dom_m_{i}{j}_{opname}"] = best[1]
                out[f"freq_{i}{j}_{opname}_GHz"] = best[2]
        rows.append(out)
    tag = f"ocelot_{label}_fit{a.fit}_flux{a.flux:.3f}" + (f"_nmax{a.nmax}" if a.nmax != 20 else "")
    path = ROOT / "data" / "branches" / f"dressed_charge_elements_{tag}.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# dressed charge matrix elements between branch states (sum over m=n-2..n+2), "
                 "normalized to bare |<e|n_q|f>|^2 (f->g, f->e) or |<g|n_q|e>|^2 (e->g)\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    for r in rows[:: max(1, a.nmax // 10)] + [rows[-1]]:
        print(f"n={r['n']:2d}  R_fg(q)={r['R_fg_q']:.3e} [m={r['dom_m_fg_q']}, {r['freq_fg_q_GHz']:.3f} GHz]  "
              f"R_fg(c)={r['R_fg_c']:.3e}  R_fe(q)={r['R_fe_q']:.4f}  R_eg(q)={r['R_eg_q']:.4f}  R_eg(c)={r['R_eg_c']:.3e}")
    use_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    n = np.arange(N)
    ax.semilogy(n, [r["R_fg_q"] for r in rows], "o-", ms=4, color=SERIES[0], label=r"$f\to g$ via $n_q$")
    ax.semilogy(n, [r["R_fg_c"] for r in rows], "s--", ms=4, color=SERIES[1], label=r"$f\to g$ via $n_c$")
    ax.set_xlabel("storage photon number n")
    ax.set_ylabel(r"$\sum_m |\langle \tilde g,m|n|\tilde f,n\rangle|^2 / |\langle e|n_q|f\rangle|^2$")
    ax.set_title(f"direct f->g opened by dressing ({label}, fit {a.fit}, $\\Phi_c$={a.flux})", fontsize=9)
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"dressed_fg_{tag}.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
