"""Lossy-coupler hypothesis for the CX^2 residuals of all 8 Ocelot pairs (diagnostic, not a fit).

The dressed ladder now includes the coupler-excited branches (c = 1). A coupler charge bath
with T1_c (scanned; not published) and an up/down ratio of 1e-3 at the 'on' frequency (the
value AWS assume in arXiv:2409.17556, p.28) acts on
  * the ancilla transitions through their coupler participation (7-12% at 'on', 2-3% at idle):
    on-gate enhancement lambda of the ancilla decay (f-excess). The idle share of the coupler
    loss is removed from the ancilla background, so the idle totals equal the measured T1s;
  * the coupler itself: heating during the gate, after which the storage rotates at the
    coupler-conditional frequency chi_sc (~ -1 MHz at 'on'): n-independent g-protocol floor.
For each pair and |alpha|^2 we read off the T1_c that would reproduce the measured fault-driven
f-excess, and report the g-excess that the same T1_c predicts from coupler heating during the
gate, next to the measured g-excess. Agreement of the two is a consistency test of the
hypothesis; nothing is tuned to the g data.

Usage: python examples/coupler_hypothesis_ocelot.py [--pairs ...]
Output: data/validation/cx2_coupler_hypothesis.csv, figures/cx2_coupler_hypothesis.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import pickle
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import (WITH_COUPLER, BathModel, LadderGrid, calibrate_cx2_to_length, calibrate_phases,
                                    initial_blocks, ladder_at, simulate_cx2)
from bosonoise.dynamics.ule import H_OVER_KB_MK, detailed_balance, flat
from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import (DELTA_B_MHZ, FLUX_GRID, NL, NMAX, PAIRS, charge_bath, load_data,  # same directory
                                    load_results)

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBARS = [2.0, 3.0, 4.0]
T1C_US = [1.5, 2.0, 3.0, 5.0, 10.0, 30.0]  # 1 us would exceed the measured idle decay of some ancillas
UP_OVER_DOWN = 1e-3  # AWS assumption (arXiv:2409.17556 p.28), applied at the 'on' coupler frequency


def coupler_grid(pair: str) -> LadderGrid:
    anc, st = pair.split("-")
    p = load_cell_params(ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml", st)
    e_cut = (NL + 3) * p.f_s + 40.0  # +10 GHz headroom for the coupler excitation
    tr = Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=8, n_q=10, ncut=25, e_cut=e_cut)
    key = hashlib.sha1(json.dumps([p.to_dict(), vars(tr), FLUX_GRID, NL, "coupler"], sort_keys=True).encode()).hexdigest()[:12]
    cache = ROOT / "runs" / "cx2" / f"ladders_coupler_{pair}_{key}.pkl"
    if cache.exists():
        return LadderGrid(pickle.loads(cache.read_bytes()))
    m = StorageCouplerAncilla(p, tr)
    lads = [ladder_at(m, f, NL, branches=WITH_COUPLER) for f in FLUX_GRID]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(lads))
    return LadderGrid(lads)


def run_pair(pair, pre, dat):
    anc, st = pair.split("-")
    t0 = time.time()
    grid = coupler_grid(pair)
    L_cx = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}[pair]
    chi = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
    pulse = calibrate_cx2_to_length(grid, L_cx * 1e3 + 0.5 / chi)
    kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kr) for nb in [1.0] + NBARS}
    corr = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0])
    on = grid.ladders[-1]
    f_c_on = on.E[3, 0] - on.E[0, 0]
    T_c = f_c_on * H_OVER_KB_MK / np.log(1.0 / UP_OVER_DOWN)
    q = charge_bath(grid, pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"])
    baths = {"nobath": BathModel({}), "q": BathModel({"q": q})}
    for T1c in T1C_US:
        c = detailed_balance(flat(1.0 / (T1c * 1e3 * on.nc01 ** 2)), T_c)
        qc = charge_bath(grid, pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"], coupler_T1_us=T1c)
        baths[f"c{T1c:g}"] = BathModel({"q": qc, "c": c})
    rows = []
    for nb in NBARS:
        P = projs[nb]
        p = {}
        for name, bath in baths.items():
            for lev, tag in ((2, "f"), (0, "g")):
                rho = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, NMAX), NMAX, phase_corr=corr)
                p[(name, tag)] = P.p_flip(rho / np.trace(rho).real)
        ffault = {name: (p[(name, "f")] - p[("nobath", "f")]) - (p[(name, "g")] - p[("nobath", "g")]) for name in baths}
        gheat = {name: p[(name, "g")] - p[("q", "g")] for name in baths}  # coupler-induced part of p_g
        row = dict(pair=pair, nbar=nb, f_c_on_GHz=f_c_on, T_c_mK=T_c, chi_sc_on_MHz=grid.chi_rel(0.5, 3) * 1e3,
                   data_fexcess=np.nan, data_gexcess=np.nan, ffault_q=ffault["q"], kerr_g=p[("nobath", "g")])
        for T1c in T1C_US:
            row[f"ffault_c{T1c:g}"] = ffault[f"c{T1c:g}"]
            row[f"gcoupler_c{T1c:g}"] = gheat[f"c{T1c:g}"]
        compare(row, dat)
        rows.append(row)
    print(f"{pair}: f_c(on) {f_c_on:.3f} GHz, T_c {T_c:.1f} mK, chi_sc(on) {grid.chi_rel(0.5, 3) * 1e3:+.3f} MHz, "
          f"min branch overlap {min(l.min_overlap for l in grid.ladders):.3f} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def compare(row, dat):
    """Data columns, the T1_c implied by the f-excess (monotonic in 1/T1_c) and the g-excess it predicts."""
    pair, nb = row["pair"], row["nbar"]
    f_d = dat[(pair, "f")].get(nb, np.nan) - dat[(pair, "g")].get(nb, np.nan)
    row["data_fexcess"] = f_d
    row["data_gexcess"] = dat[(pair, "g")].get(nb, np.nan) - dat[(pair, "static")].get(nb, np.nan)
    x = np.array([1.0 / t for t in T1C_US])  # 1/us, ascending when T1C_US descends -> sort
    fx = np.array([row[f"ffault_c{t:g}"] for t in T1C_US])
    gx = np.array([row[f"gcoupler_c{t:g}"] for t in T1C_US])
    order = np.argsort(x)
    x, fx, gx = x[order], fx[order], gx[order]
    if f_d > fx[0] and np.all(np.diff(fx) > 0) and f_d <= fx[-1]:
        xi = np.interp(f_d, fx, x)
        row["T1c_implied_us"] = 1.0 / xi
        row["gcoupler_implied"] = float(np.interp(xi, x, gx))
    else:
        row["T1c_implied_us"] = np.nan if f_d <= fx[-1] else float("-inf")
        row["gcoupler_implied"] = np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--plot-only", action="store_true", help="re-plot from the saved CSV")
    ap.add_argument("--recompare", action="store_true",
                    help="reuse the simulated columns of the saved CSV and redo only the comparison with the data")
    a = ap.parse_args()
    path = ROOT / "data" / "validation" / "cx2_coupler_hypothesis.csv"
    if a.plot_only:
        plot(list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#"))))
        return
    dat = load_data()
    if a.recompare:
        res = [r for r in load_results(path) if r["pair"] in a.pairs.split(",")]
        for r in res:
            compare(r, dat)
    else:
        pre = load_preset("ocelot")
        res = []
        for pair in a.pairs.split(","):
            res += run_pair(pair, pre, dat)
    keys = list(res[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Lossy-coupler hypothesis (diagnostic): fault-driven f-excess and coupler-induced g-protocol error vs T1_c; "
                 "implied T1_c from the f data and the g-excess it predicts. examples/coupler_hypothesis_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in res:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("\npair  nbar | f-excess data | implied T1_c (us) | g-excess: data  predicted by that T1_c (+Kerr) | x1e-3")
    for r in res:
        print(f"{r['pair']} {r['nbar']:.0f} | {r['data_fexcess'] * 1e3:5.2f} | {r['T1c_implied_us']:6.2f} | "
              f"{r['data_gexcess'] * 1e3:5.2f}  {r['gcoupler_implied'] * 1e3:5.2f} (+{r['kerr_g'] * 1e3:4.2f})")
    plot(res)
    print("saved", path)


def plot(res):
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for i, pair in enumerate(dict.fromkeys(r["pair"] for r in res)):
        sel = [r for r in res if r["pair"] == pair]
        x = np.array([float(r["nbar"]) for r in sel])
        t1 = np.array([float(r["T1c_implied_us"]) for r in sel])
        ok = np.isfinite(t1)
        col = SERIES[i % len(SERIES)]
        ax[0].plot(x[ok], t1[ok], "o-", color=col, label=pair)
        low = ~ok & np.isneginf(t1)
        ax[0].plot(x[low], np.full(low.sum(), 1.5), "v", color=col)  # data above the model at T1_c = 1.5 us
        gd = np.array([float(r["data_gexcess"]) for r in sel]) * 1e3
        gm = np.array([float(r["gcoupler_implied"]) for r in sel]) * 1e3
        m = np.isfinite(gd) & np.isfinite(gm)
        ax[1].plot(gd[m], gm[m], "o", color=col, label=pair)
    ax[0].set_yscale("log")
    ax[0].set_xlabel(r"$|\alpha|^2$")
    ax[0].set_ylabel(r"coupler $T_1$ implied by the f-excess (µs)")
    ax[0].set_title("triangles: would need T1_c < 1.5 µs", fontsize=8)
    ax[0].legend(fontsize=7, ncol=2)
    lim = [0, 0.5]
    ax[1].plot(lim, lim, color=TEXT_PRIMARY, lw=0.6)
    ax[1].set_xlabel(r"measured g-excess ($10^{-3}$)")
    ax[1].set_ylabel(r"coupler-heating g-error at implied $T_1$ ($10^{-3}$)")
    ax[1].set_title("consistency test (no fit to g data)", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"cx2_coupler_hypothesis.{ext}", dpi=160)


if __name__ == "__main__":
    main()
