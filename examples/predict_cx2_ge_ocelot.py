"""CX^2 with the ancilla in (|g> + |e>)/sqrt(2): a direct test of the e->g decay during the gate.

After the ancilla is traced out, the storage state of the g+e protocol is the equal mixture of
the g-branch and e-branch results, so
    p(g+e) - p(g) = [p_e - p_g] / 2
with p_e the bit-flip probability of a CX^2 with the ancilla prepared in |e> (the idle,
stabilization-induced errors cancel in the difference). The data side uses Fig. 16 (g+e and
g protocols; VALIDATION ONLY); the same identity for g+f is checked on the data themselves.
p_e contains (i) the deterministic rotation 2 pi r of the e branch (r = chi_ge/chi_gf; error
2 pi (r - 1), plus the e-branch Kerr shear) and (ii) e->g decays during the gate, which
convert with eta ~ 1/2 (random angle 2 pi r t/T). For the chi-matched pairs (ii) dominates at
|alpha|^2 >= 2, so the data measure the on-gate e->g rate: we report the uniform scaling
lambda_e of the idle-referenced e->g rate that the data require at each |alpha|^2 — a
photon-number-dependent T1(e) (Fig. 37d) would show up as lambda_e growing with |alpha|^2.

Usage: python examples/predict_cx2_ge_ocelot.py [--recompare]
Output: data/validation/cx2_ge_protocol.csv, figures/cx2_ge_protocol.png/.pdf
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
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases, initial_blocks, simulate_cx2
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import (DELTA_B_MHZ, NMAX, PAIRS, charge_bath, grid_for, load_data,  # same directory
                                    load_results)

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBARS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
LAM_E = [1.0, 1.5, 2.0, 3.0, 5.0]
PATH = ROOT / "data" / "validation" / "cx2_ge_protocol.csv"


def lam_key(lam: float) -> str:
    return "model_lam" + f"{lam:g}".replace(".", "p")


def simulate(pair: str, pre: dict, L: dict) -> list[dict]:
    t0 = time.time()
    anc, st = pair.split("-")
    grid = grid_for(pair)
    chi = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
    pulse = calibrate_cx2_to_length(grid, L[pair] * 1e3 + 0.5 / chi)
    kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kr) for nb in [1.0] + NBARS}
    projs_sym = {nb: AsymptoticProjector(nb) for nb in NBARS}  # symmetric map (Delta_b = 0): systematic band
    corr = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0])
    T1ge, T1fe = pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"]
    baths = {"nobath": BathModel({})}
    for lam in LAM_E:
        baths[f"lam{lam:g}"] = BathModel({"q": charge_bath(grid, T1ge, T1fe, lam=(lam, 1.0))})
    baths["standard"] = BathModel({"q": charge_bath(grid, T1ge, T1fe)}, standard=True)
    on = grid.ladders[-1]
    r = (on.E[1, 1] - on.E[1, 0] - on.E[0, 1] + on.E[0, 0]) / grid.chi_gf(0.5)
    rows = []
    for nb in NBARS:
        P, Ps = projs[nb], projs_sym[nb]
        p, ps = {}, {}
        for name, bath in baths.items():
            for lev, tag in ((0, "g"), (1, "e")):
                rho = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, NMAX), NMAX, phase_corr=corr)
                p[(name, tag)] = P.p_flip(rho / np.trace(rho).real)
                ps[(name, tag)] = Ps.p_flip(rho / np.trace(rho).real)
        model = {name: 0.5 * (p[(name, "e")] - p[(name, "g")]) for name in baths}
        model_sym = {name: 0.5 * (ps[(name, "e")] - ps[(name, "g")]) for name in baths}
        row = dict(pair=pair, nbar=nb, r=r, model_nobath=model["nobath"], model_standard=model["standard"],
                   model_nobath_sym=model_sym["nobath"], model_lam1_sym=model_sym["lam1"])
        row.update({lam_key(l): model[f"lam{l:g}"] for l in LAM_E})
        rows.append(row)
    print(f"{pair}: r = {r:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def compare(row: dict, dat: dict) -> None:
    """Data columns and the e->g scaling lambda_e the data require (inf: above the simulated range)."""
    pair, nb = row["pair"], row["nbar"]
    g = dat[(pair, "g")].get(nb, np.nan)
    d = dat[(pair, "g+e")].get(nb, np.nan) - g
    row["data_ge_minus_g"] = d
    row["data_gf_minus_g"] = dat[(pair, "g+f")].get(nb, np.nan) - g
    row["data_half_fexcess"] = 0.5 * (dat[(pair, "f")].get(nb, np.nan) - g)
    row["data_over_model_lam1"] = d / row[lam_key(1.0)]
    lams = [l for l in LAM_E if lam_key(l) in row]
    mv = np.array([row[lam_key(l)] for l in lams])
    row["lambda_e_needed"] = (float(np.interp(d, mv, lams, left=np.nan, right=np.inf))
                              if np.isfinite(d) and np.all(np.diff(mv) > 0) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompare", action="store_true",
                    help="reuse the simulated columns of the saved CSV and redo only the comparison with the data")
    a = ap.parse_args()
    dat = load_data()
    if a.recompare:
        rows = load_results(PATH)
    else:
        pre = load_preset("ocelot")
        L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
            l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
            if not l.startswith("#")) if r["color"] == "purple"}
        rows = [row for pair in PAIRS for row in simulate(pair, pre, L)]
    for row in rows:
        compare(row, dat)
    keys = list(rows[0])
    with open(PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# g+e protocol: p(g+e) - p(g) (Fig. 16, Pauli per 3-us cycle) vs model (p_e - p_g)/2; e->g rate scaled by "
                 "lambda_e. examples/predict_cx2_ge_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r_ in rows:
            fh.write(",".join(f"{r_[k]:.6g}" if isinstance(r_[k], float) else str(r_[k]) for k in keys) + "\n")
    print("\npair  nbar | data (g+e)-g | model: no bath, lambda_e=1, standard | data/model | lambda_e needed || "
          "data check (g+f)-g vs (f-g)/2  [x1e-3]")
    for r_ in rows:
        print(f"{r_['pair']} {r_['nbar']:.1f} | {r_['data_ge_minus_g'] * 1e3:7.2f} | {r_['model_nobath'] * 1e3:7.2f} "
              f"{r_[lam_key(1.0)] * 1e3:7.2f} {r_['model_standard'] * 1e3:7.2f} | {r_['data_over_model_lam1']:5.2f} | "
              f"{r_['lambda_e_needed']:.2f} || {r_['data_gf_minus_g'] * 1e3:6.3f} vs {r_['data_half_fexcess'] * 1e3:6.3f}")
    use_style()
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, constrained_layout=True)
    for ax, pair in zip(axs.flat, PAIRS):
        sel = [r_ for r_ in rows if r_["pair"] == pair]
        x = [r_["nbar"] for r_ in sel]
        ax.semilogy(x, [r_["data_ge_minus_g"] for r_ in sel], "o", mfc="none", color=TEXT_PRIMARY, label="data: p(g+e) − p(g)")
        ax.semilogy(x, [r_[lam_key(1.0)] for r_ in sel], "-", color=SERIES[0], label="dressed ULE, idle T1 (no fit)")
        ax.semilogy(x, [r_["model_nobath"] for r_ in sel], ":", color=SERIES[3], label="deterministic part (χ mismatch + Kerr)")
        ax.semilogy(x, [r_[lam_key(2.0)] for r_ in sel], "--", color=SERIES[2], label="e→g rate ×2 (diagnostic)")
        ax.set_title(f"{pair} (r = {sel[0]['r']:.2f})", fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$[p_e - p_g]/2$ per cycle")
    axs.flat[0].legend(fontsize=6, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"cx2_ge_protocol.{ext}", dpi=160)
    print("saved", PATH)


if __name__ == "__main__":
    main()
