"""Zero-fit CX^2 bit-flip excesses for all 8 Ocelot pairs: dressed-ladder ULE simulation.

For each storage-ancilla pair (per-ancilla joint static calibration, coupler prior from
A2-S3) the full CX^2 flux pulse is simulated in the dressed ladder (channels/cx2.py):
flat-top Gaussian pulse to the 'on' point, gate length set to one full f-branch turn,
AWS-like phase calibration at |alpha|^2 = 1, ULE dissipation with the ancilla charge bath
referenced to the measured idle T1_ge, T1_fe (Fig. 9c,d; flat in frequency per transition
family), then two-photon re-stabilization (asymptotic projector).

Compared with the vector-extracted Fig. 16 (VALIDATION ONLY; Pauli per 3-us cycle):
  f-excess = p(|f>) - p(|g>)     <->  model p_f - p_g
  g-excess = p(|g>) - p(static)  <->  model p_g (gate-induced only)
'standard' = same simulation with undressed, n-independent f->e and e->g decay (Hann-type).

Gate length (protocol information, not error data): CX^2 length = L_CX(g/f) + 1/(2 chi_gf)
with L_CX from the vertical lines of Fig. 12 (data/digitized/ocelot_fig12_gates.csv, purple
line; A1-S1 gives 503 ns vs 504 ns stated on p.5) and the measured chi_gf; the model's ramp
(Gaussian shoulder, ramp = 3 sigma) is then solved for so that the pulse has this length and
the f-branch makes one full turn. Assumed (not published): the ancilla effective temperature
(heating scan 40-60 mK, only for the g-protocol band).
Re-stabilization map: two-photon dissipation with the commensurate Kerr-cat term of a
detuned buffer (App. F8, Eqs. F5-F6), K/kappa_2 = -Delta_b/kappa_b with the nominal
Delta_b,0/2pi ~ -2 MHz reported by AWS (App. F8) and kappa_b of each buffer (Fig. 13a);
this map favours recovering positive phase shifts. Results for the symmetric map
(Delta_b = 0) are reported as '_sym'.
Diagnostics (not predictions): the uniform ancilla-decay scaling lambda that would be needed
to reach the data, and the ancilla transitions' coupler participation.

Usage: python examples/predict_cx2_ule_ocelot.py [--pairs A4-S5,...]
Output: data/validation/cx2_ule_excess.csv, figures/cx2_ule_{fexcess,gexcess}.png/.pdf
"""
from __future__ import annotations

import argparse
import collections
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
from bosonoise.channels.cx2 import (BathModel, LadderGrid, calibrate_cx2_to_length, calibrate_phases,
                                    initial_blocks, ladder_at, simulate_cx2)
from bosonoise.dynamics.ule import classical_dephasing, detailed_balance, flat, piecewise_flat
from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIRS = ["A1-S1", "A1-S2", "A2-S2", "A2-S3", "A3-S3", "A3-S4", "A4-S4", "A4-S5"]
RAMP_STATED = {"A1-S1": 64.0, "A4-S4": 64.0}  # App. F7 p.29 (for comparison with the solved ramp only)
FLUX_GRID = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.425, 0.45, 0.47, 0.485, 0.495, 0.5]
NL, NMAX = 17, 15  # ladder photon cutoff (>= NMAX + 2 for dm = +2 jumps), simulated photon numbers
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TEMPS = [40.0, 50.0, 60.0]  # mK, g-protocol heating band
LAMS = [1.0, 2.0, 3.0, 5.0]  # uniform ancilla-decay scaling (diagnostic only)
# Coupler T1 is not published. 10 us is the value AWS assume in their coupler-heating simulations
# (arXiv:2409.17556v2, p.28: kappa_c = 1/(10 us)) -- a literature prior, not a fit; 3 us: sensitivity.
T1_COUPLER_US = [10.0, 3.0]
DELTA_B_MHZ = -2.0  # nominal buffer detuning Delta_b,0/2pi (App. F8, p.31)


def grid_for(pair: str, nl: int = NL) -> LadderGrid:
    """Dressed ladders on FLUX_GRID up to photon number nl (cached; the storage cutoff follows nl)."""
    anc, st = pair.split("-")
    path = ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml"
    p = load_cell_params(path, st)
    e_cut = (nl + 3) * p.f_s + 30.0
    tr = Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=8, n_q=10, ncut=25, e_cut=e_cut)
    key = hashlib.sha1(json.dumps([p.to_dict(), vars(tr), FLUX_GRID, nl], sort_keys=True).encode()).hexdigest()[:12]
    cache = ROOT / "runs" / "cx2" / f"ladders_{pair}_{key}.pkl"
    if cache.exists():
        return LadderGrid(pickle.loads(cache.read_bytes()))
    m = StorageCouplerAncilla(p, tr)
    lads = [ladder_at(m, f, nl) for f in FLUX_GRID]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(lads))
    return LadderGrid(lads)


def charge_bath(grid: LadderGrid, T1ge_us: float, T1fe_us: float, T_mK=None, lam=1.0,
                coupler_T1_us: float | None = None):
    """Ancilla charge bath, flat per transition family, referenced to the idle T1s.

    lam: scaling of the rates (diagnostic); a float, or (lam_eg, lam_fe) for the e->g and the
    f->e / f->g families separately.
    coupler_T1_us: if a coupler bath (flat, T1_c at 'on') is added separately, its share of the
    measured idle decay (through the idle coupler participation) is removed from the ancilla
    background, so that the total idle rates still equal the measured 1/T1."""
    idle = grid.ladders[0]
    w_ge, w_ef = idle.E[1, 0] - idle.E[0, 0], idle.E[2, 0] - idle.E[1, 0]
    r_ef, r_ge = 1.0 / (T1fe_us * 1e3), 1.0 / (T1ge_us * 1e3)
    if coupler_T1_us is not None:
        G_c = 1.0 / (coupler_T1_us * 1e3 * grid.ladders[-1].nc01 ** 2)
        r_ef -= G_c * idle.X["c"][1, 0, 2, 0] ** 2
        r_ge -= G_c * idle.X["c"][0, 0, 1, 0] ** 2
        if min(r_ef, r_ge) <= 0:
            raise ValueError("coupler share exceeds the measured idle decay")
    g_ef = r_ef / idle.X["q"][1, 0, 2, 0] ** 2
    g_ge = r_ge / idle.X["q"][0, 0, 1, 0] ** 2
    lam_eg, lam_fe = (lam, lam) if np.isscalar(lam) else lam
    return detailed_balance(piecewise_flat([0.5 * (w_ge + w_ef)], [lam_fe * g_ef, lam_eg * g_ge]), T_mK)


def run_pair(pair: str, pre: dict, projs_sym: dict) -> list[dict]:
    anc, st = pair.split("-")
    kerr_ratio = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kerr_ratio) for nb in NBARS}
    T1ge, T1fe = pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"]
    t0 = time.time()
    grid = grid_for(pair)
    L_cx = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}[pair]
    chi_meas = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
    pulse = calibrate_cx2_to_length(grid, L_cx * 1e3 + 0.5 / chi_meas)
    ramp = pulse.ramp
    corr = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0])  # with the experiment's (asymmetric) map
    on, idle = grid.ladders[-1], grid.ladders[0]
    chi_gf, chi_ge = grid.chi_gf(0.5), (on.E[1, 1] - on.E[1, 0]) - (on.E[0, 1] - on.E[0, 0])
    shift_ge = (on.E[1, 0] - on.E[0, 0]) - (idle.E[1, 0] - idle.E[0, 0])
    shift_ef = (on.E[2, 0] - on.E[1, 0]) - (idle.E[2, 0] - idle.E[1, 0])
    # coupler participation of the ancilla transitions at 'on' (relative to the coupler's own 0-1 element)
    part_fe = (on.X["c"][1, 0, 2, 0] / on.nc01) ** 2
    part_eg = (on.X["c"][0, 0, 1, 0] / on.nc01) ** 2
    q = charge_bath(grid, T1ge, T1fe)
    # 'nobath': deterministic part only (dressed Kerr distortion of the storage during the gate)
    baths = {"nobath": BathModel({}), "ours": BathModel({"q": q}), "standard": BathModel({"q": q}, standard=True)}
    for lam in LAMS[1:]:
        ql = charge_bath(grid, T1ge, T1fe, lam=lam)
        baths[f"lam{lam:g}"] = BathModel({"q": ql})
        baths[f"stdlam{lam:g}"] = BathModel({"q": ql}, standard=True)
    for T in TEMPS:
        baths[f"T{T:g}"] = BathModel({"q": charge_bath(grid, T1ge, T1fe, T_mK=T)})
    baths["dephasing_white_50us"] = BathModel({"q": q, "Nq": classical_dephasing(1.0 / 50e3, "white")})
    for T1c in T1_COUPLER_US:  # coupler charge bath, flat, referenced to the coupler's own 0-1 element at 'on'
        qc = charge_bath(grid, T1ge, T1fe, coupler_T1_us=T1c)  # idle totals kept equal to the measured T1
        baths[f"coupler{T1c:g}us"] = BathModel({"q": qc, "c": detailed_balance(flat(1.0 / (T1c * 1e3 * on.nc01 ** 2)), None)})
    out = []
    for nb in NBARS:
        P = projs[nb]
        Ps = projs_sym[nb]
        row = dict(pair=pair, nbar=nb, kerr_ratio=kerr_ratio, ramp_ns=ramp, gate_ns=pulse.duration, chi_gf_MHz=chi_gf * 1e3,
                   chi_ratio=chi_ge / chi_gf, delta_g=corr[0], delta_f=corr[1], part_fe=part_fe, part_eg=part_eg,
                   shift_ge_MHz=shift_ge * 1e3, shift_ef_MHz=shift_ef * 1e3)
        for name, bath in baths.items():
            for lev, tag in ((2, "f"), (0, "g")):
                if name.startswith("T") and lev == 2:
                    continue
                blk = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, NMAX), NMAX, blocks=True, phase_corr=corr)
                rho = blk.sum(axis=0)
                rho = rho / np.trace(rho).real
                row[f"p{tag}_{name}"] = P.p_flip(rho)
                row[f"p{tag}_{name}_sym"] = Ps.p_flip(rho)
                if name == "ours":
                    pops = np.trace(blk, axis1=1, axis2=2).real
                    row[f"pop_{tag}_g"], row[f"pop_{tag}_e"], row[f"pop_{tag}_f"] = pops
        out.append(row)
    print(f"{pair}: gate {pulse.duration:.0f} ns (solved ramp {ramp:.1f} ns; stated {RAMP_STATED.get(pair, 'n/a')}), chi_gf {chi_gf * 1e3:+.3f} MHz, r {chi_ge / chi_gf:.3f}, "
          f"phase corr {corr[0]:+.4f}/{corr[1]:+.4f} rad, coupler participation fe {part_fe:.2e} eg {part_eg:.2e}, "
          f"ancilla shift at 'on' ge {shift_ge * 1e3:+.1f} ef {shift_ef * 1e3:+.1f} MHz "
          f"({time.time() - t0:.0f} s)", flush=True)
    return out


def load_data():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig16.csv", encoding="utf-8")
                               if not l.startswith("#")))
    dat = collections.defaultdict(dict)
    for r in rows:
        dat[(r["pair"], r["protocol"])][float(r["alpha2"])] = float(r["p_cycle"])
    return dat


def load_results(path) -> list[dict]:
    """Rows of a saved validation CSV, numeric fields as floats (for --recompare)."""
    out = []
    for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")):
        row = {}
        for k, v in r.items():
            try:
                row[k] = float(v)
            except ValueError:
                row[k] = v
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--recompare", action="store_true",
                    help="reuse the simulated columns of the saved CSV and redo only the comparison with the data")
    a = ap.parse_args()
    pairs = a.pairs.split(",")
    dat = load_data()
    if a.recompare:
        res = [r for r in load_results(ROOT / "data" / "validation" / "cx2_ule_excess.csv") if r["pair"] in pairs]
    else:
        pre = load_preset("ocelot")
        projs_sym = {nb: AsymptoticProjector(nb) for nb in NBARS}
        res = []
        for pair in pairs:
            res += run_pair(pair, pre, projs_sym)
    for r in res:
        f, g, s = (dat[(r["pair"], k)].get(r["nbar"], np.nan) for k in ("f", "g", "static"))
        r["data_fexcess"], r["data_gexcess"] = f - g, g - s
        for name in (["nobath", "ours", "standard", "dephasing_white_50us"] + [f"coupler{c:g}us" for c in T1_COUPLER_US]
                     + [f"{m}lam{l:g}" for m in ("", "std") for l in LAMS[1:]]):
            for sfx in ("", "_sym"):
                r[f"fexcess_{name}{sfx}"] = r[f"pf_{name}{sfx}"] - r[f"pg_{name}{sfx}"]
                # fault-driven part: p is linear in rho, so the deterministic (no-bath) distortion separates
                r[f"ffault_{name}{sfx}"] = (r[f"pf_{name}{sfx}"] - r[f"pf_nobath{sfx}"]) - (r[f"pg_{name}{sfx}"] - r[f"pg_nobath{sfx}"])
        # uniform decay scaling needed to reach the data (diagnostic): interpolate the log of the
        # fault-driven f-excess in lambda (the g-protocol data bound the deterministic part, paper App. C.4)
        d = r["data_fexcess"]
        for tag, first in (("", "ours"), ("std", "standard")):
            fx = np.array([r[f"ffault_{first}"]] + [r[f"ffault_{tag}lam{l:g}"] for l in LAMS[1:]])
            ok = d > 0 and fx[0] > 0 and np.all(np.diff(fx) > 0)
            r["lambda_needed" + ("_std" if tag else "")] = \
                float(np.interp(np.log(d), np.log(fx), LAMS, left=np.nan, right=np.nan)) if ok else np.nan
    vd = ROOT / "data" / "validation"
    vd.mkdir(parents=True, exist_ok=True)
    keys = list(res[0])
    with open(vd / "cx2_ule_excess.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Zero-fit dressed-ladder ULE simulation of the CX^2 cycle vs vector-extracted Fig. 16 "
                 "(Pauli per 3-us cycle). Generated by examples/predict_cx2_ule_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in res:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("\npair   nbar | f-excess x1e-3: data | ours (sym) | std (sym) | coupler 10/3 us | lambda needed ours, std "
          "| g-excess x1e-3: data | Kerr distortion (sym), faults T=0 | total 40 / 50 / 60 mK")
    for r in res:
        if r["nbar"] in (2.0, 3.0, 4.0):
            print(f"{r['pair']} {r['nbar']:.1f} | {r['data_fexcess'] * 1e3:5.2f} | {r['ffault_ours'] * 1e3:5.2f} "
                  f"({r['ffault_ours_sym'] * 1e3:5.2f}) | {r['ffault_standard'] * 1e3:5.2f} ({r['ffault_standard_sym'] * 1e3:5.2f}) "
                  f"| {r['ffault_coupler10us'] * 1e3:5.2f} {r['ffault_coupler3us'] * 1e3:5.2f} "
                  f"| {r['lambda_needed']:.2f} {r['lambda_needed_std']:.2f} | {r['data_gexcess'] * 1e3:5.2f} | "
                  f"Kerr {r['pg_nobath'] * 1e3:5.2f} ({r['pg_nobath_sym'] * 1e3:5.2f}) faults {(r['pg_ours'] - r['pg_nobath']) * 1e3:5.2f} "
                  + " ".join(f"{r[k] * 1e3:5.2f}" for k in ("pg_T40", "pg_T50", "pg_T60")))
    use_style()
    for kind in ("fexcess", "gexcess"):
        fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, constrained_layout=True)
        for ip, pair in enumerate(pairs):
            sel = [r for r in res if r["pair"] == pair]
            x = [r["nbar"] for r in sel]
            ax = axs.flat[ip]
            if kind == "fexcess":
                ax.semilogy(x, [r["data_fexcess"] for r in sel], "o", ms=5, mfc="none", color=TEXT_PRIMARY, label="data (Fig. 16)")
                ax.semilogy(x, [r["ffault_ours"] for r in sel], "-", color=SERIES[0], label="dressed ULE (no fit)")
                ax.semilogy(x, [r["ffault_ours_sym"] for r in sel], "-", lw=0.7, color=SERIES[0],
                            label="dressed ULE, symmetric map (Delta_b = 0)")
                ax.semilogy(x, [r["ffault_standard"] for r in sel], "--", color=SERIES[1], label="undressed (Hann-type)")
                ax.semilogy(x, [r["ffault_coupler10us"] for r in sel], "-.", color=SERIES[2],
                            label="dressed + coupler bath, T1_c = 10 us (prior)")
                ax.semilogy(x, [r["ffault_lam2"] for r in sel], ":", color=SERIES[3], label="dressed, decay x2 (diagnostic)")
                ax.set_ylim(1e-5, 1e-2)
            else:
                ax.plot(x, [r["data_gexcess"] * 1e3 for r in sel], "o", ms=5, mfc="none", color=TEXT_PRIMARY, label="data (Fig. 16)")
                ax.fill_between(x, [r["pg_T40"] * 1e3 for r in sel], [r["pg_T60"] * 1e3 for r in sel], color=SERIES[0],
                                alpha=0.25, lw=0, label="dressed ULE, T_anc 40-60 mK")
                ax.plot(x, [r["pg_ours"] * 1e3 for r in sel], "-", color=SERIES[0], label="dressed ULE, T_anc = 0")
                ax.plot(x, [r["pg_nobath"] * 1e3 for r in sel], ":", color=SERIES[4], label="of which: Kerr distortion")
                ax.plot(x, [r["pg_dephasing_white_50us"] * 1e3 for r in sel], "--", color=SERIES[3],
                        label="+ white freq. noise, T_phi = 50 us (bound)")
                ax.axhline(0, color="#52514e", lw=0.6)
            ax.set_title(f"{pair} (gate {sel[0]['gate_ns']:.0f} ns, r = {sel[0]['chi_ratio']:.2f})", fontsize=9)
        for ax in axs[1]:
            ax.set_xlabel(r"$|\alpha|^2$")
        for ax in axs[:, 0]:
            ax.set_ylabel("f-excess per cycle (fault-driven)" if kind == "fexcess" else r"g-excess per cycle ($10^{-3}$)")
        axs.flat[0].legend(fontsize=6, loc="lower left" if kind == "fexcess" else "upper left")
        for ext in ("png", "pdf"):
            fig.savefig(ROOT / "figures" / f"cx2_ule_{kind}.{ext}", dpi=160)
    print("saved", vd / "cx2_ule_excess.csv")


if __name__ == "__main__":
    main()
