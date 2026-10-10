"""CX^2 f-excess with the ancilla decay during the gate taken from MEASURED coherence (paper App. C.2).

The dressed-ladder ULE model (examples/predict_cx2_ule_ocelot.py) references the ancilla charge bath to the idle T1s
(Fig. 9c, d). It underestimates the measured fault-driven f-excess of Fig. 16 by a median factor 4.3: the decay
enhancement needed during the gate is lambda ~ 2.8 (paper Sec. 6). Here the decay at 'on' is taken instead from
coherence measured with the coupler at, or pulsed to, its 'on' point. These are device characterizations, not
error-rate data; nothing is tuned to Fig. 16.

  A4  Fig. 37d (A4 on the A4-S5 coupler, App. J). T1(e) and T1(f) with the coupler held on, pulsed (one CX^2 per
      1.6 us), and pulsed with |alpha|^2 = 4 in the storage. The e->g and the f-family decay rates are scaled so that
      the model's on-point rates equal the measured ones (its own on-point dressing factor divided out).
      - Held on is the pure on-point decay.
      - The pulsed values are the closest to the CX^2 conditions, but they average the on and off segments.
  A3  Fig. 18c (A3 on the A3-S3 coupler, another cooldown). T1(e->g) versus coupler flux. Its smooth decline to Phi0/2
      (median ratio idle/on) is attributed to coupler loss. The coupler T1 at 'on' that reproduces the ratio in the
      coupler-participation model (examples/fig18c_coupler_loss_ocelot.py) then acts on every transition of the
      dressed ladder, the f-family included.

Labels: 'direct' = the measured ancilla and coupler; 'transferred' = the same ancilla on its other coupler (A4-S4,
A3-S4); 'prior' = the A3 coupler T1 used for the A1 and A2 pairs.

Usage: python examples/measured_bath_cx2_ocelot.py [--pairs A4-S5,A3-S3,...] [--plot]
Output: data/validation/cx2_measured_bath.csv, figures/cx2_measured_bath.png/.pdf
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
from scipy.optimize import brentq

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases, initial_blocks, simulate_cx2
from bosonoise.dynamics.ule import detailed_balance, flat
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, NBARS, NMAX, PAIRS, charge_bath, grid_for, load_data  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "cx2_measured_bath.csv"
CONDITIONS = ("coupler_on", "pulsed", "pulsed_alpha2_4")
LABEL = {"A4-S5": "direct", "A4-S4": "transferred", "A3-S3": "direct", "A3-S4": "transferred"}


def fig37d_means():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig37d.csv", encoding="utf-8")
                               if not l.startswith("#")))
    out = {}
    for st in ("e", "f"):
        for c in CONDITIONS:
            v = [float(r["T1_us"]) for r in rows if r["state"] == st and r["condition"] == c]
            out[(st, c)] = (float(np.mean(v)), float(np.std(v)))
    return out


def fig18c_ratio():
    """Median T1 for Phi <= 0.2 over the non-TLS median for Phi >= 0.4 (the selection of fig18c_coupler_loss_ocelot)."""
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig18c.csv", encoding="utf-8")
                               if not l.startswith("#")))
    fl = np.array([float(r["flux_Phi0"]) for r in rows])
    T1 = np.array([float(r["T1_us"]) for r in rows])
    err = np.array([float(r["T1_err_us"]) for r in rows])
    ok = err < 0.5 * T1
    idle = np.median(T1[(fl <= 0.2) & ok])
    on = np.median(T1[(fl >= 0.4) & ok & (T1 > 15)])
    return float(idle), float(on)


def coupler_T1_from_18c(pre):
    """Coupler T1 at 'on' that gives the measured idle/on T1(e->g) ratio of Fig. 18c in the A3-S3 participation model."""
    idle18, on18 = fig18c_ratio()
    target = idle18 / on18
    T1ge = pre["ancilla"]["A3"]["T1_ge_us"]
    grid = grid_for("A3-S3")
    Xq = np.array([l.X["q"][0, 0, 1, 0] for l in grid.ladders])
    Xc = np.array([l.X["c"][0, 0, 1, 0] for l in grid.ladders])
    part = (Xc / grid.ladders[-1].nc01) ** 2
    G_q = Xq ** 2 / (Xq[0] ** 2 * T1ge)

    def ratio(T1c):
        G_c = part / T1c
        scale = 1.0 - G_c[0] * T1ge
        T1 = 1.0 / (scale * G_q + G_c)
        return T1[0] / T1[-1]
    return float(brentq(lambda t: ratio(t) - target, 0.3, 300.0)), idle18, on18, target


def setup(pair, pre):
    """The pulse, phase corrections and projectors of predict_cx2_ule_ocelot.run_pair."""
    anc, st = pair.split("-")
    kerr_ratio = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
    projs = {nb: AsymptoticProjector(nb, kerr_ratio=kerr_ratio) for nb in NBARS}
    grid = grid_for(pair)
    L_cx = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}[pair]
    chi_meas = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
    pulse = calibrate_cx2_to_length(grid, L_cx * 1e3 + 0.5 / chi_meas)
    corr = calibrate_phases(grid, pulse, NMAX, 1.0, projs[1.0])
    return grid, pulse, corr, projs


def run_pair(pair, pre, T1c, t37):
    anc = pair.split("-")[0]
    T1ge, T1fe = pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"]
    t0 = time.time()
    grid, pulse, corr, projs = setup(pair, pre)
    on, idle = grid.ladders[-1], grid.ladders[0]
    df_eg = (on.X["q"][0, 0, 1, 0] / idle.X["q"][0, 0, 1, 0]) ** 2   # on-point dressing of the charge elements
    df_fe = (on.X["q"][1, 0, 2, 0] / idle.X["q"][1, 0, 2, 0]) ** 2
    q = charge_bath(grid, T1ge, T1fe)
    baths = {"nobath": BathModel({}), "zerofit": BathModel({"q": q})}
    qc = charge_bath(grid, T1ge, T1fe, coupler_T1_us=T1c)
    baths["coupler18c"] = BathModel({"q": qc, "c": detailed_balance(flat(1.0 / (T1c * 1e3 * on.nc01 ** 2)), None)})
    lams = {}
    if anc == "A4":
        for c in CONDITIONS:
            lam_eg = T1ge / (t37[("e", c)][0] * df_eg)
            lam_fe = T1fe / (t37[("f", c)][0] * df_fe)
            lams[c] = (lam_eg, lam_fe)
            baths[f"ongate_{c}"] = BathModel({"q": charge_bath(grid, T1ge, T1fe, lam=(lam_eg, lam_fe))})
    rows = []
    for nb in NBARS:
        row = dict(pair=pair, nbar=nb, label=LABEL.get(pair, "prior"), coupler_T1_us=T1c, df_eg=df_eg, df_fe=df_fe)
        for c, (le, lf) in lams.items():
            row[f"lam_eg_{c}"], row[f"lam_fe_{c}"] = le, lf
        for name, bath in baths.items():
            for lev, tag in ((2, "f"), (0, "g")):
                blk = simulate_cx2(grid, pulse, bath, initial_blocks(nb, lev, NMAX), NMAX, blocks=True, phase_corr=corr)
                rho = blk.sum(axis=0)
                rho = rho / np.trace(rho).real
                row[f"p{tag}_{name}"] = projs[nb].p_flip(rho)
        rows.append(row)
    print(f"{pair}: on-point dressing e->g {df_eg:.3f} f->e {df_fe:.3f}; "
          + "; ".join(f"{c}: lambda_eg {le:.2f} lambda_fe {lf:.2f}" for c, (le, lf) in lams.items())
          + f" ({time.time() - t0:.0f} s)", flush=True)
    return rows


def compare(rows):
    dat = load_data()
    for r in rows:
        f, g = (dat[(r["pair"], k)].get(r["nbar"], np.nan) for k in ("f", "g"))
        r["data_fexcess"] = f - g
        for name in [k[3:] for k in r if k.startswith("pf_") and k != "pf_nobath"]:
            r[f"ffault_{name}"] = (r[f"pf_{name}"] - r["pf_nobath"]) - (r[f"pg_{name}"] - r["pg_nobath"])
            d = r["data_fexcess"]
            r[f"ratio_{name}"] = d / r[f"ffault_{name}"] if r[f"ffault_{name}"] > 0 and d > 0 else np.nan
    return rows


def save(rows, header):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# " + header + "\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(("" if k not in r else f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]))
                              for k in keys) + "\n")
    print("saved", OUT)


def load():
    out = []
    for r in csv.DictReader(l for l in open(OUT, encoding="utf-8") if not l.startswith("#")):
        out.append({k: (float(v) if v not in ("", None) and k not in ("pair", "label") else v) for k, v in r.items()})
    return out


def plot(rows):
    use_style()
    pairs = [p for p in PAIRS if any(r["pair"] == p for r in rows)]
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, constrained_layout=True)
    for ip, pair in enumerate(pairs):
        sel = [r for r in rows if r["pair"] == pair]
        x = [r["nbar"] for r in sel]
        ax = axs.flat[ip]
        ax.semilogy(x, [r["data_fexcess"] for r in sel], "o", ms=5, mfc="none", color=TEXT_PRIMARY, label="data (Fig. 16)")
        ax.semilogy(x, [r["ffault_zerofit"] for r in sel], "-", color=SERIES[0], label="idle-referenced (zero fit)")
        ax.semilogy(x, [r["ffault_coupler18c"] for r in sel], "--", color=SERIES[2],
                    label=f"coupler loss from Fig. 18c (T1_c = {sel[0]['coupler_T1_us']:.1f} µs)")
        if sel[0].get("pf_ongate_coupler_on") not in (None, ""):
            lo = [r["ffault_ongate_coupler_on"] for r in sel]
            hi = [r["ffault_ongate_pulsed_alpha2_4"] for r in sel]
            ax.fill_between(x, lo, hi, color=SERIES[3], alpha=0.3, lw=0, label="measured on-gate T1 (Fig. 37d): held on … pulsed, |α|² = 4")
        ax.set_ylim(1e-5, 3e-2)
        ax.set_title(f"{pair} ({sel[0]['label']})", fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel("f-excess per cycle (fault-driven)")
    axs.flat[len(pairs) - 1].legend(fontsize=6, loc="lower left")  # the last panel (A4-S5) has every element
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"cx2_measured_bath.{ext}", dpi=160)
    print("saved figures/cx2_measured_bath.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--plot", action="store_true", help="redraw from the CSV")
    a = ap.parse_args()
    if a.plot:
        return plot(load())
    pre = load_preset("ocelot")
    T1c, idle18, on18, target = coupler_T1_from_18c(pre)
    t37 = fig37d_means()
    print(f"Fig. 18c: median T1(e) idle {idle18:.1f} us, on {on18:.1f} us, ratio {target:.2f} -> coupler T1 at 'on' "
          f"{T1c:.2f} us (A3-S3 participation model)")
    print("Fig. 37d means (us): " + ", ".join(f"T1({s}) {c} {m:.1f}+-{sd:.1f}" for (s, c), (m, sd) in t37.items()))
    rows = []
    for pair in a.pairs.split(","):
        rows += run_pair(pair, pre, T1c, t37)
    rows = compare(rows)
    save(rows, f"CX^2 fault-driven f-excess with the decay at 'on' from measured coherence (Fig. 37d for A4; coupler T1 "
               f"{T1c:.3f} us from Fig. 18c for the coupler-loss variant), vs Fig. 16 (f - g). Nothing tuned to Fig. 16. "
               "examples/measured_bath_cx2_ocelot.py")
    print("\npair   label        | data / model, median over |alpha|^2: zero-fit | coupler 18c | on-gate held / pulsed / pulsed+photons")
    for pair in dict.fromkeys(r["pair"] for r in rows):
        sel = [r for r in rows if r["pair"] == pair]
        med = lambda k: np.nanmedian([r.get(k, np.nan) for r in sel])
        extra = " / ".join(f"{med('ratio_ongate_' + c):.2f}" for c in CONDITIONS) if "ratio_ongate_coupler_on" in sel[0] else "—"
        print(f"{pair} {sel[0]['label']:12s} | {med('ratio_zerofit'):.2f} | {med('ratio_coupler18c'):.2f} | {extra}")
    plot(rows)


if __name__ == "__main__":
    main()
