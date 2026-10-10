"""Extra ancilla decay from coupler pulsing: Landau-Zener passages through TLS in the swept band (A4-S5).

On every coupler flux pulse the dressed ancilla lines sweep by the Lamb shift at 'on' (A4-S5:
e->g by -43.6 MHz, f->e by -21.5 MHz; ladders of examples/predict_cx2_ule_ocelot.py) and back.
A TLS whose frequency lies inside the swept band is crossed twice per pulse. With coupling V
(to the dressed transition) and sweep rate v = |d omega/dt| at the crossing, the probability to
leave the excitation in the TLS is P = 1 - exp(-2 pi V^2 / v) per crossing (Landau-Zener;
both in angular units). The TLS decoheres between the two crossings (pulse ~640 ns >> 1/gamma),
so the crossings add incoherently. This is a flux-switching channel (paper App. K).

Fig. 37d (VALIDATION ONLY; paper App. K): relative to the time-weighted average of the
held-on and idle rates, pulsing adds ~3.0% (f) and ~1.1% (e) decay probability per pulse
(1.6-us cycles, one CX^2-length pulse each). We compute the distribution of the LZ extra decay
per pulse over literature-prior TLS ensembles (idle-T1 conditioned) with the pulse of the
solved A4-S5 ramp (20.2 ns, 3 sigma). No parameter is tuned.

Usage: python examples/pulse_tls_lz_ocelot.py [--nreal 20000]
Output: data/tls/pulse_lz_summary.json, figures/pulse_tls_lz.png/.pdf
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.cx2 import Pulse
from bosonoise.channels.tls import AncillaBath, sample_tls_literature, surface_mean_rate_per_us
from bosonoise.plotting import SERIES, use_style

from predict_cx2_ule_ocelot import grid_for  # same directory
from tls_ensemble_ocelot import CELLS
from tls_single_map_ocelot import T1, build

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO_PI = 2 * np.pi
TARGET = dict(f=0.030, e=0.011)  # extra decay probability per pulse from Fig. 37d


def sweep_profile(grid, lines, ramp):
    """Line frequency (GHz) versus time along the up-ramp; returns t, freq for each line."""
    p = Pulse(ramp=ramp)
    t = np.linspace(0.0, ramp, 4001)
    fl = p.flux(t)
    out = {}
    for name, (i, j) in lines.items():
        f_grid = np.array([l.E[i, 0] - l.E[j, 0] for l in grid.ladders])
        out[name] = np.interp(fl, grid.fluxes, f_grid)
    return t, out


def lz_extra_decay(tls, t, freq, V_scale):
    """Sum over TLS crossed by the sweep of 2 x (1 - exp(-2 pi V^2 / v))."""
    tot = 0.0
    for fT, g01 in zip(tls.f, tls.g01):
        k = np.nonzero(np.diff(np.sign(freq - fT)))[0]
        for kk in k:
            v = TWO_PI * abs(freq[kk + 1] - freq[kk]) / (t[kk + 1] - t[kk])  # rad/ns^2
            V = TWO_PI * g01 * V_scale  # rad/ns
            tot += 2.0 * (1.0 - np.exp(-TWO_PI * V ** 2 / v))  # up and down ramp
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreal", type=int, default=20000)
    ap.add_argument("--area-um2", nargs="+", type=float, default=[0.05, 0.15])
    ap.add_argument("--ramp", type=float, default=20.2, help="solved A4-S5 ramp (ns)")
    ap.add_argument("--surf-T1-us", type=float, default=100.0)
    ap.add_argument("--idle-tol", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()
    grid = grid_for("A4-S5")
    t, fr = sweep_profile(grid, {"e": (1, 0), "f": (2, 1)}, a.ramp)
    calib, key, anc = CELLS["A4S5"]
    tab_on, tab_idle = build(calib, key)
    scale = max(0.0, 1.0 - surface_mean_rate_per_us(1.0 / a.surf_T1_us) * T1[anc][0])
    bath = AncillaBath(tab_on, tab_idle, *T1[anc], background_scale=scale)
    bg_e, bg_f = scale * 1e-3 / T1[anc][0], scale * 1e-3 / T1[anc][1]
    ti = bath.tab_idle
    Vs = {"e": abs(ti.Mq[ti.select("e->g", 0)[0]]) / bath.n01, "f": abs(ti.Mq[ti.select("f->e", 0)[0]]) / bath.n01}
    print(f"swept bands: e {fr['e'][0]:.4f} -> {fr['e'][-1]:.4f} GHz, f {fr['f'][0]:.4f} -> {fr['f'][-1]:.4f} GHz; "
          f"coupling scale e {Vs['e']:.2f}, f {Vs['f']:.2f} (relative to g01)")
    summary = {"args": vars(a), "target_fig37d": TARGET, "results": {}}
    use_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for ai, area in enumerate(a.area_um2):
        rng = np.random.default_rng(a.seed)
        xe, xf, n_try = [], [], 0
        while len(xe) < a.nreal:
            n_try += 1
            tls = sample_tls_literature(rng, (4.8, 5.4), surf_Gamma1_per_us=1.0 / a.surf_T1_us, junc_area_um2=area)
            ie, i_f = bath.idle_tls_rates(tls)
            if ie + bg_e > a.idle_tol * 1e-3 / T1[anc][0] or i_f + bg_f > a.idle_tol * 1e-3 / T1[anc][1]:
                continue
            xe.append(lz_extra_decay(tls, t, fr["e"], Vs["e"]))
            xf.append(lz_extra_decay(tls, t, fr["f"], Vs["f"]))
        xe, xf = np.array(xe), np.array(xf)
        res = dict(acceptance=len(xe) / n_try,
                   mean_extra_e=float(xe.mean()), mean_extra_f=float(xf.mean()),
                   frac_e_ge_target=float(np.mean(xe >= TARGET["e"])), frac_f_ge_target=float(np.mean(xf >= TARGET["f"])),
                   frac_both=float(np.mean((xe >= TARGET["e"]) & (xf >= TARGET["f"]))),
                   frac_any_crossing_e=float(np.mean(xe > 0)), frac_any_crossing_f=float(np.mean(xf > 0)))
        summary["results"][f"A_J={area:g}"] = res
        print(f"A_J={area:g}: mean extra per pulse e {res['mean_extra_e']:.2e}, f {res['mean_extra_f']:.2e}; "
              f"P(a TLS in band) e {100 * res['frac_any_crossing_e']:.1f}%, f {100 * res['frac_any_crossing_f']:.1f}%; "
              f"P(extra >= Fig. 37d) e {100 * res['frac_e_ge_target']:.2f}%, f {100 * res['frac_f_ge_target']:.2f}%, "
              f"both {100 * res['frac_both']:.2f}%", flush=True)
        bins = np.logspace(-6, 0, 49)
        ax.hist(xf[xf > 0], bins=bins, histtype="step", color=SERIES[ai], label=f"f, A_J = {area:g} µm²")
        ax.hist(xe[xe > 0], bins=bins, histtype="step", ls="--", color=SERIES[ai], label=f"e, A_J = {area:g} µm²")
    ax.axvline(TARGET["f"], color="#52514e", lw=0.8)
    ax.axvline(TARGET["e"], color="#52514e", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("extra decay probability per pulse (Landau–Zener through TLS)")
    ax.set_ylabel("realizations (with a TLS in the band)")
    ax.set_title("A4–S5: TLS crossed by the Lamb-shift sweep (vertical: Fig. 37d)", fontsize=9)
    ax.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"pulse_tls_lz.{ext}", dpi=160)
    with open(ROOT / "data" / "tls" / "pulse_lz_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("saved data/tls/pulse_lz_summary.json")


if __name__ == "__main__":
    main()
