"""Pulsed stabilization between gates (paper App. J): the stabilization window of the Ocelot CX^2 experiments (Fig. 16).

Per storage, zero-fit inputs only (no error data):
  kappa2 = 4 g2^2 / kappa_b (Fig. 13), kappa1 = 1/T1 and kphi = 2/T2 - 1/T1 under two-photon dissipation (Fig. 9g,h),
  kerr_ratio = -Delta_b,0 / kappa_b (App. F8, D10), storage self-Kerr under the pump K_s/2pi = 1 kHz (AWS model:
  |K_s| < 2 kHz; prior), thermal population nth = 0.5% (prior; the result is insensitive to it).
Protocols (3-us cycle, as in Fig. 16):
  static  stabilization on all cycle;
  g       the CX^2 flux pulse with the ancilla in g: stabilization off for the gate (duty d = 1 - T_CX2 / 3 us; the
          padding is not published, so this is the largest duty), the storage evolves with the g-branch dressed-ladder
          phases along the pulse (Kerr shear; the linear part and the calibrated correction delta_g as in the CX model)
          and the idle dissipation; then the stabilization window.
  g, pump-off coherence  the same with the storage's own T1, T2 WITHOUT two-photon dissipation (Fig. 9e,f) during the
          gate: the pump degrades both (Carde et al., PR Applied 23, 024073 (2025)), so the gate period is quieter.
The per-cycle Pauli bit-flip probability is the leading eigenvalue of the odd-sector cycle map (channels/window.py): it
includes relaxation left over from one window to the next, which the asymptotic projector after each gate (our CX
model) assumes complete. Diagnostic decomposition: window model vs 'projector' model (static x duty + the gate's
deterministic part from the CX model). Compared with Fig. 16 static and g (VALIDATION ONLY).

Usage: python examples/window_ocelot.py [--duty]
Output: data/validation/window_pulsed.csv, figures/window_pulsed.png/.pdf; --duty: data/validation/window_duty.csv
"""
from __future__ import annotations

import collections
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.channels.cx2 import BathModel, calibrate_cx2_to_length, calibrate_phases, initial_blocks, simulate_cx2
from bosonoise.channels.window import CatWindow
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, PAIRS, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_CYCLE = 3.0  # us
NL = 40
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
K_S_KHZ, NTH = 1.0, 0.005


def g_branch_phases(grid, pulse, delta_g: float, nmax: int, steps: int = 400):
    """Phase vector phi(n) of the g branch over the pulse (frame of the calibrated CX model) and the step list."""
    h = pulse.duration / steps
    n = np.arange(nmax + 1)
    phi = np.zeros(nmax + 1)
    incr = []
    for s in range(steps):
        E = grid.E(float(pulse.flux((s + 0.5) * h)))
        hk = E[0, :nmax + 1] - E[0, 0] - (E[0, 1] - E[0, 0]) * n
        d = 2 * np.pi * hk * h
        incr.append(d)
        phi += d
    return phi - delta_g * n, incr, h  # apply_phase(delta) == phase vector -delta * n in CatWindow.phases


def gate_map(w, incr, delta_g: float, T_gate: float, sub: int = 40):
    """Odd-sector map of the gate: off-period idle dissipation interleaved with the phase increments (sub steps),
    then the calibrated correction delta_g."""
    per = len(incr) // sub
    E_idle = w.off(T_gate / sub)
    M = np.eye(len(w.idx), dtype=complex)
    for k in range(sub):
        M = w.phases(np.sum(incr[k * per:(k + 1) * per], axis=0)) @ (E_idle @ M)
    return w.phases(-delta_g * np.arange(w.N)) @ M


def duty_scan(pre, st="S1"):
    """Pulsed stabilization without gates (the protocol of the companion paper, arXiv:2409.17556 Sec. VII, Fig. 7: a
    different device, so a qualitative check only): T_on of a 3-us cycle on, the rest off with the pump-off coherence
    and the storage Kerr. Phase tracking before the stabilization returns (CatWindow.pulsed_bitflip): 'mean' (rotate back
    the mean Kerr rotation K nbar T_off: aligned with the coherent state), 'none', 'optimal' (a control choice)."""
    s = pre["storage"][st]
    T1, T2, T1b, T2b = s["T1_under_2ph_diss_us"], s["T2_under_2ph_diss_us"], s["T1_us"], s["T2_us"]
    rows = []
    for K_kHz in (1.0, 5.0):
        K = 2 * np.pi * K_kHz * 1e-3
        for nb in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0):
            w = CatWindow(nb, kappa2=2 * np.pi * pre["two_photon_dissipation"][st]["kappa2_kHz"] * 1e-3,
                          kerr_ratio=-DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"], kappa1=1 / T1, nth=NTH,
                          kphi=max(2 / T2 - 1 / T1, 0.0), K=K, kappa1_off=1 / T1b, kphi_off=max(2 / T2b - 1 / T1b, 0.0))
            p_static = w.bitflip_per_cycle(w.on(T_CYCLE))
            rows.append(dict(cell=st, K_kHz=K_kHz, T_on_us=T_CYCLE, tracking="-", tracking_f=np.nan, nbar=nb,
                             p_cycle=p_static, T_X_ms=-2 * T_CYCLE / np.log(1 - 2 * p_static) * 1e-3))
            for T_on in (1.0, 0.2):
                for track in ("mean", "optimal", "none"):
                    p, f = w.pulsed_bitflip(T_on, T_CYCLE - T_on, K * nb * (T_CYCLE - T_on), track)
                    rows.append(dict(cell=st, K_kHz=K_kHz, T_on_us=T_on, tracking=track, tracking_f=f, nbar=nb, p_cycle=p,
                                     T_X_ms=-2 * T_CYCLE / np.log(1 - 2 * p) * 1e-3))
            sel = {(r["T_on_us"], r["tracking"]): r["T_X_ms"] for r in rows if r["K_kHz"] == K_kHz and r["nbar"] == nb}
            st_ = sel[(T_CYCLE, "-")]
            print(f"{st} K {K_kHz:.0f} kHz nbar {nb:.0f} | static {st_:.3g} ms | T_X/static, mean/optimal/none: 1 us "
                  + "/".join(f"{sel[(1.0, t)] / st_:.2f}" for t in ("mean", "optimal", "none")) + ", 200 ns "
                  + "/".join(f"{sel[(0.2, t)] / st_:.2f}" for t in ("mean", "optimal", "none")), flush=True)
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / "window_duty.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Pulsed stabilization without gates (3-us cycle, T_on on, pump-off coherence while off; Kerr tracking "
                 "mean/optimal/none): Pauli bit flip per cycle and T_X (AWS convention). Model only, Ocelot storage "
                 f"{st}. examples/window_ocelot.py --duty\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("saved", path)


def main():
    pre = load_preset("ocelot")
    if "--duty" in sys.argv:
        duty_scan(pre)
        return
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    dat = collections.defaultdict(dict)
    for r in csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig16.csv", encoding="utf-8")
                            if not l.startswith("#")):
        dat[(r["pair"], r["protocol"])][float(r["alpha2"])] = float(r["p_cycle"])
    rows = []
    for pair in PAIRS:
        anc, st = pair.split("-")
        grid = grid_for(pair, NL)
        chi = abs(pre["interactions"][f"{anc}_{st}"]["chi_gf_MHz"]) * 1e-3
        pulse = calibrate_cx2_to_length(grid, L[pair] * 1e3 + 0.5 / chi)
        kb = pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
        kr = -DELTA_B_MHZ / kb
        corr = calibrate_phases(grid, pulse, 38, 1.0, AsymptoticProjector(1.0, kerr_ratio=kr))
        s = pre["storage"][st]
        T1, T2 = s["T1_under_2ph_diss_us"], s["T2_under_2ph_diss_us"]
        T1b, T2b = s["T1_us"], s["T2_us"]  # pump off (Fig. 9e,f)
        k2 = 2 * np.pi * pre["two_photon_dissipation"][st]["kappa2_kHz"] * 1e-3  # rad/us
        pars = dict(kappa2=k2, kerr_ratio=kr, kappa1=1 / T1, nth=NTH, kphi=max(2 / T2 - 1 / T1, 0.0),
                    K=2 * np.pi * K_S_KHZ * 1e-3)
        T_gate = pulse.duration * 1e-3
        for nb in NBARS:
            w = CatWindow(nb, **pars)
            wb = CatWindow(nb, **pars, kappa1_off=1 / T1b, kphi_off=max(2 / T2b - 1 / T1b, 0.0))
            phi, incr, h = g_branch_phases(grid, pulse, corr[0], w.N - 1)
            p_g = w.bitflip_per_cycle(w.on(T_CYCLE - T_gate) @ gate_map(w, incr, corr[0], T_gate))
            p_g_bare = wb.bitflip_per_cycle(wb.on(T_CYCLE - T_gate) @ gate_map(wb, incr, corr[0], T_gate))
            p_static = w.bitflip_per_cycle(w.on(T_CYCLE))
            # projector model: gate deterministic part (CX model, no bath) + static errors for the pumped fraction
            sg = simulate_cx2(grid, pulse, BathModel({}), initial_blocks(nb, 0, 38), 38, phase_corr=corr)
            p_det = AsymptoticProjector(nb, kerr_ratio=kr).p_flip(sg)
            duty = 1 - T_gate / T_CYCLE
            rows.append(dict(pair=pair, nbar=nb, gate_ns=pulse.duration, duty=duty, model_static=p_static, model_g=p_g,
                             model_g_bare_off=p_g_bare, model_projector_g=duty * p_static + p_det, model_gate_deterministic=p_det,
                             data_static=dat[(pair, "static")].get(nb, np.nan), data_g=dat[(pair, "g")].get(nb, np.nan)))
        print(f"{pair}: done", flush=True)
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / "window_pulsed.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Stabilization windows, CX^2 experiment (3-us cycle): Pauli bit flip per cycle, window model (odd-sector "
                 "cycle map; model_g_bare_off: pump-off T1, T2 of Fig. 9e,f during the gate) vs projector model; "
                 "data_* = Fig. 16 (VALIDATION ONLY). examples/window_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("\npair nbar | static: model data | g: window model, bare off, projector model, data   [x1e-3 per 3-us cycle]")
    for r in rows:
        print(f"{r['pair']} {r['nbar']:.1f} | {r['model_static'] * 1e3:7.3f} {r['data_static'] * 1e3:7.3f} | "
              f"{r['model_g'] * 1e3:7.3f} {r['model_g_bare_off'] * 1e3:7.3f} {r['model_projector_g'] * 1e3:7.3f} "
              f"{r['data_g'] * 1e3:7.3f}")
    use_style()
    fig, axs = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True, constrained_layout=True)
    for ax, pair in zip(axs.flat, PAIRS):
        sel = [r for r in rows if r["pair"] == pair]
        x = [r["nbar"] for r in sel]
        ax.semilogy(x, [r["data_static"] for r in sel], "o", mfc="none", color=TEXT_PRIMARY, label="data static")
        ax.semilogy(x, [r["model_static"] for r in sel], "-", color=TEXT_PRIMARY, label="model static")
        ax.semilogy(x, [r["data_g"] for r in sel], "s", mfc="none", color=SERIES[1], label="data g protocol")
        ax.semilogy(x, [r["model_g"] for r in sel], "-", color=SERIES[1], label="window model (g)")
        ax.semilogy(x, [r["model_g_bare_off"] for r in sel], "--", color=SERIES[1], label="window, pump-off T1/T2 in gate")
        ax.semilogy(x, [r["model_projector_g"] for r in sel], ":", color=SERIES[1], label="projector model (g)")
        ax.set_title(pair, fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel("bit flip per 3-us cycle")
    axs.flat[0].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"window_pulsed.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
