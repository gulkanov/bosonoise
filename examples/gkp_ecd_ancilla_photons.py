"""GKP cell with ECD gates: the cavity's own photons acting on the transmon ancilla (paper App. G).

During an echoed conditional displacement the cavity is displaced to nbar = |alpha|^2 photons. In the displaced frame
the transmon sees a classical charge drive eps = 2 g sqrt(nbar) at the cavity frequency (dynamics/floquet.py; the
Stark shift agrees with the exact transmon (x) cavity model to 0.06% up to 256 photons, tests/test_floquet.py).

For each device the Floquet branches of |g>, |e>, |f> are followed from nbar = 0. A charge bath is referenced to the
measured bare T1 (flat in frequency, or Ohmic as a sensitivity case) with detailed balance at the measured thermal
population. From it come the decay of |e>, the heating and leakage out of |g>, and the e -> f leakage. No parameter is
tuned to the photon-number data.

Devices (paper App. G):
  A  Sivak et al., Nature 616, 50 (2023). Our charge-basis fit to the published dispersive data: E_J 23.05 GHz,
     E_C 203.9 MHz, g 11.70 MHz; cavity 4.479 GHz, T1 280 us, n_th 4.3%. The big ECD runs at 121-256 photons; the DAC limit is 676.
  B  Eickbusch et al., Nature Phys. 18, 1464 (2022). Printed E_J 32.33 GHz, E_C 181 MHz, g 9.12 MHz; cavity 5.26 GHz,
     T1 50 us, n_th 0.92%. Validation data:
     - T1 = 30 us at 900 photons (averaged over days, with TLS features);
     - no heating of |g> up to 2000 photons;
     - |e> leaves {g, e} above ~2500 photons and |g> above ~7500.
Offset charge n_g = 0, 0.25, 0.5 (not published).

Usage: python examples/gkp_ecd_ancilla_photons.py [--quick] [--plot]
Output: data/validation/gkp_ecd_ancilla.csv, figures/gkp_ecd_ancilla.png/.pdf
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

from bosonoise.dynamics.floquet import branch_rates, track_branches
from bosonoise.dynamics.ule import H_OVER_KB_MK, detailed_balance, flat, power_law
from bosonoise.model.transmon import Transmon
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "gkp_ecd_ancilla.csv"
DEVICES = {
    "A_SI23": dict(EJ=23.05, EC=0.2039, g=0.01170, fc=4.479, T1_us=280.0, nth=0.043, nbar_max=1200.0,
                   window=(121.0, 256.0), dac=676.0),
    "B_EI22": dict(EJ=32.33, EC=0.181, g=0.00912, fc=5.26, T1_us=50.0, nth=0.0092, nbar_max=10000.0,
                   window=(900.0, 900.0), dac=None),
}
NGS = (0.0, 0.25, 0.5)
NLEV, K, KMAX = 30, 16, 10


def bath(T1_us, nth, f01, n01, kind):
    T_mK = f01 * H_OVER_KB_MK / np.log((1 - nth) / nth)
    gamma0 = 1.0 / (T1_us * 1e3 * n01 ** 2)  # 1/ns, so that the bare e -> g rate is 1/T1
    emission = flat(gamma0) if kind == "flat" else power_law(gamma0, f01, 1.0)
    return detailed_balance(emission, T_mK), T_mK


def run(name, dev, ng, steps):
    t0 = time.time()
    tr = Transmon(EJ=dev["EJ"], EC=dev["EC"], ng=ng, ncut=45)
    E, n = tr.spectrum(NLEV)
    f01 = E[1] - E[0]
    eps_grid = 2 * dev["g"] * np.sqrt(np.linspace(0.0, np.sqrt(dev["nbar_max"]), steps) ** 2)
    fb = track_branches(E, n, dev["fc"], eps_grid, K, states=(0, 1, 2))
    baths = {kind: bath(dev["T1_us"], dev["nth"], f01, abs(n[0, 1]), kind)[0] for kind in ("flat", "ohmic")}
    rows = []
    for s, eps in enumerate(eps_grid):
        row = dict(device=name, ng=ng, nbar=(eps / (2 * dev["g"])) ** 2, eps_GHz=eps,
                   w_g=fb.bare_weight[s, 0], w_e=fb.bare_weight[s, 1], w_f=fb.bare_weight[s, 2],
                   stark_ge_MHz=1e3 * ((fb.q[s, 1] - fb.q[s, 0]) - f01))
        for kind, gam in baths.items():
            r = branch_rates(fb, s, n, dev["fc"], gam, KMAX)
            e_out = sum(v for (a, b), v in r.items() if a == 1)
            g_out = sum(v for (a, b), v in r.items() if a == 0)
            row[f"T1e_us_{kind}"] = 1e-3 / e_out
            row[f"e_to_g_per_us_{kind}"] = 1e3 * r[(1, ("branch", 0))]
            row[f"e_to_f_per_us_{kind}"] = 1e3 * r[(1, ("branch", 2))]
            row[f"g_out_per_us_{kind}"] = 1e3 * g_out
            row[f"g_to_e_per_us_{kind}"] = 1e3 * r[(0, ("branch", 1))]
            row[f"e_leak_per_us_{kind}"] = 1e3 * (e_out - r[(1, ("branch", 0))])
            # the same device with the f level as a computational state (as the chi-matched (g, f) ancilla of Ocelot):
            # f -> g is closed for charge noise without photons and opened by dressing with one photon (total parity)
            row[f"f_to_g_per_us_{kind}"] = 1e3 * r[(2, ("branch", 0))]
            row[f"f_to_e_per_us_{kind}"] = 1e3 * r[(2, ("branch", 1))]
        rows.append(row)
    print(f"{name} n_g = {ng}: {steps} steps, levels {NLEV}, K {K} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def threshold(rows, key, level=0.5):
    """First nbar at which the bare weight of a branch drops below `level`."""
    for r in rows:
        if r[key] < level:
            return r["nbar"]
    return np.nan


def summarize(rows):
    for name, dev in DEVICES.items():
        for ng in NGS:
            sel = [r for r in rows if r["device"] == name and r["ng"] == ng]
            if not sel:
                continue
            tg, te = threshold(sel, "w_g"), threshold(sel, "w_e")
            t1_0 = sel[0]["T1e_us_flat"]
            at = lambda x, k: float(np.interp(x, [r["nbar"] for r in sel], [r[k] for r in sel]))
            msg = (f"{name} n_g {ng}: branch weight < 0.5 for |e> at nbar {te:.0f}, |g> at {tg:.0f}; "
                   f"T1(e) bare {t1_0:.1f} us")
            for x in sorted({dev["window"][0], dev["window"][1], dev["dac"] or dev["window"][1]}):
                msg += (f"; at {x:.0f}: T1(e) {at(x, 'T1e_us_flat'):.1f} (flat) / {at(x, 'T1e_us_ohmic'):.1f} (ohmic) us, "
                        f"heating g {at(x, 'g_out_per_us_flat'):.2e}/us, e->f {at(x, 'e_to_f_per_us_flat'):.2e}/us, "
                        f"f->g {at(x, 'f_to_g_per_us_flat'):.2e}/us (f->e {at(x, 'f_to_e_per_us_flat'):.2e})")
            if name == "B_EI22":
                msg += f"; heating g at 2000: {at(2000, 'g_out_per_us_flat'):.2e}/us"
            print(msg)


def save(rows):
    keys = list(rows[0])
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Floquet branches and Floquet-Markov rates of the transmon ancilla driven by the displaced cavity "
                 "(eps = 2 g sqrt(nbar)); charge bath referenced to the bare T1. examples/gkp_ecd_ancilla_photons.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    print("saved", OUT)


def load():
    rows = []
    for r in csv.DictReader(l for l in open(OUT, encoding="utf-8") if not l.startswith("#")):
        rows.append({k: (v if k == "device" else float(v)) for k, v in r.items()})
    return rows


def plot(rows):
    use_style()
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row_i, (name, dev) in enumerate(DEVICES.items()):
        ax_w, ax_t, ax_r = axs[row_i]
        for j, ng in enumerate(NGS):
            sel = [r for r in rows if r["device"] == name and r["ng"] == ng]
            if not sel:
                continue
            x = np.array([r["nbar"] for r in sel])
            ls = ["-", "--", ":"][j]
            ax_w.plot(x, [r["w_g"] for r in sel], ls, color=SERIES[0], label=f"|g⟩ branch, n_g = {ng}")
            ax_w.plot(x, [r["w_e"] for r in sel], ls, color=SERIES[1], label=f"|e⟩ branch, n_g = {ng}")
            t1_0 = sel[0]["T1e_us_flat"]
            ax_t.plot(x, [r["T1e_us_flat"] / t1_0 for r in sel], ls, color=SERIES[0], label=f"flat bath, n_g = {ng}")
            ax_t.plot(x, [r["T1e_us_ohmic"] / sel[0]["T1e_us_ohmic"] for r in sel], ls, color=SERIES[2],
                      label=f"Ohmic bath, n_g = {ng}")
            ax_r.semilogy(x, np.maximum([r["g_out_per_us_flat"] for r in sel], 1e-12), ls, color=SERIES[3],
                          label=f"heating/leakage out of |g⟩, n_g = {ng}")
            ax_r.semilogy(x, np.maximum([r["e_to_f_per_us_flat"] for r in sel], 1e-12), ls, color=SERIES[4],
                          label=f"e → f, n_g = {ng}")
        lo, hi = dev["window"]
        for ax in (ax_w, ax_t, ax_r):
            if hi > lo:
                ax.axvspan(lo, hi, color="0.85", zorder=0)
            if dev["dac"]:
                ax.axvline(dev["dac"], color="0.5", lw=0.8, ls=":")
            ax.set_xlabel(r"cavity photons $\bar n=|\alpha|^2$")
        if name == "B_EI22":
            ax_w.axvline(2500, color=SERIES[1], lw=0.8, ls="-.")
            ax_w.axvline(7500, color=SERIES[0], lw=0.8, ls="-.")
            ax_t.plot([900], [30 / 50], "o", ms=7, mfc="none", color=TEXT_PRIMARY, label="measured 30/50 µs (EI22)")
            ax_r.axvspan(0, 2000, color="0.92", zorder=0)
        ax_w.set_ylabel("population of the bare state")
        ax_t.set_ylabel(r"$T_1(e)$ / bare")
        ax_r.set_ylabel(r"rate (1/µs)")
        ax_w.set_title(f"{name}: Floquet branches", fontsize=9)
        ax_t.set_title(f"{name}: decay of |e⟩ (bath referenced to bare T1)", fontsize=9)
        ax_r.set_title(f"{name}: heating and leakage", fontsize=9)
        for ax in (ax_w, ax_t, ax_r):
            ax.legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"gkp_ecd_ancilla.{ext}", dpi=160)
    print("saved figures/gkp_ecd_ancilla.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarse grid, n_g = 0 only (smoke test; writes to runs/)")
    ap.add_argument("--plot", action="store_true", help="redraw from the CSV")
    a = ap.parse_args()
    if a.plot:
        return plot(load())
    rows = []
    for name, dev in DEVICES.items():
        for ng in (NGS[:1] if a.quick else NGS):
            rows += run(name, dev, ng, steps=60 if a.quick else 600)
    summarize(rows)
    if a.quick:
        return
    save(rows)
    plot(rows)


if __name__ == "__main__":
    main()
