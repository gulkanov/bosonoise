"""Single-TLS response map: which TLS (frequency, coupling) produces an F9-like direct f->g peak?

For one TLS at frequency f_T with coupling g01 (to the bare ancilla 0-1 charge element) and
linewidth gamma, compute the F9 observable of Ocelot Fig. 18b -- Delta P_g after an 800 ns
pulse at the 'on' point versus storage mean photon number |alpha|^2 -- in the dressed A4-S4
model (static calibration, background from the measured idle T1 of A4). The map shows, as a
function of f_T, the peak height max Delta P_g over |alpha|^2 in [0, 20] and its position.
This is a response function of the model, NOT a fit: no parameter is tuned to F9.

Usage: python examples/tls_single_map_ocelot.py [--cell S4] [--g01-MHz 0.5] [--T1tls-us 1.0]
Output: figures/tls_single_map_A4<cell>_g<g01>MHz_T1tls<T1>us.png/.pdf, data/tls/single_map_A4<cell>_g<g01>MHz_T1tls<T1>us.csv
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.tls import AncillaBath, TLSEnsemble, poisson_average
from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, use_style
from bosonoise.spectra.transitions import dressed_transitions

ROOT = pathlib.Path(__file__).resolve().parents[1]
T1 = {"A4": (41.0, 25.0), "A2": (25.0, 24.0)}  # (T1_ge, T1_fe) us, Ocelot Fig. 9c,d (coupler idle)


def build(calib, cell, nmax=40, e_cut=250.0):
    p = load_cell_params(ROOT / calib, cell)
    m = StorageCouplerAncilla(p, Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=8, n_q=12, ncut=25, e_cut=e_cut))
    t0 = time.time()
    tab_on = dressed_transitions(m, 0.5, nmax=nmax)
    m_idle = StorageCouplerAncilla(p, Truncation(n_s=10, n_c=8, n_q=10, ncut=25))
    tab_idle = dressed_transitions(m_idle, 0.0, nmax=3)
    print(f"  transition tables: dim {m.basis.dim}, {time.time() - t0:.0f} s")
    return tab_on, tab_idle


def f9_curve(bath, tls, nbar, t=800.0):
    pops = bath.f_state_populations(tls, t)  # (n, 3): P_g, P_e, P_f
    avg, tail = poisson_average(pops, nbar)
    return avg, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="data/calibration/ocelot_A4_fcmin6.569.yaml")
    ap.add_argument("--cell", default="S4")
    ap.add_argument("--anc", default="A4")
    ap.add_argument("--g01-MHz", type=float, default=0.5)
    ap.add_argument("--T1tls-us", type=float, default=1.0, help="TLS decay time; gamma = 1/T1tls (FWHM, angular)")
    ap.add_argument("--fmin", type=float, default=4.95)
    ap.add_argument("--fmax", type=float, default=5.35)
    ap.add_argument("--df-MHz", type=float, default=0.2)
    a = ap.parse_args()
    label = f"{a.anc}{a.cell}"
    print(f"cell {label}")
    tab_on, tab_idle = build(a.calib, a.cell)
    bath = AncillaBath(tab_on, tab_idle, *T1[a.anc])
    nbar = np.linspace(0, 20, 101)
    none = TLSEnsemble(np.array([]), np.array([]), np.array([]))
    base, tail = f9_curve(bath, none, nbar)
    print(f"  no-TLS background: dP_g(10) = {base[50, 0] - base[0, 0]:.2e}; Poisson tail at 20: {tail[-1]:.1e}")
    fT = np.arange(a.fmin, a.fmax, a.df_MHz * 1e-3)
    g01, gam = a.g01_MHz * 1e-3, 1e-3 / a.T1tls_us
    peak, where, peak_e = np.zeros(len(fT)), np.zeros(len(fT)), np.zeros(len(fT))
    t0 = time.time()
    for i, f in enumerate(fT):
        tls = TLSEnsemble(np.array([f]), np.array([g01]), np.array([gam]))
        avg, _ = f9_curve(bath, tls, nbar)
        dPg = avg[:, 0] - avg[0, 0]
        dPe = avg[:, 1] - avg[0, 1]
        k = int(np.argmax(dPg))
        peak[i], where[i], peak_e[i] = dPg[k], nbar[k], dPe.max()
    print(f"  scanned {len(fT)} TLS frequencies in {time.time() - t0:.0f} s")
    # channel bands swept by n = 0..20 at the 'on' point
    bands = {}
    for pair in ("f->g", "f->e", "e->g"):
        d = tab_on.dominant(pair)
        sel = d["n"] <= 20
        bands[pair] = (d["freq"][sel].min(), d["freq"][sel].max())
        print(f"  band {pair}: {bands[pair][0]:.4f} - {bands[pair][1]:.4f} GHz (dominant channel, n=0..20)")
    out = ROOT / "data" / "tls"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"A4{a.cell}_g{a.g01_MHz:g}MHz_T1tls{a.T1tls_us:g}us" if a.anc == "A4" else f"{label}_g{a.g01_MHz:g}MHz_T1tls{a.T1tls_us:g}us"
    with open(out / f"single_map_{tag}.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# single-TLS response map, cell {label}, g01={a.g01_MHz} MHz, T1_TLS={a.T1tls_us} us, 800 ns at 'on'\n")
        fh.write("f_TLS_GHz,max_dPg,nbar_at_max,max_dPe\n")
        for row in zip(fT, peak, where, peak_e):
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
    use_style()
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, constrained_layout=True)
    ax[0].semilogy(fT, np.maximum(peak, 1e-6), color=SERIES[0], label=r"max $\Delta P_g$ (direct $f\to g$ + $f\to e\to g$)")
    ax[0].semilogy(fT, np.maximum(peak_e, 1e-6), color=SERIES[1], label=r"max $\Delta P_e$")
    ax[0].axhline(0.097, color="#52514e", lw=0.8, ls="--")
    ax[0].text(fT[2], 0.11, "F9 peak height (Ocelot Fig. 18b)", fontsize=7, color="#52514e")
    ax[0].set_ylabel("peak population change")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].plot(fT, where, ".", ms=2, color=SERIES[0])
    ax[1].set_ylabel(r"$|\alpha|^2$ at the peak")
    ax[1].set_xlabel("TLS frequency (GHz)")
    for x in ax:
        for pair, c in zip(bands, SERIES[2:5]):
            x.axvspan(*bands[pair], color=c, alpha=0.12, lw=0)
    ax[0].set_title(f"Single TLS ({a.g01_MHz:g} MHz, T1={a.T1tls_us:g} µs), cell {label}, 800 ns at 'on' "
                    f"(bands: f→g+photon, f→e, e→g for n≤20)", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"tls_single_map_{tag}.{ext}", dpi=160)
    i = int(np.argmax(peak))
    print(f"  largest peak {peak[i]:.3f} at f_TLS={fT[i]:.4f} GHz, |alpha|^2={where[i]:.1f}")
    for thr in (0.05, 0.02):
        w = (peak >= thr).sum() * a.df_MHz
        print(f"  frequency width where max dP_g >= {thr}: {w:.1f} MHz")


if __name__ == "__main__":
    main()
