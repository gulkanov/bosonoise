"""Zero-fit prediction of the f-excess of the CX^2 bit-flip probability for all 8 Ocelot pairs.

Observable (model-independent, from vector-extracted Fig. 16; Pauli probability per 3-us
cycle, p = [1 - exp(-2 T_cycle / T_X)] / 2 with T_X the inverse Pauli rate, the p_cycle column of
data/digitized/ocelot_fig16.csv, paper App. C.2):  f-excess = p(ancilla prepared in |f>) - p(ancilla in |g>).

Model (no parameter tuned to error data):
  * per-ancilla joint static calibrations (data/calibration/ocelot_A{k}_fcmin6.569.yaml);
  * dressed transition tables at the 'on' point; flat charge bath per transition family,
    referenced to the measured idle T1_ge, T1_fe of each ancilla (Fig. 9c,d, taken at the time
    of the CX^2 characterization);
  * CX^2 duration T = 1/|chi_gf| with the measured chi_gf (Fig. 12); f->e faults convert with
    eta_fe(nbar, r = chi_ge/chi_gf) (asymptotic projector, uniform fault time); direct f->g and
    double decay f->e->g convert with eta = 1/2 (uniform storage angle);
  * Poisson average over the storage photon number.
"Standard model" (Hann et al. type): same, but without dressing (bare idle rates, no direct
f->g channel).

Usage: python examples/predict_cx2_fexcess_ocelot.py
Output: data/validation/cx2_fexcess.csv, figures/cx2_fexcess_prediction.png/.pdf
Flat-top rate model; superseded by the dressed-ladder ULE simulation (examples/predict_cx2_ule_ocelot.py),
kept as an independent cross-check (paper App. C.1).
"""
from __future__ import annotations

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.model import StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset
from bosonoise.spectra.transitions import dressed_transitions

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAIRS = ["A1-S1", "A1-S2", "A2-S2", "A2-S3", "A3-S3", "A3-S4", "A4-S4", "A4-S5"]
NBARS = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
NMAX = 14


def dressed_rates(pair, pre):
    anc, st = pair.split("-")
    p = load_cell_params(ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml", st)
    e_cut = (NMAX + 3) * p.f_s + 30.0
    m = StorageCouplerAncilla(p, Truncation(n_s=int(e_cut / p.f_s) + 3, n_c=8, n_q=10, ncut=25, e_cut=e_cut))
    tab = dressed_transitions(m, 0.5, nmax=NMAX)
    mi = StorageCouplerAncilla(p, Truncation(n_s=8, n_c=8, n_q=10, ncut=25))
    idle = dressed_transitions(mi, 0.0, nmax=2)
    ref_eg = idle.Mq[idle.select("e->g", 0)[0]] ** 2
    ref_fe = idle.Mq[idle.select("f->e", 0)[0]] ** 2
    T1ge, T1fe = pre["ancilla"][anc]["T1_ge_us"], pre["ancilla"][anc]["T1_ef_us"]
    rates = {}
    for pair_t, ref, T1 in (("f->e", ref_fe, T1fe), ("f->g", ref_fe, T1fe), ("e->g", ref_eg, T1ge)):
        r = np.zeros(NMAX + 1)
        for n in range(NMAX + 1):
            rows = np.nonzero((tab.pair == pair_t) & (tab.n == n) & (tab.freq > 0))[0]
            r[n] = (tab.Mq[rows] ** 2).sum() / ref / (T1 * 1e3)  # 1/ns
        rates[pair_t] = r
    return rates, T1ge, T1fe


def main():
    pre = load_preset("ocelot")
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig16.csv", encoding="utf-8")
                               if not l.startswith("#")))
    dat = collections.defaultdict(dict)
    for r in rows:
        dat[(r["pair"], r["protocol"])][float(r["alpha2"])] = float(r["p_cycle"])
    proj = {nb: AsymptoticProjector(nb) for nb in NBARS}
    out = []
    use_style()
    fig, axs = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, constrained_layout=True)
    for ip, pair in enumerate(PAIRS):
        anc, st = pair.split("-")
        it = pre["interactions"][f"{anc}_{st}"]
        chi_ge, chi_gf = abs(it["chi_ge_MHz"]), abs(it["chi_gf_MHz"])
        T = 1e3 / chi_gf  # ns, CX^2 = full 2pi rotation for |f>
        r = chi_ge / chi_gf
        rates, T1ge, T1fe = dressed_rates(pair, pre)
        n = np.arange(NMAX + 1)
        for nb in NBARS:
            w = poisson.pmf(n, nb)
            w = w / w.sum()
            P_fe = w @ (rates["f->e"] * T)
            P_fg = w @ (rates["f->g"] * T)
            P_dd = w @ (rates["f->e"] * rates["e->g"] * T ** 2 / 2)
            eta_fe = proj[nb].eta_uniform(2 * np.pi * (r - 1))
            ours = 0.5 * (P_fg + P_dd) + eta_fe * P_fe
            # standard model: bare idle rates, no dressing channel
            Pfe0, Pdd0 = T / (T1fe * 1e3), T ** 2 / (2 * T1fe * T1ge * 1e6)
            std = 0.5 * Pdd0 + eta_fe * Pfe0
            data = dat[(pair, "f")].get(nb, np.nan) - dat[(pair, "g")].get(nb, np.nan)
            out.append(dict(pair=pair, nbar=nb, data_fexcess=data, ours=ours, standard=std, direct=0.5 * P_fg,
                            double=0.5 * P_dd, mismatch=eta_fe * P_fe, eta_fe=eta_fe, T_ns=T, r=r))
        sel = [o for o in out if o["pair"] == pair]
        ax = axs.flat[ip]
        ax.semilogy(NBARS, [o["data_fexcess"] for o in sel], "o", ms=5, mfc="none", color=TEXT_PRIMARY, label="data (Fig. 16)")
        ax.semilogy(NBARS, [o["ours"] for o in sel], "-", color=SERIES[0], label="this work (dressed, no fit)")
        ax.semilogy(NBARS, [o["standard"] for o in sel], "--", color=SERIES[1], label="standard model")
        ax.semilogy(NBARS, [o["direct"] for o in sel], ":", color=SERIES[2], label="direct f→g part")
        ax.set_title(f"{pair} (χ ratio {r:.2f})", fontsize=9)
        ax.set_ylim(1e-5, 1e-2)
        print(f"{pair}: T={T:.0f} ns r={r:.3f} | " + "  ".join(
            f"nbar={o['nbar']:g}: data {o['data_fexcess'] * 1e3:.2f} ours {o['ours'] * 1e3:.2f} std {o['standard'] * 1e3:.2f} (direct {o['direct'] * 1e3:.2f})"
            for o in sel if o["nbar"] in (2.0, 3.0, 4.0)))
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel("f-excess per cycle (Pauli)")
    axs.flat[0].legend(fontsize=6, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"cx2_fexcess_prediction.{ext}", dpi=160)
    vd = ROOT / "data" / "validation"
    vd.mkdir(parents=True, exist_ok=True)
    with open(vd / "cx2_fexcess.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# zero-fit prediction of p_f - p_g per CX^2 cycle (Pauli), vs vector-extracted Fig. 16\n")
        keys = list(out[0])
        fh.write(",".join(keys) + "\n")
        for o in out:
            fh.write(",".join(f"{o[k]:.5g}" if isinstance(o[k], float) else str(o[k]) for k in keys) + "\n")
    print("saved", vd / "cx2_fexcess.csv")


if __name__ == "__main__":
    main()
