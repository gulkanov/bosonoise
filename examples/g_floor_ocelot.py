"""The g-protocol floor of the Ocelot CX^2 cycles: duty-cycle-corrected reference and ancilla heating.

The static reference of Fig. 16 keeps the two-photon stabilization on for the whole 3-us
cycle, while in the gate protocols it is off during the CX^2 pulse. At |alpha|^2 <= 1.5 the
g protocol has fewer bit flips than the static reference in 5/8 pairs (mean p_g/p_static =
0.81-0.93 for A1-S1, A1-S2, A2-S2, A4-S4, A4-S5, close to the pump duty cycle d = 1 - T_gate/3 us;
1.13-1.48 for A2-S3, A3-S3, A3-S4): part of the static errors is caused by the stabilization
itself. The gate-induced part of the g protocol is estimated as
    floor = p(g) - d p(static),      d = 1 - T_CX2 / 3 us  (padding not published: d is an upper bound),
robust at |alpha|^2 >= 3 where d p(static) is small. We report the floor per pair, test its
|alpha|^2 dependence, and convert it into the ancilla effective temperature that the dressed
ULE model needs for heating during the gate (the model's Kerr term subtracted; thermal
rates from examples/predict_cx2_ule_ocelot.py at 40/50/60 mK, interpolated in exp(-hf/kT)).
The implied temperature is an upper bound (other heating adds). VALIDATION data only.

Usage: python examples/g_floor_ocelot.py
Output: data/validation/g_floor.csv, figures/g_floor.png/.pdf
"""
from __future__ import annotations

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBARS = [2.0, 2.5, 3.0, 3.5, 4.0]


def main():
    dig = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig16.csv", encoding="utf-8")
                              if not l.startswith("#")))
    d = collections.defaultdict(dict)
    for r in dig:
        d[(r["pair"], r["protocol"])][float(r["alpha2"])] = float(r["p_cycle"])
    ule = {(r["pair"], float(r["nbar"])): r for r in csv.DictReader(
        l for l in open(ROOT / "data" / "validation" / "cx2_ule_excess.csv", encoding="utf-8") if not l.startswith("#"))}
    pairs = list(dict.fromkeys(k[0] for k in ule))
    rows = []
    for pair in pairs:
        gate = float(ule[(pair, 2.0)]["gate_ns"])
        duty = 1.0 - gate / 3000.0
        low = [d[(pair, "g")][a] / d[(pair, "static")][a] for a in (1.0, 1.5) if a in d[(pair, "static")]]
        for nb in NBARS:
            pg, ps = d[(pair, "g")].get(nb, np.nan), d[(pair, "static")].get(nb, np.nan)
            floor = pg - duty * ps
            u = ule[(pair, nb)]
            kerr = float(u["pg_ours"])
            th = np.array([float(u[f"pg_T{t}"]) - kerr for t in (40, 50, 60)])
            slope, icpt = np.polyfit(1.0 / np.array([40.0, 50.0, 60.0]), np.log(th), 1)
            need = floor - kerr
            T_imp = slope / (np.log(need) - icpt) if need > 0 else np.nan
            rows.append(dict(pair=pair, nbar=nb, gate_ns=gate, duty=duty, low_nbar_ratio_g_over_static=np.mean(low),
                             data_p_g=pg, data_p_static=ps, floor=floor, model_kerr=kerr, T_eff_implied_mK=T_imp))
    path = ROOT / "data" / "validation" / "g_floor.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# g-protocol floor = p(g) - d p(static), d = 1 - T_CX2/3us; implied ancilla T_eff for heating during the gate "
                 "(model Kerr term subtracted). examples/g_floor_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.5g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("pair  duty  <p_g/p_static>(|a|^2<=1.5) | floor x1e-3 at |a|^2 = 2 .. 4 | slope (3-4) | implied T_eff (mK) at 3 / 3.5 / 4")
    by_anc = collections.defaultdict(list)
    for pair in pairs:
        sel = [r for r in rows if r["pair"] == pair]
        fl = np.array([r["floor"] for r in sel])
        hi = [r for r in sel if r["nbar"] >= 3.0 and np.isfinite(r["floor"])]
        slope = np.polyfit([r["nbar"] for r in hi], [r["floor"] for r in hi], 1)[0] if len(hi) >= 2 else np.nan
        T = [r["T_eff_implied_mK"] for r in hi]
        by_anc[pair.split("-")[0]] += [t for t in T if np.isfinite(t)]
        print(f"{pair} {sel[0]['duty']:.2f}  {sel[0]['low_nbar_ratio_g_over_static']:.2f} | "
              + " ".join(f"{x * 1e3:6.3f}" for x in fl) + f" | {slope * 1e3:+.3f}/photon | "
              + " / ".join(f"{t:.0f}" for t in T))
    for anc, v in by_anc.items():
        print(f"{anc}: implied T_eff median {np.median(v):.0f} mK (range {min(v):.0f}-{max(v):.0f})")
    use_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, pair in enumerate(pairs):
        sel = [r for r in rows if r["pair"] == pair]
        x = [r["nbar"] for r in sel]
        ax[0].plot(x, [r["floor"] * 1e3 for r in sel], "o-", color=SERIES[i % len(SERIES)], label=pair)
        ax[1].plot(x, [(r["data_p_g"] - r["data_p_static"]) * 1e3 for r in sel], "o-", color=SERIES[i % len(SERIES)])
    ax[0].set_title("p(g) − d·p(static): gate-induced floor", fontsize=9)
    ax[1].set_title("p(g) − p(static): apparent growth (reference artefact)", fontsize=9)
    for a in ax:
        a.axhline(0, color="#52514e", lw=0.6)
        a.set_xlabel(r"$|\alpha|^2$")
        a.set_ylabel(r"per cycle ($10^{-3}$)")
    ax[0].legend(fontsize=7, ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"g_floor.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
