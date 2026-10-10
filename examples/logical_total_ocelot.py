"""Overall logical error per cycle of the Ocelot repetition cat code, eps_L = (eps_bit + eps_phase)/2 (AWS Fig. 5a
definition), model against the raw data, with the optimal photon number.

Data (VALIDATION ONLY; Zenodo 10.5281/zenodo.14257632, Putterman et al., CC BY 4.0): eps_bit from our fits of the
bit-flip runs (zenodo_logical_bitflip.csv), eps_phase from our decoding of the phase-flip runs with erasure information
(zenodo_phase_logical.csv).
Model:
  zero-fit   eps_bit = stabilizer gates (zero-fit ULE fault tables) + idle from the window model (static, pumped rates,
             window_pulsed.csv, rescaled to the 2.8-us cycle); eps_phase = Stim model, zero-fit inputs
             (phase_logical_model.csv).
  lambda3    gates with the on-gate ancilla decay enhanced by lambda = 3 (derived from CX^2, logical and erasure data,
             sections 6.14-6.15) + the data-derived idle band (logical_bitflip.csv); eps_phase = Stim model, lambda3.
  data-anchored phase with the lambda3 bit part (the phase part is a check of the model structure).

Usage: python examples/logical_total_ocelot.py
Output: data/validation/logical_total.csv, figures/logical_total.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORAGE_PAIR = {"S1": "A1-S1", "S2": "A1-S2", "S3": "A2-S3", "S4": "A3-S4", "S5": "A4-S5"}
SECTION_CATS = {"d3_S1-S3": ("S1", "S2", "S3"), "d3_S3-S5": ("S3", "S4", "S5"), "d5_S1-S5": ("S1", "S2", "S3", "S4", "S5")}


def rows(name):
    return list(csv.DictReader(l for l in open(ROOT / "data" / "validation" / name, encoding="utf-8") if not l.startswith("#")))


def main():
    bit_data = {(r["section"], float(r["nbar"])): float(r["data_eps_exact"]) for r in rows("zenodo_logical_bitflip.csv")}
    ph_data = {(r["section"], float(r["nbar"])): float(r["eps_L_phase"]) for r in rows("zenodo_phase_logical.csv")
               if r["erasure_info"] == "1"}
    ph_model = {(r["section"], float(r["nbar"]), r["variant"]): float(r["model_eps"]) for r in rows("phase_logical_model.csv")}
    lb = {(r["section"], float(r["nbar"])): r for r in rows("logical_bitflip.csv") if r["protocol"] == "g+f"}
    win = {(r["pair"], float(r["nbar"])): float(r["model_static"]) for r in rows("window_pulsed.csv")}
    out = []
    for (section, nbar), eb in sorted(bit_data.items()):
        if (section, nbar) not in ph_data or (section, nbar) not in lb:
            continue
        r = lb[(section, nbar)]
        idle_zf = 1.0
        for c in SECTION_CATS[section]:
            idle_zf *= 1 - 2 * win[(STORAGE_PAIR[c], nbar)] * 2.8 / 3.0
        gates_zf = float(r["gates_zero-fit"])
        bit_zf = 0.5 * (1 - (1 - 2 * gates_zf) * idle_zf)
        bit_l3 = 0.5 * (float(r["total_lam3_lo"]) + float(r["total_lam3_hi"]))
        row = dict(section=section, nbar=nbar, data_eps_bit=eb, data_eps_phase=ph_data[(section, nbar)],
                   data_eps_L=0.5 * (eb + ph_data[(section, nbar)]),
                   zero_fit_eps_bit=bit_zf, zero_fit_eps_phase=ph_model[(section, nbar, "zero-fit")],
                   lambda3_eps_bit=bit_l3, lambda3_eps_phase=ph_model[(section, nbar, "lambda3")],
                   anchored_eps_phase=ph_model[(section, nbar, "data-anchored")])
        row["zero_fit_eps_L"] = 0.5 * (row["zero_fit_eps_bit"] + row["zero_fit_eps_phase"])
        row["lambda3_eps_L"] = 0.5 * (row["lambda3_eps_bit"] + row["lambda3_eps_phase"])
        row["anchored_eps_L"] = 0.5 * (row["lambda3_eps_bit"] + row["anchored_eps_phase"])
        out.append(row)
    keys = list(out[0])
    path = ROOT / "data" / "validation" / "logical_total.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Overall logical error per cycle eps_L = (eps_bit + eps_phase)/2: data (our fits/decoding of Zenodo "
                 "10.5281/zenodo.14257632, Putterman et al., CC BY 4.0; VALIDATION ONLY) vs model variants. "
                 "examples/logical_total_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in out:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("section | nbar* and min eps_L: data | zero-fit | lambda3 | anchored-phase")
    for section in SECTION_CATS:
        sel = [r for r in out if r["section"] == section]
        best = lambda k: min(sel, key=lambda r: r[k])
        print(f"{section} | " + " | ".join(f"{best(k)['nbar']:.1f}: {best(k)[k] * 100:.2f}%"
                                          for k in ("data_eps_L", "zero_fit_eps_L", "lambda3_eps_L", "anchored_eps_L")))
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for ax, section in zip(axs, SECTION_CATS):
        sel = sorted((r for r in out if r["section"] == section), key=lambda r: r["nbar"])
        x = [r["nbar"] for r in sel]
        ax.plot(x, [r["data_eps_L"] for r in sel], "o", mfc="none", color=TEXT_PRIMARY, label="data eps_L")
        ax.plot(x, [r["data_eps_bit"] for r in sel], "v", ms=3, color="0.5", label="data bit / phase")
        ax.plot(x, [r["data_eps_phase"] for r in sel], "^", ms=3, color="0.5")
        ax.plot(x, [r["zero_fit_eps_L"] for r in sel], "-", color=SERIES[0], label="model, zero-fit")
        ax.plot(x, [r["lambda3_eps_L"] for r in sel], "--", color=SERIES[1], label="model, lambda = 3")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_title(section, fontsize=9)
    axs[0].set_ylabel("logical error per cycle")
    axs[0].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"logical_total.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
