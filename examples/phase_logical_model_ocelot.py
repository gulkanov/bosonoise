"""Logical phase-flip rate of the Ocelot repetition code: our Stim model (dem/repetition.py) against the raw data decoded
with the same kind of decoder (examples/zenodo_phase_decode_ocelot.py -> data/validation/zenodo_phase_logical.csv).

The experiment is mirrored exactly: sections d3 S1-S3, d3 S3-S5, d5 S1-S5; the photon numbers and run lengths of the
data; 15000 shots per run; decoded with the model's own DEM (with heralds, correlated matching); <X_L(0)X_L(N)> fitted
with A exp(-N/tau) + B; eps = 1/(2 tau). Variants:
  data-anchored  p_z from the T1,eff of Fig. 30, p_meas and p_erase from the raw syndromes (like AWS App. I1);
                 a check of the model structure, not a prediction.
  zero-fit       p_z = kappa_1 |alpha|^2 (1 + 2 n_th) T_cycle with the storage T1 of Fig. 9 (pump on during readout +
                 reset, ~1/3 of the 2.8-us cycle; pump off otherwise), n_th = 0.5% (prior);
                 p_meas = SPAM 1.5% (App. F3) + the dispersive-phase syndrome error (syndrome_phase.csv, gate fixed);
                 p_erase = our f->e herald over the gates and the superposition window (zenodo_erasures.csv model column)
                 + f->e during the readout of odd outcomes (half the outcomes, readout + integration time / T1_ef).
  lambda3        as zero-fit with the on-gate ancilla decay enhanced by lambda = 3 (sections 6.14-6.15) in the herald.
Share of phase flips between a cat's two CX gates 0.2 in all variants (timing: ~0.6 us of 2.8 us, pump off).

Usage: python examples/phase_logical_model_ocelot.py
Output: data/validation/phase_logical_model.csv, figures/phase_logical_model.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib
import time
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.dem.repetition import PhaseFlipNoise, matching_for, phase_flip_circuit
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

from zenodo_erasure_ocelot import SECTIONS, ZEN
from zenodo_phase_decode_ocelot import F_MID, SPAM, T_CYCLE_US, cats_of, data_anchored_noise, fit_eps, read_csv

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHOTS = 15000
DUTY_ON = 1 / 3
NTH = 0.005
INTEGRATION_NS = 80.0


def run_lengths(folder, nbar):
    d = ZEN / folder / f"nbar_{nbar:.1f}"
    return sorted(int(p.name.split("_")[-1]) for p in d.iterdir() if p.name.startswith("num_cycles_"))


def zero_fit_noise(pre, section, nbar, sp_rows, er_rows, herald_col):
    cats, ancs = cats_of(section), SECTIONS[section][1]
    pz = []
    for c in cats:
        s = pre["storage"][c]
        k1 = DUTY_ON / s["T1_under_2ph_diss_us"] + (1 - DUTY_ON) / s["T1_us"]
        pz.append(0.5 * (1 - np.exp(-2 * k1 * (nbar * (1 + 2 * NTH) + NTH) * T_CYCLE_US)))
    pm, pe = [], []
    for a in ancs:
        disp = float(next(r["model_syndrome_error_fixed_gate"] for r in sp_rows if r["ancilla"] == a and float(r["nbar"]) == nbar))
        pm.append(SPAM + disp)
        anc = pre["ancilla"][a]
        ro = 0.5 * (1 - np.exp(-(anc["readout_ns"] + INTEGRATION_NS) * 1e-3 / anc["T1_ef_us"]))
        her = float(next(r[herald_col] for r in er_rows if r["section"] == section and float(r["nbar"]) == nbar
                         and r["ancilla"] == a))
        pe.append(her + ro - her * ro)
    return PhaseFlipNoise(p_z=pz, p_meas=pm, p_erase=pe, f_mid=F_MID, p_init=SPAM, p_final=SPAM)


def simulate(noise, d, Ns, seed):
    rows = []
    for k, N in enumerate(Ns):
        circ = phase_flip_circuit(d, N, noise)
        det, obs = circ.compile_detector_sampler(seed=seed + k).sample(SHOTS, separate_observables=True)
        pred = matching_for(circ, True).decode_batch(det.astype(np.uint8), enable_correlations=True)
        rows.append((N, float(np.mean(pred[:, 0] != obs[:, 0])), SHOTS))
    return rows


def main():
    pre = load_preset("ocelot")
    det_rows, er_rows, sp_rows = read_csv("zenodo_detection.csv"), read_csv("zenodo_erasures.csv"), read_csv("syndrome_phase.csv")
    data = {(r["section"], float(r["nbar"])): r for r in read_csv("zenodo_phase_logical.csv") if r["erasure_info"] == "1"}
    out = []
    for section, (folder, ancs) in SECTIONS.items():
        d = len(cats_of(section))
        for nbar in sorted({float(p.name.split("_")[1]) for p in (ZEN / folder).iterdir() if p.is_dir()}):
            Ns = run_lengths(folder, nbar)
            variants = {"data-anchored": data_anchored_noise(section, nbar, det_rows, er_rows),
                        "zero-fit": zero_fit_noise(pre, section, nbar, sp_rows, er_rows, "model_herald_window_zero-fit"),
                        "lambda3": zero_fit_noise(pre, section, nbar, sp_rows, er_rows, "model_herald_window_lam3")}
            t0 = time.time()
            for name, noise in variants.items():
                Nr, P, n = zip(*simulate(noise, d, Ns, seed=zlib.crc32(f"{section}|{nbar:g}|{name}".encode()) % 100000))
                eps, deps, popt = fit_eps(Nr, P, np.asarray(n))
                dd = data.get((section, nbar))
                out.append(dict(section=section, nbar=nbar, variant=name, model_eps=eps, model_eps_err=deps,
                                mean_p_z=float(np.mean(noise.p_z)), mean_p_meas=float(np.mean(noise.p_meas)),
                                mean_p_erase=float(np.mean(noise.p_erase)),
                                data_eps=float(dd["eps_L_phase"]) if dd else np.nan,
                                data_eps_err=float(dd["eps_err"]) if dd else np.nan))
            sel = [r for r in out if r["section"] == section and r["nbar"] == nbar]
            print(f"{section} nbar {nbar:.1f} | data {sel[0]['data_eps'] * 100:.2f}% | " + "  ".join(
                f"{r['variant']} {r['model_eps'] * 100:.2f}% (pz {r['mean_p_z']:.3f} pm {r['mean_p_meas']:.3f} "
                f"pe {r['mean_p_erase']:.3f})" for r in sel) + f"  ({time.time() - t0:.0f} s)", flush=True)
    keys = list(out[0])
    path = ROOT / "data" / "validation" / "phase_logical_model.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Logical phase-flip error per cycle: our Stim model (variants) vs the raw data decoded the same way "
                 "(data_*: Zenodo 10.5281/zenodo.14257632, Putterman et al., CC BY 4.0; VALIDATION ONLY). "
                 "examples/phase_logical_model_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in out:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    styles = {"data-anchored": ("--", SERIES[2]), "zero-fit": ("-", SERIES[0]), "lambda3": (":", SERIES[1])}
    for ax, section in zip(axs, SECTIONS):
        sel = sorted((r for r in out if r["section"] == section and r["variant"] == "zero-fit"), key=lambda r: r["nbar"])
        ax.errorbar([r["nbar"] for r in sel], [r["data_eps"] for r in sel], yerr=[r["data_eps_err"] for r in sel],
                    fmt="o", mfc="none", color=TEXT_PRIMARY, label="data (raw syndromes, our decoder)")
        for name, (ls, colr) in styles.items():
            sel = sorted((r for r in out if r["section"] == section and r["variant"] == name), key=lambda r: r["nbar"])
            ax.plot([r["nbar"] for r in sel], [r["model_eps"] for r in sel], ls, color=colr, label=f"model, {name}")
        ax.set_yscale("log")
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_title(section, fontsize=9)
    axs[0].set_ylabel("logical phase-flip error per cycle")
    axs[0].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"phase_logical_model.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
