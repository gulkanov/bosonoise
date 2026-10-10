"""Logical phase-flip rate of the Ocelot repetition code from the raw syndromes (Zenodo 10.5281/zenodo.14257632,
Putterman et al., CC BY 4.0), decoded with the DEM of our Stim model (dem/repetition.py).

For every section (d3 S1-S3, d3 S3-S5, d5 S1-S5), photon number and run length N the raw shots are turned into the
detection events of phase_flip_circuit(d, N): syndrome changes (round 0 against the measured initial parities, the last
round against the final parities) and herald bits (syndrome value 1 = erasure, ancilla read in e; its syndrome bit is
set to 1 as in AWS App. H2). Observable: the parity change of the first cat, X_L = X_1.

Decoder weights (a 'data-anchored' model, like AWS's App. I1, and clearly not a prediction):
  p_z of each cat from the T1,eff of Fig. 30 (rep-code run), p_meas of each ancilla = the time-like detector-pair
  probability without erasures (zenodo_detection.csv), p_erase = the erasure fraction (zenodo_erasures.csv),
  initial/final parity readout errors 1.5% (App. F3 SPAM), share of phase flips between a cat's two CX gates 0.2
  (space-like pairs are 0.8 of kappa_1,eff |alpha|^2 T_cycle, section 6.16.2).
Decoded with the heralds (correlated matching: erased time-like edges) and without (heralds zeroed, erasures only in the
averaged weights). <X_L(0) X_L(N)> = 1 - 2 P_fail(N) is fitted with A exp(-N/tau) + B (offset allowed, as AWS for X_L),
epsilon = 1/(2 tau) per cycle (= T_cycle/(2 T_X)). VALIDATION ONLY.

Usage: python examples/zenodo_phase_decode_ocelot.py
Output: data/validation/zenodo_phase_logical.csv (fitted), data/validation/zenodo_phase_runs.csv (per run),
        figures/zenodo_phase_logical.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

from bosonoise.dem.repetition import PhaseFlipNoise, matching_for, phase_flip_circuit
from bosonoise.plotting import SERIES, use_style

from zenodo_detection_ocelot import T1_EFF_US  # same directory (Fig. 30, d5 run)
from zenodo_erasure_ocelot import ERASED, ODD, SECTIONS, ZEN

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_CYCLE_US = 2.8
SPAM = 0.015
F_MID = 0.2


def cats_of(section):
    a, b = section.split("_")[1].split("-")
    return [f"S{k}" for k in range(int(a[1]), int(b[1]) + 1)]


def read_csv(name):
    return list(csv.DictReader(l for l in open(ROOT / "data" / "validation" / name, encoding="utf-8")
                               if not l.startswith("#")))


def data_anchored_noise(section, nbar, det_rows, er_rows):
    cats = cats_of(section)
    ancs = SECTIONS[section][1]
    pz = [0.5 * (1 - np.exp(-2 * nbar * T_CYCLE_US / T1_EFF_US[c])) for c in cats]
    pm = [float(next(r["data_p_pair"] for r in det_rows if r["section"] == section and float(r["nbar"]) == nbar
                     and r["kind"] == "time" and r["detectors"] == a)) for a in ancs]
    pe = [float(next(r["data_E_all"] for r in er_rows if r["section"] == section and float(r["nbar"]) == nbar
                     and r["ancilla"] == a)) for a in ancs]
    return PhaseFlipNoise(p_z=pz, p_meas=pm, p_erase=pe, f_mid=F_MID, p_init=SPAM, p_final=SPAM)


def detection_events(table, cats, ancs, N):
    """(dets, obs, n_heralds) in the detector order of phase_flip_circuit(d, N) (every ancilla has heralds)."""
    d = len(cats)
    col = lambda name: np.asarray(table.column(name).to_numpy(), dtype=np.int8)
    init = np.stack([col(f"{c}_initial_state") for c in cats], axis=1).astype(bool)
    fin = np.stack([col(f"{c}_final_state") for c in cats], axis=1).astype(bool)
    raw = np.stack([np.stack([col(f"{a}_syndrome_{t}") for a in ancs], axis=1) for t in range(N)], axis=1)  # (S, N, d-1)
    s = (raw == ODD) | (raw == ERASED)
    h = raw == ERASED
    S = raw.shape[0]
    blocks = []
    for t in range(N):
        prev = (init[:, :-1] ^ init[:, 1:]) if t == 0 else s[:, t - 1]
        blocks.append(s[:, t] ^ prev)
        blocks.append(h[:, t])
    blocks.append(s[:, N - 1] ^ fin[:, :-1] ^ fin[:, 1:])
    dets = np.concatenate(blocks, axis=1)
    obs = (init[:, 0] ^ fin[:, 0])[:, None]
    assert dets.shape == (S, (d - 1) * (2 * N + 1))
    return dets, obs, int(h.sum())


def decay(N, A, tau, B):
    return A * np.exp(-N / tau) + B


def fit_eps(Ns, P, n):
    Ns, P = np.asarray(Ns, float), np.asarray(P, float)
    y = 1 - 2 * P
    sig = 2 * np.sqrt(np.clip(P * (1 - P), 1e-12, None) / n)
    p0 = (1.0, max(Ns.max() / 3, 1.0), 0.0)
    popt, pcov = curve_fit(decay, Ns, y, p0=p0, sigma=sig, absolute_sigma=True, maxfev=20000)
    tau, dtau = popt[1], float(np.sqrt(pcov[1, 1]))
    return 1 / (2 * tau), dtau / (2 * tau ** 2), popt


def main():
    det_rows, er_rows = read_csv("zenodo_detection.csv"), read_csv("zenodo_erasures.csv")
    per_run, fits = [], []
    for section, (folder, ancs) in SECTIONS.items():
        cats = cats_of(section)
        d = len(cats)
        for nb_dir in sorted((ZEN / folder).iterdir()):
            if not nb_dir.is_dir():
                continue
            nbar = float(nb_dir.name.split("_")[1])
            noise = data_anchored_noise(section, nbar, det_rows, er_rows)
            t0 = time.time()
            res = {True: [], False: []}
            for f in sorted(nb_dir.iterdir(), key=lambda q: int(q.name.split("_")[-1]) if q.name.startswith("num_cycles_") else -1):
                if not f.name.startswith("num_cycles_"):
                    continue
                N = int(f.name.split("_")[-1])
                table = pq.read_table(f)
                dets, obs, nh = detection_events(table, cats, ancs, N)
                circ = phase_flip_circuit(d, N, noise)
                assert circ.num_detectors == dets.shape[1]
                for use_h in (True, False):
                    m = matching_for(circ, use_heralds=use_h)
                    shots = dets.astype(np.uint8)
                    if not use_h:
                        shots = shots.copy()
                        for t in range(N):  # zero the herald block of every round
                            shots[:, (2 * t + 1) * (d - 1):(2 * t + 2) * (d - 1)] = 0
                    pred = m.decode_batch(shots, enable_correlations=use_h)
                    fail = float(np.mean(pred[:, 0] != obs[:, 0]))
                    res[use_h].append((N, fail, len(obs)))
                    per_run.append(dict(section=section, nbar=nbar, erasure_info=int(use_h), num_cycles=N, shots=len(obs),
                                        p_fail=fail, heralds=nh))
            for use_h in (True, False):
                Ns, P, n = zip(*res[use_h])
                eps, deps, popt = fit_eps(Ns, P, np.asarray(n))
                fits.append(dict(section=section, nbar=nbar, erasure_info=int(use_h), eps_L_phase=eps, eps_err=deps,
                                 fit_A=popt[0], fit_tau_cycles=popt[1], fit_B=popt[2]))
            print(f"{section} nbar {nbar:.1f}: eps_L,phase with erasure info {fits[-2]['eps_L_phase'] * 100:.3f} "
                  f"+- {fits[-2]['eps_err'] * 100:.3f} %, without {fits[-1]['eps_L_phase'] * 100:.3f} % "
                  f"({time.time() - t0:.0f} s)", flush=True)
    out = ROOT / "data" / "validation" / "zenodo_phase_logical.csv"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Logical phase-flip error per cycle of the Ocelot repetition code, decoded from the raw syndromes of "
                 "Zenodo 10.5281/zenodo.14257632 (Putterman et al., CC BY 4.0; attribution required) with the DEM of our "
                 "data-anchored Stim model; fits of <X_L(0)X_L(N)> = A exp(-N/tau) + B, eps = 1/(2 tau). "
                 "VALIDATION ONLY. examples/zenodo_phase_decode_ocelot.py\n")
        keys = list(fits[0])
        fh.write(",".join(keys) + "\n")
        for r in fits:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    out2 = ROOT / "data" / "validation" / "zenodo_phase_runs.csv"
    with open(out2, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Per-run logical failure probability (same decoding). examples/zenodo_phase_decode_ocelot.py\n")
        keys = list(per_run[0])
        fh.write(",".join(keys) + "\n")
        for r in per_run:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    use_style()
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    for k, section in enumerate(SECTIONS):
        for use_h, mk, ls in ((1, "s", "-"), (0, "o", "--")):
            sel = sorted((r for r in fits if r["section"] == section and r["erasure_info"] == use_h), key=lambda r: r["nbar"])
            ax.errorbar([r["nbar"] for r in sel], [r["eps_L_phase"] for r in sel], yerr=[r["eps_err"] for r in sel],
                        fmt=mk + ls, ms=4, mfc="none" if not use_h else None, color=SERIES[k],
                        label=f"{section}" + (", erasure info" if use_h else ""))
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_ylabel("logical phase-flip error per cycle")
    ax.set_title("Ocelot raw syndromes decoded with our DEM (data-anchored weights)", fontsize=9)
    ax.legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"zenodo_phase_logical.{ext}", dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
