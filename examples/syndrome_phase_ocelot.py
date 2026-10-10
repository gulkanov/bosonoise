"""Syndrome dephasing by photon-number-dependent dispersive phases during the CX gates (Ocelot stabilizers).

The syndrome is the relative phase of the ancilla branches g and f after the two CX gates. For storage photon number
n the f branch accumulates Phi(n) = 2 pi int [E_f(n) - E_g(n)](t) dt along the calibrated flux pulse; an ideal CX
gives pi n + const. The dressed ladder (dispersive nonlinearity chi(n), Kerr differences between branches) makes
Phi(n) - pi n - const = delta(n) nonlinear in n. For cats with photon-number spread, the contrast of the syndrome is
C = |sum_n P(n) exp(i delta(n))| per cat (even or odd cat: P(n) Poisson restricted to one parity; the result is
averaged over both), after removing the mean phase (the ancilla phase is calibrated in situ at each |alpha|^2, App. G3),
and the syndrome error per round is (1 - C_a C_b) / 2. Unitary part only; model columns, no fit. Two calibrations:
  fixed gate: the phase per photon is the one of the model pulse at low n (gate length fixed across |alpha|^2, as in the
              experiment: AWS App. H5 attribute the growth of the time-like error to exactly this, via the conditional
              Kerr terms K_g a^2 a^2 |g><g|, K_f ... |f><f|, strongest for the smallest storage-ancilla detuning, A4);
  retuned:    the linear part re-fitted at each |alpha|^2 (gate length tuned per photon number, AWS's proposed fix):
              only the curvature of Phi(n) remains.
Compared with the measurement error per round from the raw syndromes (examples/zenodo_detection_ocelot.py; VALIDATION).

Usage: python examples/syndrome_phase_ocelot.py
Output: data/validation/syndrome_phase.csv
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
from scipy.stats import poisson

from bosonoise.channels.cx2 import calibrate_cx2_to_length

from predict_cx2_ule_ocelot import grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NL = 40
STAB = {"A1": ("S1", "S2"), "A2": ("S2", "S3"), "A3": ("S3", "S4"), "A4": ("S4", "S5")}
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]


def branch_phases(pair: str, L_cx_us: float, dt: float = 0.25):
    """Phi_f(n) - Phi_g(n) (rad) along the calibrated half-turn CX pulse, n = 0 .. NL."""
    grid = grid_for(pair, NL)
    pulse = calibrate_cx2_to_length(grid, L_cx_us * 1e3, turns=0.5)
    t = np.arange(0.0, pulse.duration + dt / 2, dt)
    acc = np.zeros(NL + 1)
    for i, ti in enumerate(t):
        w = dt if 0 < i < len(t) - 1 else dt / 2  # trapezoid
        E = grid.E(float(pulse.flux(ti)))
        acc += w * (E[2] - E[0])
    return 2 * np.pi * acc, pulse.duration


def contrast(delta: np.ndarray, nbar: float, retune: bool = False) -> float:
    n = np.arange(len(delta))
    c = []
    for parity in (0, 1):  # even and odd cats; the mean phase is calibrated away
        P = poisson.pmf(n, nbar) * (n % 2 == parity)
        P /= P.sum()
        d = delta
        if retune:  # remove the best P-weighted linear phase around nbar
            m = P > 1e-12
            d = delta - np.polyval(np.polyfit(n[m], delta[m], 1, w=np.sqrt(P[m])), n)
        c.append(abs(np.sum(P * np.exp(1j * d))))
    return float(np.mean(c))


def main():
    L = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}
    det = {}
    dpath = ROOT / "data" / "validation" / "zenodo_detection.csv"
    if dpath.exists():
        for r in csv.DictReader(l for l in open(dpath, encoding="utf-8") if not l.startswith("#")):
            if r["kind"] == "time" and r["section"] == "d5_S1-S5":
                det[(r["detectors"], float(r["nbar"]))] = float(r["data_p_pair"])
    rows = []
    for anc, cats in STAB.items():
        deltas = {}
        for cat in cats:
            phi, T = branch_phases(f"{anc}-{cat}", L[f"{anc}-{cat}"])
            n = np.arange(NL + 1)
            # remove the ideal pi n and a constant; what is left is the nonlinear (unitary) syndrome phase error
            lin = np.polyfit(n[:3], phi[:3], 1)  # phase per photon set by the low-n ladder (the calibration)
            deltas[cat] = phi - np.polyval(lin, n)
            curv = np.polyfit(n[:15], phi[:15], 2)[0]
            print(f"{anc}-{cat}: gate {T:.0f} ns, phase per photon {lin[0]:.4f} rad (pi = 3.1416), curvature "
                  f"{curv * 1e3:.1f} mrad/photon^2, delta(n) at n = 4 / 10 / 20: {deltas[cat][4]:+.4f} / {deltas[cat][10]:+.4f} / "
                  f"{deltas[cat][20]:+.4f} rad")
        for nb in NBARS:
            Ca, Cb = contrast(deltas[cats[0]], nb), contrast(deltas[cats[1]], nb)
            Ra, Rb = contrast(deltas[cats[0]], nb, True), contrast(deltas[cats[1]], nb, True)
            d = det.get((anc, nb), np.nan)
            rows.append(dict(ancilla=anc, nbar=nb, model_syndrome_error_fixed_gate=0.5 * (1 - Ca * Cb),
                             model_syndrome_error_retuned=0.5 * (1 - Ra * Rb), data_measurement_error=d,
                             data_excess_over_nbar1=d - det.get((anc, 1.0), np.nan)))
    path = ROOT / "data" / "validation" / "syndrome_phase.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Syndrome error per round from photon-number-dependent dispersive phases (dressed ladder, unitary, ancilla phase "
                 "calibrated at each |alpha|^2); data_measurement_error = time-like detector-pair probability of the d = 5 run "
                 "(Zenodo 10.5281/zenodo.14257632, CC BY 4.0; VALIDATION ONLY). examples/syndrome_phase_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("\nancilla nbar | model syndrome error: fixed gate, retuned | data measurement error per round (d = 5), excess over 1")
    for r in rows:
        print(f"{r['ancilla']} {r['nbar']:4.1f} | {r['model_syndrome_error_fixed_gate']:.2e} {r['model_syndrome_error_retuned']:.2e} | "
              f"{r['data_measurement_error']:.3e} {r['data_excess_over_nbar1']:+.3e}")
    print("saved", path)


if __name__ == "__main__":
    main()
