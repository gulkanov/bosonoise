"""chi-matching and the opened f -> g cost per gate (paper Sec. 3.1, Eq. (5), Table 4, Fig. 1(c)).

A conditional pi rotation lasts T = pi / |chi_gf| (angular units), and the opened branching per photon is
rho ~ |chi_ge| alpha / (2 |Delta (Delta + alpha)|) (examples/universal_law.py). The opened probability per gate and photon
is therefore

    rho Gamma_fe T = r * c(x) * Gamma_fe / alpha,   r = chi_ge / chi_gf,   c(x) = 1 / (4 |x (1 + x)|),   x = Delta / alpha,

with alpha and Delta as ordinary frequencies. chi-matching (r = 1), which keeps a single f -> e decay from rotating the
cat, fixes x, and with it the cost.

  transmon  a directly coupled transmon (charge basis, n_g = 0) at E_J / E_C = 50, 95, 150; dispersive shifts of g, e, f
            to second order in the coupling with counter-rotating terms (as for the GKP devices in universal_law.py);
            r(x) on a grid and the matching point x* (r = 1).
  ocelot    the eight storage-ancilla pairs at the CX^2 'on' point: x and alpha from the dressed ladders of the full
            storage-coupler-ancilla model, chi_ge / chi_gf from the same ladders (model) and from Fig. 12 of Putterman
            et al. (measured, presets/ocelot.yaml).
  drive_matched
            the g/f ancilla of Reinhold et al., Nat. Phys. 16, 822 (2020), whose chi_e = chi_f is set by an
            error-transparency sideband drive, so that the detuning is free: the published f_ge, cavity frequency,
            alpha, chi_e, chi_f and T1_ef (Supplementary table, arXiv:1907.12327v1); likewise the ancilla of
            Rosenblum et al., Science 361, 266 (2018), matched by a sideband drive with
            the cavity 2 GHz below (Table S4 of arXiv:1803.00102v4); r_model from a directly coupled transmon with that
            f_ge and alpha; rho from the dispersive law; the static (undriven) r enters the cost.
            drive_matched_bound: the same transmons matched by the detuning.
  scan      r(x) and the cost 2 pi r c(x) (in units of Gamma_fe / alpha with angular alpha) of the directly coupled
            transmons on a dense grid of both signs of Delta, outside the straddling region -1.5 < x < 0.2 (paper Fig. 1(c)).

Usage: python examples/chi_matching.py
Output: data/validation/chi_matching.csv, data/validation/chi_matching_scan.csv
"""
from __future__ import annotations

import pathlib

import numpy as np
import yaml
from scipy.optimize import brentq

from bosonoise.model.transmon import Transmon

from predict_cx2_ule_ocelot import PAIRS, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "chi_matching.csv"
SCAN = ROOT / "data" / "validation" / "chi_matching_scan.csv"
PRESET = ROOT / "src" / "bosonoise" / "presets" / "ocelot.yaml"
KEYS = ["kind", "name", "EJ_over_EC", "x", "alpha_MHz", "r_model", "r_measured", "cost_coeff", "per_gate_T1fe_25us",
        "T1fe_us", "per_gate", "rho_law"]
X_GRID = [0.25, 0.37, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
# dense grid for the figure: cavity above the ancilla (x > 0) and below it beyond the e-f resonance (x < -1)
X_SCAN = np.concatenate([-np.geomspace(12.0, 1.5, 40), np.geomspace(0.2, 4.0, 60)])
T1FE_US = 25.0
ALPHA_REF_MHZ = 200.0
# Drive-matched g/f ancillas, published parameters (GHz, us): |alpha|, |chi_e|, |chi_f| (chi_ge, chi_gf in our notation).
# Reinhold et al. 2020, Supplementary table (arXiv:1907.12327v1); Rosenblum et al. 2018, Table S4 (arXiv:1803.00102v4),
# matched by an off-resonant sideband drive at a cavity 2 GHz below the ancilla.
DRIVE_MATCHED = {"Reinhold 2020": dict(f_ge=4.2, f_c=4.5, alpha=0.137, chi_ge=0.9e-3, chi_gf=1.2e-3, T1fe_us=47.0),
                 "Rosenblum 2018": dict(f_ge=6.5, f_c=4.5, alpha=0.210, chi_ge=0.093e-3, chi_gf=0.236e-3, T1fe_us=23.0)}


def transmon_r(EJ_over_EC, EC=0.2, g=0.005):
    """r(x) = chi_ge / chi_gf of a directly coupled transmon, second order in g with counter-rotating terms."""
    E, n = Transmon(EJ=EJ_over_EC * EC, EC=EC, ng=0.0, ncut=40).spectrum(20)
    f_ge, alpha = E[1] - E[0], (E[1] - E[0]) - (E[2] - E[1])

    def shift(i, w):
        return g ** 2 * sum(abs(n[i, k]) ** 2 * (1 / (E[i] - E[k] - w) - 1 / (E[k] - E[i] - w))
                            for k in range(len(E)) if k != i)

    def r(x):
        w = f_ge + x * alpha
        s0, s1, s2 = shift(0, w), shift(1, w), shift(2, w)
        return (s1 - s0) / (s2 - s0)

    return r, alpha


def transmon_for(f_ge, alpha):
    """E_J/E_C and E_C (GHz) of a charge-basis transmon (n_g = 0) with the given g-e frequency and anharmonicity."""
    def spec(R):
        E, _ = Transmon(EJ=R, EC=1.0, ng=0.0, ncut=40).spectrum(4)
        return E[1] - E[0], (E[1] - E[0]) - (E[2] - E[1])

    R = brentq(lambda R: spec(R)[1] / spec(R)[0] - alpha / f_ge, 20.0, 500.0)
    return R, f_ge / spec(R)[0]


def cost(x, r):
    return r / (4 * abs(x * (1 + x)))


def main():
    rows, scan = [], []
    for ratio in (50, 95, 150):
        r, alpha = transmon_r(ratio)
        for x in X_SCAN:
            rr = r(x)
            scan.append((ratio, x, rr, 2 * np.pi * cost(x, rr)))
        xs = brentq(lambda x: r(x) - 1, 0.5, 1.5)
        for x in X_GRID + [xs]:
            rr = r(x)
            c = cost(x, rr)
            rows.append(dict(kind="transmon" if x != xs else "transmon_match", name=f"E_J/E_C = {ratio}",
                             EJ_over_EC=ratio, x=x, alpha_MHz=alpha * 1e3, r_model=rr, r_measured=float("nan"),
                             cost_coeff=c, per_gate_T1fe_25us=c / (T1FE_US * 1e-6) / (ALPHA_REF_MHZ * 1e6)))
        print(f"transmon E_J/E_C = {ratio}: alpha {alpha * 1e3:.1f} MHz, chi-matching at x* = {xs:.3f}, "
              f"cost {cost(xs, 1.0):.4f} Gamma_fe/alpha = Gamma_fe/({1 / cost(xs, 1.0):.2f} alpha)")
    meas = yaml.safe_load(open(PRESET, encoding="utf-8"))["interactions"]
    for pair in PAIRS:
        on = grid_for(pair).ladders[-1]
        E = on.E  # E[b, n]: b = g, e, f (coupler in 0)
        f_ge, f_ef, f_s = E[1, 0] - E[0, 0], E[2, 0] - E[1, 0], E[0, 1] - E[0, 0]
        alpha, delta = f_ge - f_ef, f_s - f_ge
        chi_ge = (E[1, 1] - E[1, 0]) - (E[0, 1] - E[0, 0])
        chi_gf = (E[2, 1] - E[2, 0]) - (E[0, 1] - E[0, 0])
        m = meas[pair.replace("-", "_")]
        r_meas = m["chi_ge_MHz"] / m["chi_gf_MHz"]
        x = delta / alpha
        c = cost(x, r_meas)
        rows.append(dict(kind="ocelot", name=pair, EJ_over_EC=float("nan"), x=x, alpha_MHz=alpha * 1e3,
                         r_model=chi_ge / chi_gf, r_measured=r_meas, cost_coeff=c,
                         per_gate_T1fe_25us=c / (T1FE_US * 1e-6) / (alpha * 1e9)))
        print(f"Ocelot {pair}: x = Delta/alpha = {x:.3f}, alpha {alpha * 1e3:.1f} MHz, r model {chi_ge / chi_gf:.3f}, "
              f"measured {r_meas:.3f}; per gate and photon at T1_fe = 25 us: {rows[-1]['per_gate_T1fe_25us']:.2e}")
    for row in rows:
        row.update(T1fe_us=T1FE_US, per_gate=row["per_gate_T1fe_25us"], rho_law=float("nan"))
    for label, rd in DRIVE_MATCHED.items():
        ratio, EC = transmon_for(rd["f_ge"], rd["alpha"])
        r, alpha = transmon_r(ratio, EC=EC)
        delta = rd["f_c"] - rd["f_ge"]
        xs = brentq(lambda x: r(x) - 1, 0.5, 1.5)
        for kind, name, x, r_used, r_meas, rho in (
                ("drive_matched", label, delta / rd["alpha"], rd["chi_ge"] / rd["chi_gf"],
                 rd["chi_ge"] / rd["chi_gf"], rd["chi_ge"] * rd["alpha"] / (2 * abs(delta * (delta + rd["alpha"])))),
                ("drive_matched_bound", f"{label} matched by detuning", xs, 1.0, float("nan"), float("nan"))):
            c = cost(x, r_used)
            rows.append(dict(kind=kind, name=name, EJ_over_EC=ratio, x=x, alpha_MHz=alpha * 1e3, r_model=r(x),
                             r_measured=r_meas, cost_coeff=c, per_gate_T1fe_25us=c / (T1FE_US * 1e-6) / (alpha * 1e9),
                             T1fe_us=rd["T1fe_us"], per_gate=c / (rd["T1fe_us"] * 1e-6) / (alpha * 1e9), rho_law=rho))
            print(f"{name}: E_J/E_C = {ratio:.1f} (E_C {EC * 1e3:.1f} MHz, alpha {alpha * 1e3:.1f} MHz), x = {x:.3f}, "
                  f"r model {r(x):.3f}, measured {r_meas:.3f}, rho (law) {rho:.3g}; cost Gamma_fe/({1 / c:.2f} alpha) "
                  f"= {2 * np.pi * c:.3f} Gamma_fe/alpha_angular, per gate and photon at T1_fe = {rd['T1fe_us']:.0f} us: "
                  f"{rows[-1]['per_gate']:.3g}")
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# chi-matching and the opened f -> g cost per gate and photon, r c(x) Gamma_fe / alpha with r = chi_ge/chi_gf, "
                 "c(x) = 1/(4|x(1+x)|), x = Delta/alpha; transmon: direct coupling, 2nd order with counter-rotating terms "
                 "(per_gate at alpha/2pi = 200 MHz); ocelot: model ladders at 'on', r_measured from Fig. 12; "
                 "drive_matched: Reinhold et al. 2020 and Rosenblum et al. 2018 (published parameters, static r; per_gate "
                 "at their T1_ef), drive_matched_bound: the same transmons matched by the detuning; with angular alpha the "
                 "cost is 2 pi cost_coeff Gamma_fe / alpha. examples/chi_matching.py\n")
        fh.write(",".join(KEYS) + "\n")
        for r in rows:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in KEYS) + "\n")
    print("saved", OUT)
    with open(SCAN, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Directly coupled transmon (charge basis, n_g = 0, 2nd order in the coupling with counter-rotating terms): "
                 "r = chi_ge/chi_gf and the opened f -> g cost per conditional pi rotation and photon, "
                 "cost_angular = 2 pi r c(x) in units of Gamma_fe/alpha (angular alpha), c(x) = 1/(4|x(1+x)|), x = Delta/alpha; "
                 "the straddling region -1.5 < x < 0.2 is left out. examples/chi_matching.py\n")
        fh.write("EJ_over_EC,x,r,cost_angular\n")
        for ratio, x, rr, ca in scan:
            fh.write(f"{ratio},{x:.6g},{rr:.6g},{ca:.6g}\n")
    print("saved", SCAN)


if __name__ == "__main__":
    main()
