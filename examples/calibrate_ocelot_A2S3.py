"""Calibrate the storage-coupler-ancilla model of the Ocelot A2-S3 cell (static data only).

Two calibrations are produced (paper App. A.2, Table 2):

  A  "reproduce Fig. 7": targets = Fig. 7 curves of arXiv:2409.13025 (AWS circuit model of
     A2-S3, vector-extracted) + measured idle frequencies (Fig. 9b) + coupler range (App. D3).
  B  "data-anchored": as A, but chi_ge, chi_gf at the "on" point replaced by the *measured*
     values of Fig. 12 (A2<->S3: -1.18, -1.14 MHz).

Usage:  python examples/calibrate_ocelot_A2S3.py
Outputs: data/calibration/ocelot_A2S3_fit{A,B}.yaml
"""
from __future__ import annotations

import datetime as dt
import pathlib
import sys

import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.spectra.calibrate import Target, fit, params_to_yaml_dict, targets_from_fig7_csv
from bosonoise.spectra.dispersive import static_observables

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "digitized" / "ocelot_fig7.csv"
OUT = ROOT / "data" / "calibration"

FIT_TRUNC = Truncation(n_s=8, n_c=6, n_q=8, ncut=20)
BIG_TRUNC = Truncation(n_s=12, n_c=9, n_q=11, ncut=30)

# initial guess (order-of-magnitude estimates)
GUESS = CellParams(f_s=5.31, EJ_q=18.9, EC_q=0.195, EJmax_c=42.8, EC_c=0.25, d_c=0.549,
                   g_sc=0.060, g_sq=0.0045, g_cq=0.173, K_s_extra=1.7e-6)

FIG7_FLUXES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

EXTRA = [
    Target("f_c", 0.0, 9.0, 0.05, "Ocelot App. D3 p.19: coupler 9.0 GHz at Phi=0 (A2-S3 model)"),
    Target("f_c", 0.5, 6.6, 0.05, "Ocelot App. D3 p.19: coupler 6.6 GHz at Phi0/2 (A2-S3 model)"),
    Target("f_s", 0.0, 5.31, 0.005, "Ocelot Fig. 9b p.22: S3 frequency, couplers idle (measured)"),
    Target("f_ge", 0.0, 5.21, 0.005, "Ocelot Fig. 9b p.22: A2 frequency, couplers idle (measured)"),
]
MEASURED_ON = [
    Target("chi_ge", 0.5, -1.18e-3, 1e-5, "Ocelot Fig. 12 p.24: A2<->S3 chi_ge/2pi = -1.18(1) MHz (measured)"),
    Target("chi_gf", 0.5, -1.14e-3, 1e-5, "Ocelot Fig. 12 p.24: A2<->S3 chi_gf/2pi = -1.14(1) MHz (measured)"),
]


def run(label: str, targets: list, base: CellParams = GUESS) -> dict:
    print(f"\n=== calibration {label}: {len(targets)} targets ===")
    res = fit(base, targets, trunc=FIT_TRUNC, verbose=0)
    print(res.message, "cost", res.cost)
    print(res.table())
    # convergence: same parameters, much larger truncation
    small = StorageCouplerAncilla(res.params, FIT_TRUNC)
    big = StorageCouplerAncilla(res.params, BIG_TRUNC)
    conv = {}
    for fl in (0.0, 0.25, 0.5):
        a, b = static_observables(small, fl), static_observables(big, fl)
        conv[fl] = {k: float(b[k] - a[k]) for k in ("f_s", "f_ge", "f_ef", "f_c", "chi_ge", "chi_gf", "K_g", "K_e", "K_f")}
    worst = {k: max(abs(conv[fl][k]) for fl in conv) for k in conv[0.0]}
    print("truncation check (big - fit truncation), max over flux, GHz:", {k: f"{v:.2e}" for k, v in worst.items()})
    g01 = {fl: {k: float(v) for k, v in big.g01(fl).items()} for fl in (0.0, 0.5)}
    rec = dict(
        label=label,
        date=str(dt.date.today()),
        units="GHz (f = omega/2pi); d_c dimensionless",
        params=params_to_yaml_dict(res.params),
        stderr_statistical=res.stderr,
        g01_couplings_GHz=g01,
        cost=res.cost, success=res.success, message=res.message,
        fit_truncation=vars(FIT_TRUNC), check_truncation=vars(BIG_TRUNC),
        truncation_max_abs_change_GHz=worst,
        targets=[dict(quantity=t.quantity, flux=t.flux, value=t.value, sigma=t.sigma, source=t.source,
                      model=float(t.value + r * t.sigma), pull=float(r))
                 for t, r in zip(res.targets, res.residuals)],
        note=("Static-spectroscopy calibration only (no error-rate data; paper App. A.2). K_s_extra is a spectator "
              "(buffer/ATS) storage Kerr absent from the 3-mode reduction."),
    )
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tA = targets_from_fig7_csv(str(CSV), FIG7_FLUXES) + EXTRA
    recA = run("A (reproduce Fig. 7)", tA)
    with open(OUT / "ocelot_A2S3_fitA.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(recA, fh, sort_keys=False, allow_unicode=True)

    # B: the measured on-point chi anchors the scale; the Fig. 7 chi(flux) curves keep only
    # their shape information (10% relative sigma).
    tB = []
    for t in tA:
        if t.quantity in ("chi_ge", "chi_gf"):
            if t.flux == 0.5:
                continue
            t = Target(t.quantity, t.flux, t.value, max(0.1 * abs(t.value), 5e-6), t.source + " [sigma 10%]")
        tB.append(t)
    tB += MEASURED_ON
    baseB = CellParams(**recA["params"])
    recB = run("B (data-anchored chi at on-point)", tB, base=baseB)
    with open(OUT / "ocelot_A2S3_fitB.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(recB, fh, sort_keys=False, allow_unicode=True)
    print("\nparameters A vs B:")
    for k in recA["params"]:
        a, b = recA["params"][k], recB["params"][k]
        print(f"  {k:10s} A={a:12.6g}  B={b:12.6g}  rel.diff={(b - a) / (abs(a) + 1e-30):+.3%}")


if __name__ == "__main__":
    sys.exit(main())
