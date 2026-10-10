"""Joint static calibration of the Ocelot cells A4-S4 and A4-S5 (shared ancilla A4).

Why these cells: A4-S4 is the pair of the direct f->g anomaly (Ocelot App. F9, Fig. 18b)
and the worst chi-matched pair (chi_ge/chi_gf ~ 1.3); A4-S5 is the well-matched pair of
the App. J analysis (lambda ~ 2.5). Both share ancilla A4, so S5 is a natural control.

Data (static only, Ocelot arXiv:2409.13025v2):
  f_A4 = 5.32, f_S4 = 5.36, f_S5 = 5.41 GHz (Fig. 9b, p.22, couplers idle)
  A4<->S4: chi_ge = -1.88, chi_gf = -1.45 MHz;  A4<->S5: chi_ge = -1.70, chi_gf = -1.65 MHz
  (Fig. 12, p.24, coupler at its minimum-frequency 'on' point, +-0.01 MHz)

Not published -> PRIORS (flagged in the output):
  * couplers C7, C8: E_Jmax, E_C and SQUID asymmetry taken from the A2-S3 calibration
    (fit A); the minimum coupler frequency f_c,min is scanned over the published range
    6.25-6.75 GHz (--fcmin) by adjusting the asymmetry d_c;
  * coupling structure: g_cq and the ratio g_sq/g_sc taken from A2-S3 fit A.

Fitted (6 parameters, 8 targets): shared EJ_q, EC_q of A4; per cell f_s and g_sc.

Usage:  python examples/calibrate_ocelot_A4.py [--fcmin 6.25]
Output: data/calibration/ocelot_A4_fcmin<..>.yaml. Without --fcmin (the A2-S3 value 6.569 GHz) it writes the same
file as calibrate_ocelot_ancilla.py --anc A4, with the same numbers; that script is the canonical producer of
ocelot_A4_fcmin6.569.yaml (tools/reproduce.py), this one is used for the prior scan (--fcmin 6.25 and 6.75).
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import numpy as np
import yaml
from scipy.optimize import brentq, least_squares

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation, TunableTransmon
from bosonoise.spectra.dispersive import static_observables

ROOT = pathlib.Path(__file__).resolve().parents[1]
TR = Truncation(n_s=8, n_c=6, n_q=8, ncut=20)
BIG = Truncation(n_s=12, n_c=9, n_q=11, ncut=30)

DATA = {
    "f_A4": (5.32, 0.005, "Ocelot Fig. 9b p.22 (measured, couplers idle)"),
    "S4": dict(f_s=(5.36, 0.005, "Ocelot Fig. 9b p.22"),
               chi_ge=(-1.88e-3, 1e-5, "Ocelot Fig. 12 p.24, A4<->S4"),
               chi_gf=(-1.45e-3, 1e-5, "Ocelot Fig. 12 p.24, A4<->S4")),
    "S5": dict(f_s=(5.41, 0.005, "Ocelot Fig. 9b p.22"),
               chi_ge=(-1.70e-3, 1e-5, "Ocelot Fig. 12 p.24, A4<->S5"),
               chi_gf=(-1.65e-3, 1e-5, "Ocelot Fig. 12 p.24, A4<->S5")),
}


def d_for_fcmin(EJmax, EC, fcmin):
    """SQUID asymmetry d such that the bare coupler f01 at Phi0/2 equals fcmin."""
    return brentq(lambda d: TunableTransmon(EJmax, EC, d).at(0.5).f01() - fcmin, 0.05, 0.95, xtol=1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fcmin", type=float, default=None, help="bare coupler f01 at Phi0/2 (GHz); default: A2-S3 value")
    a = ap.parse_args()
    ref = yaml.safe_load(open(ROOT / "data" / "calibration" / "ocelot_A2S3_fitA.yaml", encoding="utf-8"))
    pA = CellParams(**ref["params"])
    fcmin_ref = TunableTransmon(pA.EJmax_c, pA.EC_c, pA.d_c).at(0.5).f01()
    fcmin = fcmin_ref if a.fcmin is None else a.fcmin
    d_c = d_for_fcmin(pA.EJmax_c, pA.EC_c, fcmin)
    r_sq = pA.g_sq / pA.g_sc
    print(f"priors from A2-S3 fit A: EJmax_c={pA.EJmax_c:.3f} EC_c={pA.EC_c:.4f} g_cq={pA.g_cq:.4f} "
          f"g_sq/g_sc={r_sq:.4f}; f_c,min: A2-S3 {fcmin_ref:.3f} GHz -> using {fcmin:.3f} GHz (d_c={d_c:.4f})")

    def cells(x):
        EJq, ECq, fs4, gsc4, fs5, gsc5 = x
        base = pA.with_(EJ_q=EJq, EC_q=ECq, d_c=d_c)
        return {"S4": base.with_(f_s=fs4, g_sc=gsc4, g_sq=r_sq * gsc4),
                "S5": base.with_(f_s=fs5, g_sc=gsc5, g_sq=r_sq * gsc5)}

    def resid(x, trunc=TR, detail=False):
        out, rows = [], []
        for name, p in cells(x).items():
            m = StorageCouplerAncilla(p, trunc)
            o0, o5 = static_observables(m, 0.0), static_observables(m, 0.5)
            items = [("f_ge", 0.0, o0["f_ge"], DATA["f_A4"]), ("f_s", 0.0, o0["f_s"], DATA[name]["f_s"]),
                     ("chi_ge", 0.5, o5["chi_ge"], DATA[name]["chi_ge"]), ("chi_gf", 0.5, o5["chi_gf"], DATA[name]["chi_gf"])]
            for q, fl, val, (tv, sig, src) in items:
                out.append((val - tv) / sig)
                rows.append(dict(cell=f"A4-{name}", quantity=q, flux=fl, target=tv, sigma=sig, model=float(val),
                                 pull=float((val - tv) / sig), source=src))
            if detail:
                rows.append(dict(cell=f"A4-{name}", derived=dict(
                    f_ef=float(o0["f_ef"]), alpha_q_MHz=float(o0["alpha_q"] * 1e3), f_c_idle=float(o0["f_c"]),
                    f_c_on=float(o5["f_c"]), f_s_on=float(o5["f_s"]), f_ge_on=float(o5["f_ge"]),
                    detuning_on_MHz=float((o5["f_s"] - o5["f_ge"]) * 1e3),
                    K_g_on_kHz=float(o5["K_g"] * 1e6), K_e_on_kHz=float(o5["K_e"] * 1e6), K_f_on_kHz=float(o5["K_f"] * 1e6))))
        return (np.array(out), rows) if detail else np.array(out)

    x0 = np.array([pA.EJ_q * (5.32 / 5.21) ** 2, pA.EC_q, 5.36, pA.g_sc, 5.41, pA.g_sc])
    res = least_squares(resid, x0, x_scale=np.abs(x0), diff_step=1e-4,
                        bounds=([5, 0.08, 4.5, 0, 4.5, 0], [60, 0.5, 6.5, 0.5, 6.5, 0.5]))
    r, rows = resid(res.x, detail=True)
    rbig = resid(res.x, trunc=BIG)
    print(res.message, "cost", round(float(res.cost), 4))
    for row in rows:
        if "quantity" in row:
            print(f"  {row['cell']} {row['quantity']:7s} Phi={row['flux']:.1f} target {row['target']:+.6f} model {row['model']:+.6f} pull {row['pull']:+.2f}")
        else:
            print(f"  {row['cell']} derived: " + ", ".join(f"{k}={v:.4f}" for k, v in row["derived"].items()))
    print("truncation check: max |pull change| =", float(np.max(np.abs(rbig - r))))
    cp = cells(res.x)
    out = dict(
        label="A4 cells joint static calibration", date=str(dt.date.today()),
        units="GHz (f = omega/2pi)",
        priors=dict(source="data/calibration/ocelot_A2S3_fitA.yaml (A2-S3 fit A)",
                    EJmax_c=pA.EJmax_c, EC_c=pA.EC_c, g_cq=pA.g_cq, g_sq_over_g_sc=r_sq,
                    f_c_min_bare_GHz=float(fcmin), d_c=float(d_c),
                    note="coupler C7/C8 parameters and coupling structure NOT published; prior transferred from A2-S3"),
        fitted=dict(zip(["EJ_q", "EC_q", "f_s_S4", "g_sc_S4", "f_s_S5", "g_sc_S5"], map(float, res.x))),
        cells={k: {f: float(getattr(v, f)) for f in v.__dataclass_fields__} for k, v in cp.items()},
        targets=[row for row in rows if "quantity" in row],
        derived=[row for row in rows if "derived" in row],
        cost=float(res.cost), truncation=vars(TR), truncation_check_max_pull_change=float(np.max(np.abs(rbig - r))),
    )
    path = ROOT / "data" / "calibration" / f"ocelot_A4_fcmin{fcmin:.3f}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    print("saved", path)


if __name__ == "__main__":
    main()
