"""Joint static calibration of the two Ocelot cells that share one ancilla (A1..A4).

Ancilla Ak couples to storages S_k and S_{k+1}. Data (static only) are read from the preset
(src/bosonoise/presets/ocelot.yaml): idle frequencies (Fig. 9b) and the measured chi_ge,
chi_gf at the 'on' point (Fig. 12). Couplers and coupling structure are NOT published:
E_Jmax, E_C of the coupler, g_cq and the ratio g_sq/g_sc are taken from the A2-S3 fit A
(prior); the coupler minimum frequency is a prior (--fcmin, default: A2-S3 value).
Fitted: shared E_J, E_C of the ancilla; per cell f_s and g_sc (6 parameters, 8 targets).
Generalizes examples/calibrate_ocelot_A4.py (same method; A4 results agree).

Usage: python examples/calibrate_ocelot_ancilla.py --anc A1 [--fcmin 6.569]
Output: data/calibration/ocelot_<anc>_fcmin<..>.yaml
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import numpy as np
import yaml
from scipy.optimize import brentq, least_squares

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation, TunableTransmon
from bosonoise.presets import load_preset
from bosonoise.spectra.dispersive import static_observables

ROOT = pathlib.Path(__file__).resolve().parents[1]
TR = Truncation(n_s=8, n_c=6, n_q=8, ncut=20)
BIG = Truncation(n_s=12, n_c=9, n_q=11, ncut=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anc", required=True, choices=["A1", "A2", "A3", "A4"])
    ap.add_argument("--fcmin", type=float, default=None)
    a = ap.parse_args()
    k = int(a.anc[1])
    stor = [f"S{k}", f"S{k + 1}"]
    pre = load_preset("ocelot")
    f_anc = pre["ancilla"][a.anc]["f_ge_GHz"]
    data = {"f_anc": (f_anc, 0.005, "Ocelot Fig. 9b p.22")}
    for s in stor:
        it = pre["interactions"][f"{a.anc}_{s}"]
        data[s] = dict(f_s=(pre["storage"][s]["f_GHz"], 0.005, "Ocelot Fig. 9b p.22"),
                       chi_ge=(it["chi_ge_MHz"] * 1e-3, 1e-5, f"Ocelot Fig. 12 p.24, {a.anc}<->{s}"),
                       chi_gf=(it["chi_gf_MHz"] * 1e-3, 1e-5, f"Ocelot Fig. 12 p.24, {a.anc}<->{s}"))
    ref = yaml.safe_load(open(ROOT / "data" / "calibration" / "ocelot_A2S3_fitA.yaml", encoding="utf-8"))
    pA = CellParams(**ref["params"])
    fcmin_ref = TunableTransmon(pA.EJmax_c, pA.EC_c, pA.d_c).at(0.5).f01()
    fcmin = fcmin_ref if a.fcmin is None else a.fcmin
    d_c = brentq(lambda d: TunableTransmon(pA.EJmax_c, pA.EC_c, d).at(0.5).f01() - fcmin, 0.05, 0.95, xtol=1e-10)
    r_sq = pA.g_sq / pA.g_sc

    def cells(x):
        EJq, ECq, fs1, g1, fs2, g2 = x
        base = pA.with_(EJ_q=EJq, EC_q=ECq, d_c=d_c)
        return {stor[0]: base.with_(f_s=fs1, g_sc=g1, g_sq=r_sq * g1), stor[1]: base.with_(f_s=fs2, g_sc=g2, g_sq=r_sq * g2)}

    def resid(x, trunc=TR, detail=False):
        out, rows = [], []
        for s, p in cells(x).items():
            m = StorageCouplerAncilla(p, trunc)
            o0, o5 = static_observables(m, 0.0), static_observables(m, 0.5)
            for q, fl, val, (tv, sig, src) in [("f_ge", 0.0, o0["f_ge"], data["f_anc"]), ("f_s", 0.0, o0["f_s"], data[s]["f_s"]),
                                               ("chi_ge", 0.5, o5["chi_ge"], data[s]["chi_ge"]),
                                               ("chi_gf", 0.5, o5["chi_gf"], data[s]["chi_gf"])]:
                out.append((val - tv) / sig)
                rows.append(dict(cell=f"{a.anc}-{s}", quantity=q, flux=fl, target=tv, sigma=sig, model=float(val),
                                 pull=float((val - tv) / sig), source=src))
            if detail:
                rows.append(dict(cell=f"{a.anc}-{s}", derived=dict(
                    f_ef=float(o0["f_ef"]), alpha_q_MHz=float(o0["alpha_q"] * 1e3), f_c_on=float(o5["f_c"]),
                    detuning_idle_MHz=float((o0["f_s"] - o0["f_ge"]) * 1e3),
                    detuning_on_MHz=float((o5["f_s"] - o5["f_ge"]) * 1e3),
                    alpha_on_MHz=float(o5["alpha_q"] * 1e3), chi_ge_on_MHz=float(o5["chi_ge"] * 1e3))))
        return (np.array(out), rows) if detail else np.array(out)

    x0 = np.array([pA.EJ_q * (f_anc / 5.21) ** 2, pA.EC_q, data[stor[0]]["f_s"][0], pA.g_sc, data[stor[1]]["f_s"][0], pA.g_sc])
    res = least_squares(resid, x0, x_scale=np.abs(x0), diff_step=1e-4,
                        bounds=([5, 0.08, 4.5, 0, 4.5, 0], [60, 0.5, 6.5, 0.5, 6.5, 0.5]))
    r, rows = resid(res.x, detail=True)
    rbig = resid(res.x, trunc=BIG)
    print(f"{a.anc}: {res.message} cost {res.cost:.3f}; truncation check max |dpull| {np.max(np.abs(rbig - r)):.3g}")
    for row in rows:
        if "quantity" in row:
            print(f"  {row['cell']} {row['quantity']:7s} target {row['target']:+.6f} model {row['model']:+.6f} pull {row['pull']:+.2f}")
        else:
            print(f"  {row['cell']} derived: " + ", ".join(f"{kk}={v:.4f}" for kk, v in row["derived"].items()))
    cp = cells(res.x)
    out = dict(label=f"{a.anc} cells joint static calibration", date=str(dt.date.today()), units="GHz",
               priors=dict(source="data/calibration/ocelot_A2S3_fitA.yaml (A2-S3 fit A)", EJmax_c=float(pA.EJmax_c),
                           EC_c=float(pA.EC_c), g_cq=float(pA.g_cq), g_sq_over_g_sc=float(r_sq),
                           f_c_min_bare_GHz=float(fcmin), d_c=float(d_c),
                           note="coupler and coupling-structure parameters NOT published; prior from A2-S3"),
               fitted=dict(zip(["EJ_q", "EC_q", f"f_s_{stor[0]}", f"g_sc_{stor[0]}", f"f_s_{stor[1]}", f"g_sc_{stor[1]}"],
                               map(float, res.x))),
               cells={kk: {f: float(getattr(v, f)) for f in v.__dataclass_fields__} for kk, v in cp.items()},
               targets=[row for row in rows if "quantity" in row], derived=[row for row in rows if "derived" in row],
               cost=float(res.cost), truncation=vars(TR))
    path = ROOT / "data" / "calibration" / f"ocelot_{a.anc}_fcmin{fcmin:.3f}.yaml"
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)
    print("saved", path)


if __name__ == "__main__":
    main()
