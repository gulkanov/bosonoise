"""One lossy coupler for two effects? The coupler T1 at 'on' from the storage decay measured during the gates (paper
App. C.2).

At the CX 'on' point the storage hybridizes with the coupler and the ancilla, so it loses photons through their baths.
The measured storage decay during the gates is therefore an independent handle on the coupler T1 at 'on' that the CX^2
model needs for the ancilla decay (examples/measured_bath_cx2_ocelot.py, coupler18c variant: T1_c = 2.37 us from the
A3 T1 of Fig. 18c). No error-rate data are used; the inputs are storage lifetimes.

Inputs (device characterizations):
  (a) A1-S1, main text p.5: storage T1,eff from CX^2 phase flips 63 +- 2 us with the gates (one CX^2 per 3-us cycle) and
      68 +- 2 us without.
  (b) Repetition-code run (d = 5): T1,eff of S1..S5 from Fig. 30 (60 / 55 / 51 / 61 / 59 us) against the pumped and
      pump-off T1 of Fig. 9 (presets/ocelot.yaml). With the pump on for 1/3 of the 2.8-us cycle (readout and reset),
      the pump-off part implies a storage T1 of 60 / 50 / 50 / 63 / 61 us against an idle pump-off T1 of
      76 / 71 / 61 / 75 / 68 us (Fig. 9; cf. paper App. H). Each storage spends the CX gates of its one or two ancillas at 'on'
      (Fig. 12 lengths) in that part of the cycle.
Model: storage decay at 'on' = ancilla bath (flat, referenced to the idle T1(e->g), as in the zero-fit CX^2 model) +
coupler bath (flat, 1/T1_c on the coupler 0-1 element), through the dressed storage elements of the ladders at 'on'
(examples/predict_cx2_ule_ocelot.py). The extra decay over idle is attributed to the time at 'on'; T1_c is solved per
input and compared across inputs.

Usage: python examples/coupler_from_storage_ocelot.py
Output: data/validation/coupler_from_storage.csv
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import PAIRS, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "coupler_from_storage.csv"
T_CYCLE_US, DUTY_ON = 2.8, 1 / 3                      # rep-code cycle; pump on during readout and reset (6.20.3)
T1_EFF_US = {"S1": 60.0, "S2": 55.0, "S3": 51.0, "S4": 61.0, "S5": 59.0}   # Fig. 30 (p.45), d = 5 run
A1S1 = dict(with_cx2=(63.0, 2.0), without=(68.0, 2.0), cycle_us=3.0)       # main text p.5 (presets: validation)


def pair_elements(pair, pre):
    """Storage-loss elements (n = 1 -> 0, ancilla in g and in f) at 'on' and idle, the bath scales, the CX length."""
    anc = pair.split("-")[0]
    grid = grid_for(pair)
    on, idle = grid.ladders[-1], grid.ladders[0]
    T1ge = pre["ancilla"][anc]["T1_ge_us"]
    G_q = 1.0 / (T1ge * idle.X["q"][0, 0, 1, 0] ** 2)        # 1/us per |n_q element|^2 (flat, idle-referenced)
    out = dict(pair=pair)
    for lev, tag in ((0, "g"), (2, "f")):
        out[f"q_on_{tag}"] = on.X["q"][lev, 0, lev, 1] ** 2
        out[f"q_idle_{tag}"] = idle.X["q"][lev, 0, lev, 1] ** 2
        out[f"c_on_{tag}"] = (on.X["c"][lev, 0, lev, 1] / on.nc01) ** 2   # times 1/T1_c
    out["G_q"] = G_q
    out["L_cx_us"] = {r["pair"]: float(r["L_CX_us"]) for r in csv.DictReader(
        l for l in open(ROOT / "data" / "digitized" / "ocelot_fig12_gates.csv", encoding="utf-8")
        if not l.startswith("#")) if r["color"] == "purple"}[pair]
    return out


def extra_rate_on(e, T1c_us, lev="g"):
    """Extra storage loss rate at 'on' over idle (1/us): ancilla bath plus coupler bath with T1_c."""
    return e["G_q"] * (e[f"q_on_{lev}"] - e[f"q_idle_{lev}"]) + e[f"c_on_{lev}"] / T1c_us


def solve_T1c(target_rate, terms):
    """T1_c such that sum_k w_k * extra_rate_on(e_k, T1_c) = target_rate (terms = [(w_k, e_k, lev)])."""
    anc = sum(w * e["G_q"] * (e[f"q_on_{lev}"] - e[f"q_idle_{lev}"]) for w, e, lev in terms)
    cpl = sum(w * e[f"c_on_{lev}"] for w, e, lev in terms)
    rest = target_rate - anc
    return (cpl / rest if rest > 0 else np.inf), anc, cpl


def main():
    pre = load_preset("ocelot")
    el = {p: pair_elements(p, pre) for p in PAIRS}
    rows = []
    print("pair    L_CX   storage loss at 'on' over idle: ancilla bath (1/ms)  coupler term x T1_c (us/ms)  [g / f]")
    for p, e in el.items():
        print(f"{p}  {e['L_cx_us']:.3f}  "
              f"{1e3 * e['G_q'] * (e['q_on_g'] - e['q_idle_g']):7.3f} / {1e3 * e['G_q'] * (e['q_on_f'] - e['q_idle_f']):7.3f}   "
              f"{1e3 * e['c_on_g']:8.3f} / {1e3 * e['c_on_f']:8.3f}")

    # (a) A1-S1 with and without CX^2 gates (one CX^2 per 3-us cycle; ancilla in g in that experiment)
    (Tw, dTw), (To, dTo) = A1S1["with_cx2"], A1S1["without"]
    gate_us = el["A1-S1"]["L_cx_us"] + 0.5 / (abs(pre["interactions"]["A1_S1"]["chi_gf_MHz"]))
    for sgn, tag in ((0, "central"), (-1, "low"), (+1, "high")):
        extra = 1 / (Tw + sgn * dTw) - 1 / (To - sgn * dTo)   # sgn = +1 -> smaller extra rate
        target = extra * A1S1["cycle_us"] / gate_us            # rate while at 'on'
        T1c, anc, cpl = solve_T1c(target, [(1.0, el["A1-S1"], "g")])
        rows.append(dict(input="A1-S1 with/without CX2 (p.5)", case=tag, storage="S1", target_rate_per_ms=1e3 * target,
                         ancilla_part_per_ms=1e3 * anc, T1c_us=T1c))

    # (b) repetition-code run: extra pump-off storage decay over the idle pump-off T1, attributed to the CX time at 'on'
    st = pre["storage"]
    for s in ("S1", "S2", "S3", "S4", "S5"):
        k_eff = 1 / T1_EFF_US[s]
        k_off_implied = (k_eff - DUTY_ON / st[s]["T1_under_2ph_diss_us"]) / (1 - DUTY_ON)
        extra = k_off_implied - 1 / st[s]["T1_us"]
        pairs = [p for p in PAIRS if p.endswith(s)]
        t_off = (1 - DUTY_ON) * T_CYCLE_US
        for lev in ("g", "f"):
            terms = [(el[p]["L_cx_us"] / t_off, el[p], lev) for p in pairs]
            T1c, anc, cpl = solve_T1c(extra, terms)
            rows.append(dict(input=f"rep-code T1,eff (Fig. 30) vs idle (Fig. 9), ancilla in {lev}", case="central",
                             storage=s, target_rate_per_ms=1e3 * extra, ancilla_part_per_ms=1e3 * anc, T1c_us=T1c))
    keys = list(rows[0])
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Coupler T1 at 'on' implied by the storage decay measured during the gates (target_rate = extra storage "
                 "loss rate to explain; ancilla_part = the part from the idle-referenced ancilla bath). "
                 "examples/coupler_from_storage_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.4g}" for k in keys) + "\n")
    print("saved", OUT)
    for r in rows:
        print(f"{r['input']:52s} {r['case']:8s} {r['storage']}: extra {r['target_rate_per_ms']:7.3f}/ms, ancilla part "
              f"{r['ancilla_part_per_ms']:6.3f}/ms -> T1_c = {r['T1c_us']:.2f} us")


if __name__ == "__main__":
    main()
