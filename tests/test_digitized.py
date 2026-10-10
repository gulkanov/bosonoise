"""Sanity checks of the vector-extracted validation data (regression guard for tools/extract_ocelot_*)."""
import csv
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIG = ROOT / "data" / "digitized"


def _rows(name):
    return list(csv.DictReader(l for l in open(DIG / name, encoding="utf-8") if not l.startswith("#")))


def test_fig16_structure_and_convention():
    rows = _rows("ocelot_fig16.csv")
    pairs = {r["pair"] for r in rows}
    assert len(pairs) == 8
    for r in rows:  # T_X = inverse Pauli rate: p = [1 - exp(-2 T_cycle / T_X)] / 2 per 3-us cycle
        T = float(r["T_X_s"])
        assert abs(float(r["p_cycle"]) - 0.5 * (1 - np.exp(-2 * 3e-6 / T))) < 1e-3 * float(r["p_cycle"])
    # A1-S1, |alpha|^2 = 2, ancilla in g+f: the main text quotes (3.5 +- 0.4)e-3 per cycle
    gf = [float(r["p_cycle"]) for r in rows if r["pair"] == "A1-S1" and r["protocol"] == "g+f" and r["alpha2"] == "2.0"][0]
    assert abs(gf - 3.5e-3) < 0.6e-3


def test_physical_flip_times_are_inverse_pauli_rates():
    """The Fig. 15(d) fit decay constants are half the Fig. 2(c) bit-flip times of the same data, and
    |alpha|^2 T_phi equals the storage T1 under dissipation (tools/check_ocelot_flip_time_convention.py)."""
    for r in _rows("ocelot_fig15d_fits.csv"):
        assert abs(float(r["ratio"]) - 2.0) < 0.1
    tphi = [(float(r["alpha2"]), float(r["time_us"])) for r in _rows("ocelot_fig2c.csv") if r["quantity"] == "T_phi"]
    assert len(tphi) == 7 and abs(np.mean([a * T for a, T in tphi]) - 63.0) < 4.0  # T1,eff = 63 +- 2 us (p.5)
    # Fig. 2(c) is the A1-S1 panel of Fig. 16
    f2 = {(r["protocol"], r["alpha2"]): float(r["time_us"]) for r in _rows("ocelot_fig2c.csv") if r["quantity"] == "T_X"}
    f16 = {(r["protocol"], r["alpha2"]): 1e6 * float(r["T_X_s"]) for r in _rows("ocelot_fig16.csv") if r["pair"] == "A1-S1"}
    ratios = [f2[k] / f16[k] for k in f2 if k in f16]
    assert len(ratios) >= 25 and max(abs(np.log(ratios))) < 0.05


def test_fig12_gate_lengths_match_stated_cx2():
    L = {(r["pair"], r["color"]): float(r["L_CX_us"]) for r in _rows("ocelot_fig12_gates.csv")}
    assert abs((L[("A1-S1", "purple")] + 0.5 / 2.37) - 0.504) < 0.003  # CX^2 = 504 ns (p.5)
    lengths = [v for (p, c), v in L.items() if c == "purple"]
    assert abs(min(lengths) - 0.292) < 0.002 and abs(max(lengths) - 0.552) < 0.002  # 292-552 ns (p.6)


def test_fig18c_and_fig37d_summaries():
    t1 = {round(float(r["flux_Phi0"]), 3): float(r["T1_us"]) for r in _rows("ocelot_fig18c.csv")}
    assert len(t1) == 31
    assert 30 < np.median([v for k, v in t1.items() if k <= 0.2]) < 45
    d = {}
    for r in _rows("ocelot_fig37d.csv"):
        d.setdefault((r["state"], r["condition"]), []).append(float(r["T1_us"]))
    assert len(d) == 6 and all(len(v) == 10 for v in d.values())
    assert np.mean(d[("f", "pulsed_alpha2_4")]) < np.mean(d[("f", "coupler_on")]) < 25  # below the idle 25 us
