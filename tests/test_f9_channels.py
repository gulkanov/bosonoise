"""The F9 channel table of A4 (examples/f9_bayes_ocelot.py prepare): the statements the papers make from it."""
import csv
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "tls" / "f9_bayes_channels_A4.csv"


def _load():
    lines = TABLE.read_text(encoding="utf-8").splitlines()
    checks = json.loads(re.search(r"Checks: (\{.*?\})\.", lines[0]).group(1))
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    return checks, rows


def _col(rows, calib, cell, key):
    return np.array([float(r[key]) for r in rows if r["calib"] == calib and r["cell"] == cell])


def test_forward_model_matches_ancilla_bath():
    checks, _ = _load()
    assert checks["none"] < 1e-12  # identical formula without a defect
    assert checks["tls"] < 1e-4  # one defect: only the far-detuned channels that the vectorized model omits differ
    assert abs(checks["bare_g_cos_f"] - 0.0997) < 1e-3


def test_bands_of_the_two_storages():
    _, rows = _load()
    for calib in ("6.569", "6.25", "6.75"):
        fA4, fA5 = _col(rows, calib, "S4", "fA_GHz"), _col(rows, calib, "S5", "fA_GHz")
        fB4, fB5 = _col(rows, calib, "S4", "fB_GHz"), _col(rows, calib, "S5", "fB_GHz")
        # opened channel: the S5 band (n = 0..40) lies below the S4 band (n = 0..20), tens of MHz away
        assert fA4[:21].min() - fA5.max() > 0.03
        # critical-current channel: both bands start at the same omega_gf (n = 0) and are swept at -1.3 / -1.6 MHz
        assert abs(fB4[0] - fB5[0]) < 1e-3
        sweep4, sweep5 = (fB4[20] - fB4[0]) / 20, (fB5[20] - fB5[0]) / 20
        assert -1.4e-3 < sweep4 < -1.2e-3 and -1.7e-3 < sweep5 < -1.5e-3


def test_coupling_profiles():
    _, rows = _load()
    aA, cB = _col(rows, "6.569", "S4", "aA"), _col(rows, "6.569", "S4", "cB")
    assert aA[20] / aA[0] > 3  # the opened element grows with the photon number
    assert abs(cB[20] / cB[0] - 1) < 0.05  # the cos(phi) element barely changes
