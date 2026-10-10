import csv
import pathlib

import numpy as np

from bosonoise.presets import available, load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_ocelot_preset_loads_and_is_consistent():
    assert "ocelot" in available()
    d = load_preset("ocelot")
    assert set(d["storage"]).issuperset({"S1", "S2", "S3", "S4", "S5"})
    assert set(d["ancilla"]).issuperset({"A1", "A2", "A3", "A4"})
    # every section with numbers declares a source
    for sec in ("storage", "ancilla", "buffer", "two_photon_dissipation", "readout", "interactions"):
        assert "_source" in d[sec], sec
    # kappa2 = 4 g2^2 / kappa_b (adiabatic elimination of the buffer) holds for all five cells
    for i in range(1, 6):
        g2 = d["two_photon_dissipation"][f"S{i}"]["g2_kHz"] * 1e-3
        k2 = d["two_photon_dissipation"][f"S{i}"]["kappa2_kHz"] * 1e-3
        kb = d["buffer"][f"B{i}"]["kappa_b_MHz"]
        assert abs(4 * g2 ** 2 / kb - k2) / k2 < 0.03
    # measured chi ratios in the reported range 0.85 - 1.3
    ratios = [v["chi_ge_MHz"] / v["chi_gf_MHz"] for k, v in d["interactions"].items() if not k.startswith("_")
              and isinstance(v, dict)]
    assert len(ratios) == 8 and 0.84 < min(ratios) and max(ratios) < 1.31


def test_fig7_extraction_is_self_consistent():
    path = ROOT / "data" / "digitized" / "ocelot_fig7.csv"
    rows = list(csv.reader(line for line in open(path, encoding="utf-8") if not line.startswith("#")))
    head, data = rows[0], np.array(rows[1:], float)
    c = {h: data[:, i] for i, h in enumerate(head)}
    assert data.shape[0] == 91
    # the plotted ratio curve equals chi_ge/chi_gf of panel (c) where chi is not tiny
    m = np.abs(c["chi_gf_MHz"]) > 0.2
    np.testing.assert_allclose(c["chi_ge_MHz"][m] / c["chi_gf_MHz"][m], c["chi_ratio_numerics"][m], rtol=0.01)
    # naive Eq. (D5) with the plotted dressed frequencies reproduces the plotted naive curve
    D = c["omega_s_GHz"] - c["omega_a_ge_GHz"]
    K = c["omega_a_ef_GHz"] - c["omega_a_ge_GHz"]  # anharmonicity K_q = f_ef - f_ge
    naive = (D - 2 * K) / (2 * D - K)
    np.testing.assert_allclose(naive, c["chi_ratio_naive_D5"], rtol=0.01)


def test_fig18b_extraction_populations_sum_to_one():
    path = ROOT / "data" / "digitized" / "ocelot_fig18b.csv"
    rows = list(csv.reader(line for line in open(path, encoding="utf-8") if not line.startswith("#")))
    data = np.array(rows[1:], float)
    assert data.shape == (101, 4)
    np.testing.assert_allclose(data[:, 1:].sum(axis=1), 1.0, atol=3e-3)
    np.testing.assert_allclose(data[:, 0], np.linspace(0, 20, 101), atol=0.02)


def test_load_cell_params_single_and_multi():
    from bosonoise.model.cell import load_cell_params

    p = load_cell_params(ROOT / "data" / "calibration" / "ocelot_A2S3_fitA.yaml")
    assert 5.2 < p.f_s < 5.4
    q4 = load_cell_params(ROOT / "data" / "calibration" / "ocelot_A4_fcmin6.569.yaml", "S4")
    q5 = load_cell_params(ROOT / "data" / "calibration" / "ocelot_A4_fcmin6.569.yaml", "S5")
    assert q4.EJ_q == q5.EJ_q and q4.EC_q == q5.EC_q  # shared ancilla A4
    assert q4.f_s < q5.f_s


# Literature presets of the reference noise models (allcat, kerrcat); no number read as a string.
def _leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, path + (str(k),))
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v, path)
    else:
        yield path, x


def test_presets_load_and_no_number_is_a_string():
    names = available()
    assert {"ocelot", "ocelot_rep_noise", "companion", "allcat", "kerrcat"} <= set(names)
    for name in names:
        for path, leaf in _leaves(load_preset(name)):
            if isinstance(leaf, str):
                try:
                    float(leaf)
                except ValueError:
                    continue
                raise AssertionError(f"{name}: {'.'.join(path)} = {leaf!r} is a number read as a string")


def test_kerrcat_preset_tables_are_consistent():
    ops = load_preset("kerrcat")["operations"]
    assert {float(t["nbar"]) for t in ops.values()} == {1.0, 6.25}
    for t in ops.values():
        n = len(t["kappa_over_K"])
        for key in ("cx_infidelity", "cx_bias", "prep_infidelity", "meas_infidelity"):
            assert len(t[key]) == n and all(isinstance(v, float) for v in t[key])
        assert np.all(np.diff(t["kappa_over_K"]) > 0)   # interpolation needs increasing kappa/K


def test_allcat_preset_reproduces_the_le_regent_syndrome_error():
    p = load_preset("allcat")
    nb, eta = 8.0, 1e-3
    za = nb * eta + p["cnot"]["Z_ancilla_nonadiabatic"]["value"] / nb
    zzd = p["cnot"]["Z_data_fraction"]["value"] * nb * eta
    q = 2 * (za + zzd) + (p["ancilla"]["prep_steps"]["value"] + p["ancilla"]["meas_steps"]["value"]) * nb * eta
    assert abs(q - (0.318 / nb + 6 * nb * eta)) < 1e-12   # LR23 p. 8
    assert len(p["cnot"]["bitflip_paulis"]["value"]) == 12
