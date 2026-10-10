"""Exact diagonalization vs. second-order perturbation theory (no RWA) in the weak-coupling limit."""
import numpy as np

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.spectra import static_observables


def pt2_shift(E_q, n_q, g, f_s, q, n):
    """Second-order energy shift of |q, n> for V = g n_q (a + a^dag), counter-rotating terms included."""
    tot = 0.0
    for k in range(len(E_q)):
        m2 = (g * n_q[q, k]) ** 2
        if m2 == 0:
            continue
        if n > 0:  # |k, n-1>
            tot += m2 * n / (E_q[q] - E_q[k] + f_s)
        tot += m2 * (n + 1) / (E_q[q] - E_q[k] - f_s)  # |k, n+1>
    return tot


def test_chi_matches_second_order_pt():
    # decoupled coupler; weak direct storage-ancilla coupling, ~100 MHz detuning
    p = CellParams(f_s=5.33, EJ_q=18.87, EC_q=0.198, EJmax_c=60.0, EC_c=0.17, d_c=0.55,
                   g_sc=0.0, g_sq=0.0015, g_cq=0.0)
    m = StorageCouplerAncilla(p, Truncation(n_s=8, n_c=2, n_q=12, ncut=25))
    o = static_observables(m, 0.0)
    E_q, n_q = p.ancilla(25).spectrum(12)
    g = p.g_sq
    f = {q: [pt2_shift(E_q, n_q, g, p.f_s, q, n) for n in range(3)] for q in range(3)}
    fs = {q: f[q][1] - f[q][0] for q in range(3)}
    chi_ge_pt, chi_gf_pt = fs[1] - fs[0], fs[2] - fs[0]
    assert abs(o["chi_ge"] - chi_ge_pt) < 0.01 * abs(chi_ge_pt)
    assert abs(o["chi_gf"] - chi_gf_pt) < 0.01 * abs(chi_gf_pt)
    # sanity: chi negative and of order g01^2 / Delta
    assert o["chi_ge"] < 0 and o["chi_gf"] < 0


def test_uncoupled_limit_is_trivial():
    p = CellParams(f_s=5.3, EJ_q=18.87, EC_q=0.198, EJmax_c=60.0, EC_c=0.17, d_c=0.55,
                   g_sc=0.0, g_sq=0.0, g_cq=0.0)
    o = static_observables(StorageCouplerAncilla(p, Truncation(n_s=5, n_c=3, n_q=4)), 0.3)
    assert abs(o["chi_ge"]) < 1e-12 and abs(o["chi_gf"]) < 1e-12
    assert abs(o["K_g"]) < 1e-12
    assert np.isclose(o["f_s"], 5.3)
