import numpy as np

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.spectra import branch_analysis, find_resonances

BASE = dict(f_s=5.31, EJ_q=18.87, EC_q=0.198, EJmax_c=60.8, EC_c=0.171, d_c=0.546)


def test_uncoupled_branches_are_bare_ladders():
    p = CellParams(**BASE, g_sc=0.0, g_sq=0.0, g_cq=0.0)
    m = StorageCouplerAncilla(p, Truncation(n_s=12, n_c=3, n_q=5))
    br = branch_analysis(m, 0.5, nmax=10)
    np.testing.assert_allclose(br.overlap, 1.0, atol=1e-12)
    for b, (c, q) in enumerate(br.starts):
        np.testing.assert_allclose(br.mean_q[b], q, atol=1e-12)
        np.testing.assert_allclose(br.mean_s[b], br.n, atol=1e-12)
        np.testing.assert_allclose(np.diff(br.energy[b]), 5.31, atol=1e-9)
    assert find_resonances(br) == []


def test_weak_coupling_branches_follow_photon_number():
    p = CellParams(**BASE, g_sc=0.02, g_sq=0.003, g_cq=0.05)
    m = StorageCouplerAncilla(p, Truncation(n_s=16, n_c=4, n_q=6, e_cut=90.0))
    br = branch_analysis(m, 0.5, nmax=10)
    assert np.all(br.overlap[:, 1:] > 0.9)
    np.testing.assert_allclose(br.mean_s, np.broadcast_to(br.n, br.mean_s.shape), atol=0.05)
