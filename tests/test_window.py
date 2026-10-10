"""Stabilization-window model (channels/window.py): odd-sector cycle maps of a pulsed dissipative cat."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from bosonoise.channels.conversion import logical_states
from bosonoise.channels.window import CatWindow, _commutator, _dissipator, odd_indices, stabilization_superop


def test_noise_free_stabilization_conserves_the_logical_z():
    w = CatWindow(2.0, kappa2=0.2, kappa1=0.0, nth=0.0, kphi=0.0, K=0.0)
    assert w.static_TX() > 1e8  # Z_L is exactly conserved by D[a^2 - nbar]
    assert w.bitflip_per_cycle(w.on(3.0)) < 1e-10


def test_bit_flip_time_grows_with_the_cat_size():
    TX = [CatWindow(nb, kappa2=0.2, kappa1=1 / 60, kphi=0.01).static_TX() for nb in (1.0, 2.0, 3.0)]
    assert TX[0] < TX[1] < TX[2]


def test_odd_sector_eigenvalue_matches_the_full_evolution():
    """<Z> of the full (unrestricted) evolution decays at the leading odd-sector eigenvalue."""
    nb, N = 2.0, 24
    w = CatWindow(nb, N=N, kappa2=0.3, kappa1=1 / 40, kphi=0.02, K=0.01)
    a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
    ad, num = a.T.tocsr(), (a.T @ a).tocsr()
    L2 = (a @ a - nb * sp.identity(N, format="csr")).tocsr()
    Lfull = (0.3 * _dissipator(L2, N) + (1 / 40) * _dissipator(a, N) + 0.02 * _dissipator(num, N)
             + _commutator(0.005 * (ad @ ad @ a @ a).tocsr(), N)).toarray()
    z, o = logical_states(nb, N)
    rho = np.outer(z, z.conj()).reshape(-1, order="F")
    Zop = (np.outer(z, z.conj()) - np.outer(o, o.conj()))
    t1, t2 = 20.0, 40.0
    zt = [np.real(np.trace(Zop @ (la.expm(Lfull * t) @ rho).reshape(N, N, order="F"))) for t in (t1, t2)]
    rate_full = -np.log(zt[1] / zt[0]) / (t2 - t1)
    ev = la.eigvals(w.L_stab + w.L_idle)
    rate_odd = -ev[np.argmax(ev.real)].real
    assert abs(rate_full - rate_odd) < 0.02 * rate_odd
    assert len(odd_indices(N)) == N * N // 2


def test_off_period_rates_default_to_the_on_rates_and_can_be_set_separately():
    w = CatWindow(2.0, kappa2=0.2, kappa1=1 / 60, kphi=0.01, K=0.005)
    assert np.allclose(w.off(0.5), la.expm(w.L_idle * 0.5))
    quiet = CatWindow(2.0, kappa2=0.2, kappa1=1 / 60, kphi=0.01, K=0.0, kappa1_off=0.0, kphi_off=0.0)
    assert np.allclose(quiet.off(0.5), np.eye(len(quiet.idx)))  # no noise, no Kerr: nothing happens while off
    assert np.allclose(quiet.L_idle, CatWindow(2.0, kappa2=0.2, kappa1=1 / 60, kphi=0.01).L_idle)


def test_exact_buffer_elimination_limits_and_trace():
    N, nb, k2, kb = 20, 2.0, 0.8, 67.0
    a = sp.diags(np.sqrt(np.arange(1, N)), 1, shape=(N, N), format="csr")
    A = (a @ a - nb * sp.identity(N, format="csr")).tocsr()
    # delta_b = chi_sb = 0: exactly kappa2 D[a^2 - nbar]
    S0 = stabilization_superop(N, nb, k2, kappa_b=kb)
    assert abs(S0 - k2 * _dissipator(A, N)).max() < 1e-12
    # small detuning: the kerr_ratio form to first order in delta_b / kappa_b
    d = 0.02 * kb
    S1 = stabilization_superop(N, nb, k2, kappa_b=kb, delta_b=d)
    S2 = stabilization_superop(N, nb, k2, kerr_ratio=-d / kb)
    # the leading difference is the Lorentzian factor (kappa_b/2)^2/(delta_b^2 + (kappa_b/2)^2) = 1 - 4 (delta_b/kappa_b)^2
    assert abs(S1 - S2).max() < 5 * (d / kb) ** 2 * abs(S2).max()
    # cross-Kerr: trace preserving (vec(I)^dag S = 0) and exact code states stay dark only at chi_sb = 0
    S3 = stabilization_superop(N, nb, k2, kappa_b=kb, chi_sb=0.06, delta_b=0.3)
    tr = np.eye(N).reshape(-1, order="F")
    assert np.max(np.abs(tr @ S3.toarray())) < 1e-10
    w = CatWindow(nb, N=N, kappa2=k2, kappa1=0.0, kappa_b=kb, chi_sb=0.0)
    assert w.static_TX() > 1e8


def test_pulsed_bitflip_matches_the_dense_cycle_map_and_tracking_modes():
    w = CatWindow(3.0, kappa2=0.5, kappa1=1 / 60, kphi=0.005, K=2 * np.pi * 2e-3, kappa1_off=1 / 70, kphi_off=0.002)
    rot = 2 * np.pi * 2e-3 * 3.0 * 2.0
    p1, f1 = w.pulsed_bitflip(1.0, 2.0, rot, "mean")
    dense = w.bitflip_per_cycle(w.on(1.0) @ w.phases(-rot * np.arange(w.N)) @ w.off(2.0))
    assert f1 == 1.0 and abs(p1 - dense) < 1e-9 * max(dense, 1e-12) + 1e-14
    p0, _ = w.pulsed_bitflip(1.0, 2.0, rot, "none")
    assert abs(p0 - w.bitflip_per_cycle(w.on(1.0) @ w.off(2.0))) < 1e-12
    po, fo = w.pulsed_bitflip(1.0, 2.0, rot, "optimal")
    assert po <= min(p0, p1) * (1 + 1e-6) and -0.5 <= fo <= 1.5
