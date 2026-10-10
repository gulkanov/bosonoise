"""Time-dependent cell: static equivalence and adiabatic limit."""
import numpy as np

from bosonoise.dynamics.cell_td import TimeDependentCell
from bosonoise.dynamics.flux import flat_top_gaussian
from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation

P = CellParams(f_s=5.31385, EJ_q=18.8677, EC_q=0.198275, EJmax_c=62.11, EC_c=0.1674, d_c=0.5463,
               g_sc=0.05438, g_sq=0.01061, g_cq=0.16324)


def test_static_equivalence_at_reference_flux_and_convergence_elsewhere():
    tr = Truncation(n_s=4, n_c=10, n_q=5, ncut=20)
    td = TimeDependentCell(P, tr, flux_ref=0.5)
    st = StorageCouplerAncilla(P, Truncation(n_s=4, n_c=10, n_q=5, ncut=20))
    E_td, _ = td.eigh(0.5)
    E_st, _ = st.eigh(0.5)
    np.testing.assert_allclose(E_td[:20], E_st[:20], atol=1e-9)
    # away from the reference flux the fixed coupler basis converges with n_c
    errs = []
    stn = StorageCouplerAncilla(P, Truncation(n_s=4, n_c=14, n_q=5, ncut=20))
    for nc in (6, 10, 14):
        tdn = TimeDependentCell(P, Truncation(n_s=4, n_c=nc, n_q=5, ncut=20), flux_ref=0.5)
        errs.append(np.max(np.abs(tdn.eigh(0.0)[0][:12] - stn.eigh(0.0)[0][:12])))
    # measured 2026-09-29: 156 MHz, 6.1 MHz, 0.2 MHz -> the fixed coupler basis needs n_c >~ 14
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-3  # GHz


def test_slow_pulse_is_adiabatic_for_ground_state():
    # small system; dt must resolve the fast coupler transitions (see propagate docstring)
    td = TimeDependentCell(P, Truncation(n_s=1, n_c=8, n_q=3, ncut=20), flux_ref=0.25)
    E0, V0 = td.eigh(0.0)
    pulse = lambda t: flat_top_gaussian(t, 0.0, 5.0, 10.0, 0.5, n_sigma=3.0)  # noqa: E731
    t = np.arange(0.0, 65.0 + 1e-9, 0.0125)
    psi_mid = td.propagate(V0[:, 0], pulse, t, method="midpoint")[-1]
    psi_mag = td.propagate(V0[:, 0], pulse, t, method="magnus4")[-1]
    assert 1 - abs(np.vdot(V0[:, 0], psi_mid)) ** 2 < 1e-6
    assert abs(abs(np.vdot(psi_mid, psi_mag)) - 1) < 1e-8


def test_apt_matches_brute_force_for_ramp_exchange():
    from bosonoise.dynamics.apt import adiabatic_path, apt_amplitudes

    td = TimeDependentCell(P, Truncation(n_s=3, n_c=8, n_q=4, ncut=20), flux_ref=0.25)
    E0, V0 = td.eigh(0.0)
    i0 = int(np.argmax(np.abs(V0[td.static.basis.index[(1, 0, 0)], :]) ** 2))  # |g, n=1>
    ie = int(np.argmax(np.abs(V0[td.static.basis.index[(0, 0, 1)], :]) ** 2))  # |e, n=0>
    path = adiabatic_path(td, np.linspace(0, 0.5, 201), i0, n_targets=6)
    k = int(np.nonzero(path.targets0 == ie)[0][0])
    pulse = lambda t: flat_top_gaussian(t, 0.0, 20.0, 2.0, 0.5, n_sigma=4.0)  # noqa: E731
    t = np.arange(0.0, 36.0 + 1e-9, 0.0125)
    p_apt = abs(apt_amplitudes(path, pulse, t)[k]) ** 2
    psi = td.propagate(V0[:, i0], pulse, t, method="midpoint")[-1]
    p_bf = abs(np.vdot(V0[:, ie], psi)) ** 2
    assert 1e-4 < p_bf < 1e-2
    assert abs(p_apt - p_bf) / p_bf < 0.05  # measured 2026-09-29: 7.59e-4 vs 7.79e-4
