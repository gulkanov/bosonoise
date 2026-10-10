"""Irrotational-gauge SQUID Hamiltonian: static equivalence with the TunableTransmon model."""
import numpy as np
import pytest

from bosonoise.dynamics.flux import IrrotationalSquid, flat_top_gaussian, squid_hamiltonian
from bosonoise.model import TunableTransmon


@pytest.mark.parametrize("flux", [0.0, 0.13, 0.25, 0.4, 0.5])
@pytest.mark.parametrize("cap", ["area", 0.3])
def test_static_spectrum_is_gauge_invariant(flux, cap):
    EJmax, EC, d = 62.1, 0.167, 0.546
    sq = IrrotationalSquid.from_tunable(EJmax, d, cap)
    E = np.linalg.eigvalsh(squid_hamiltonian(EC, sq, flux, ncut=25))
    Eref, _ = TunableTransmon(EJmax, EC, d, ncut=25).at(flux).eigensys(8)
    np.testing.assert_allclose(E[:8] - E[0], Eref - Eref[0], atol=1e-8)


def test_area_capacitance_ratio_suppresses_sin_term():
    sq = IrrotationalSquid.from_tunable(62.1, 0.546, "area")
    for fl in (0.01, 0.05):
        A, B = sq.AB(fl)
        assert abs(B) < 1e-3 * abs(A)  # B = O(phi_e^3) for C proportional to E_J
    sq_bad = IrrotationalSquid.from_tunable(62.1, 0.546, 0.5)
    A, B = sq_bad.AB(0.05)
    assert abs(B) > 10 * abs(sq.AB(0.05)[1])


def test_flat_top_gaussian_shape():
    t = np.linspace(-10, 300, 3101)
    y = flat_top_gaussian(t, t_start=0.0, t_flat=200.0, sigma=16.0, amp=0.5, n_sigma=3.0)
    assert y[t < 0].max() == 0.0 and np.isclose(y.max(), 0.5)
    assert np.all(np.diff(y[(t >= 0) & (t <= 48)]) >= -1e-15)  # monotonic rise
    assert np.isclose(y[np.argmin(np.abs(t - 48.0))], 0.5, atol=1e-3)
    assert y[np.argmin(np.abs(t - 296.0))] == 0.0
