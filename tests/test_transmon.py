import numpy as np
import pytest

from bosonoise.model import Transmon, TunableTransmon


@pytest.mark.parametrize("EJ,EC,ng", [(18.9, 0.198, 0.0), (18.9, 0.198, 0.37), (60.8, 0.171, 0.0), (10.0, 0.3, 0.25)])
def test_transmon_against_scqubits(EJ, EC, ng):
    scq = pytest.importorskip("scqubits")
    ours = Transmon(EJ, EC, ng=ng, ncut=25)
    E, n = ours.spectrum(8)
    ref = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=25, truncated_dim=8)
    Eref = ref.eigenvals(evals_count=8)
    np.testing.assert_allclose(E, Eref - Eref[0], atol=1e-9)
    nref = ref.matrixelement_table("n_operator", evals_count=8)
    np.testing.assert_allclose(np.abs(n), np.abs(nref), atol=1e-8)


def test_charge_matrix_real_symmetric_and_parity():
    E, n = Transmon(18.9, 0.198, ng=0.0, ncut=25).spectrum(10)
    np.testing.assert_allclose(n, n.T, atol=1e-12)
    # at ng = 0 the eigenstates have definite parity: <i|n|j> = 0 for i - j even
    for i in range(10):
        for j in range(10):
            if (i - j) % 2 == 0:
                assert abs(n[i, j]) < 1e-10


def test_tunable_transmon_flux_dependence():
    tt = TunableTransmon(EJmax=60.0, EC=0.17, d=0.55)
    assert np.isclose(tt.EJ(0.0), 60.0)
    assert np.isclose(tt.EJ(0.5), 0.55 * 60.0)
    assert np.isclose(tt.EJ(0.3), tt.EJ(-0.3)) and np.isclose(tt.EJ(0.3), tt.EJ(0.7))
    scq = pytest.importorskip("scqubits")
    ref = scq.TunableTransmon(EJmax=60.0, EC=0.17, d=0.55, flux=0.37, ng=0.0, ncut=25, truncated_dim=5)
    Eref = ref.eigenvals(evals_count=5)
    E, _ = tt.at(0.37).spectrum(5)
    np.testing.assert_allclose(E, Eref - Eref[0], atol=1e-9)


def test_sign_convention_continuity():
    """Matrix elements vary smoothly with flux (no random eigenvector sign flips)."""
    tt = TunableTransmon(EJmax=60.0, EC=0.17, d=0.55, ncut=25)
    prev = None
    for fl in np.linspace(0, 0.5, 26):
        _, n = tt.at(fl).spectrum(6)
        if prev is not None:
            assert np.max(np.abs(n - prev)) < 0.2
        prev = n
