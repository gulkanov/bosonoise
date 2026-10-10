import numpy as np
import scipy.linalg as la

from bosonoise.dynamics.ule import (H_OVER_KB_MK, classical_dephasing, detailed_balance, flat, jump_operator,
                                    lindblad_rhs, piecewise_flat)


def _liouvillian(H, Ls):
    d = H.shape[0]
    basis = np.eye(d * d)
    cols = [lindblad_rhs(H, Ls, b.reshape(d, d)).ravel() for b in basis]
    return np.array(cols).T


def test_golden_rule_and_detailed_balance_qubit():
    f01, T = 5.0, 60.0
    E = np.array([0.0, f01])
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    rate = detailed_balance(flat(1e-4), T)
    L = jump_operator(E, X, rate)
    assert abs(abs(L[0, 1]) ** 2 - 1e-4) < 1e-12  # emission 1 -> 0
    boltz = np.exp(-f01 * H_OVER_KB_MK / T)
    assert abs(abs(L[1, 0]) ** 2 - 1e-4 * boltz) < 1e-12  # absorption 0 -> 1
    M = _liouvillian(np.diag(E), [L])
    w, v = la.eig(M)
    rho = v[:, np.argmin(abs(w))].reshape(2, 2)
    rho = rho / np.trace(rho)
    assert abs(rho[1, 1].real / rho[0, 0].real - boltz) < 1e-6  # Gibbs state


def test_ule_jump_preserves_coherent_state_for_harmonic_ladder():
    # H = f a^dag a, X = a + a^dag, flat emission and no absorption: L = sqrt(G) a (lowering part only)
    N, f = 30, 5.0
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    L = jump_operator(f * np.arange(N), a + a.T, detailed_balance(flat(2.0), None))
    assert np.allclose(L, np.sqrt(2.0) * a)


def test_piecewise_and_dephasing_shapes():
    r = piecewise_flat([5.0], [1.0, 2.0])
    assert np.allclose(r(np.array([4.9, 5.1, -4.9, -5.1])), [1.0, 2.0, 1.0, 2.0])
    white = classical_dephasing(0.01, "white")
    assert np.allclose(white(np.array([0.0, 0.1, -0.1])), 0.02)
    onef = classical_dephasing(0.01, "1/f", f_ref=1e-3)
    assert abs(onef(np.array(0.1)) - 0.02 * 1e-2) < 1e-12
