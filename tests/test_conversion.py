import numpy as np
import pytest

from bosonoise.channels.conversion import AsymptoticProjector, coherent, logical_states


@pytest.fixture(scope="module")
def proj2():
    return AsymptoticProjector(2.0)


def test_logical_states_orthonormal_and_close_to_coherent():
    z, o = logical_states(3.0, 40)
    assert abs(np.vdot(z, z) - 1) < 1e-12 and abs(np.vdot(o, o) - 1) < 1e-12
    assert abs(np.vdot(z, o)) < 1e-12
    assert abs(np.vdot(coherent(np.sqrt(3.0), 40), z)) ** 2 > 1 - 1e-4  # overlap ~ 1 - e^{-2 nbar}/...


def test_code_space_is_invariant(proj2):
    for v in (proj2.zero, proj2.one, (proj2.zero + 1j * proj2.one) / np.sqrt(2)):
        rho = np.outer(v, v.conj())
        assert np.abs(proj2.asymptotic(rho) - rho).max() < 1e-10


def test_flip_probability_symmetries(proj2):
    th = np.array([0.0, 0.3, np.pi / 2, np.pi - 0.3, np.pi])
    p = proj2.flip_probability(th)
    assert abs(p[0]) < 1e-10 and abs(p[-1] - 1) < 1e-10 and abs(p[2] - 0.5) < 1e-10
    assert abs(p[1] + p[3] - 1) < 1e-10  # P(theta) + P(pi - theta) = 1
    assert abs(proj2.eta_uniform(2 * np.pi) - 0.5) < 1e-6


def test_flip_decreases_with_nbar():
    ps = [AsymptoticProjector(nb).flip_probability([np.pi / 4])[0] for nb in (1.0, 2.0, 4.0)]
    assert ps[0] > ps[1] > ps[2] > 0


def test_p_flip_pads_small_input(proj2):
    z, _ = logical_states(2.0, 16)
    assert proj2.p_flip(np.outer(z, z.conj()) / np.vdot(z, z).real) < 1e-4  # truncation error only


def test_buffer_click_statistics_are_consistent():
    """No-click projector (Ferrari et al.): code states never click; a rotated coherent state clicks, more so for larger
    angles and efficiencies; the n-click distribution sums to one, starts with the no-click value, and its flip
    marginal equals the unconditional flip probability (monitoring does not change the dynamics)."""
    P0 = AsymptoticProjector(3.0)
    Ph = AsymptoticProjector(3.0, eta_detect=0.5)
    assert abs(sum(Ph.silent_split(np.outer(P0.zero, P0.zero.conj()))) - 1) < 1e-9
    silent = []
    for th in (0.2, 0.8, np.pi / 2):
        c = coherent(np.sqrt(3.0) * np.exp(1j * th), P0.N)
        rho = np.outer(c, c.conj()) / np.vdot(c, c).real
        D = Ph.click_distribution(rho, 40)
        assert abs(D.sum() - 1) < 1e-4 and abs(D[0].sum() - sum(Ph.silent_split(rho))) < 1e-9
        assert abs(D[:, 1].sum() - P0.p_flip(rho)) < 1e-4
        silent.append(D[0].sum())
    assert silent[0] > silent[1] > silent[2]


def test_p_flip_truncates_large_input_only_if_the_tail_is_empty(proj2):
    """A small cat simulated with a large common cutoff is truncated; a real tail is an error."""
    N = proj2.N + 15
    c = coherent(np.sqrt(2.0), N)
    rho = np.outer(c, c.conj())
    ref = proj2.p_flip(rho[:proj2.N, :proj2.N] / np.trace(rho[:proj2.N, :proj2.N]).real)
    assert abs(proj2.p_flip(rho) - ref) < 1e-12
    big = coherent(np.sqrt(0.8 * proj2.N), N)
    with pytest.raises(ValueError):
        proj2.p_flip(np.outer(big, big.conj()))


def test_detuned_buffer_map_is_asymmetric_but_keeps_code_space():
    """Ocelot App. F8: K/kappa_2 = -Delta_b/kappa_b; Delta_b < 0 favours recovering positive phase shifts."""
    P = AsymptoticProjector(2.0, kerr_ratio=0.1)
    for v in (P.zero, P.one):
        rho = np.outer(v, v.conj())
        assert np.abs(P.asymptotic(rho) - rho).max() < 1e-10
    p_pos, p_neg = P.flip_probability([0.5, -0.5])
    assert p_pos < p_neg
    Q = AsymptoticProjector(2.0, kerr_ratio=-0.1)
    assert abs(Q.flip_probability([-0.5])[0] - p_pos) < 1e-10  # mirror symmetry
