"""Floquet analysis of the charge-driven transmon (dynamics/floquet.py)."""
import numpy as np

from bosonoise.dynamics.floquet import charge_fourier, rate, track_branches
from bosonoise.dynamics.ule import detailed_balance, flat
from bosonoise.model.transmon import Transmon

# device A of paper App. G (Sivak et al. 2023; our charge-basis fit): E_J, E_C, g, cavity frequency
TR = Transmon(EJ=23.05, EC=0.2039, ng=0.0, ncut=30)
G, FC = 0.01170, 4.479


def _spec(nlev=14):
    return TR.spectrum(nlev)


def test_undriven_branches_and_rates_are_the_bare_ones():
    E, n = _spec()
    fb = track_branches(E, n, FC, [0.0], K=6, states=(0, 1, 2))
    assert np.allclose(fb.q[0], E[:3], atol=1e-12)
    gam = detailed_balance(flat(1.0 / 300e3), 90.0)
    md = fb.modes[0]
    r10 = rate(fb.q[0][1], md[1], fb.q[0][0], md[0], n, FC, gam, kmax=3)
    assert abs(r10 - abs(n[0, 1]) ** 2 * gam(E[1] - E[0])) < 1e-15
    r01 = rate(fb.q[0][0], md[0], fb.q[0][1], md[1], n, FC, gam, kmax=3)
    assert abs(r01 / r10 - np.exp(-(E[1] - E[0]) * 47.992430 / 90.0)) < 1e-9


def test_rates_do_not_depend_on_the_replica_of_the_final_state():
    E, n = _spec()
    eps = 2 * G * np.sqrt(200.0)
    fb = track_branches(E, n, FC, np.linspace(0, eps, 9), K=10, states=(0, 1))
    md, q = fb.modes[-1], fb.q[-1]
    gam = detailed_balance(flat(1.0 / 300e3), 90.0)
    r = rate(q[1], md[1], q[0], md[0], n, FC, gam, kmax=6)
    shifted = np.roll(md[0], 1, axis=0)  # replica q + f_d: phi_m -> phi_(m-1) (the edge block is negligible)
    shifted[0] = 0.0
    r2 = rate(q[1], md[1], q[0] + FC, shifted, n, FC, gam, kmax=6)
    assert abs(r2 / r - 1) < 1e-6
    X = charge_fourier(md[0], md[1], n, 3)
    assert abs(abs(X[3]) - abs(n[0, 1])) < 0.05 * abs(n[0, 1])  # weak dressing: X^0 close to the bare element


def test_stark_shift_matches_the_exact_transmon_cavity_model():
    """The n-dependent part of the e-g splitting: Floquet with eps = 2 g sqrt(n) against exact diagonalization of
    transmon (x) cavity with g n (a + a^dag), at n = 10, 20, 40 photons."""
    E, n = _spec(12)
    nmax = 70
    a = np.diag(np.sqrt(np.arange(1, nmax + 1)), 1)
    Hc = FC * np.diag(np.arange(nmax + 1))
    H = np.kron(np.diag(E), np.eye(nmax + 1)) + np.kron(np.eye(len(E)), Hc) + G * np.kron(n, a + a.T)
    w, V = np.linalg.eigh(H)

    def dressed(i, m):
        return w[np.argmax(np.abs(V[i * (nmax + 1) + m, :]))]
    split = lambda m: dressed(1, m) - dressed(0, m)
    ns = [10, 20, 40]
    fb = track_branches(E, n, FC, np.concatenate([np.linspace(0, 2 * G * np.sqrt(40), 41)]), K=10, states=(0, 1),
                        keep_rest=False)
    for m in ns:
        quantum = split(m) - split(0)
        eps = 2 * G * np.sqrt(m)
        qf = np.interp(eps, fb.eps, fb.q[:, 1] - fb.q[:, 0]) - (E[1] - E[0])
        assert abs(qf / quantum - 1) < 0.06, (m, qf, quantum)


def test_one_photon_decay_follows_the_first_order_dressing_formula_and_the_parity_rule():
    """The one-photon emission sideband (k = +1) of f -> g (equal parity) equals the first-order dressing amplitude
    M_fg = g sqrt(n+1) sum_k ... (within 6% up to 533 photons, the rest is higher-order dressing), and it grows with
    the photon number; e -> g (opposite parity) has no odd sideband at all. Device A at 133 and 533 photons."""
    from bosonoise.dynamics.floquet import one_photon_element

    E, n = _spec(14)
    nbars = np.array([133.0, 533.0])
    fb = track_branches(E, n, FC, 2 * G * np.sqrt(np.linspace(0, nbars[-1], 41)), K=12, states=(0, 1, 2),
                        keep_rest=False)
    for nb in nbars:
        s = int(np.argmin(np.abs((fb.eps / (2 * G)) ** 2 - nb)))
        nb_s = (fb.eps[s] / (2 * G)) ** 2
        X_fg = charge_fourier(fb.modes[s][0], fb.modes[s][2], n, 1)   # k = -1, 0, +1
        X_eg = charge_fourier(fb.modes[s][0], fb.modes[s][1], n, 1)
        M = one_photon_element(E, n, G, FC, 2, 0, nb_s)
        assert abs(abs(X_fg[2]) / abs(M) - 1) < 0.06, (nb, X_fg, M)
        assert abs(X_fg[1]) < 1e-12                                     # no f -> g without a photon (parity)
        assert abs(X_eg[0]) < 1e-12 and abs(X_eg[2]) < 1e-12           # no one-photon e -> g (parity)
        assert abs(one_photon_element(E, n, G, FC, 1, 0, nb_s)) < 1e-12
