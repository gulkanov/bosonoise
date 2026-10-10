import numpy as np

from bosonoise.channels.tls import AncillaBath, TLSEnsemble, poisson_average, two_level_transfer
from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.spectra.transitions import dressed_transitions

P = CellParams(f_s=5.31385, EJ_q=18.8677, EC_q=0.198275, EJmax_c=62.11, EC_c=0.1674, d_c=0.5463,
               g_sc=0.05438, g_sq=0.01061, g_cq=0.16324)


def test_two_level_transfer_limits():
    t = np.linspace(0, 2000, 7)
    assert np.allclose(two_level_transfer(0.0, 0.3, 0.01, t), 0.0)
    V = 2e-3
    np.testing.assert_allclose(two_level_transfer(V, 0.0, 0.0, t), np.sin(V * t) ** 2, atol=1e-12)
    # weak coupling: golden rule with a Lorentzian, rate = V^2 gamma / (Delta^2 + gamma^2/4),
    # valid for t >> 1/gamma (the exact result lags by a ~2/gamma transient)
    V, gam = 1e-4, 1e-2
    for D in (0.0, 0.02):
        rate = V ** 2 * gam / (D ** 2 + gam ** 2 / 4)
        tt = 1e5
        np.testing.assert_allclose(two_level_transfer(V, D, gam, tt), 1 - np.exp(-rate * tt), rtol=1e-2)


def test_uncoupled_transition_table_is_bare():
    p0 = P.with_(g_sc=0.0, g_sq=0.0, g_cq=0.0)
    m = StorageCouplerAncilla(p0, Truncation(n_s=8, n_c=3, n_q=5))
    tab = dressed_transitions(m, 0.5, nmax=4)
    anc = p0.ancilla(20).spectrum(3)
    same = tab.m == tab.n
    np.testing.assert_allclose(np.abs(tab.Mq[same & (tab.pair == "e->g")]), abs(anc[1][0, 1]), atol=1e-12)
    np.testing.assert_allclose(np.abs(tab.Mq[same & (tab.pair == "f->e")]), abs(anc[1][1, 2]), atol=1e-12)
    assert np.allclose(tab.Mq[~same], 0.0) and np.allclose(tab.Mq[tab.pair == "f->g"], 0.0)
    np.testing.assert_allclose(tab.freq[same & (tab.pair == "e->g")], anc[0][1], atol=1e-12)


def test_background_reproduces_idle_T1_and_poisson_average():
    m = StorageCouplerAncilla(P, Truncation(n_s=10, n_c=6, n_q=6))
    idle = dressed_transitions(m, 0.0, nmax=3)
    bath = AncillaBath(idle, idle, T1_ge_us=41.0, T1_fe_us=25.0)
    none = TLSEnsemble(np.array([]), np.array([]), np.array([]))
    re = bath.state_rates(none, 100.0, "e")
    rf = bath.state_rates(none, 100.0, "f")
    np.testing.assert_allclose(re[0], 1e-3 / 41.0, rtol=1e-9)
    assert abs(rf[0] - 1e-3 / 25.0) / (1e-3 / 25.0) < 0.02  # f->g adds a small dressing-opened part
    avg, tail = poisson_average(np.arange(4.0), np.array([0.0, 1.0]))
    assert np.isclose(avg[0], 0.0) and np.isclose(avg[1] + tail[1] * 0, np.sum(np.arange(4.0) * np.exp(-1) / np.array([1, 1, 2, 6])))


def test_literature_sampler_reproduces_reference_densities():
    from bosonoise.channels.tls import sample_tls_literature, surface_mean_rate_per_us

    rng = np.random.default_rng(1)
    n_surf, n_junc, W, R = 0, 0, 1.0, 400
    for _ in range(R):
        e = sample_tls_literature(rng, (5.0, 5.0 + W))
        g_MHz = e.g01 * 1e3
        n_surf += np.sum((g_MHz >= 0.1) & (g_MHz <= 0.4))
        n_junc += np.sum(g_MHz >= 0.5)
    # literature prior (paper App. E.4): ~2.4-2.5 /GHz surface TLS above 0.1 MHz (Weeden law, T1_TLS = 100 us,
    # alpha = 10, cutoff 0.4 MHz) and ~0.56 /GHz junction TLS above 0.5 MHz (sigma_J = 3, A_J = 0.05)
    assert abs(n_surf / (R * W) - 2.4) < 0.4
    assert abs(n_junc / (R * W) - 0.56) < 0.15
    assert np.isclose(surface_mean_rate_per_us(1e-2, 10.0, 0.01, 0.4), 2e-2 * np.log(40) / 10)
