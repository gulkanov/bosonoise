"""Phenomenological ("standard model") CX bit-flip baselines.

Hann et al., "Hybrid cat-transmon architecture for scalable, hardware-efficient quantum
error correction", PRX Quantum 6, 030305 (2025), arXiv:2410.23363. For a chi-matched CX
in the (g, f) manifold with gate time pi/chi, their Eq. (15) (p. 10 of the published
version; Eq. (11) of arXiv v1) gives the large-|alpha|^2 lower bound on the cat bit-flip
probability per CX:

    p_bit,CX >= 1/2 [ (gamma_up/2)(pi/chi) + ((gamma_down/2)(pi/chi))^2 ]

(first term: transmon heating; second: double decay). It is independent of |alpha|^2.
Rates in 1/ns, chi in rad/ns (angular). We use it as the baseline against which
photon-number-dependent mechanisms are compared.
"""
from __future__ import annotations

import numpy as np


def hann_cx_bitflip_bound(chi_MHz: float, T1_us: float, gamma_up_per_us: float = 0.0) -> dict:
    """Eq. (15) of Hann et al. (PRX Quantum 6, 030305); chi_MHz = |chi|/2pi, T1 = 1/gamma_down."""
    chi = 2 * np.pi * abs(chi_MHz) * 1e-3  # rad/ns
    T = np.pi / chi  # ns
    g_dn = 1e-3 / T1_us
    g_up = gamma_up_per_us * 1e-3
    heating = 0.5 * (g_up / 2) * T
    double_decay = 0.5 * ((g_dn / 2) * T) ** 2
    return dict(gate_ns=T, heating=heating, double_decay=double_decay, total=heating + double_decay)
