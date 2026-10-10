"""Ramp-induced storage<->ancilla exchange in the CX^2 of all 8 Ocelot pairs (first-order APT).

The spectral content of the coupler flux ramp at the storage-ancilla detuning converts
storage photons into ancilla excitations (Ocelot App. D3/F7; paper App. C.4): |g,n> -> |e,n-1>
(parity-even: driven by the cos(phi_c) part of dH/dPhi in the irrotational gauge) and
|f,n> -> |e,n+1>. To first order the amplitudes scale as sqrt(n) and sqrt(n+1).

In a chi-matched CX^2 an exchange at the ramp-UP leaves the ancilla in |e> for the whole
gate, so the storage turns by 2 pi r (r = chi_ge/chi_gf) instead of 0 (g protocol) or 2 pi
(f protocol): the error angle is 2 pi (r - 1). An exchange at the ramp-DOWN happens after
the rotation and is harmless for the storage. We therefore use the up-ramp amplitude alone:
    p_ramp(g) ~ nbar P_up(|g,1> -> |e,0>) P_flip(2 pi (r-1)),
    p_ramp(f) ~ (nbar+1) P_up(|f,0> -> |e,1>) P_flip(2 pi (r-1)),
with the asymmetric re-stabilization map (App. F8). Ramp: flat-top Gaussian edge (3 sigma) of
the length solved in predict_cx2_ule_ocelot.py (from the Fig. 12 gate lengths), and the
stated 64 ns for A1-S1 and A4-S4. The waveform is not published: the numbers are indicative
(the exchange is exponentially sensitive to sigma * detuning).

Usage: python examples/ramp_exchange_cx2_ocelot.py
Output: data/validation/cx2_ramp_exchange.csv (model only)
"""
from __future__ import annotations

import csv
import pathlib
import time

import numpy as np

from bosonoise.channels.conversion import AsymptoticProjector
from bosonoise.dynamics.apt import adiabatic_path, apt_amplitudes
from bosonoise.dynamics.cell_td import TimeDependentCell
from bosonoise.dynamics.flux import flat_top_gaussian
from bosonoise.model import Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.presets import load_preset

from predict_cx2_ule_ocelot import DELTA_B_MHZ, PAIRS, RAMP_STATED  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBARS = [2.0, 3.0, 4.0]


def up_ramp_probability(td, path, target_bare, sigma):
    """|c|^2 after a Gaussian up-ramp (0 -> Phi0/2, 3 sigma) followed by a 60-ns hold at 'on'."""
    E0, V0 = td.eigh(0.0)
    k_idx = int(np.argmax(np.abs(V0[td.static.basis.index[target_bare], :]) ** 2))
    hit = np.nonzero(path.targets0 == k_idx)[0]
    if len(hit) == 0:
        return np.nan
    pulse = lambda t: flat_top_gaussian(t, 0.0, 1e6, sigma, 0.5, n_sigma=3.0)  # noqa: E731
    t = np.arange(0.0, 3 * sigma + 60.0, min(0.01, sigma / 50))
    return float(abs(apt_amplitudes(path, pulse, t)[hit[0]]) ** 2)


def main():
    pre = load_preset("ocelot")
    ule = {(r["pair"], float(r["nbar"])): r for r in csv.DictReader(
        l for l in open(ROOT / "data" / "validation" / "cx2_ule_excess.csv", encoding="utf-8") if not l.startswith("#"))}
    rows = []
    for pair in PAIRS:
        anc, st = pair.split("-")
        t0 = time.time()
        p = load_cell_params(ROOT / "data" / "calibration" / f"ocelot_{anc}_fcmin6.569.yaml", st)
        td = TimeDependentCell(p, Truncation(n_s=4, n_c=12, n_q=5, ncut=25), flux_ref=0.25)
        E0, V0 = td.eigh(0.0)
        idx = lambda lab: int(np.argmax(np.abs(V0[td.static.basis.index[lab], :]) ** 2))  # noqa: E731
        path_g = adiabatic_path(td, np.linspace(0, 0.5, 201), idx((1, 0, 0)), n_targets=10)
        path_f = adiabatic_path(td, np.linspace(0, 0.5, 201), idx((0, 0, 2)), n_targets=10)
        r = float(ule[(pair, 2.0)]["chi_ratio"])
        kr = -DELTA_B_MHZ / pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"]
        ramps = {"solved": float(ule[(pair, 2.0)]["ramp_ns"])}
        if pair in RAMP_STATED:
            ramps["stated"] = RAMP_STATED[pair]
        for tag, ramp in ramps.items():
            s = ramp / 3.0
            Pg = up_ramp_probability(td, path_g, (0, 0, 1), s)
            Pf = up_ramp_probability(td, path_f, (1, 0, 1), s)
            for nb in NBARS:
                eta = float(AsymptoticProjector(nb, kerr_ratio=kr).flip_probability([2 * np.pi * (r - 1)])[0])
                rows.append(dict(pair=pair, ramp=tag, ramp_ns=ramp, sigma_ns=s, r=r, nbar=nb, P_up_g1e0=Pg, P_up_f0e1=Pf,
                                 eta=eta, p_ramp_g=nb * Pg * eta, p_ramp_f=(nb + 1) * Pf * eta,
                                 data_gexcess=float(ule[(pair, nb)]["data_gexcess"])))
        det_g = (E0[idx((1, 0, 0))] - E0[idx((0, 0, 1))]) * 1e3
        print(f"{pair}: idle |g,1>-|e,0> {det_g:+.1f} MHz; " + "; ".join(
            f"{tag} ramp {ramp:.1f} ns: P_up(g1->e0) {x['P_up_g1e0']:.1e}, P_up(f0->e1) {x['P_up_f0e1']:.1e}, "
            f"eta(2pi(r-1)) at nbar=4 {x['eta']:.1e}"
            for tag, ramp in ramps.items() for x in [next(y for y in rows[::-1] if y['pair'] == pair and y['ramp'] == tag)])
            + f" ({time.time() - t0:.0f} s)", flush=True)
    path = ROOT / "data" / "validation" / "cx2_ramp_exchange.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Ramp-induced exchange (first-order APT, up-ramp) and its CX^2 bit-flip contribution. "
                 "examples/ramp_exchange_cx2_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.5g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print("\npair  ramp  nbar | p_ramp(g) p_ramp(f) x1e-3 | data g-excess")
    for r in rows:
        print(f"{r['pair']} {r['ramp']:6s} {r['nbar']:.0f} | {r['p_ramp_g'] * 1e3:.2e} {r['p_ramp_f'] * 1e3:.2e} | {r['data_gexcess'] * 1e3:.2f}")
    print("saved", path)


if __name__ == "__main__":
    main()
