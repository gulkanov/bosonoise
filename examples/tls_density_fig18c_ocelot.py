"""TLS density of Ocelot's own ancillas from the T1 dips of Fig. 18c, and the probability of an F9-sized anomaly
(paper App. E.4, Table 6).

Fig. 18c (A3 on the A3-S3 coupler, another cooldown) samples the ancilla T1 at 31 coupler fluxes. Through the coupler's
Lamb shift the e-g frequency moves by 43 MHz (5.210 -> 5.167 GHz in our A3 calibration), so the scan is a coarse TLS
spectroscopy. Three points fall below half of the smooth trend (Phi = 0.233, 0.450, 0.483 Phi0).

Forward model, nothing fitted to F9 or to error data:
  * smooth T1(Phi): the coupler-participation model of examples/measured_bath_cx2_ocelot.py (coupler T1 2.37 us at 'on',
    reproducing the median idle/on ratio of the same data);
  * TLS: Poisson number at density rho per GHz, uniform in [5.14, 5.24] GHz, coupling g01 log-uniform 0.02-2 MHz and TLS
    T1 log-uniform 0.3-10 us (the broad prior of the F9 ensembles, examples/tls_ensemble_ocelot.py); the extra decay at
    each sampled frequency from the exact two-level transfer (channels/tls.two_level_transfer) over a time equal to the
    smooth T1, converted to a rate;
  * a dip = T1 below half of the smooth trend. The observed count is compared with the simulated distribution.
The density range consistent with the data is then fed to the F9 ensemble (A4-S4, 800 ns, same prior).

Usage: python examples/tls_density_fig18c_ocelot.py [--nreal 4000]
Output: data/tls/density_fig18c.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np
from scipy.interpolate import PchipInterpolator

from bosonoise.channels.tls import TLSEnsemble, sample_tls_literature, two_level_transfer
from bosonoise.presets import load_preset

from measured_bath_cx2_ocelot import coupler_T1_from_18c  # same directory
from predict_cx2_ule_ocelot import grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tls" / "density_fig18c.csv"
WINDOW = (5.14, 5.24)
DIP = 0.5


def fig18c():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig18c.csv", encoding="utf-8")
                               if not l.startswith("#")))
    fl = np.array([float(r["flux_Phi0"]) for r in rows])
    T1 = np.array([float(r["T1_us"]) for r in rows])
    err = np.array([float(r["T1_err_us"]) for r in rows])
    o = np.argsort(fl)
    return fl[o], T1[o], err[o]


def smooth_model(pre, flux):
    """f_ge(Phi) and the smooth T1(Phi) of the A3-S3 coupler-participation model (coupler T1 from Fig. 18c)."""
    T1c, idle18, on18, target = coupler_T1_from_18c(pre)
    grid = grid_for("A3-S3")
    fl = np.array([l.flux for l in grid.ladders])
    fge = PchipInterpolator(fl, [l.E[1, 0] - l.E[0, 0] for l in grid.ladders])(flux)
    Xq = np.array([l.X["q"][0, 0, 1, 0] for l in grid.ladders])
    Xc = np.array([l.X["c"][0, 0, 1, 0] for l in grid.ladders])
    T1ge = pre["ancilla"]["A3"]["T1_ge_us"]
    part = (Xc / grid.ladders[-1].nc01) ** 2
    G_q = Xq ** 2 / (Xq[0] ** 2 * T1ge)
    G_c = part / T1c
    scale = 1.0 - G_c[0] * T1ge
    T1s = 1.0 / (scale * G_q + G_c)
    # the participation model is normalized to the measured idle median: rescale to the data's idle level
    T1s *= idle18 / T1s[0]
    return fge, PchipInterpolator(fl, T1s)(flux), T1c


def tls_rates(tls: TLSEnsemble, f_k, t_us):
    """Extra decay rate (1/us) of the e-g transition at frequencies f_k from the TLS of one realization."""
    if tls.size == 0:
        return np.zeros_like(f_k)
    V = 2 * np.pi * tls.g01[None, :]                     # rad/ns (g01 in GHz)
    D = 2 * np.pi * (f_k[:, None] - tls.f[None, :])      # rad/ns
    t = (t_us * 1e3)[:, None]                            # ns
    P = two_level_transfer(V, D, tls.gamma[None, :], t)
    rate = -np.log(np.clip(1 - P, 1e-300, 1.0)) / t     # 1/ns
    return 1e3 * rate.sum(axis=1)


def sample(rng, rho):
    N = rng.poisson(rho * (WINDOW[1] - WINDOW[0]))
    f = rng.uniform(*WINDOW, N)
    g = np.exp(rng.uniform(np.log(2e-5), np.log(2e-3), N))          # GHz (0.02-2 MHz)
    T1 = np.exp(rng.uniform(np.log(300.0), np.log(1e4), N))          # ns
    return TLSEnsemble(f, g, 1.0 / T1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nreal", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261001)
    a = ap.parse_args()
    pre = load_preset("ocelot")
    flux, T1, err = fig18c()
    f_k, T1s, T1c = smooth_model(pre, flux)
    ok = err < 0.5 * T1
    ratio = T1 / T1s
    dips_obs = int(np.sum(ok & (ratio < DIP)))
    print(f"A3 e-g frequency {f_k.max():.4f} -> {f_k.min():.4f} GHz over the scan ({1e3 * (f_k.max() - f_k.min()):.0f} MHz); "
          f"smooth T1 {T1s[0]:.1f} -> {T1s[-1]:.1f} us (coupler T1 {T1c:.2f} us)")
    for fl, f, t, s, o in zip(flux, f_k, T1, T1s, ok):
        if o and t / s < DIP:
            print(f"  dip at Phi = {fl:.4f} ({f:.4f} GHz): T1 {t:.1f} us, {t / s:.2f} of the smooth trend")
    print(f"observed dips (T1 < {DIP} x smooth, reliable points): {dips_obs} of {int(ok.sum())} points")

    rng = np.random.default_rng(a.seed)
    rows = []
    # broad prior: density rho per GHz; literature prior (surface + junction TLS of paper App. E.4, the defaults
    # of tls_ensemble_ocelot.py --prior literature: surface TLS-limited T1 100 us, junction area 0.05 um^2) scaled by m
    grid = [("broad", rho, (lambda r=rho: sample(rng, r))) for rho in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)]
    grid += [("literature", m, (lambda m=m: sample_tls_literature(rng, WINDOW, surf_Gamma1_per_us=1e-2 * m,
                                                                     junc_area_um2=0.05 * m)))
             for m in (1.0, 3.0, 10.0, 30.0, 100.0)]
    idx = np.flatnonzero(ok)                      # reliable points in flux order
    for prior, scale, draw in grid:
        counts = np.empty(a.nreal, int)
        isolated = np.zeros(a.nreal, bool)        # no two dips at neighbouring reliable points, as observed
        for i in range(a.nreal):
            tls = draw()
            G = 1.0 / T1s + tls_rates(tls, f_k, T1s)
            d = ((1.0 / G) / T1s < DIP)[idx]
            counts[i] = int(d.sum())
            isolated[i] = not np.any(d[1:] & d[:-1])
        p_eq = float(np.mean(counts == dips_obs))
        p_ge = float(np.mean(counts >= dips_obs))
        p_iso = float(np.mean((counts == dips_obs) & isolated))
        rows.append(dict(prior=prior, scale=scale, mean_dips=float(counts.mean()), P_equal_observed=p_eq,
                         P_at_least_observed=p_ge, P_equal_observed_isolated=p_iso))
        unit = "/GHz" if prior == "broad" else " x literature"
        print(f"{prior:10s} {scale:6.1f}{unit}: mean dips {counts.mean():5.2f}; P(N = {dips_obs}) = {p_eq:.3f}; "
              f"P(N >= {dips_obs}) = {p_ge:.3f}; P(N = {dips_obs}, isolated) = {p_iso:.3f}")
    keys = list(rows[0])
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# TLS density from the T1 dips of Ocelot Fig. 18c (A3): observed {dips_obs} dips below {DIP} x the smooth "
                 "trend; simulated with the broad prior (scale = TLS per GHz; g01 0.02-2 MHz, T1_TLS 0.3-10 us) and the "
                 "literature prior (scale = multiplier of its densities). examples/tls_density_fig18c_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(r[k] if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
