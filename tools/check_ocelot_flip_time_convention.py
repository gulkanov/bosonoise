"""How Putterman et al. (arXiv:2409.13025v2) define the *physical* cat bit-flip and phase-flip times.

Three vector-extracted checks, all on the A1-S1 CX^2 data of the main text (validation only):

1. Fig. 15(d), p.28 ("example exponential fits for determining the bit-flip times under a CX^2
   gate ... correspond to the data in Fig. 2 with the ancilla prepared in |g> + |f>"): the drawn
   AWS fit curves are exact exponentials |P(+-alpha) - 1/2| = A exp(-t / tau); we read tau.
2. Fig. 2(c), p.4: the reported bit-flip times T_X of the same data (= the A1-S1 panel of Fig. 16).
   Result: T_X = 2 tau, i.e. T_X is the inverse Pauli flip rate 1/Gamma (<Z> ~ exp(-2 Gamma t)),
   not the decay constant of <Z>.
3. Fig. 2(c) inset: phase-flip times T_phi; |alpha|^2 T_phi = T1 of the storage under two-photon
   dissipation (Fig. 9g: S1 = 59 us; main text: T1,eff = 63 +- 2 us). Photon loss flips the
   parity at the Pauli rate kappa_1 |alpha|^2, so T_phi is again the inverse Pauli rate.

Consequence: the Pauli flip probability per cycle of the physical data is
p = [1 - exp(-2 T_cycle / T_X)] / 2 ~= T_cycle / T_X -- the value AWS quote in the text -- while
logical lifetimes (Eq. C2, fitted decay constants of <Z_L>, <X_L>) give eps = T_cycle / (2 T_L).

Usage: python tools/check_ocelot_flip_time_convention.py literature/ocelot_2409.13025.pdf
Output: data/digitized/ocelot_fig2c.csv, data/digitized/ocelot_fig15d_fits.csv (VALIDATION ONLY)
"""
import pathlib
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure, LTLine

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_CYCLE_US = 3.0

# Fig. 2(c) main axes (pt, y measured from the page top): x ticks |alpha|^2 = 1 ... 7, log-y decade ticks
F2_X1, F2_PT_PER_UNIT = 357.79, 30.195
F2_Y_1E4, F2_PT_PER_DECADE = 184.77, 43.165
F2_SERIES = {  # (number of path points, fill colour) -> protocol
    (9, 0.0): "static", (9, (1.0, 0.666, 0.371)): "g", (9, 0.653): "g+e", (5, (0.255, 0.856, 0.917)): "g+f"}
# Fig. 2(c) inset: x ticks 1 ... 4, linear y ticks 0, 20, 40, 60 us
IN_X1, IN_PT_PER_UNIT = 478.59, 20.07
IN_Y0, IN_PT_PER_20US = 278.23, 14.807
# Fig. 15(d): x ticks 0 ... 1000 cycles, y ticks 0.0, 0.5, 1.0
F15_X0, F15_PT_PER_CYCLE = 345.26, 35.18 / 200
F15_Y0, F15_PT_PER_UNIT = 260.71, 57.10
F15_SERIES = {(0.291, 0.595, 0.789): (1.0, "P(+a)"), (0.947, 0.268, 0.196): (1.0, "P(-a)"),
              (0.031, 0.188, 0.42): (3.0, "P(+a)"), (0.404, 0.0, 0.051): (3.0, "P(-a)")}


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def _col(c):
    return tuple(round(v, 3) for v in c) if isinstance(c, (tuple, list)) else c


def fig2c(pdf):
    page = next(extract_pages(pdf, page_numbers=[3]))
    H = page.bbox[3]
    main, inset = {}, {}
    for x in _walk(page):
        if isinstance(x, LTCurve) and x.fill and 3 <= len(x.pts) <= 10:
            xs, ys = [p[0] for p in x.pts], [H - p[1] for p in x.pts]
            if max(xs) - min(xs) > 5:
                continue
            key = (len(x.pts), _col(x.non_stroking_color))
            if key not in F2_SERIES:
                continue
            cx, cy = np.mean(xs), np.mean(ys)
            a2 = 1.0 + (cx - F2_X1) / F2_PT_PER_UNIT
            if not (0.95 < a2 < 4.05 and abs(a2 * 2 - round(a2 * 2)) < 0.02 and 170 < cy < 300):
                continue  # legend, other panels
            main[(F2_SERIES[key], round(a2 * 2) / 2)] = 1e4 * 10 ** ((F2_Y_1E4 - cy) / F2_PT_PER_DECADE)
        if isinstance(x, LTLine) and _col(x.stroking_color) == (0.09, 0.745, 0.812):
            (xa, ya), (xb, yb) = x.pts
            if abs(xa - xb) < 0.05 and 470 < xa < 545:  # inset error bars: centre = marker
                a2 = 1.0 + (xa - IN_X1) / IN_PT_PER_UNIT
                inset[round(a2 * 2) / 2] = (IN_Y0 - (H - 0.5 * (ya + yb))) / IN_PT_PER_20US * 20.0
    return main, inset


def fig15d(pdf):
    page = next(extract_pages(pdf, page_numbers=[27]))
    H = page.bbox[3]
    taus = {}
    for x in _walk(page):
        if isinstance(x, LTCurve) and len(x.pts) in (21, 22) and _col(x.stroking_color) in F15_SERIES:
            xs = np.array([p[0] for p in x.pts])
            ys = np.array([H - p[1] for p in x.pts])
            if not (xs.min() > 336 and 190 < ys.min() and ys.max() < 272):
                continue  # legend lines
            t_us = (xs - F15_X0) / F15_PT_PER_CYCLE * T_CYCLE_US
            d = np.abs((F15_Y0 - ys) / F15_PT_PER_UNIT - 0.5)
            ok = d > 0.02
            slope, icpt = np.polyfit(t_us[ok], np.log(d[ok]), 1)
            if t_us.max() < 0.99 * 1000 * T_CYCLE_US or np.abs(np.log(d[ok]) - slope * t_us[ok] - icpt).max() > 0.01:
                continue  # not one of the four drawn fit curves (they are exact exponentials over 1000 cycles)
            a2, lab = F15_SERIES[_col(x.stroking_color)]
            taus.setdefault(a2, []).append(-1.0 / slope)
    return {a2: float(np.mean(v)) for a2, v in taus.items()}


def main(pdf):
    main_pts, inset = fig2c(pdf)
    taus = fig15d(pdf)
    dd = ROOT / "data" / "digitized"
    with open(dd / "ocelot_fig2c.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted Fig. 2(c) of Putterman et al., arXiv:2409.13025v2, p.4 (A1-S1 CX^2 cycles, 3 us). "
                 "VALIDATION ONLY.\n# T_X = reported bit-flip time (inverse Pauli rate, see tools/"
                 "check_ocelot_flip_time_convention.py); T_phi = inset phase-flip time (ancilla g+f).\n")
        fh.write("quantity,protocol,alpha2,time_us\n")
        for (prot, a2), T in sorted(main_pts.items()):
            fh.write(f"T_X,{prot},{a2:.1f},{T:.5g}\n")
        for a2, T in sorted(inset.items()):
            fh.write(f"T_phi,g+f,{a2:.1f},{T:.4g}\n")
    with open(dd / "ocelot_fig15d_fits.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Decay constants of the AWS exponential fit curves drawn in Fig. 15(d), p.28 (A1-S1, ancilla g+f, "
                 "|P(+-alpha) - 1/2| ~ exp(-t/tau)), vs the bit-flip time T_X reported for the same data in Fig. 2(c). "
                 "VALIDATION ONLY.\n")
        fh.write("alpha2,tau_fit_us,T_X_fig2c_us,ratio\n")
        for a2, tau in sorted(taus.items()):
            T = main_pts[("g+f", a2)]
            fh.write(f"{a2:.1f},{tau:.4g},{T:.4g},{T / tau:.3f}\n")
    print("Fig. 15(d) fit decay constants vs Fig. 2(c) bit-flip times (A1-S1, g+f):")
    for a2, tau in sorted(taus.items()):
        T = main_pts[("g+f", a2)]
        print(f"  |alpha|^2 = {a2:.0f}: tau = {tau:.0f} us, T_X = {T:.0f} us, T_X / tau = {T / tau:.2f}")
    print("Fig. 2(c) inset: |alpha|^2 * T_phi (should equal the storage T1 under dissipation, Fig. 9g S1 = 59 us):")
    print("  " + ", ".join(f"{a2:.1f}: {a2 * T:.1f} us" for a2, T in sorted(inset.items())))
    print(f"  mean {np.mean([a2 * T for a2, T in inset.items()]):.1f} us (main text: T1,eff = 63 +- 2 us)")


if __name__ == "__main__":
    main(sys.argv[1])
