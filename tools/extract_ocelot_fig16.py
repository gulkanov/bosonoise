"""Extract the vector data of Fig. 16 of Putterman et al., arXiv:2409.13025v2 (page 29).

Fig. 16 (top row): cat bit-flip time T_X versus |alpha|^2 = 1.0 ... 4.0 under repeated CX^2
cycles (3 us) for all 8 storage-ancilla pairs, for five protocols: static stabilization
(black), ancilla prepared in |g> (orange), |g>+|e> (grey), |f> (green), |g>+|f> (cyan).
Data were taken ~5 days after the repetition-code run. VALIDATION DATA ONLY.

Marker centroids give the data points (error bars not extracted). Axes: log y from the
decade ticks (10^-2 ... 10^-5 at fixed page coordinates, checked on the 2e-3 minor tick);
x from the |alpha|^2 = 1 and 3 ticks of each panel. The reported bit-flip time is the inverse
Pauli flip rate, T_X = 1/Gamma = 2x the decay constant of <Z> (AWS fit curves of Fig. 15(d) vs
Fig. 2(c); tools/check_ocelot_flip_time_convention.py). The Pauli flip probability per 3-us
cycle is therefore p = [1 - exp(-2 T_cycle / T_X)] / 2 ~= T_cycle / T_X, the number AWS quote (paper App. C.2).
Requires pdfminer.six (MIT license; not a dependency of the bosonoise package).

Usage: python tools/extract_ocelot_fig16.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig16.csv
"""
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure

PANELS = ["A1-S1", "A1-S2", "A2-S2", "A2-S3", "A3-S3", "A3-S4", "A4-S4", "A4-S5"]
FRAME_X = [(110.08, 160.83), (163.37, 214.13), (216.67, 267.42), (269.96, 320.72),
           (323.26, 374.01), (376.55, 427.31), (429.85, 480.6), (483.14, 533.9)]
FRAME_Y = (62.95, 144.07)  # top row (bit-flip times), top-down page coordinates
Y_DECADE = (77.46, -2.0, 17.885)  # y of 1e-2, its log10, points per decade
X_UNIT = 15.385  # points per unit |alpha|^2 (ticks at 1 and 3 are 30.77 pt apart)
COLORS = {(0.0, 10): "static", ((1.0, 0.498, 0.055), 10): "g", (0.502, 10): "g+e",
          ((0.173, 0.627, 0.173), 10): "f", ((0.09, 0.745, 0.812), 5): "g+f"}
T_CYCLE = 3e-6


def p_cycle(T_X, T_cycle=T_CYCLE):
    """Pauli flip probability per cycle for an inverse Pauli rate T_X: <Z> ~ (1 - 2p)^n = exp(-2 n T_cycle / T_X)."""
    return 0.5 * (1.0 - np.exp(-2.0 * T_cycle / T_X))


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[28]))  # printed page 29
    H = page.bbox[3]
    y0, l0, dec = Y_DECADE
    assert abs(10 ** (l0 - 1 + (y0 + dec - 89.96) / dec) - 2e-3) < 2e-5  # minor tick at 2e-3
    pts = {}
    for x in _walk(page):
        if not (isinstance(x, LTCurve) and x.stroke and len(x.pts) in (5, 10)):
            continue
        c = tuple(round(v, 3) for v in x.stroking_color) if isinstance(x.stroking_color, (list, tuple)) else x.stroking_color
        key = COLORS.get((c, len(x.pts)))
        if key is None:
            continue
        xs = np.array([p[0] for p in x.pts])
        ys = np.array([H - p[1] for p in x.pts])
        if len(xs) == 10:
            xs, ys = xs[:-1], ys[:-1]  # closed marker path
        cx, cy = xs.mean(), ys.mean()
        if not (FRAME_Y[0] < cy < FRAME_Y[1]):
            continue
        for i, (xa, xb) in enumerate(FRAME_X):
            if xa < cx < xb:
                a2 = 1.0 + (cx - (xa + 2.30)) / X_UNIT  # the |alpha|^2 = 1 tick sits 2.30 pt inside each frame
                T = 10 ** (l0 - (cy - y0) / dec)
                pts.setdefault((PANELS[i], key), []).append((round(a2 * 2) / 2, T))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 16 of Putterman et al., arXiv:2409.13025v2, p.29. VALIDATION ONLY.\n"
                 "# T_X in s (marker centroids) = inverse Pauli flip rate; p_cycle = [1 - exp(-2 * 3 us / T_X)] / 2 = "
                 "Pauli flip probability per 3-us cycle. Extracted by tools/extract_ocelot_fig16.py.\n")
        fh.write("pair,protocol,alpha2,T_X_s,p_cycle\n")
        for (pair, prot) in sorted(pts):
            for a2, T in sorted(pts[(pair, prot)]):
                fh.write(f"{pair},{prot},{a2:.1f},{T:.5g},{p_cycle(T):.5g}\n")
    print(f"wrote {out}")
    for pair in PANELS:
        counts = {prot: len(pts.get((pair, prot), [])) for prot in COLORS.values()}
        print(f"  {pair}: {counts}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
