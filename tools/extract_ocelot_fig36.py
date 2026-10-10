"""Extract the vector data of Fig. 36 of Putterman et al., arXiv:2409.13025v2 (page 52).

Fig. 36: CX^2 bit-flip probability per cycle for the A4-S5 gate with initial ancilla state
|f>, versus |alpha|^2 = 1.0 ... 4.0: measured values (black markers with error bars) and AWS
master-equation simulations in which the independently measured ancilla T1 values are
divided by a factor beta = 1.0, 1.5, 2.0, 2.5 (coloured curves). VALIDATION DATA ONLY.

Axis mapping: x from the 7 equally spaced tick marks (|alpha|^2 = 1.0 ... 4.0); y is
logarithmic, from the 1e-3 and 1e-2 major ticks (a minor tick at 2e-3 checks the mapping).
Data values are the centres of the (linear-space symmetric) error bars.
Requires pdfminer.six (MIT license; not a dependency of the bosonoise package).

Usage: python tools/extract_ocelot_fig36.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig36.csv
"""
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure, LTLine

X_TICKS = ([104.66, 134.36, 164.05, 193.74, 223.44, 253.13, 282.82], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
Y_MAJOR = ((153.67, -3.0), (70.55, -2.0))  # (top-down pdf y, log10 value)
BETA = {(0.122, 0.467, 0.706): 1.0, (1.0, 0.498, 0.055): 1.5, (0.173, 0.627, 0.173): 2.0, (0.839, 0.153, 0.157): 2.5}


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def ylog(Y):
    (y1, l1), (y2, l2) = Y_MAJOR
    return 10 ** (l1 + (Y - y1) * (l2 - l1) / (y2 - y1))


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[51]))  # printed page 52
    H = page.bbox[3]
    assert abs(ylog(128.65) - 2e-3) / 2e-3 < 0.01, "log-axis check failed"
    px = np.polyfit(X_TICKS[0], X_TICKS[1], 1)
    curves, bars = {}, []
    for x in _walk(page):
        if isinstance(x, LTCurve) and x.stroke and isinstance(x.stroking_color, (list, tuple)) and len(x.pts) == 7:
            c = tuple(round(v, 3) for v in x.stroking_color)
            if c in BETA:
                pts = np.array([(a, H - b) for a, b in x.pts])
                curves[BETA[c]] = (np.polyval(px, pts[:, 0]), ylog(pts[:, 1]))
        if isinstance(x, LTLine) and abs(x.linewidth - 1.02) < 0.05 and not isinstance(x.stroking_color, (list, tuple)):
            (x0, y0), (x1, y1) = x.pts
            if abs(x0 - x1) < 0.01 and x0 < 300 and 60 < H - y0 < 170:
                bars.append((np.polyval(px, x0), ylog(H - y0), ylog(H - y1)))
    assert set(curves) == set(BETA.values()) and len(bars) == 7, (curves.keys(), len(bars))
    bars.sort()
    a2 = np.array([b[0] for b in bars])
    lo = np.array([min(b[1], b[2]) for b in bars])
    hi = np.array([max(b[1], b[2]) for b in bars])
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 36 of Putterman et al., arXiv:2409.13025v2, p.52 (A4-S5, CX^2, ancilla |f>).\n"
                 "# data = centre of the error bar (linear space); sim_beta* = AWS simulations with ancilla T1 / beta.\n"
                 "# VALIDATION ONLY. Extracted by tools/extract_ocelot_fig36.py.\n")
        fh.write("alpha2,data,data_err," + ",".join(f"sim_beta{b:g}" for b in sorted(curves)) + "\n")
        for i in range(7):
            row = [a2[i], 0.5 * (lo[i] + hi[i]), 0.5 * (hi[i] - lo[i])] + [np.interp(a2[i], *curves[b]) for b in sorted(curves)]
            fh.write(",".join(f"{v:.5g}" for v in row) + "\n")
    print(f"wrote {out}")
    for i in range(7):
        print(f"  |alpha|^2={a2[i]:.1f}: data {0.5 * (lo[i] + hi[i]):.2e} +- {0.5 * (hi[i] - lo[i]):.1e}; "
              + ", ".join(f"beta{b:g}={np.interp(a2[i], *curves[b]):.2e}" for b in sorted(curves)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
