"""Extract Fig. 7(b) (p.10) of Putterman et al., "Preserving phase coherence and linearity in cat qubits with exponential
bit-flip suppression", PRX 15, 011070 (2025), arXiv:2409.17556v2: bit-flip times under pulsed stabilization.

T_cycle = 3 us; stabilization on for T_on of each cycle. Series:
  static     markers black (static stabilization)
  Ton_1us    markers blue (T_off/T_cycle = 0.67)
  Ton_200ns  markers orange (T_off/T_cycle = 0.93)
The y axis is the bit-flip time 2 T_Z in s (the paper's App. F convention: the inverse Pauli rate, as Ocelot's
physical flip times). Markers: centroids of the marker outlines; error bars: the vertical bar end points (caps).
Dashed lines: AWS simulations of the three protocols (with the white-noise dephasing rate reduced by 2 to match their
static data, K_s/2pi = 1.1 kHz, chi_sb/2pi = 156 kHz; App. G4), dotted brown: their T_on = 1 us simulation with
K_s/2pi = 5 kHz. VALIDATION ONLY (error-rate data; never used for calibration).

Axis calibration from the vector tick marks: x ticks |alpha|^2 = 1 ... 8 at 99.91 + 26.486 (n - 1) pt; log y ticks
1e-4 ... 1e0 s at 511.61 + 30.27 k pt (page coordinates, origin bottom left). Curves are clipped to the axes.

Usage: python tools/extract_companion_fig7b.py literature/2409.17556.pdf data/digitized
Output: companion_fig7b.csv in the given directory
"""
import pathlib
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure, LTLine

PAGE = 9  # zero-based index of printed page 10
X1, X_PER = 99.91, 26.486  # x of |alpha|^2 = 1, pt per unit
Y0, Y_PER = 632.69, 30.27  # y of 1 s, pt per decade
AXES = (94.61, 511.61, 290.6, 653.85)
MARKER_STROKE = {0.0: "static", (0.122, 0.467, 0.706): "Ton_1us", (1.0, 0.498, 0.055): "Ton_200ns"}
CURVES = {0.0: "static", (0.122, 0.467, 0.706): "Ton_1us", (1.0, 0.498, 0.055): "Ton_200ns",
          (0.627, 0.322, 0.176): "Ton_1us_Ks_5kHz"}
LEGEND_Y = 525.0  # legend row (markers at y ~ 520) lies below this


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def _col(c):
    return tuple(round(v, 3) for v in c) if isinstance(c, (tuple, list)) else (round(c, 3) if isinstance(c, float) else c)


def to_data(x, y):
    return 1.0 + (x - X1) / X_PER, 10 ** ((y - Y0) / Y_PER)


def extract(pdf):
    page = next(extract_pages(pdf, page_numbers=[PAGE]))
    markers, bars, curves = [], [], []
    for o in _walk(page):
        x0, y0, x1, y1 = o.bbox
        if not (AXES[0] - 5 < x0 and x1 < AXES[2] + 10 and AXES[1] - 5 < y0 and y1 < AXES[3] + 10):
            continue
        if isinstance(o, LTCurve) and len(o.pts) == 10 and o.stroke and not o.fill and _col(o.stroking_color) in MARKER_STROKE:
            cy = 0.5 * (y0 + y1)
            if cy > LEGEND_Y:
                markers.append((MARKER_STROKE[_col(o.stroking_color)], 0.5 * (x0 + x1), cy))
        elif isinstance(o, LTLine) and o.stroke and not o.fill and abs(x1 - x0) < 0.01 and y1 - y0 > 0.3 \
                and _col(o.stroking_color) in MARKER_STROKE and round(o.linewidth, 2) == 0.43:
            if y0 > LEGEND_Y or y1 > LEGEND_Y + 3:
                bars.append((MARKER_STROKE[_col(o.stroking_color)], x0, y0, y1))
        elif isinstance(o, LTCurve) and o.dashing_style and o.dashing_style[0] and _col(o.stroking_color) in CURVES \
                and len(o.pts) > 40:
            curves.append((CURVES[_col(o.stroking_color)], o.pts))
    rows = []
    for series, cx, cy in markers:
        a2, T = to_data(cx, cy)
        # the error bar of this marker: same series, same x, spanning the marker centre
        cand = [b for b in bars if b[0] == series and abs(b[1] - cx) < 0.3 and b[2] - 1e-6 <= cy <= b[3] + 1e-6]
        lo = hi = np.nan
        if cand:
            _, _, by0, by1 = max(cand, key=lambda b: b[3] - b[2])
            lo, hi = 10 ** ((by0 - Y0) / Y_PER), 10 ** ((by1 - Y0) / Y_PER)
        rows.append(dict(series=series, kind="data", alpha2=round(a2 * 2) / 2, alpha2_raw=a2, T_bitflip_s=T,
                         T_low_s=lo, T_high_s=hi))
    for series, pts in curves:
        for px, py in pts:
            if AXES[0] <= px <= AXES[2] and AXES[1] <= py <= AXES[3]:
                a2, T = to_data(px, py)
                rows.append(dict(series=series, kind="aws_sim", alpha2=a2, alpha2_raw=a2, T_bitflip_s=T,
                                 T_low_s=np.nan, T_high_s=np.nan))
    return rows


def main():
    pdf, out = sys.argv[1], pathlib.Path(sys.argv[2])
    rows = extract(pdf)
    rows.sort(key=lambda r: (r["kind"], r["series"], r["alpha2_raw"]))
    path = out / "companion_fig7b.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Putterman et al., PRX 15, 011070 (2025), arXiv:2409.17556v2, Fig. 7(b) p.10: bit-flip time 2T_Z (s) under "
                 "pulsed stabilization, T_cycle = 3 us; kind data = markers (+ error-bar ends), aws_sim = their simulation "
                 "curves. Vector extraction: tools/extract_companion_fig7b.py. VALIDATION ONLY.\n")
        fh.write("series,kind,alpha2,alpha2_raw,T_bitflip_s,T_low_s,T_high_s\n")
        for r in rows:
            fh.write(f"{r['series']},{r['kind']},{r['alpha2']:.4g},{r['alpha2_raw']:.4f},{r['T_bitflip_s']:.5g},"
                     f"{r['T_low_s']:.5g},{r['T_high_s']:.5g}\n")
    data = [r for r in rows if r["kind"] == "data"]
    for s in ("static", "Ton_1us", "Ton_200ns"):
        sel = sorted((r for r in data if r["series"] == s), key=lambda r: r["alpha2"])
        print(s, len(sel), " ".join(f"{r['alpha2']:g}:{r['T_bitflip_s']:.3g}" for r in sel))
    print("curve points:", {s: sum(1 for r in rows if r["kind"] == "aws_sim" and r["series"] == s) for s in CURVES.values()})
    print("saved", path)


if __name__ == "__main__":
    main()
