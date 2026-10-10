"""Extract the CX gate lengths (vertical dashed lines) of Fig. 12 of Putterman et al., arXiv:2409.13025v2 (page 24).

Each of the 8 panels (storage conditional phase vs flux pulse length) carries two vertical
lines, the CX gate lengths in the g/e and g/f manifolds used in the repetition-code
experiment. Colors: red (1,0,0) and purple (0.502,0,0.502). The caption assigns purple to
g/e and red to g/f; for 7 of 8 panels the ordering of the lines versus the fitted chi_ge,
chi_gf is the opposite, and for A1-S1 the stated CX^2 length (504 ns, p.5) equals
L(purple) + 1/(2 chi_gf) = 503 ns but L(red) + 1/(2 chi_gf) = 491 ns. We therefore record both
lines with their colors and use purple as the g/f gate, as the text on p. 25 does. Protocol (timing)
information only; not error-rate data.

x axis from the tick marks 0.25 ... 1.50 us under each bottom panel (18.578 pt per 0.25 us).
Requires pdfminer.six.

Usage: python tools/extract_ocelot_fig12_gates.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig12_gates.csv
"""
import sys

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure, LTLine

PANELS = {(0, 0): "A1-S1", (0, 1): "A2-S2", (0, 2): "A3-S3", (0, 3): "A4-S4",
          (1, 0): "A1-S2", (1, 1): "A2-S3", (1, 2): "A3-S4", (1, 3): "A4-S5"}
COLORS = {(1.0, 0.0, 0.0): "red", (0.502, 0.0, 0.502): "purple"}


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[23]))  # printed page 24
    H = page.bbox[3]
    ticks, lines = [], []
    for x in _walk(page):
        if not isinstance(x, (LTCurve, LTLine)):
            continue
        xs = [p[0] for p in x.pts]
        ys = [H - p[1] for p in x.pts]
        if max(xs) - min(xs) > 0.05:
            continue
        h = max(ys) - min(ys)
        if 1 < h <= 5 and 240 < min(ys) < 250:
            ticks.append(min(xs))
        c = tuple(round(v, 3) for v in x.stroking_color) if isinstance(x.stroking_color, (list, tuple)) else None
        if h > 50 and c in COLORS:
            lines.append((COLORS[c], min(xs), min(ys)))
    ticks = sorted(set(round(t, 1) for t in ticks))
    first = [t for i, t in enumerate(ticks) if i == 0 or t - ticks[i - 1] > 20]  # x of the 0.25-us tick per column
    assert len(first) == 4, first
    pt_per_us = 18.578 / 0.25
    rows = []
    for color, xpos, ytop in lines:
        col = max(i for i, x0 in enumerate(first) if xpos > x0 - 10)
        row = 0 if ytop < 150 else 1
        L = 0.25 + (xpos - first[col]) / pt_per_us
        rows.append((PANELS[(row, col)], color, L))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 12 of Putterman et al., arXiv:2409.13025v2, p.24: CX gate lengths (us).\n"
                 "# Caption: purple = g/e, red = g/f; the chi ordering and the A1-S1 CX^2 length (504 ns) indicate the opposite.\n")
        fh.write("pair,color,L_CX_us\n")
        for pair, color, L in sorted(rows):
            fh.write(f"{pair},{color},{L:.4f}\n")
    for r in sorted(rows):
        print(r)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
