"""Extract Fig. 37d of Putterman et al., arXiv:2409.13025v2 (page 53): ancilla A4 T1 on the A4-S5 coupler.

Six series of repeated T1 measurements (10 iterations each; App. J2): T1 of |e> (blue) and
|f> (orange) with (solid) the coupler held at the 'on' position, (dashed, dash pattern
[3.881, 1.678]) the coupler pulsed to 'on' for one CX^2 length every 1.6 us, (dotted,
[1.049, 1.731]) the same with the storage prepared at |alpha|^2 = 4. The connecting polylines
have one vertex per data point. y axis: ticks 0 ... 60 us, 15.18 pt per 10 us (0 at y = 276.78
top-down). VALIDATION ONLY.

Usage: python tools/extract_ocelot_fig37d.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig37d.csv
"""
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure

Y0, PT_PER_10US = 276.78, 15.18
X0, PT_PER_ITER = 348.05, 22.38
COLORS = {(0.122, 0.467, 0.706): "e", (1.0, 0.498, 0.055): "f"}
DASH = {(): "coupler_on", (3.881, 1.678): "pulsed", (1.049, 1.731): "pulsed_alpha2_4"}


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[52]))  # printed page 53
    H = page.bbox[3]
    series = {}
    for x in _walk(page):
        if not (isinstance(x, LTCurve) and x.stroke and len(x.pts) == 10):
            continue
        xs = np.array([p[0] for p in x.pts])
        if not (abs(xs[0] - X0) < 0.1 and np.allclose(np.diff(xs), PT_PER_ITER, atol=0.05)):
            continue  # markers and other panels
        c = tuple(round(v, 3) for v in x.stroking_color)
        dash = tuple(round(v, 3) for v in (x.dashing_style[0] if x.dashing_style else []))
        key = (COLORS[c], DASH[dash])
        series[key] = (Y0 - (H - np.array([p[1] for p in x.pts]))) / PT_PER_10US * 10.0
    assert len(series) == 6, series.keys()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 37d of Putterman et al., arXiv:2409.13025v2, p.53 (A4, A4-S5 coupler). "
                 "VALIDATION ONLY.\nstate,condition,iteration,T1_us\n")
        for (state, cond), T1 in sorted(series.items()):
            for i, v in enumerate(T1):
                fh.write(f"{state},{cond},{i},{v:.2f}\n")
    for (state, cond), T1 in sorted(series.items()):
        print(f"T1({state}) {cond:16s}: mean {T1.mean():.1f} us, std {T1.std(ddof=1):.1f} us, values {np.round(T1, 1)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
