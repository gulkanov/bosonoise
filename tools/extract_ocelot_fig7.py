"""Extract the vector data of Fig. 7 of Putterman et al., arXiv:2409.13025v2 (page 19).

Fig. 7 is embedded in the PDF as vector graphics (matplotlib output), so the curves
can be recovered exactly: each curve is a polyline of 91 points. The axis mapping is
reconstructed from the major tick marks, whose values were read off the rendered page
(tick labels are drawn as glyph outlines, not as text).

NOTE: Fig. 7 shows the *model* predictions of AWS's circuit-quantization (DMRG-X) model
for the A2-S3 CX gate, whose parameters were tuned by AWS to the measured spectrum.
These are not raw measurements.

Requires pdfminer.six (MIT license; not a dependency of the bosonoise package).

Usage:  python tools/extract_ocelot_fig7.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig7.csv
"""
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure

RED = (0.886, 0.29, 0.2)
BLUE = (0.204, 0.541, 0.741)
PURPLE = (0.596, 0.557, 0.835)

# x axis (shared): major ticks 0.25 ... 0.50 at these PDF x-coordinates
X_TICKS_PDF = [377.93, 411.48, 445.03, 478.58, 512.12, 545.67]
X_TICKS_VAL = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# y axes: (major tick PDF y-coordinates, tick values), read from the rendered figure
PANELS = {
    "a": dict(ypdf=[125.84, 104.22, 82.60, 60.98], yval=[4.8, 5.0, 5.2, 5.4], yrange=(60, 132),
              curves={RED: "omega_s_GHz", BLUE: "omega_a_ge_GHz", PURPLE: "omega_a_ef_GHz"}),
    "b": dict(ypdf=[199.64, 179.56, 159.49, 139.41], yval=[1.0, 1.1, 1.2, 1.3], yrange=(139, 210),
              curves={RED: "chi_ratio_numerics", BLUE: "chi_ratio_naive_D5"}),
    "c": dict(ypdf=[279.23, 249.27, 219.31], yval=[-1.0, -0.5, 0.0], yrange=(217, 289),
              curves={RED: "chi_ge_MHz", BLUE: "chi_gf_MHz"}),
    "d": dict(ypdf=[366.55, 348.98, 331.41, 313.84, 296.28], yval=[-5.0, -2.5, 0.0, 2.5, 5.0],
              yrange=(296, 367),
              curves={RED: "K_s_g_kHz", BLUE: "K_s_e_kHz", PURPLE: "K_s_f_kHz"}),
}


def _col(c):
    return tuple(round(v, 3) for v in c) if c else None


def linmap(ref_pdf, ref_val):
    p = np.polyfit(ref_pdf, ref_val, 1)
    resid = np.polyval(p, ref_pdf) - np.asarray(ref_val)
    assert np.max(np.abs(resid)) < 1e-3 * (max(ref_val) - min(ref_val)), "non-linear axis?"
    return lambda z: np.polyval(p, z)


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def extract(pdf_path):
    page = next(extract_pages(pdf_path, page_numbers=[18]))  # printed page 19
    H = page.bbox[3]  # pdfminer y axis points up; tick constants use a top-down y axis
    fx = linmap(X_TICKS_PDF, X_TICKS_VAL)
    out = {}
    for x in _walk(page):
        if not (isinstance(x, LTCurve) and x.stroke and isinstance(x.stroking_color, (list, tuple))):
            continue
        c = _col(x.stroking_color)
        if c not in (RED, BLUE, PURPLE) or len(x.pts) < 20:  # skip markers and legend handles
            continue
        pts = np.array([(px, H - py) for px, py in x.pts])
        ymid = pts[:, 1].mean()
        for pan in PANELS.values():
            if pan["yrange"][0] <= ymid <= pan["yrange"][1]:
                fy = linmap(pan["ypdf"], pan["yval"])
                out[pan["curves"][c]] = (fx(pts[:, 0]), fy(pts[:, 1]))
    return out


def main(pdf_path, csv_path):
    curves = extract(pdf_path)
    names = [n for pan in PANELS.values() for n in pan["curves"].values()]
    missing = [n for n in names if n not in curves]
    assert not missing, f"missing curves: {missing}"
    x0 = curves[names[0]][0]
    for n in names:
        assert len(curves[n][0]) == len(x0) and np.allclose(curves[n][0], x0, atol=1e-4), n
    header = (
        "# Vector-extracted from Fig. 7 of Putterman et al., arXiv:2409.13025v2, p. 19 (A2-S3 CX gate).\n"
        "# These are AWS circuit-quantization-model predictions (not raw data). Extracted by\n"
        "# tools/extract_ocelot_fig7.py. Axis mapping from major tick marks; precision ~0.1% of axis span.\n"
        "# flux in units of Phi_0; frequencies in GHz (omega/2pi); chi in MHz (chi/2pi); Kerr in kHz (K/2pi).\n"
        "# Legend in the figure labels chi_gf as 'chi_ef'; the caption and text define it as chi_gf.\n"
    )
    cols = ["flux_Phi0"] + names
    data = np.column_stack([x0] + [curves[n][1] for n in names])
    with open(csv_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        fh.write(",".join(cols) + "\n")
        for row in data:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
    print(f"wrote {csv_path}: {data.shape[0]} rows")
    for i in (0, len(x0) // 2, -1):
        print({c: round(float(v), 5) for c, v in zip(cols, data[i])})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
