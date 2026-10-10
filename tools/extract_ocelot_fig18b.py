"""Extract the vector data of Fig. 18(b) of Putterman et al., arXiv:2409.13025v2 (page 31).

Fig. 18(b): A4-S4 interaction, ancilla prepared in |f>, 800 ns coupler flux pulse to the
'on' position, storage in a coherent state with mean photon number |alpha|^2; classified
populations of |g>, |e>, |f> versus |alpha|^2 (0..20). Measured in a DIFFERENT cooldown
than the repetition-code run (App. F9). VALIDATION DATA ONLY -- never used for calibration.

The markers' connecting polylines (101 points each) are extracted; error bars are not.
Requires pdfminer.six (MIT license; not a dependency of the bosonoise package).

Usage:  python tools/extract_ocelot_fig18b.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig18b.csv
"""
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure

BLUE, ORANGE, GREEN = (0.122, 0.467, 0.706), (1.0, 0.498, 0.055), (0.173, 0.627, 0.173)
NAMES = {BLUE: "P_g", ORANGE: "P_e", GREEN: "P_f"}
# axis ticks (top-down page coordinates, points) read from the vector tick marks
X_TICKS = ([362.71, 399.05, 435.40, 471.74, 508.08], [0, 5, 10, 15, 20])
Y_TICKS = ([386.15, 366.49, 346.84, 327.18], [0.0, 0.25, 0.50, 0.75])
PANEL = (355.45, 309.81, 515.34, 388.88)  # x0, y_top, x1, y_bottom of panel (b)


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def _lin(ref_pdf, ref_val):
    p = np.polyfit(ref_pdf, ref_val, 1)
    assert np.max(np.abs(np.polyval(p, ref_pdf) - np.asarray(ref_val))) < 1e-3 * (max(ref_val) - min(ref_val))
    return lambda z: np.polyval(p, z)


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[30]))  # printed page 31
    H = page.bbox[3]
    fx, fy = _lin(*X_TICKS), _lin(*Y_TICKS)
    curves = {}
    for x in _walk(page):
        if not (isinstance(x, LTCurve) and x.stroke and isinstance(x.stroking_color, (list, tuple))):
            continue
        c = tuple(round(v, 3) for v in x.stroking_color)
        if c not in NAMES or len(x.pts) < 50:
            continue
        pts = np.array([(px, H - py) for px, py in x.pts])
        if not (PANEL[0] <= pts[:, 0].min() and pts[:, 0].max() <= PANEL[2]
                and PANEL[1] <= pts[:, 1].min() and pts[:, 1].max() <= PANEL[3]):
            continue
        curves[NAMES[c]] = (fx(pts[:, 0]), fy(pts[:, 1]))
    assert set(curves) == {"P_g", "P_e", "P_f"}, curves.keys()
    x0 = curves["P_g"][0]
    for k in curves:
        assert np.allclose(curves[k][0], x0, atol=1e-3)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 18(b) of Putterman et al., arXiv:2409.13025v2, p.31 (A4-S4, |f> prepared,\n"
                 "# 800 ns flux pulse to 'on', coherent storage state). Different cooldown than the rep-code run.\n"
                 "# VALIDATION ONLY. Extracted by tools/extract_ocelot_fig18b.py; error bars not extracted.\n")
        fh.write("alpha2,P_g,P_e,P_f\n")
        for i in range(len(x0)):
            fh.write(f"{x0[i]:.4f},{curves['P_g'][1][i]:.5f},{curves['P_e'][1][i]:.5f},{curves['P_f'][1][i]:.5f}\n")
    s = np.array([curves[k][1] for k in ("P_g", "P_e", "P_f")]).sum(axis=0)
    print(f"wrote {out}: {len(x0)} points; P_g+P_e+P_f in [{s.min():.3f}, {s.max():.3f}]")
    ig = int(np.argmax(curves["P_g"][1]))
    print(f"P_g: {curves['P_g'][1][0]:.3f} at 0, max {curves['P_g'][1][ig]:.3f} at |alpha|^2={x0[ig]:.1f}, "
          f"{curves['P_g'][1][-1]:.3f} at 20")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
