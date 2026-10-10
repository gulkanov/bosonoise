"""Extract Fig. 18c of Putterman et al., arXiv:2409.13025v2 (page 31): ancilla A3 T1 vs coupler flux.

Fig. 18c (App. F9, a DIFFERENT cooldown than the repetition-code run): the ancilla A3 is
prepared in |e>, the coupler of the A3-S3 interaction is pulsed to flux Phi for a variable
duration, and P(1) is measured. The heat map is drawn as vector cells whose fill colours
follow matplotlib's 'Blues' colour map (checked: max RGB residual 8e-4 over all cells; the
colour bar spans 0..1).

Axes: flux from the y ticks (0.0 ... 0.5 Phi0, 18.49 pt per 0.1 Phi0); delay from the x ticks
(20 ... 100 us, 29.38 pt per 20 us). Each cell centre gives (Phi, t, P(1)).
Per flux row we fit P(t) = A exp(-t/T1) + B (a fit within the validation data themselves,
to summarise them; nothing of the model is fitted). VALIDATION ONLY.
Requires pdfminer.six and matplotlib.

Usage: python tools/extract_ocelot_fig18c.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig18c.csv
"""
import sys

import matplotlib
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure, LTRect
from scipy.optimize import curve_fit

FLUX_TICK0, FLUX_PT_PER_0P1 = 518.89, 18.49  # y (top-down) of Phi = 0.0; pitch
DELAY_TICK20, DELAY_PT_PER_20 = 384.57, 29.38  # x of t = 20 us; pitch


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def main(pdf, out):
    page = next(extract_pages(pdf, page_numbers=[30]))  # printed page 31
    H = page.bbox[3]
    ref = matplotlib.colormaps["Blues"](np.linspace(0, 1, 2001))[:, :3]
    cells = []
    for x in _walk(page):
        if isinstance(x, (LTRect, LTCurve)) and x.fill and not x.stroke:
            x0, y0, x1, y1 = x.bbox
            c = x.non_stroking_color
            if (355 < x0 < 520 and 424 < H - y1 < 521 and x1 - x0 < 20 and abs((y1 - y0) - 3.082) < 0.01
                    and isinstance(c, (list, tuple)) and len(c) == 3):
                d = np.linalg.norm(ref - np.array(c), axis=1)
                assert d.min() < 3e-3, "cell colour not on the Blues map"
                P = np.linspace(0, 1, 2001)[d.argmin()]
                flux = (FLUX_TICK0 - 0.5 * ((H - y1) + (H - y0))) / FLUX_PT_PER_0P1 * 0.1
                t = 20.0 + (0.5 * (x0 + x1) - DELAY_TICK20) / DELAY_PT_PER_20 * 20.0
                cells.append((round(flux, 4), t, P))
    cells = np.array(cells)
    rows = []
    for fl in np.unique(cells[:, 0]):
        sel = cells[cells[:, 0] == fl]
        sel = sel[np.argsort(sel[:, 1])]
        t, P = sel[:, 1], sel[:, 2]
        try:
            (A, T1, B), cov = curve_fit(lambda t, A, T1, B: A * np.exp(-t / T1) + B, t, P, p0=(0.9, 30.0, 0.0),
                                        bounds=([0, 0.1, -0.2], [1.5, 1e3, 0.5]))
            err = float(np.sqrt(cov[1, 1]))
        except RuntimeError:
            A = T1 = B = err = np.nan
        rows.append((fl, T1, err, A, B, len(t)))
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 18c of Putterman et al., arXiv:2409.13025v2, p.31 (A3, coupler of A3-S3; "
                 "different cooldown). VALIDATION ONLY.\n# T1 from P(t) = A exp(-t/T1) + B per flux row "
                 "(tools/extract_ocelot_fig18c.py).\n")
        fh.write("flux_Phi0,T1_us,T1_err_us,A,B,n_points\n")
        for r in rows:
            fh.write(",".join(f"{v:.5g}" for v in r) + "\n")
    print(f"wrote {out}: {len(cells)} cells, {len(rows)} flux rows")
    for r in rows:
        print(f"  flux {r[0]:.4f}: T1 = {r[1]:.1f} +- {r[2]:.1f} us (A {r[3]:.2f}, B {r[4]:+.3f}, {r[5]} delays)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
