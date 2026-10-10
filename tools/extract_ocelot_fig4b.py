"""Extract Fig. 4b (p.8) and Fig. 31 (p.45) of Putterman et al., arXiv:2409.13025v2: logical bit-flip error per cycle.

epsilon_L,bit-flip = p_X + p_Y = T_cycle / (2 T_Z) (Eq. C2; T_Z = fitted decay constant of <Z_1 ... Z_d>,
T_cycle = 2.8 us; the right axis of Fig. 4b, logical T_Z, confirms this mapping) for the two
distance-3 sections (red: S1-S3 with A1, A2; blue: S3-S5 with A3, A4) and the distance-5 code
(purple), |alpha|^2 = 1 ... 4.
  Fig. 4b: data (markers, averaged over the initial states |alpha>^d and |-alpha>^d) and the AWS
           phenomenological model of App. I 2 (dotted lines, Eq. I2).
  Fig. 31: the two initial states separately, for the ancilla prepared in (|g> + |f>)/sqrt2 every
           cycle (circles; the Fig. 4b run) and in |g> every cycle (triangles; control run without
           f-related errors). The two initial states are combined as in the main text (mean T_Z).
Marker centroids; error bars are smaller than the markers. VALIDATION ONLY.

Usage: python tools/extract_ocelot_fig4b.py literature/ocelot_2409.13025.pdf data/digitized
Output: ocelot_fig4b.csv, ocelot_fig31.csv in the given directory
"""
import pathlib
import sys

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve, LTFigure

T_CYCLE_US = 2.8
SECTIONS = {"purple": "d5_S1-S5", "red": "d3_S1-S3", "blue": "d3_S3-S5"}
# Fig. 4b (page index 7): x ticks 1.0 ... 4.0 (26.13 pt per 0.5); log y ticks 5e-2, 1e-2, 5e-3
F4 = dict(page=7, x0=100.19, pt_half=26.13, y_ref=256.85, y_ref_val=1e-2, pt_decade=101.7,
          markers={(0.874, 0.861, 0.948): "purple", (0.927, 0.543, 0.485): "red", (0.395, 0.67, 0.834): "blue"},
          lines={(0.596, 0.557, 0.835): "purple", (0.886, 0.29, 0.2): "red", (0.204, 0.541, 0.741): "blue"})
# Fig. 31 (page index 44): x ticks 1.0 ... 4.0 (26.34 pt per 0.5); log y ticks 1e-2, 1e-3
F31 = dict(page=44, x0=361.25, pt_half=26.34, y_ref=490.36, y_ref_val=1e-2, pt_decade=74.47,
           markers={(0.874, 0.861, 0.948): "purple", (0.927, 0.543, 0.485): "red", (0.395, 0.67, 0.834): "blue"})


def _walk(obj):
    for x in obj:
        if isinstance(x, LTFigure):
            yield from _walk(x)
        else:
            yield x


def _col(c):
    return tuple(round(v, 3) for v in c) if isinstance(c, (tuple, list)) else c


def _xy(fig, cx, cy):
    a2 = 1.0 + (cx - fig["x0"]) / fig["pt_half"] * 0.5
    return a2, fig["y_ref_val"] * 10 ** ((fig["y_ref"] - cy) / fig["pt_decade"])


def _on_grid(a2):
    return 0.95 < a2 < 4.05 and abs(a2 * 2 - round(a2 * 2)) < 0.05


def fig4b(pdf):
    page = next(extract_pages(pdf, page_numbers=[F4["page"]]))
    H = page.bbox[3]
    data, model = {}, {}
    for x in _walk(page):
        if not isinstance(x, LTCurve):
            continue
        xs, ys = [p[0] for p in x.pts], [H - p[1] for p in x.pts]
        if len(x.pts) in (9, 10) and x.fill and _col(x.non_stroking_color) in F4["markers"]:
            a2, eps = _xy(F4, np.mean(xs), np.mean(ys))
            if _on_grid(a2):  # excludes the legend markers (|alpha|^2 ~ 3.45)
                data[(SECTIONS[F4["markers"][_col(x.non_stroking_color)]], round(a2 * 2) / 2)] = eps
        if len(x.pts) == 7 and _col(x.stroking_color) in F4["lines"] and x.dashing_style and x.dashing_style[0]:
            for px, py in zip(xs, ys):  # dotted = phenomenological model
                a2, eps = _xy(F4, px, py)
                model[(SECTIONS[F4["lines"][_col(x.stroking_color)]], round(a2 * 2) / 2)] = eps
    return data, model


def fig31(pdf):
    page = next(extract_pages(pdf, page_numbers=[F31["page"]]))
    H = page.bbox[3]
    pts = {}
    for x in _walk(page):
        if not (isinstance(x, LTCurve) and x.fill and _col(x.non_stroking_color) in F31["markers"]):
            continue
        xs, ys = [p[0] for p in x.pts], [H - p[1] for p in x.pts]
        if max(xs) - min(xs) > 5 or len(x.pts) not in (4, 5, 9, 10):
            continue  # the legend squares are 5.6 pt wide, the markers 4.0 pt
        a2, eps = _xy(F31, np.mean(xs), np.mean(ys))
        if not _on_grid(a2):
            continue  # legend
        prot = "g+f" if len(x.pts) in (9, 10) else "g"
        pts.setdefault((SECTIONS[F31["markers"][_col(x.non_stroking_color)]], prot, round(a2 * 2) / 2), []).append(eps)
    return pts


def main(pdf, outdir):
    out = pathlib.Path(outdir)
    data, model = fig4b(pdf)
    with open(out / "ocelot_fig4b.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 4b of Putterman et al., arXiv:2409.13025v2, p.8: logical bit-flip error per "
                 "cycle eps = p_X + p_Y = T_cycle/(2 T_Z), T_cycle = 2.8 us. series: data (markers) or aws_model "
                 "(dotted, App. I 2 Eq. I2). VALIDATION ONLY. tools/extract_ocelot_fig4b.py\n")
        fh.write("section,series,alpha2,eps_L_bitflip\n")
        for series, d in (("data", data), ("aws_model", model)):
            for (sec, a2) in sorted(d):
                fh.write(f"{sec},{series},{a2:.1f},{d[(sec, a2)]:.5g}\n")
    pts = fig31(pdf)
    with open(out / "ocelot_fig31.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Vector-extracted from Fig. 31 of Putterman et al., arXiv:2409.13025v2, p.45: logical bit-flip error per "
                 "cycle for the ancilla prepared in g+f (the Fig. 4b run) or g (control) every cycle. eps_mean combines the "
                 "two initial states via the mean T_Z (as in the main text). VALIDATION ONLY. tools/extract_ocelot_fig4b.py\n")
        fh.write("section,protocol,alpha2,n_markers,eps_state1,eps_state2,eps_mean\n")
        for (sec, prot, a2) in sorted(pts):
            e = sorted(pts[(sec, prot, a2)])
            e2 = e + [np.nan] * (2 - len(e))
            mean = 1.0 / np.mean([1.0 / v for v in e])  # eps ~ 1/T_Z: average the lifetimes
            fh.write(f"{sec},{prot},{a2:.1f},{len(e)},{e2[0]:.5g},{e2[1]:.5g},{mean:.5g}\n")
    for sec in SECTIONS.values():
        print(sec, "data ", " ".join(f"{data.get((sec, a), np.nan) * 1e3:6.2f}" for a in np.arange(1, 4.01, 0.5)))
        print(sec, "AWS  ", " ".join(f"{model.get((sec, a), np.nan) * 1e3:6.2f}" for a in np.arange(1, 4.01, 0.5)))
        for prot in ("g+f", "g"):
            vals = [pts.get((sec, prot, a)) for a in np.arange(1, 4.01, 0.5)]
            print(sec, f"F31 {prot:3s}", " ".join(f"{1e3 / np.mean([1 / u for u in v]):6.2f}" if v else "   nan" for v in vals),
                  " n=", [len(v) if v else 0 for v in vals])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
