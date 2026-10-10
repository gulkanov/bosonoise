"""Where do a bosonic code's own photons open ancilla decay channels? (paper Sec. 3.2, Fig. 1(d))

The one-photon-assisted decay |f, n> -> |g, n+1> of a transmon ancilla is forbidden for charge noise without dressing
(total parity: f and g have the same parity, so the charge operator connects them only with an odd change of the
photon number). Dressing by the code's mode opens it with a branching per photon

    rho(n) = |<g~, n+1| n_q |f~, n>|^2 / ((n + 1) |<e|n_q|f>|^2),

so that the rate is rho (n + 1) times the f -> e scale (for a flat charge bath). The same channel for a qubit ancilla
(e -> g, opposite parity) does not exist at any photon number (paper Sec. 2.1, App. G).

  Ocelot  exact dressed ladders of the storage-coupler-ancilla cell at the coupler 'on' point (the CX^2 gate), all 8
          pairs (examples/predict_cx2_ule_ocelot.py ladders, n = 0 ... 15).
  GKP     first-order dressing (dynamics/floquet.one_photon_element, checked against the Floquet sidebands to 1-5%),
          devices A (Sivak et al. 2023) and B (Eickbusch et al. 2022) of paper App. G, as if their
          ancilla used the f level.

Usage: python examples/universal_map.py [--plot]
Output: data/validation/universal_map.csv, figures/universal_map.png/.pdf
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.dynamics.floquet import one_photon_element
from bosonoise.model.transmon import Transmon
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

from predict_cx2_ule_ocelot import PAIRS, grid_for  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "validation" / "universal_map.csv"
GKP = {  # paper App. G (device A: our charge-basis fit); operating photon numbers of the large ECD displacement
    "GKP A (Sivak 2023)": dict(EJ=23.05, EC=0.2039, g=0.01170, fc=4.479, nbar=(121.0, 256.0)),
    "GKP B (Eickbusch 2022)": dict(EJ=32.33, EC=0.181, g=0.00912, fc=5.26, nbar=(900.0, 900.0)),
}
OCELOT_NBAR = (1.0, 4.0)  # |alpha|^2 range of the Ocelot repetition-code data


def ocelot_rows():
    rows = []
    for pair in PAIRS:
        on = grid_for(pair).ladders[-1]
        Xq = on.X["q"]
        ref = abs(Xq[1, 0, 2, 0]) ** 2
        nmax = Xq.shape[1] - 1
        for n in range(nmax):
            R = abs(Xq[0, n + 1, 2, n]) ** 2 / ref
            rows.append(dict(system=f"Ocelot {pair}", kind="ocelot", n=n, rho=R / (n + 1), branching=R,
                             parity_forbidden_check=abs(Xq[0, n, 2, n]) ** 2 / ref))
    return rows


def gkp_rows():
    rows = []
    for name, d in GKP.items():
        E, nq = Transmon(EJ=d["EJ"], EC=d["EC"], ng=0.0, ncut=40).spectrum(20)
        ref = nq[1, 2] ** 2
        for n in (0, 1, 10, 30, 100, 121, 256, 300, 600, 900):
            M = one_photon_element(E, nq, d["g"], d["fc"], 2, 0, n)
            rows.append(dict(system=name, kind="gkp", n=n, rho=M ** 2 / ((n + 1) * ref), branching=M ** 2 / ref,
                             parity_forbidden_check=one_photon_element(E, nq, d["g"], d["fc"], 1, 0, n) ** 2 / nq[0, 1] ** 2))
    return rows


def save(rows):
    keys = list(rows[0])
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Dressing-opened one-photon f -> g branching rho = |<g,n+1|n|f,n>|^2/((n+1)|<e|n|f>|^2); Ocelot from the "
                 "exact ladders at 'on', GKP from first-order dressing; parity_forbidden_check: Ocelot same-n f->g, GKP "
                 "one-photon e->g (both ~0). examples/universal_map.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) if isinstance(r[k], str) else f"{r[k]:.6g}" for k in keys) + "\n")
    print("saved", OUT)


def load():
    return [{k: (v if k in ("system", "kind") else float(v)) for k, v in r.items()}
            for r in csv.DictReader(l for l in open(OUT, encoding="utf-8") if not l.startswith("#"))]


def plot(rows):
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    systems = list(dict.fromkeys(r["system"] for r in rows))
    for i, s in enumerate(systems):
        sel = [r for r in rows if r["system"] == s]
        rho0 = sel[0]["rho"] if sel[0]["n"] == 0 else sel[1]["rho"]
        if sel[0]["kind"] == "ocelot":
            lo, hi = OCELOT_NBAR
            ax.plot([lo, hi], [rho0, rho0], "-", lw=6, alpha=0.6, color=SERIES[0],
                    label="Ocelot pairs at the CX² 'on' point (exact ladders)" if i == 0 else None)
        else:
            lo, hi = GKP[s]["nbar"]
            c = SERIES[2] if "A" in s.split()[1] else SERIES[3]
            ax.plot([lo * 0.9, hi * 1.1], [rho0, rho0], "-", lw=6, alpha=0.7, color=c,
                    label=f"{s}: big ECD displacement, if the ancilla used f")
    nb = np.geomspace(0.5, 3e3, 100)
    for b, ls in ((1e-3, ":"), (1e-2, "--"), (1e-1, "-.")):
        ax.plot(nb, b / nb, ls, color=TEXT_PRIMARY, lw=0.8)
        ax.text(nb[-1] * 0.5, b / (nb[-1] * 0.5) * 1.3, f"f→g / f→e = {b:g}", fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"photons in the code mode during the gate, $\bar n$")
    ax.set_ylabel(r"dressing-opened f→g branching per photon, $\rho$")
    ax.set_title("Total parity and dressing: one-photon f→g is open for equal-parity ancilla levels;\n"
                 "the qubit ancilla's e→g has no such channel at any photon number", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"universal_map.{ext}", dpi=160)
    print("saved figures/universal_map.png/.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="redraw from the CSV")
    a = ap.parse_args()
    if a.plot:
        return plot(load())
    rows = ocelot_rows() + gkp_rows()
    for s in dict.fromkeys(r["system"] for r in rows):
        sel = [r for r in rows if r["system"] == s]
        r0 = next(r for r in sel if r["n"] in (0, 1))
        big = max(r["parity_forbidden_check"] for r in sel)
        print(f"{s:28s} rho(n=0..1) = {r0['rho']:.3e} per photon; largest parity-forbidden element {big:.1e}")
    save(rows)
    plot(rows)


if __name__ == "__main__":
    main()
