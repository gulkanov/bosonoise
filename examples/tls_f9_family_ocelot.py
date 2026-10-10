"""TLS-assisted direct f->g in A4-S4: family of predictions vs Ocelot Fig. 18b (no fit).

(a) For a single TLS placed at evenly spaced positions across the band swept by the
    dressing-opened f->g + storage-photon channel (n = 0..20), plot the predicted
    Delta P_g(|alpha|^2) after 800 ns at 'on', together with the F9 data. No curve is
    selected or tuned: the TLS positions are a uniform grid over the band.
(b) Interior-peak height (peak at |alpha|^2 in [2, 18]) versus TLS frequency for several
    couplings g01 (T1_TLS = 1 us) and (c) for several TLS lifetimes (g01 = 0.5 MHz).

Usage: python examples/tls_f9_family_ocelot.py
Output: figures/tls_f9_family_A4S4.png/.pdf, data/tls/f9_family_A4S4.csv (a), data/tls/f9_family_peaks_A4S4.csv (b, c)
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.channels.tls import AncillaBath, TLSEnsemble, poisson_average
from bosonoise.plotting import BLUE_ORDINAL, SERIES, TEXT_SECONDARY, use_style

from tls_single_map_ocelot import T1, build  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBAR = np.linspace(0, 20, 101)


def curve(bath, f, g01, gam, t=800.0):
    tls = TLSEnsemble(np.array([f]), np.array([g01]), np.array([gam]))
    avg, _ = poisson_average(bath.f_state_populations(tls, t), NBAR)
    return avg[:, 0] - avg[0, 0]


def interior_peak(d):
    k = int(np.argmax(d))
    return d[k] if 2.0 <= NBAR[k] <= 18.0 else 0.0


def main():
    tab_on, tab_idle = build("data/calibration/ocelot_A4_fcmin6.569.yaml", "S4")
    bath = AncillaBath(tab_on, tab_idle, *T1["A4"])
    dom = tab_on.dominant("f->g")
    sel = dom["n"] <= 20
    band = (dom["freq"][sel].min(), dom["freq"][sel].max())
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig18b.csv", encoding="utf-8")
                               if not l.startswith("#")))
    a2 = np.array([float(r["alpha2"]) for r in rows])
    dPg_data = np.array([float(r["P_g"]) for r in rows]) - float(rows[0]["P_g"])

    use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    positions = np.linspace(band[0] + 0.1 * (band[1] - band[0]), band[1] - 0.1 * (band[1] - band[0]), 6)
    fam = []
    for i, f in enumerate(positions):
        d = curve(bath, f, 0.5e-3, 1e-3)
        fam.append(d)
        ax[0].plot(NBAR, d, color=BLUE_ORDINAL[i % len(BLUE_ORDINAL)], lw=1.2,
                   label=f"TLS at {f:.4f} GHz" if i in (0, len(positions) - 1) else None)
    ax[0].plot(a2, dPg_data, "o", ms=3, color=SERIES[1], label="Ocelot Fig. 18b (A4-S4)")
    ax[0].set_xlabel(r"$|\alpha|^2$")
    ax[0].set_ylabel(r"$\Delta P_g$ after 800 ns at 'on'")
    ax[0].set_title("(a) one TLS (0.5 MHz, T1 1 µs) at 6 positions across the band", fontsize=8)
    ax[0].legend(fontsize=7)

    fgrid = np.arange(band[0] - 0.004, band[1] + 0.004, 0.0002)
    peaks_vs_f = {}
    for j, g in enumerate((0.1, 0.2, 0.5, 1.0)):
        h = [interior_peak(curve(bath, f, g * 1e-3, 1e-3)) for f in fgrid]
        peaks_vs_f[f"g01_{g:g}MHz"] = h
        ax[1].plot(fgrid, h, color=SERIES[j], label=f"g01 = {g:g} MHz")
    ax[1].set_title("(b) interior peak height, T1_TLS = 1 µs", fontsize=8)
    for j, T1t in enumerate((0.3, 1.0, 3.0, 10.0)):
        h = [interior_peak(curve(bath, f, 0.5e-3, 1e-3 / T1t)) for f in fgrid]
        peaks_vs_f[f"T1_{T1t:g}us"] = h
        ax[2].plot(fgrid, h, color=SERIES[j], label=f"T1_TLS = {T1t:g} µs")
    ax[2].set_title("(c) interior peak height, g01 = 0.5 MHz", fontsize=8)
    for x in ax[1:]:
        x.axhline(0.097, color=TEXT_SECONDARY, lw=0.8, ls="--")
        x.set_xlabel("TLS frequency (GHz)")
        x.set_ylabel(r"max $\Delta P_g$ (peak at $2\leq|\alpha|^2\leq18$)")
        x.legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"tls_f9_family_A4S4.{ext}", dpi=160)
    out = ROOT / "data" / "tls"
    with open(out / "f9_family_A4S4.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# predicted Delta P_g(nbar) for one TLS (g01 0.5 MHz, T1 1 us) at uniform positions across the f->g+photon band\n")
        fh.write("nbar," + ",".join(f"f{f:.4f}" for f in positions) + "\n")
        for i, nb in enumerate(NBAR):
            fh.write(f"{nb:.2f}," + ",".join(f"{d[i]:.5g}" for d in fam) + "\n")
    with open(out / "f9_family_peaks_A4S4.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# interior peak height of Delta P_g (peak at 2 <= |alpha|^2 <= 18, else 0) against the TLS frequency, for couplings "
                 "g01 (T1_TLS = 1 us) and lifetimes T1_TLS (g01 = 0.5 MHz)\n")
        fh.write("f_GHz," + ",".join(peaks_vs_f) + "\n")
        for i, f in enumerate(fgrid):
            fh.write(f"{f:.5f}," + ",".join(f"{h[i]:.5g}" for h in peaks_vs_f.values()) + "\n")
    peaks = [(interior_peak(d), NBAR[int(np.argmax(d))]) for d in fam]
    print("band (n=0..20):", [round(b, 4) for b in band])
    print("family interior peaks (height, nbar):", [(round(h, 3), round(nb, 1)) for h, nb in peaks])
    print("data peak: 0.097 at 10.2")


if __name__ == "__main__":
    main()
