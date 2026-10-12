"""The fifteen figures of the paper at print size, drawn only from stored data products (no computation beyond
interpolation). Widths 3.4 in (one column) and 7.0 in (two columns); text 6.3-7.5 pt at that size; white background. The
diagnostic figures of the individual scripts are unchanged.

  paper    name              content                                                   data from the targets
  Fig. 1   prx_channel       the opened channel: (a) total-parity rule, (b) law        universal_law, chi_matching (incl.
                             against the numerical branching, (c) cost per gate and    the scan), universal_map
                             photon, (d) exposure of each cell's gates
  Fig. 2   prx_conversion    conversion of single ancilla faults into cat errors, and  fault_conversion,
                             the direct f -> g against the double decay (three baths)  fault_conversion_better_ancilla
  Fig. 3   prx_consequences  floor c0 + c1 nbar, bias ceiling, optimal cat size        bias_ceiling, logical_projection
  Fig. 4   prx_cx2_ratios    CX^2 f-excess, data/model per pair in three steps         cx2_measured_bath, f9_strength_cx2
  Fig. 5   prx_calibration   A2-S3 model against the published circuit-model curves    fig7_ocelot_reproduction
  Fig. 6   prx_rfg           squared opened element of A4-S4 against n                 dressed_fg_A4S4_fitA_flux0.500_nmax40
  Fig. 7   prx_branches      branches of A4-S4 at the operating point                  branches_A4S4_flux0.500_fitA_ng0.00
  Fig. 8   prx_atlas         crossings of the A2-S3 branches along the flux pulse      atlas_preview
  Fig. 9   prx_cx2_model     CX^2 f-excess against the idle-referenced model variants  cx2_ule
  Fig. 10  prx_cx2_measured  CX^2 f-excess with measured on-gate coherence (and the    cx2_measured_bath, f9_strength_cx2
                             f -> g strength favoured by F9 for the A4 pairs)
  Fig. 11  prx_faults        logical-relevant error per stabilizer by fault sequence   fault_conversion
  Fig. 12  prx_tls_family    one defect in the opened band of A4-S4                    tls_f9_family
  Fig. 13  prx_f9_bayes      one-defect fits of F9 with the smooth channel free        f9_bayes_smooth_A, f9_bayes_smooth_B
  Fig. 14  prx_projection    repetition-code projection: min eps_L, n*, penalty        logical_projection
  Fig. 15  prx_gkp           GKP devices A, B: branches, T1(e), heating, leakage       gkp_ecd_ancilla

Usage: python examples/prx_figures.py [name ...]   (default: all)
Output: figures/prx_<name>.png/.pdf
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

from bosonoise.plotting import BLUE_ORDINAL, SERIES, TEXT_PRIMARY, TEXT_SECONDARY, ordinal_cmap, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
COL, FULL = 3.4, 7.0  # column and text widths (inches)
PAIRS = ["A1-S1", "A1-S2", "A2-S2", "A2-S3", "A3-S3", "A3-S4", "A4-S4", "A4-S5"]
SHADE = "#9a9893"


def rows(path):
    return list(csv.DictReader(l for l in open(ROOT / path, encoding="utf-8") if not l.startswith("#")))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def col(sel, key):
    return np.array([num(r.get(key)) for r in sel])


def dash(pair):
    return pair.replace("-", "–")


def style():
    use_style()
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.5, "lines.linewidth": 1.1, "lines.markersize": 3.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.pad": 2.0, "ytick.major.pad": 2.0, "axes.labelpad": 2.0, "axes.titlepad": 3.0,
        "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3, "legend.columnspacing": 1.0, "pdf.fonttype": 42,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


def title(ax, s):
    ax.set_title(s, loc="left")


def tag(ax, s, x=0.02, y=0.96, ha="left", va="top", size=7.5):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=size)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"prx_{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"saved figures/prx_{name}.png/.pdf")


# ---------------------------------------------------------------- general theory
def ladder_panel(ax):
    """|i, n> levels (columns n and n+1) with the charge-noise transitions allowed or forbidden by total parity
    (schematic, for a mode slightly above the ancilla's g-e frequency, Delta > 0, as in Ocelot)."""
    E = {"g": 0.0, "e": 1.0, "f": 1.85}
    w = 1.15  # photon energy in units of the g-e spacing: |g,n+1> just above |e,n>, |e,n+1> above |f,n>
    xc = {0: 0.0, 1: 1.25}
    for k, levels in ((0, ("g", "e", "f")), (1, ("g", "e"))):
        for lev in levels:
            y = E[lev] + k * w
            ax.plot([xc[k] - 0.25, xc[k] + 0.25], [y, y], "-", color=TEXT_PRIMARY, lw=1.6)
            ax.text(xc[k] + (0.28 if k else -0.28), y, rf"$|{lev},{'n' if k == 0 else 'n{+}1'}\rangle$", va="center",
                    ha="left" if k else "right", fontsize=7)

    def arrow(x0, y0, x1, y1, color, ls="-", lw=1.5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1))

    arrow(-0.15, E["e"], -0.15, E["g"], SERIES[0])                         # e,n -> g,n (T1)
    arrow(-0.15, E["f"], -0.15, E["e"], SERIES[0])                         # f,n -> e,n
    arrow(0.05, E["f"], 0.05, E["g"], TEXT_SECONDARY, ls=":", lw=1.1)      # f,n -> g,n forbidden
    ax.text(0.05, 0.5 * (E["e"] + E["g"]) - 0.05, "×", color=SERIES[7], fontsize=12, ha="center", va="center",
            fontweight="bold")
    arrow(0.2, E["f"] - 0.02, xc[1] - 0.25, E["g"] + w + 0.03, SERIES[1], lw=2.2)   # f,n -> g,n+1 opened
    ax.text(-0.42, 2.0, r"opened, $\propto n{+}1$", color=SERIES[1], fontsize=7, ha="left", va="bottom")
    arrow(0.2, E["e"], xc[1] - 0.25, E["g"] + w - 0.03, SERIES[2], ls="--", lw=1.1)  # e,n -> g,n+1
    ax.text(0.28, 0.72, "frequency\nnoise only", color=SERIES[2], fontsize=6.8, ha="left", va="center")
    ax.text(0.62, -0.5, "charge noise is odd under total parity:\n" +
            r"$|\tilde\imath,n\rangle\to|\tilde\jmath,n{+}k\rangle$ needs $i-j+k$ odd", ha="center", fontsize=7)
    ax.set_xlim(-0.75, 1.95)
    ax.set_ylim(-0.85, 2.4)
    ax.axis("off")


def cost_panel(ax):
    """Opened probability per conditional pi rotation and photon, 2 pi r c(x) Gamma_fe/alpha (paper Eq. (5)), against
    |Delta|/alpha (chi_matching.py)."""
    sc = [r for r in rows("data/validation/chi_matching_scan.csv") if num(r["EJ_over_EC"]) == 95]
    x = np.array([num(r["x"]) for r in sc])
    r = np.array([num(r["r"]) for r in sc])
    cost = np.array([num(r["cost_angular"]) for r in sc])
    pos = x > 0
    neg = (x < 0) & (r > 0.2)  # beyond the e-f resonance; r -> 0 and changes sign near Delta = -1.5 alpha
    ax.plot(x[pos], cost[pos], "-", color=TEXT_PRIMARY, lw=1.1, label=r"transmon, $\Delta>0$")
    ax.plot(-x[neg], cost[neg], "--", color=TEXT_PRIMARY, lw=1.0, label=r"transmon, $\Delta<-\alpha$")
    win = pos & (r >= 0.8) & (r <= 1.2)
    ax.plot(x[win], cost[win], "-", color=SERIES[2], lw=4.0, alpha=0.45, solid_capstyle="butt",
            label=r"$0.8\leq r\leq 1.2$")
    cm = rows("data/validation/chi_matching.csv")
    m = next(q for q in cm if q["kind"] == "transmon_match" and num(q["EJ_over_EC"]) == 95)
    ax.plot(num(m["x"]), 2 * np.pi * num(m["cost_coeff"]), "*", color=SERIES[1], ms=10, zorder=5,
            label=r"$\chi$-matched by the detuning")
    oc = [q for q in cm if q["kind"] == "ocelot"]
    ax.plot([num(q["x"]) for q in oc], [2 * np.pi * num(q["cost_coeff"]) for q in oc], "o", color=SERIES[0], ms=3.6,
            zorder=4, label="Ocelot, eight pairs")
    dm = [q for q in cm if q["kind"] == "drive_matched"]
    ax.plot([abs(num(q["x"])) for q in dm], [2 * np.pi * num(q["cost_coeff"]) for q in dm], "s", color=SERIES[6],
            ms=4.5, zorder=4, label="matched by a drive")
    # labels on the side of the marker away from the curve: above-right of the solid line, below-left of the dashed one
    place = {"Reinhold": dict(xytext=(4, 4), ha="left", va="bottom"),
             "Rosenblum": dict(xytext=(-5, -4), ha="right", va="top")}
    for q in dm:
        name = q["name"].split()[0]
        ax.annotate(name, (abs(num(q["x"])), 2 * np.pi * num(q["cost_coeff"])), textcoords="offset points",
                    fontsize=6.3, color=SERIES[6], **place.get(name, dict(xytext=(0, 6), ha="center")))
    ax.text(0.97, 0.97, "qubit ancilla $(g,e)$:\nno charge-noise channel", transform=ax.transAxes, fontsize=6.5,
            ha="right", va="top", color=TEXT_SECONDARY, style="italic")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.18, 15)
    ax.set_ylim(3e-3, 20)
    ax.set_xlabel(r"$|\Delta|/\alpha$")
    ax.set_ylabel(r"$\varepsilon$ per gate and photon $[\Gamma_{fe}/\alpha]$")
    ax.legend(fontsize=6.0, loc="lower left", handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)


def map_panel(ax, legend_loc="upper right"):
    """Opened branching per photon at the photon numbers of each cell's gates, with lines of constant rho*nbar. Returns
    a function that writes the line labels; call it after the final layout (fig.canvas.draw())."""
    from universal_map import GKP, OCELOT_NBAR  # same directory: operating photon numbers
    um = rows("data/validation/universal_map.csv")
    for i, s in enumerate(dict.fromkeys(r["system"] for r in um)):
        sel = [r for r in um if r["system"] == s]
        rho0 = next(num(r["rho"]) for r in sel if num(r["n"]) == 0)
        if sel[0]["kind"] == "ocelot":
            lo, hi = OCELOT_NBAR
            ax.plot([lo, hi], [rho0, rho0], "-", lw=3.5, alpha=0.55, color=SERIES[0], solid_capstyle="butt",
                    label="Ocelot, eight pairs (CX gates)" if i == 0 else None)
        else:
            lo, hi = GKP[s]["nbar"]
            dev = s.split()[1]
            ax.plot([lo * 0.9, hi * 1.1], [rho0, rho0], "-", lw=3.5, alpha=0.8, solid_capstyle="butt",
                    color=SERIES[2] if dev == "A" else SERIES[3],
                    label=rf"GKP device {dev}, ECD gates (if its ancilla used $f$)")
    nb = np.geomspace(0.5, 3e3, 100)
    lines = ((1e-3, ":"), (1e-2, "--"), (1e-1, "-."))
    for b, ls in lines:
        ax.plot(nb, b / nb, ls, color=TEXT_SECONDARY, lw=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 3e3)
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlabel(r"photons in the code mode during the gate, $\bar n$")
    ax.set_ylabel(r"opened branching per photon, $\rho$")
    ax.legend(loc=legend_loc)

    def label_lines():
        for b, _ in lines:
            x0 = 30.0 if b > 1e-3 else 8.0
            p0, p1 = ax.transData.transform([(x0, b / x0), (3 * x0, b / (3 * x0))])
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            ax.text(x0 * 1.25, b / (x0 * 1.25) * 1.15, rf"$\rho\bar n=10^{{{int(round(np.log10(b)))}}}$", fontsize=6.3,
                    rotation=angle, rotation_mode="anchor", ha="left", va="bottom", color=TEXT_SECONDARY)

    return label_lines


def law_panel(ax):
    """Numerical opened branching per photon against the dispersive law, Eq. (3): Ocelot from the dressed ladders of the
    cell model at n = 0 (filled) and n = 10 (open), the GKP devices from the first-order element (all levels,
    counter-rotating terms included)."""
    ul = rows("data/validation/universal_law.csv")
    oc = [r for r in ul if r["kind"] == "ocelot"]
    gk = [r for r in ul if r["kind"] == "gkp"]
    lim = np.array([1e-6, 3e-2])
    ax.fill_between(lim, lim / 1.3, lim * 1.3, color=TEXT_PRIMARY, alpha=0.08, lw=0)
    ax.plot(lim, lim, "-", color=TEXT_PRIMARY, lw=0.8)
    ax.plot(col(oc, "rho_law"), col(oc, "rho_exact"), "o", ms=3.8, color=SERIES[0], label=r"Ocelot, $n=0$")
    ax.plot(col(oc, "rho_law"), col(oc, "rho_exact_n10"), "o", ms=3.8, mfc="none", mew=0.9, color=SERIES[0],
            label=r"Ocelot, $n=10$")
    ax.plot(col(gk, "rho_law"), col(gk, "rho_exact"), "s", ms=3.8, color=SERIES[2], label="GKP (first order)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(r"law, $|\chi_{ge}|\,\alpha\,/\,2|\Delta(\Delta+\alpha)|$")
    ax.set_ylabel(r"numerical $\rho$")
    ax.legend(loc="upper left")


def channel():
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 4.75), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ladder_panel(axs[0, 0])
    title(axs[0, 0], "(a) total parity on the dressed ladder")
    law_panel(axs[0, 1])
    title(axs[0, 1], "(b) the law against the numerical branching")
    cost_panel(axs[1, 0])
    title(axs[1, 0], "(c) cost per gate and photon")
    label_lines = map_panel(axs[1, 1])
    title(axs[1, 1], "(d) exposure of the gates of bosonic-code cells")
    fig.canvas.draw()
    label_lines()
    save(fig, "channel")


def rfg():
    m = rows("data/branches/dressed_charge_elements_ocelot_A4S4_fitA_flux0.500_nmax40.csv")
    n = col(m, "n")
    fig, ax = plt.subplots(figsize=(COL, 2.3), constrained_layout=True)
    ax.semilogy(n, col(m, "R_fg_q"), "o-", ms=2.4, color=SERIES[0], label=r"through the ancilla charge $\hat n_q$")
    ax.semilogy(n, col(m, "R_fg_c"), "s--", ms=2.4, color=SERIES[1], label=r"through the coupler charge $\hat n_c$")
    ax.set_xlim(-1, 41)
    ax.set_xlabel(r"storage photon number $n$")
    ax.set_ylabel(r"$R_{fg}(n)$")
    ax.legend(loc="lower right")
    save(fig, "rfg")


def gkp():
    from gkp_ecd_ancilla_photons import DEVICES, NGS  # same directory: windows and limits of the two devices
    g = rows("data/validation/gkp_ecd_ancilla.csv")
    styles = dict(zip(NGS, ("-", "--", ":")))
    fig, axs = plt.subplots(2, 3, figsize=(FULL, 4.0), constrained_layout=True)
    for i, (name, dev) in enumerate(DEVICES.items()):
        aw, at, ar = axs[i]
        for ng, ls in styles.items():
            sel = [r for r in g if r["device"] == name and num(r["ng"]) == ng]
            if not sel:
                continue
            x = col(sel, "nbar")
            aw.plot(x, col(sel, "w_g"), ls, color=SERIES[0], lw=1.0)
            aw.plot(x, col(sel, "w_e"), ls, color=SERIES[1], lw=1.0)
            tf, to = col(sel, "T1e_us_flat"), col(sel, "T1e_us_ohmic")
            at.plot(x, tf / tf[0], ls, color=SERIES[0], lw=1.0)
            at.plot(x, to / to[0], ls, color=SERIES[2], lw=1.0)
            ar.semilogy(x, np.maximum(col(sel, "g_out_per_us_flat"), 1e-12), ls, color=SERIES[3], lw=1.0)
            ar.semilogy(x, np.maximum(col(sel, "e_to_f_per_us_flat"), 1e-12), ls, color=SERIES[4], lw=1.0)
        lo, hi = dev["window"]
        for ax in (aw, at, ar):
            if hi > lo:
                ax.axvspan(lo, hi, color=SHADE, alpha=0.2, lw=0, zorder=0)
            if dev["dac"]:
                ax.axvline(dev["dac"], color=TEXT_SECONDARY, lw=0.7, ls=":")
            ax.set_xlim(0, dev["nbar_max"])
            ax.set_xlabel(r"cavity photons $\bar n=|\alpha|^2$")
        aw.set_ylabel("weight of the bare state")
        at.set_ylabel(r"$T_1(e)\,/\,T_1(e)|_{\bar n=0}$")
        ar.set_ylabel(r"rate ($\mu$s$^{-1}$)")
        title(aw, f"({'ad'[i]}) device {name[0]}")
        title(at, f"({'be'[i]})")
        title(ar, f"({'cf'[i]})")
    aw, at, ar = axs[1]  # device B: the measured facts
    aw.axvline(2500, color=SERIES[1], lw=0.8, ls="-.")
    aw.axvline(7500, color=SERIES[0], lw=0.8, ls="-.")
    at.plot([900], [30.0 / 50.0], "o", ms=4.5, mfc="none", mew=1.0, color=TEXT_PRIMARY)
    at.annotate("measured", xy=(900, 0.6), xytext=(1500, 0.08), fontsize=6.3,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=TEXT_PRIMARY))
    ar.axvspan(0, 2000, color=SERIES[3], alpha=0.12, lw=0, zorder=0)
    axs[0, 0].legend([Line2D([], [], color=SERIES[0]), Line2D([], [], color=SERIES[1])],
                     [r"$|g\rangle$ branch", r"$|e\rangle$ branch"], loc="lower left")
    axs[0, 1].legend([Line2D([], [], color=SERIES[0]), Line2D([], [], color=SERIES[2])], ["flat bath", "Ohmic bath"],
                     loc="lower left")
    axs[0, 2].legend([Line2D([], [], color=SERIES[3]), Line2D([], [], color=SERIES[4])],
                     [r"heating out of $|g\rangle$", r"$e\to f$ leakage"], loc="lower left")
    axs[1, 0].legend([Line2D([], [], color=TEXT_PRIMARY, ls=ls, lw=1.0) for ls in styles.values()],
                     [r"$n_g=0$", r"$n_g=1/4$", r"$n_g=1/2$"], loc="upper right")
    save(fig, "gkp")


# ---------------------------------------------------------------- the Ocelot cell
def calibration():
    m = rows("data/validation/fig7_ocelot_model.csv")
    pub = rows("data/digitized/ocelot_fig7.csv")[::6]  # thinned markers
    fl = col(m, "flux_Phi0")
    A = {k[:-2]: col(m, k) for k in m[0] if k.endswith("_A")}
    B = {k[:-2]: col(m, k) for k in m[0] if k.endswith("_B")}
    xp = col(pub, "flux_Phi0")
    mk = dict(ls="none", marker="o", mfc="none", mew=0.8, ms=3.0)
    meas = dict(color=TEXT_PRIMARY, marker="s", ms=3.0, ls="none", capsize=0, elinewidth=0.8)
    fig, ax = plt.subplots(5, 1, figsize=(COL, 5.4), sharex=True, constrained_layout=True,
                           gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.0, 1.0, 0.8]))
    for j, (k, c, lab) in enumerate([("f_s", "omega_s_GHz", r"$f_s$"), ("f_ge", "omega_a_ge_GHz", r"$f_{ge}$"),
                                     ("f_ef", "omega_a_ef_GHz", r"$f_{ef}$")]):
        ax[0].plot(fl, A[k], color=SERIES[j], label=lab)
        ax[0].plot(xp, col(pub, c), color=SERIES[j], **mk)
    h, l = ax[0].get_legend_handles_labels()
    h += [Line2D([], [], color=TEXT_PRIMARY), Line2D([], [], color=TEXT_PRIMARY, **mk)]
    l += ["this work (fit A)", "published circuit model"]
    ax[0].legend(h, l, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 0.17))
    ax[0].set_ylabel("frequency (GHz)")
    ax[0].set_ylim(4.94, 5.37)
    ok = np.abs(A["chi_gf"]) > 20e-6  # the ratio is undefined where both shifts vanish (near 0.13 Phi_0)
    D, K = A["f_s"] - A["f_ge"], A["f_ef"] - A["f_ge"]
    ax[1].plot(fl[ok], (A["chi_ge"] / A["chi_gf"])[ok], color=SERIES[0], label="exact")
    ax[1].plot(fl[ok], ((D - 2 * K) / (2 * D - K))[ok], color=SERIES[1], label="perturbative")
    okB = np.abs(B["chi_gf"]) > 20e-6
    ax[1].plot(fl[okB], (B["chi_ge"] / B["chi_gf"])[okB], ":", color=SERIES[0], label="exact, fit B")
    ax[1].plot(xp, col(pub, "chi_ratio_numerics"), color=SERIES[0], **mk)
    ax[1].plot(xp, col(pub, "chi_ratio_naive_D5"), color=SERIES[1], **mk)
    r = 1.18 / 1.14
    ax[1].errorbar([0.5], [r], yerr=[r * np.hypot(0.01 / 1.18, 0.01 / 1.14)], label="measured", **meas)
    ax[1].set_ylabel(r"$\chi_{ge}/\chi_{gf}$")
    ax[1].set_ylim(1.0, 1.3)
    ax[1].legend(loc="lower left", ncol=2)
    for j, (k, c, lab) in enumerate([("chi_ge", "chi_ge_MHz", r"$\chi_{ge}$"), ("chi_gf", "chi_gf_MHz", r"$\chi_{gf}$")]):
        ax[2].plot(fl, A[k] * 1e3, color=SERIES[j], label=lab)
        ax[2].plot(fl, B[k] * 1e3, ":", color=SERIES[j])
        ax[2].plot(xp, col(pub, c), color=SERIES[j], **mk)
    ax[2].errorbar([0.5, 0.5], [-1.18, -1.14], yerr=[0.01, 0.01], **meas)
    ax[2].set_ylabel(r"$\chi/2\pi$ (MHz)")
    ax[2].set_ylim(-1.35, 0.3)
    ax[2].legend(loc="lower left")
    for j, (k, c, lab) in enumerate([("K_g", "K_s_g_kHz", r"$K_{s|g}$"), ("K_e", "K_s_e_kHz", r"$K_{s|e}$"),
                                     ("K_f", "K_s_f_kHz", r"$K_{s|f}$")]):
        ax[3].plot(fl, A[k] * 1e6, color=SERIES[j], label=lab)
        ax[3].plot(xp, col(pub, c), color=SERIES[j], **mk)
    ax[3].set_ylabel(r"$K/2\pi$ (kHz)")
    ax[3].set_ylim(-4.5, 5.0)
    ax[3].legend(loc="lower left", ncol=3)
    ax[4].plot(fl, A["f_c"], color=SERIES[0], label=r"$f_c$, this work")
    ax[4].errorbar([0.0, 0.5], [9.0, 6.6], yerr=[0.05, 0.05], label="published values", **meas)
    ax[4].set_ylabel(r"$f_c$ (GHz)")
    ax[4].set_xlabel(r"coupler flux $\Phi_c/\Phi_0$")
    ax[4].legend(loc="lower left")
    for i, a in enumerate(ax):
        tag(a, f"({'abcde'[i]})", x=0.98 if i == 4 else 0.02, ha="right" if i == 4 else "left")
    save(fig, "calibration")


def branches():
    b = rows("data/branches/ocelot_A4S4_flux0.500_fitA_ng0.00.csv")
    fig, ax = plt.subplots(4, 1, figsize=(COL, 4.3), sharex=True, constrained_layout=True,
                           gridspec_kw=dict(height_ratios=[1.25, 1.0, 1.0, 0.8]))
    for k, nm in enumerate("gef"):
        sel = [r for r in b if r["branch"] == nm]
        n = col(sel, "n")
        c = SERIES[k]
        ax[0].plot(n, col(sel, "dev_from_linear_MHz"), "o-", ms=2.2, color=c, label=rf"$|{nm}\rangle$")
        ax[1].semilogy(n, np.abs(col(sel, "mean_q") - k) + 1e-12, "o-", ms=2.2, color=c)
        ax[2].semilogy(n, col(sel, "mean_c"), "o-", ms=2.2, color=c)
        ax[3].semilogy(n[1:], 1.0 - col(sel, "overlap")[1:], "o-", ms=2.2, color=c)
    ax[0].set_ylabel(r"$\delta E_k(n)$ (MHz)")
    ax[0].set_ylim(-38, 7)
    ax[0].legend(loc="lower left", ncol=3)
    ax[1].set_ylabel(r"$|\langle\hat N_q\rangle-k|$")
    ax[2].set_ylabel(r"$\langle\hat N_c\rangle$")
    ax[3].set_ylabel(r"$1-$overlap")
    ax[3].set_xlabel(r"storage photon number $n$")
    ax[3].set_xticks(range(0, 21, 5))
    ax[3].set_xlim(-0.7, 20.7)
    for i, a in enumerate(ax):
        bottom = i in (1, 2)  # the curves of (b) and (c) run along the top
        tag(a, f"({'abcd'[i]})", x=0.98, ha="right", y=0.06 if bottom else 0.94, va="bottom" if bottom else "top")
    save(fig, "branches")


def atlas():
    xs = rows("data/atlas/crossings_ocelot_A2S3_fitA_dflux0.01.csv")
    fl, fc = col(xs, "flux"), col(xs, "f_c01_bare_GHz")
    o = np.argsort(fl)
    fl, fc = fl[o], fc[o]

    def to_fc(x):
        return np.interp(x, fl, fc)

    def to_flux(y):
        return np.interp(y, fc[::-1], fl[::-1])

    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.25), sharey=True, constrained_layout=True)
    cmap = ordinal_cmap(6)
    for k, (nm, ax) in enumerate(zip("gef", axs)):
        sel = [r for r in xs if r["branch"] == nm]
        order = col(sel, "order")
        sc = ax.scatter(col(sel, "flux"), col(sel, "n"), c=order, cmap=cmap, vmin=0.5, vmax=6.5, s=16.0 / order + 2.0,
                        edgecolors="none")
        ax.set_xlim(0, 0.5)
        ax.set_xlabel(r"coupler flux $\Phi_c/\Phi_0$")
        sec = ax.secondary_xaxis("top", functions=(to_fc, to_flux))
        sec.set_xticks([8.5, 8.0, 7.5, 7.0])
        sec.set_xlabel(r"bare coupler $f_{01}$ (GHz)")
        sec.tick_params(labelsize=7, pad=1)
        tag(ax, f"({'abc'[k]}) ${nm}$ branch, {len(sel)} crossings", x=0.03, y=0.03, va="bottom", size=7)
    axs[0].set_ylabel(r"storage photon number $n$")
    axs[0].set_ylim(0, 21.5)
    axs[0].set_yticks(range(0, 21, 5))
    fig.colorbar(sc, ax=axs, label="order", ticks=range(1, 7), shrink=0.95, pad=0.01, aspect=25)
    save(fig, "atlas")


def cx2_panels():
    fig, axs = plt.subplots(2, 4, figsize=(FULL, 3.6), sharex=True, sharey=True, constrained_layout=True)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel("excess flips per cycle")
    axs[0, 0].set_ylim(1e-5, 3e-2)
    axs[0, 0].set_xticks([1, 2, 3, 4])
    return fig, axs


def cx2_model():
    res = rows("data/validation/cx2_ule_excess.csv")
    fig, axs = cx2_panels()
    for ip, pair in enumerate(PAIRS):
        sel = [r for r in res if r["pair"] == pair]
        x = col(sel, "nbar")
        ax = axs.flat[ip]
        ax.semilogy(x, col(sel, "data_fexcess"), "o", ms=3.5, mfc="none", mew=0.9, color=TEXT_PRIMARY, label="data")
        ax.semilogy(x, col(sel, "ffault_ours"), "-", color=SERIES[0], lw=1.5, label="dressed model, idle-referenced")
        ax.semilogy(x, col(sel, "ffault_ours_sym"), "-", color=SERIES[0], lw=0.6,
                    label=r"the same, symmetric re-stabilization ($\Delta_b=0$)")
        ax.semilogy(x, col(sel, "ffault_standard"), "--", color=SERIES[1], label="undressed baseline")
        ax.semilogy(x, col(sel, "ffault_coupler10us"), "-.", color=SERIES[2],
                    label=r"dressed, coupler bath $T_{1,c}=10~\mu$s (prior)")
        ax.semilogy(x, col(sel, "ffault_lam2"), ":", color=SERIES[3], lw=1.3, label=r"dressed, decay rates $\times2$")
        tag(ax, f"{dash(pair)}\n{num(sel[0]['gate_ns']):.0f} ns, $r={num(sel[0]['chi_ratio']):.2f}$", x=0.97, y=0.04,
            ha="right", va="bottom", size=6.5)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=3)
    save(fig, "cx2_model")


def cx2_measured():
    mb = rows("data/validation/cx2_measured_bath.csv")
    f9 = rows("data/validation/f9_strength_cx2.csv")
    fig, axs = cx2_panels()
    for ip, pair in enumerate(PAIRS):
        sel = [r for r in mb if r["pair"] == pair]
        x = col(sel, "nbar")
        ax = axs.flat[ip]
        ax.semilogy(x, col(sel, "data_fexcess"), "o", ms=3.5, mfc="none", mew=0.9, color=TEXT_PRIMARY, label="data")
        ax.semilogy(x, col(sel, "ffault_zerofit"), "-", color=SERIES[0], label=r"idle-referenced $T_1$")
        ax.semilogy(x, col(sel, "ffault_coupler18c"), "--", color=SERIES[2],
                    label=rf"coupler loss, $T_{{1,c}}={num(sel[0]['coupler_T1_us']):.2f}~\mu$s")
        if sel[0].get("ffault_ongate_coupler_on"):
            ax.fill_between(x, col(sel, "ffault_ongate_coupler_on"), col(sel, "ffault_ongate_pulsed_alpha2_4"),
                            color=SERIES[3], alpha=0.35, lw=0, label=r"measured on-gate $T_1$ (held to pulsed)")
        s9 = [r for r in f9 if r["pair"] == pair]
        if s9:
            x9 = col(s9, "nbar")
            ax.fill_between(x9, col(s9, "ffault_ongate_fg5"), col(s9, "ffault_ongate_fg11"), color=SERIES[4], alpha=0.35,
                            lw=0, label=r"on-gate $T_1$, $f\to g$ strength of F9 ($s=5$–11)")
            ax.semilogy(x9, col(s9, "ffault_ongate_fg8"), "-", color=SERIES[4], lw=0.8)
        tag(ax, f"{dash(pair)}\n{sel[0]['label']}", x=0.97, y=0.04, ha="right", va="bottom", size=6.5)
    h, l = axs.flat[-1].get_legend_handles_labels()  # A4-S5 has every element
    fig.legend(h, l, loc="outside upper center", ncol=3)
    save(fig, "cx2_measured")


def faults():
    fc = rows("data/validation/fault_conversion.csv")
    groups = [("deterministic", r"deterministic (Kerr shear, $\chi(n)$)", TEXT_SECONDARY),
              ("f->g only", r"direct $f\to g$", SERIES[1]),
              ("double decay f->e->g", r"double decay $f\to e\to g$", SERIES[6]),
              ("f->e only", r"$f\to e$ only", SERIES[4]),
              ("e->g only", r"$e\to g$ only", SERIES[2]),
              ("other", "other", SERIES[0])]
    stabs = list(dict.fromkeys(r["stabilizer"] for r in fc))
    fig, axs = plt.subplots(1, 4, figsize=(FULL, 2.4), sharey=True, constrained_layout=True)
    for k, (ax, st) in enumerate(zip(axs, stabs)):
        for v, ls in (("zero-fit", "-"), ("lam3", "--")):
            sel = [r for r in fc if r["stabilizer"] == st and r["variant"] == v]
            x = sorted({num(r["nbar"]) for r in sel})
            for key, lab, c in groups:
                y = [sum(num(r["p_fault"]) * num(r["odd"]) for r in sel if num(r["nbar"]) == nb and r["fault"] == key)
                     for nb in x]
                y = np.where(np.array(y) > 0, y, np.nan)  # 'other' is a difference and can vanish
                ax.semilogy(x, y, ls, color=c, lw=1.0, label=lab if ls == "-" else None)
        anc, cats = st.split(":")
        tag(ax, f"({'abcd'[k]}) {anc}: {cats[:2]}, {cats[2:]}", size=7)
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_xticks([2, 4, 8, 12, 16])
        ax.set_ylim(1e-7, 1e-2)
    axs[0].set_ylabel("odd-flip probability per measurement")
    h, l = axs[0].get_legend_handles_labels()
    h += [Line2D([], [], color=TEXT_PRIMARY, lw=1.0), Line2D([], [], color=TEXT_PRIMARY, lw=1.0, ls="--")]
    l += ["idle-referenced", r"$\lambda=3$"]
    fig.legend(h, l, loc="outside upper center", ncol=4)
    save(fig, "faults")


# ---------------------------------------------------------------- the direct f -> g decay (F9)
def tls_family():
    fam = rows("data/tls/f9_family_A4S4.csv")
    pk = rows("data/tls/f9_family_peaks_A4S4.csv")
    d = rows("data/digitized/ocelot_fig18b.csv")
    nb = col(fam, "nbar")
    keys = [k for k in fam[0] if k != "nbar"]
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.25), constrained_layout=True)
    ax = axs[0]
    for i, k in enumerate(keys):
        ax.plot(nb, col(fam, k), color=BLUE_ORDINAL[i], lw=1.0,
                label=f"defect at {float(k[1:]):.3f} GHz" if i in (0, len(keys) - 1) else None)
    a2, Pg = col(d, "alpha2"), col(d, "P_g")
    base = Pg[a2 < 2].mean()  # baseline as in the text and Fig. bayes: mean of the points at |alpha|^2 < 2
    ipk = int(np.argmax(Pg))
    peak = Pg[ipk - 3:ipk + 4].mean() - base  # seven-point mean at the peak (paper App. E.1: 0.084)
    ax.plot(a2, Pg - base, "o", ms=2.2, color=SERIES[1], label="data (A4–S4)")
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_ylabel(r"$\Delta P_g$ after 800 ns")
    ax.set_ylim(-0.006, 0.142)
    ax.legend(loc="upper left")
    title(ax, r"(a) $g_{01}=0.5$ MHz, $T_{1,\rm TLS}=1~\mu$s")
    f = col(pk, "f_GHz")
    for j, g in enumerate((0.1, 0.2, 0.5, 1.0)):
        axs[1].plot(f, col(pk, f"g01_{g:g}MHz"), color=SERIES[j], lw=1.0, label=rf"$g_{{01}}={g:g}$ MHz")
    for j, T1 in enumerate((0.3, 1.0, 3.0, 10.0)):
        axs[2].plot(f, col(pk, f"T1_{T1:g}us"), color=SERIES[j], lw=1.0, label=rf"$T_{{1,\rm TLS}}={T1:g}~\mu$s")
    title(axs[1], r"(b) $T_{1,\rm TLS}=1~\mu$s")
    title(axs[2], r"(c) $g_{01}=0.5$ MHz")
    for ax, loc in ((axs[1], "upper left"), (axs[2], "lower center")):
        ax.axhline(peak, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.text(0.98, peak, "data", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.3,
                color=TEXT_SECONDARY)
        ax.set_xlabel("defect frequency (GHz)")
        ax.set_ylabel(r"interior peak of $\Delta P_g$")
        ax.set_xticks([4.99, 5.00, 5.01, 5.02])
        ax.legend(loc=loc)
    axs[2].set_ylim(-0.003, 0.11)
    save(fig, "tls_family")


def f9_bayes():
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.25), constrained_layout=True)
    smooth = {k: json.loads((ROOT / "data" / "tls" / f"f9_bayes_smooth_{k}.json").read_text(encoding="utf-8"))
              for k in "AB"}
    names = {"A": "(a) A: charge-coupled defect", "B": "(b) B: critical-current defect"}
    for j, k in enumerate("AB"):
        cur = rows(f"data/tls/f9_bayes_smooth_{k}_curves.csv")
        a2, Pg = col(cur, "alpha2"), col(cur, "P_g")
        base = Pg[a2 < 2].mean()
        s_map = smooth[k]["reference"]["MAP"]["s"]
        ax = axs[j]
        ax.plot(a2, Pg - base, "o", ms=2.2, color=TEXT_PRIMARY, label="data (A4–S4)")
        ax.plot(a2, col(cur, "MAP") - base, "-", color=SERIES[j], lw=1.3, label=rf"defect + smooth channel, $s={s_map:g}$")
        ax.plot(a2, col(cur, "smooth_MAP_s") - base, "--", color=SERIES[j], lw=1.0, label="smooth channel alone")
        ax.plot(a2, col(cur, "smooth_s1") - base, ":", color=TEXT_SECONDARY, lw=1.0, label=r"smooth channel, $s=1$")
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_ylabel(r"$\Delta P_g$ after 800 ns")
        ax.set_ylim(-0.012, 0.178)
        ax.legend(loc="upper left")
        title(ax, names[k])
    ax = axs[2]
    variants = (("reference", "-", "reference"), ("binomial noise", "--", "binomial weights"),
                ("|alpha|^2 >= 2 only", ":", r"$|\alpha|^2\geq2$ only"))
    for j, k in enumerate("AB"):
        for vname, ls, lab in variants:
            sv = smooth[k][vname]
            s_ = np.array(sv["s"])
            sf = np.linspace(0, s_[-1], 601)
            # the log-marginal is known on the grid of s (step 1 near the peak); a cubic spline draws it smoothly. Display
            # only: the quantiles and Bayes factors in the text use log-linear interpolation, which differs by <= 0.04.
            lf = CubicSpline(s_, sv["log_marginal_per_s"])(sf)
            p = np.exp(lf - lf.max())
            ax.plot(sf, p / (p.sum() * (sf[1] - sf[0])), ls, color=SERIES[j], lw=1.1)
    ax.axvline(1.0, color=TEXT_SECONDARY, lw=0.8, ls="-.")
    ax.text(1.5, 0.6, "idle-referenced", transform=ax.get_xaxis_transform(), fontsize=6.3, color=TEXT_SECONDARY,
            rotation=90, ha="left", va="center")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 0.335)
    ax.set_xlabel(r"strength $s$ of the smooth $f\to g$ channel")
    ax.set_ylabel("posterior density")
    fits = ax.legend([Line2D([], [], color=TEXT_PRIMARY, ls=ls, lw=1.1) for _, ls, _ in variants],
                     [lab for *_, lab in variants], loc="upper right")
    ax.add_artist(fits)
    ax.legend([Line2D([], [], color=SERIES[j], lw=1.1) for j in (0, 1)], ["hypothesis A", "hypothesis B"],
              loc="center right", bbox_to_anchor=(1.0, 0.5))
    title(ax, "(c)")
    save(fig, "f9_bayes")


# ---------------------------------------------------------------- logical consequences and the windows
def projection():
    summ = rows("data/validation/logical_projection_summary.csv")
    styles = {"zero-fit": "-", "lambda3": "--", "phenomenology": ":"}
    model_label = {"zero-fit": "ours, idle-referenced", "lambda3": r"ours, $\lambda=3$", "phenomenology": "constant floor"}
    scen_label = {"as built (T1 57 us; T_c 2.8 us)": r"as built: $T_1=57~\mu$s, cycle $2.8~\mu$s",
                  "T1 300 us; T_c 2.8 us": r"$T_1=300~\mu$s, cycle $2.8~\mu$s",
                  "T1 1 ms; T_c 2.8 us": r"$T_1=1$ ms, cycle $2.8~\mu$s",
                  "T1 1 ms; T_c 1 us": r"$T_1=1$ ms, cycle $1~\mu$s"}
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.4), constrained_layout=True)
    for i, scen in enumerate(scen_label):
        for model, ls in styles.items():
            sel = [r for r in summ if r["scenario"] == scen and r["model"] == model]
            d = col(sel, "d")
            axs[0].plot(d, col(sel, "eps_L_min"), ls, marker="o", ms=2.2, lw=1.0, color=SERIES[i],
                        label=scen_label[scen] if model == "zero-fit" else None)
            axs[2].plot(d, col(sel, "eps_ratio_nmax"), ls, marker="o", ms=2.2, lw=1.0, color=SERIES[i])
    for model, ls in styles.items():
        axs[0].plot([], [], ls, color=TEXT_PRIMARY, lw=1.0, label=model_label[model])
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r"$\min_{\bar n}\varepsilon_L$ per cycle")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=4)
    title(axs[0], r"(a) logical error at the optimal $\bar n$")
    scen = "T1 1 ms; T_c 1 us"
    for j, (model, ls) in enumerate(styles.items()):
        sel = [r for r in summ if r["scenario"] == scen and r["model"] == model]
        d = col(sel, "d") + 0.25 * (j - 1)
        c = (SERIES[6], SERIES[4], SERIES[7])[j]  # hues not used for the scenarios of (a) and (c)
        axs[1].fill_between(d, col(sel, "nbar_lo_10"), col(sel, "nbar_hi_10"), color=c, alpha=0.15, lw=0)
        axs[1].plot(d, col(sel, "nbar_star"), ls, marker="o", ms=2.2, lw=1.0, color=c, label=model_label[model])
    axs[1].set_ylabel(r"optimal $\bar n$")
    axs[1].set_ylim(0, 17)
    axs[1].set_yticks([0, 4, 8, 12, 16])
    axs[1].legend(loc="upper left")
    title(axs[1], r"(b) $T_1=1$ ms, cycle $1~\mu$s")
    axs[2].axhline(1.1, color=TEXT_PRIMARY, lw=0.6, ls=":")
    axs[2].set_yscale("log")
    axs[2].set_ylabel(r"$\varepsilon_L(\bar n=16)\,/\,\min_{\bar n}\varepsilon_L$")
    title(axs[2], "(c) penalty for an oversized cat")
    for ax in axs:
        ax.set_xlabel("distance $d$")
        ax.set_xticks([3, 5, 7, 9, 11, 13, 15])
    save(fig, "projection")



# ---------------------------------------------------------------- figures of the main text
STABS = ["A1:S1S2", "A2:S2S3", "A3:S3S4", "A4:S4S5"]
BATHS = (("lam0.333", ":", r"$\lambda=1/3$"), ("zero-fit", "-", r"$\lambda=1$ (idle-referenced)"), ("lam3", "--", r"$\lambda=3$"))


def fault_rows():
    fc = rows("data/validation/fault_conversion.csv")
    better = ROOT / "data" / "validation" / "fault_conversion_better_ancilla.csv"
    if better.exists():
        fc += rows("data/validation/fault_conversion_better_ancilla.csv")
    return fc


def contribution(fc, st, variant, nb, fault):
    """p(F) P(odd | F) per stabilizer measurement, summed over the rows of one fault sequence."""
    return sum(num(r["p_fault"]) * num(r["odd"]) for r in fc if r["stabilizer"] == st and r["variant"] == variant
               and num(r["nbar"]) == nb and r["fault"] == fault)


def conversion():
    fc = fault_rows()
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.35), constrained_layout=True,
                            gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.05]))
    # (a) conversion of one fault of each class at |alpha|^2 = 4: P(odd | F), open = heralded (ancilla read in e)
    ax = axs[0]
    classes = (("f->e only", "zero-fit", r"$f\to e$"), ("f->g only", "zero-fit", r"direct $f\to g$"),
               ("double decay f->e->g", "zero-fit", "double\ndecay"), ("heating only", "T50", "heating\n(50 mK)"))
    for k, st in enumerate(STABS):
        for i, (fault, variant, _) in enumerate(classes):
            sel = [r for r in fc if r["stabilizer"] == st and r["variant"] == variant and num(r["nbar"]) == 4.0
                   and r["fault"] == fault]
            if not sel:
                continue
            p = sum(num(r["p_fault"]) for r in sel)
            odd = sum(num(r["p_fault"]) * num(r["odd"]) for r in sel) / p
            herald = sum(num(r["p_fault"]) * num(r["herald"]) for r in sel) / p
            ax.semilogy(i + 0.13 * (k - 1.5), max(odd, 1.2e-6), "o", ms=4.0, color=SERIES[k], mew=1.0,
                        mfc="none" if herald > 0.5 else SERIES[k], label=st.split(":")[0] if i == 0 else None)
    ax.set_xticks(range(len(classes)), [c[2] for c in classes])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlim(-0.5, len(classes) - 0.5)
    ax.set_ylim(1e-6, 2.0)
    ax.set_ylabel(r"$P(\mathrm{cat\ flip}\mid\mathrm{fault})$")
    leg = ax.legend(loc="center left", ncol=2, columnspacing=0.6, bbox_to_anchor=(0.36, 0.5))
    ax.add_artist(leg)
    ax.legend([Line2D([], [], ls="none", marker="o", ms=4.0, color=TEXT_PRIMARY),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc="none", mew=1.0, color=TEXT_PRIMARY)],
              ["unheralded", "heralded"], loc="lower right")
    title(ax, r"(a) one fault, $|\alpha|^2=4$")
    # (b) A4: the opened channel against the double decay for three baths
    ax = axs[1]
    st = "A4:S4S5"
    for variant, ls, lab in BATHS:
        x = sorted({num(r["nbar"]) for r in fc if r["stabilizer"] == st and r["variant"] == variant})
        if not x:
            continue
        ax.semilogy(x, [contribution(fc, st, variant, nb, "f->g only") for nb in x], ls, color=SERIES[1], lw=1.1)
        ax.semilogy(x, [contribution(fc, st, variant, nb, "double decay f->e->g") for nb in x], ls, color=SERIES[6], lw=1.1)
    x = sorted({num(r["nbar"]) for r in fc if r["stabilizer"] == st and r["variant"] == "zero-fit"})
    det = [next(num(r["odd"]) for r in fc if r["stabilizer"] == st and r["variant"] == "zero-fit" and num(r["nbar"]) == nb
                and r["fault"] == "deterministic") for nb in x]
    ax.semilogy(x, det, "-", color=TEXT_SECONDARY, lw=0.9)
    ax.legend([Line2D([], [], color=SERIES[1], lw=1.1), Line2D([], [], color=SERIES[6], lw=1.1),
               Line2D([], [], color=TEXT_SECONDARY, lw=0.9)],
              [r"direct $f\to g$", r"double decay $f\to e\to g$", r"deterministic (Kerr, $\chi(n)$)"], loc="lower right")
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_xticks([2, 4, 8, 12, 16])
    ax.set_ylim(1e-6, 3e-3)
    ax.set_ylabel("odd-flip probability per measurement")
    title(ax, "(b) A4: S4, S5")
    # (c) the ratio of the two for the four stabilizers
    ax = axs[2]
    for k, st in enumerate(STABS):
        for variant, ls, _ in BATHS:
            x = sorted({num(r["nbar"]) for r in fc if r["stabilizer"] == st and r["variant"] == variant})
            if not x:
                continue
            y = [contribution(fc, st, variant, nb, "f->g only") / contribution(fc, st, variant, nb, "double decay f->e->g")
                 for nb in x]
            ax.semilogy(x, y, ls, color=SERIES[k], lw=1.1, label=st.split(":")[0] if variant == "zero-fit" else None)
    ax.axhline(1.0, color=TEXT_PRIMARY, lw=0.6)
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_xticks([2, 4, 8, 12, 16])
    ax.set_ylabel(r"direct $f\to g$ / double decay")
    leg = ax.legend(loc="upper left", ncol=2, columnspacing=0.6)
    ax.add_artist(leg)
    ax.legend([Line2D([], [], color=TEXT_PRIMARY, ls=ls, lw=1.1) for _, ls, _ in BATHS], [lab for *_, lab in BATHS],
              loc="lower right")
    title(ax, "(c) direct over double decay")
    save(fig, "conversion")


def consequences():
    b = rows("data/validation/bias_ceiling.csv")
    s = rows("data/validation/bias_ceiling_summary.csv")
    lp = rows("data/validation/logical_projection.csv")
    fig, axs = plt.subplots(2, 2, figsize=(FULL, 4.3), constrained_layout=True)
    cats = ["S1", "S2", "S3", "S4", "S5"]

    def cat_rows(cat, var):
        return sorted((r for r in b if r["kind"] == "cat" and r["name"] == cat and r["variant"] == var),
                      key=lambda r: num(r["nbar"]))

    def summ(cat, var):
        return next(r for r in s if r["kind"] == "cat" and r["name"] == cat and r["variant"] == var)

    # (a) the CX gate part of cat S3: c0 + c1 nbar against a constant floor
    ax = axs[0, 0]
    for var, c, lab in (("zero-fit", SERIES[0], "idle-referenced"), ("lam3", SERIES[1], r"$\lambda=3$")):
        sel = cat_rows("S3", var)
        nb, gate = col(sel, "nbar"), col(sel, "gate")
        ax.semilogy(nb, gate, "-", color=c, lw=1.3, label=f"CX gate part, {lab}")
        ax.semilogy(nb, col(sel, "total"), ":", color=c, lw=1.0)
        sm = summ("S3", var)
        x = np.linspace(0, 16.5, 50)
        ax.semilogy(x, num(sm["c0"]) + num(sm["c1"]) * x, "-", color=c, lw=0.6, alpha=0.6)
        phen = gate[list(nb).index(8.0)] / num(sm["ratio_to_phenomenology_8"])  # constant floor anchored at 3-4
        ax.semilogy([3, 16.5], [phen, phen], "--", color=c, lw=0.9,
                    label="constant floor (anchored at $\\bar n=3$–4)" if var == "zero-fit" else None)
    ax.set_ylim(1e-4, 2e-2)
    ax.set_ylabel("bit flips per cycle, cat S3")
    ax.legend(loc="upper right")
    title(ax, "(a) the floor grows as $c_0+c_1\\bar n$")
    # (b) all five cats: total bit flips per cycle against the exponential suppression
    ax = axs[0, 1]
    for i, cat in enumerate(cats):
        for var, ls in (("zero-fit", "-"), ("lam3", "--")):
            sel = cat_rows(cat, var)
            ax.semilogy(col(sel, "nbar"), col(sel, "total"), ls, color=SERIES[i], lw=1.0,
                        label=cat if ls == "-" else None)
    nn = np.linspace(1, 16.5, 100)
    ax.semilogy(nn, 0.5 * np.exp(-2 * nn), ":", color=TEXT_PRIMARY, lw=1.0)
    ax.set_ylim(1e-8, 5e-2)
    ax.set_ylabel("bit flips per cat and cycle")
    leg = ax.legend(loc="upper right", ncol=3, columnspacing=0.8)
    ax.add_artist(leg)
    ax.legend([Line2D([], [], color=TEXT_PRIMARY, lw=1.0), Line2D([], [], color=TEXT_PRIMARY, lw=1.0, ls="--"),
               Line2D([], [], color=TEXT_PRIMARY, lw=1.0, ls=":")],
              ["idle-referenced", r"$\lambda=3$", r"$\frac{1}{2}e^{-2\bar n}$"], loc="lower left")
    title(ax, "(b) gate part plus data-derived idle part")
    # (c) the noise bias and its ceiling kappa_1 T_c / c_1 (idle-referenced bath)
    ax = axs[1, 0]
    for i, cat in enumerate(cats):
        sel = cat_rows(cat, "zero-fit")
        ax.semilogy(col(sel, "nbar"), col(sel, "bias"), "-", color=SERIES[i], lw=1.0, label=cat)
        ax.axhline(num(summ(cat, "zero-fit")["ceiling"]), color=SERIES[i], lw=0.7, ls=":")
    ax.set_ylabel(r"bias $\Gamma_Z/\Gamma_X$ (idle-referenced)")
    ax.legend(loc="lower right", ncol=3, columnspacing=0.8)
    title(ax, r"(c) the bias approaches $\kappa_1T_c/c_1$ (dotted)")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.axvspan(4, 16.5, color=SHADE, alpha=0.1, lw=0, zorder=0)
        ax.set_xlim(0.5, 16.5)
        ax.set_xlabel(r"$\bar n=|\alpha|^2$")
    axs[0, 1].text(13.0, 1.5e-8, "beyond the data", fontsize=6.3, ha="center", va="bottom", color=TEXT_SECONDARY)
    # (d) logical error against nbar relative to its minimum: a true minimum against a plateau
    ax = axs[1, 1]
    scen = "T1 1 ms; T_c 1 us"
    for d, c in ((3, SERIES[0]), (9, SERIES[2]), (15, SERIES[3])):
        for model, ls in (("lambda3", "-"), ("phenomenology", "--")):
            sel = sorted((num(r["nbar"]), num(r["eps_L"])) for r in lp
                         if r["scenario"] == scen and r["model"] == model and int(r["d"]) == d)
            nb = np.array([x for x, _ in sel])
            e = np.array([y for _, y in sel])
            ax.plot(nb, e / e.min(), ls, color=c, lw=1.2 if model == "lambda3" else 0.9,
                    label=rf"$d={d}$" if model == "lambda3" else None)
    ax.axhline(1.1, color=TEXT_SECONDARY, lw=0.6, ls=":")
    leg = ax.legend(loc="upper left", bbox_to_anchor=(0.12, 1.0))
    ax.add_artist(leg)
    ax.legend([Line2D([], [], color=TEXT_PRIMARY, lw=1.2), Line2D([], [], color=TEXT_PRIMARY, lw=0.9, ls="--")],
              [r"ours ($\lambda=3$)", "constant floor"], loc="upper left", bbox_to_anchor=(0.42, 1.0))
    ax.set_xlim(1, 16)
    ax.set_ylim(0.95, 2.2)
    ax.set_xlabel(r"$\bar n$")
    ax.set_ylabel(r"$\varepsilon_L\,/\,\min_{\bar n}\varepsilon_L$")
    tag(ax, r"$T_1=1$ ms, cycle $1~\mu$s", x=0.435, y=0.78, size=6.5)
    title(ax, "(d) logical error: a minimum, not a plateau")
    save(fig, "consequences")


def cx2_ratios():
    mb = rows("data/validation/cx2_measured_bath.csv")
    f9 = rows("data/validation/f9_strength_cx2.csv")
    fig, ax = plt.subplots(figsize=(COL, 2.75), constrained_layout=True)
    marks = {"direct": "D", "transferred": "s", "prior": "^"}
    names = {"direct": "measured on-gate decay: on this pair", "transferred": "… on the other pair of the ancilla",
             "prior": "… taken from A3 (prior)"}
    for i, pair in enumerate(PAIRS):
        sel = [r for r in mb if r["pair"] == pair and num(r["nbar"]) >= 2]

        def med(k):
            return np.nanmedian([num(r.get(k)) for r in sel])

        z = med("ratio_zerofit")
        best = med("ratio_ongate_pulsed_alpha2_4") if sel[0].get("ratio_ongate_pulsed_alpha2_4") else med("ratio_coupler18c")
        lab = sel[0]["label"]
        ax.plot([i, i], [best, z], "-", color=TEXT_SECONDARY, lw=0.8)
        ax.plot(i, z, "o", mfc="none", mew=1.0, color=SERIES[0], ms=4.5, label="idle-referenced bath")
        ax.plot(i, best, marks[lab], color=SERIES[1], ms=4.5, label=names[lab])
    for pair in ("A4-S4", "A4-S5"):
        sel = [r for r in f9 if r["pair"] == pair and num(r["nbar"]) >= 2]

        def med(k):
            return np.nanmedian([num(r[k]) for r in sel])

        m8, lo, hi = med("ratio_ongate_fg8"), med("ratio_ongate_fg11"), med("ratio_ongate_fg5")
        ax.errorbar(PAIRS.index(pair) + 0.25, m8, yerr=[[m8 - lo], [hi - m8]], fmt="*", color=SERIES[3], ms=6.5,
                    capsize=2, lw=0.8, label=r"… with the $f\to g$ strength of F9 ($s=5$–11)")
    h, l = ax.get_legend_handles_labels()
    keep = dict(zip(l, h))
    order = ["idle-referenced bath", names["direct"], names["transferred"], names["prior"],
             r"… with the $f\to g$ strength of F9 ($s=5$–11)"]
    ax.legend([keep[k] for k in order if k in keep], [k for k in order if k in keep], loc="upper left", fontsize=6.2)
    ax.axhspan(0.5, 2.0, color=SERIES[2], alpha=0.1, lw=0)
    ax.axhline(1.0, color=TEXT_PRIMARY, lw=0.7)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 60)
    ax.set_xticks(range(len(PAIRS)), [dash(p) for p in PAIRS], rotation=45, fontsize=6.5)
    ax.set_ylabel(r"$f$ excess, data / model ($|\alpha|^2\geq2$)")
    save(fig, "cx2_ratios")


FIGURES = dict(channel=channel, rfg=rfg, gkp=gkp, calibration=calibration, branches=branches, atlas=atlas,
               cx2_model=cx2_model, cx2_measured=cx2_measured, faults=faults, tls_family=tls_family, f9_bayes=f9_bayes,
               projection=projection, conversion=conversion, consequences=consequences, cx2_ratios=cx2_ratios)


def main():
    names = sys.argv[1:] or list(FIGURES)
    style()
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
