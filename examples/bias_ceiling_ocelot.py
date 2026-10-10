"""Bit-flip channel of the Ocelot syndrome cycle beyond |alpha|^2 = 4: the floor, n*, the bias ceiling, and the
comparison with the phenomenological models (paper Sec. 5, Fig. 3).

Input: data/validation/stabilizer_faults_large_nbar.csv (examples/stabilizer_faults_ocelot.py with ladder cutoff 40 and
simulation cutoff 38; |alpha|^2 = 1 ... 16; variants zero-fit, lam3, T50).

Per cat j and 2.8-us cycle:
  gate part   p_X,gate(j, n) = sum over the (one or two) stabilizers acting on j of that stabilizer's X probability on j;
  idle part   p_idle(j, n): DATA-DERIVED at |alpha|^2 <= 4 (static reference of Fig. 16, Pauli, geometric mean over the two
              pairs containing j); beyond 4 continued as A e^{-B n} with B fitted on 2 <= n <= 4 (labelled extrapolation;
              the fitted B ~ 1 is below the kappa_1-limited asymptote B = 2, so the continuation overestimates the idle
              part and gives an upper estimate of n*).
  n*_gate     minimum of the gate part alone (model only); n*_idle with the continued idle part. The true n* lies in between
              if the idle bit-flip rate falls at least as fast as the continuation.
  bias        rate ratio eta = Gamma_Z / Gamma_X = kappa_1,eff n T_cycle / p_X (T1,eff per storage from Fig. 30, d = 5 run).
  floor       p_X,gate = c0 + c1 n fitted on n = 8 ... 16; ceiling eta_max = kappa_1,eff T_cycle / c1 (reached for n >> c0/c1).
  logical     eps_L(section) = 1/2 [1 - prod_s (1 - 2 p_odd,s) prod_j (1 - 2 p_idle,j)] (correlated X_a X_b are harmless).
Phenomenology: 'AWS' = App. I form (|alpha|^2-independent CX term + idle), its CX term taken from our own model at
|alpha|^2 = 3-4 (the range AWS fit), so only the functional form differs; 'Hann' = constant floor (heating + double decay,
their Eq. 11) likewise anchored at 3-4. Ratio = our gate part / phenomenological CX term at larger |alpha|^2.

Usage: python examples/bias_ceiling_ocelot.py
Output: data/validation/bias_ceiling.csv, data/validation/bias_ceiling_summary.csv, figures/bias_ceiling.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_REP_US = 2.8
STAB = {"A1": ("S1", "S2"), "A2": ("S2", "S3"), "A3": ("S3", "S4"), "A4": ("S4", "S5")}
CATS = ["S1", "S2", "S3", "S4", "S5"]
T1_EFF_US = {"S1": 60.0, "S2": 55.0, "S3": 51.0, "S4": 61.0, "S5": 59.0}  # Fig. 30 (p.45), d = 5 run legends
SECTIONS = {"d3_S1-S3": (("A1", "A2"), ("S1", "S2", "S3")), "d3_S3-S5": (("A3", "A4"), ("S3", "S4", "S5")),
            "d5_S1-S5": (("A1", "A2", "A3", "A4"), tuple(CATS))}


def _rows(path):
    return list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))


def p_pauli(T_X_us, T_c_us):
    return 0.5 * (1.0 - np.exp(-2.0 * T_c_us / T_X_us))


def idle_model():
    """Per cat: data points (n <= 4) and the continuation A e^{-B n} fitted on 2 <= n <= 4."""
    TX = {}
    for r in _rows(ROOT / "data" / "digitized" / "ocelot_fig16.csv"):
        if r["protocol"] == "static":
            TX.setdefault(r["pair"], {})[float(r["alpha2"])] = 1e6 * float(r["T_X_s"])
    out = {}
    for cat in CATS:
        pairs = [f"{a}-{cat}" for a, cs in STAB.items() if cat in cs]
        pts = {}
        for nb in sorted({n for p in pairs for n in TX[p]}):
            vals = [TX[p][nb] for p in pairs if nb in TX[p]]
            if len(vals) == len(pairs):  # both pairs measured
                pts[nb] = p_pauli(float(np.exp(np.mean(np.log(vals)))), T_REP_US)
        xs = np.array([n for n in pts if 2.0 <= n <= 4.0])
        B, lnA = np.polyfit(xs, np.log([pts[n] for n in xs]), 1)
        out[cat] = dict(points=pts, A=float(np.exp(lnA)), B=float(-B))
    return out


def idle_at(idle, cat, n):
    d = idle[cat]
    if n in d["points"]:
        return d["points"][n], False
    return d["A"] * np.exp(-d["B"] * n), True


def main():
    rows = _rows(ROOT / "data" / "validation" / "stabilizer_faults_large_nbar.csv")
    S = {(r["stabilizer"].split(":")[0], r["variant"], float(r["nbar"])): r for r in rows}
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    nbars = sorted({float(r["nbar"]) for r in rows})
    idle = idle_model()
    out = []
    for v in variants:
        for cat in CATS:
            for n in nbars:
                gate = 0.0
                for a, (ca, cb) in STAB.items():
                    if cat == ca:
                        gate += float(S[(a, v, n)]["p_x_a"])
                    if cat == cb:
                        gate += float(S[(a, v, n)]["p_x_b"])
                pid, extrap = idle_at(idle, cat, n)
                kappa_T = T_REP_US / T1_EFF_US[cat]
                out.append(dict(kind="cat", name=cat, variant=v, nbar=n, gate=gate, idle=pid, idle_extrapolated=int(extrap),
                                total=gate + pid, bias=kappa_T * n / (gate + pid), bias_gate_only=kappa_T * n / gate))
        for sec, (ancs, cats) in SECTIONS.items():
            for n in nbars:
                L = np.array([float(S[(a, v, n)]["p_odd"]) for a in ancs])
                pid = np.array([idle_at(idle, c, n)[0] for c in cats])
                gates = 0.5 * (1 - np.prod(1 - 2 * L))
                out.append(dict(kind="logical", name=sec, variant=v, nbar=n, gate=float(gates),
                                idle=float(0.5 * (1 - np.prod(1 - 2 * pid))), idle_extrapolated=int(n > 4),
                                total=float(0.5 * (1 - np.prod(1 - 2 * L) * np.prod(1 - 2 * pid))), bias=np.nan,
                                bias_gate_only=np.nan))
    # summaries: n*, floor fit, ceiling, phenomenology ratios
    summ = []
    for v in variants:
        for kind, names in (("cat", CATS), ("logical", list(SECTIONS))):
            for name in names:
                sel = sorted((r for r in out if r["kind"] == kind and r["name"] == name and r["variant"] == v),
                             key=lambda r: r["nbar"])
                x = np.array([r["nbar"] for r in sel])
                g = np.array([r["gate"] for r in sel])
                t = np.array([r["total"] for r in sel])
                big = x >= 8
                c1, c0 = np.polyfit(x[big], g[big], 1)
                ref = g[(x >= 3) & (x <= 4)].mean()  # the phenomenological constant, anchored where AWS fit
                row = dict(kind=kind, name=name, variant=v, nstar_gate=float(x[np.argmin(g)]), nstar_total=float(x[np.argmin(t)]),
                           pmin_total=float(t.min()), c0=float(c0), c1=float(c1), floor_at_16=float(g[x == x.max()][0]),
                           ratio_to_phenomenology_8=float(g[x == 8][0] / ref), ratio_to_phenomenology_12=float(g[x == 12][0] / ref),
                           ratio_to_phenomenology_16=float(g[x == 16][0] / ref))
                if kind == "cat":
                    kT = T_REP_US / T1_EFF_US[name]
                    row.update(idle_B=idle[name]["B"], ceiling=kT / c1 if c1 > 0 else np.inf,
                               bias_at_4=float([r["bias"] for r in sel if r["nbar"] == 4][0]),
                               bias_at_16=float([r["bias"] for r in sel if r["nbar"] == 16][0]))
                summ.append(row)
    vd = ROOT / "data" / "validation"
    keys = list(dict.fromkeys(k for r in out for k in r))
    with open(vd / "bias_ceiling.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Bit-flip channel beyond |alpha|^2 = 4: per-cat and logical gate parts (model), idle part (data-derived at "
                 "|alpha|^2 <= 4, exponential continuation beyond: idle_extrapolated = 1), bias = Gamma_Z/Gamma_X. "
                 "examples/bias_ceiling_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in out:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    skeys = list(dict.fromkeys(k for r in summ for k in r))
    with open(vd / "bias_ceiling_summary.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Summary: n* (gate only / with continued idle), floor c0 + c1 n (fit n = 8-16), ceiling kappa_1 T_c / c1, "
                 "ratio of the gate part to a |alpha|^2-independent phenomenology anchored at |alpha|^2 = 3-4. "
                 "examples/bias_ceiling_ocelot.py\n")
        fh.write(",".join(skeys) + "\n")
        for r in summ:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in skeys) + "\n")
    print("variant   name      | n*_gate n*_total p_min(x1e-3) | c0 (x1e-3) c1 (x1e-3/photon) | ratio to phenomenology at 8/12/16 "
          "| idle B | ceiling | bias at 4 / 16")
    for r in summ:
        extra = (f" | {r['idle_B']:.2f} | {r['ceiling']:8.0f} | {r['bias_at_4']:6.0f} / {r['bias_at_16']:6.0f}"
                 if r["kind"] == "cat" else "")
        print(f"{r['variant']:8s} {r['name']:9s} | {r['nstar_gate']:4.1f} {r['nstar_total']:5.1f} {r['pmin_total'] * 1e3:7.3f} | "
              f"{r['c0'] * 1e3:6.3f} {r['c1'] * 1e3:7.4f} | {r['ratio_to_phenomenology_8']:4.2f} {r['ratio_to_phenomenology_12']:4.2f} "
              f"{r['ratio_to_phenomenology_16']:4.2f}{extra}")
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for i, cat in enumerate(CATS):
        for v, ls in (("zero-fit", "-"), ("lam3", "--")):
            sel = sorted((r for r in out if r["kind"] == "cat" and r["name"] == cat and r["variant"] == v), key=lambda r: r["nbar"])
            if not sel:
                continue
            x = [r["nbar"] for r in sel]
            axs[0].semilogy(x, [r["total"] for r in sel], ls, color=SERIES[i], label=f"{cat} ({v})" if ls == "-" else None)
            axs[1].semilogy(x, [r["bias"] for r in sel], ls, color=SERIES[i], label=cat if ls == "-" else None)
    nn = np.linspace(1, 16, 100)
    axs[0].semilogy(nn, 0.5 * np.exp(-2 * nn), ":", color=TEXT_PRIMARY, label=r"$\frac{1}{2}e^{-2\bar n}$ (architecture assumption)")
    axs[0].axvline(4, color="#9a9893", lw=0.6)
    axs[0].text(4.1, 2e-2, "data range ends", fontsize=7, color="#52514e")
    axs[0].set_ylim(1e-8, 5e-2)
    axs[0].set_ylabel("cat bit flip per 2.8-us cycle (gate + idle)")
    axs[0].set_title("per cat: solid zero-fit, dashed λ = 3", fontsize=9)
    axs[1].set_ylabel(r"bias $\Gamma_Z/\Gamma_X$")
    axs[1].set_title("noise bias per cat (idle continued beyond 4)", fontsize=9)
    for j, sec in enumerate(SECTIONS):
        for v, ls in (("zero-fit", "-"), ("lam3", "--")):
            sel = sorted((r for r in out if r["kind"] == "logical" and r["name"] == sec and r["variant"] == v), key=lambda r: r["nbar"])
            if sel:
                axs[2].semilogy([r["nbar"] for r in sel], [r["total"] for r in sel], ls, color=SERIES[j],
                                label=f"{sec}" if ls == "-" else None)
    axs[2].set_ylabel(r"logical bit flip per cycle $\epsilon_L$")
    axs[2].set_title("logical bit-flip floor (solid zero-fit, dashed λ = 3)", fontsize=9)
    for ax in axs:
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"bias_ceiling.{ext}", dpi=160)
    print("saved", vd / "bias_ceiling.csv")


if __name__ == "__main__":
    main()
