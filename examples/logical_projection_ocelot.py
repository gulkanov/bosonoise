"""p_L(d, nbar) of the transmon-ancilla repetition cat code: the first-principles bit-flip channel against the AWS
phenomenology; optimal photon number nbar* and the logical floor (paper App. F, Table 7, Fig. 14).

Bit flips per cycle, eps_bit = 1/2 [1 - prod_s (1 - 2 L_s) prod_j (1 - 2 p_idle,j)] over d - 1 stabilizers and d cats;
the d = 5 device is repeated periodically (stabilizers A1..A4, cats S1..S5):
  ours           L_s = P(odd number of X) per stabilizer measurement from the ULE fault tables to |alpha|^2 = 16
                 (stabilizer_faults_large_nbar.csv; variants zero-fit and lambda3), log-linear in nbar between points;
  phenomenology  AWS App. I2: a constant CX part per gate, rebuilt from Fig. 16 (logical_bitflip_ocelot.recipe_cx);
  idle (both)    DATA-DERIVED: Fig. 16 static at |alpha|^2 <= 4, continued as A e^{-B nbar} fitted on 2..4
                 (bias_ceiling_ocelot.idle_model); the same in every model, so the comparison isolates the gates.
Phase flips per cycle: the Stim table eps_phase(d, p_Z) (phase_flip_scaling.csv: p_meas 1.6%, p_erase 5%, noiseless
boundaries), log-log in p_Z, p_Z = [1 - exp(-2 nbar T_c / T1)] / 2.
Scenarios: storage T1,eff = 57 us (as built, Fig. 30 mean), 300 us, 1 ms at T_c = 2.8 us; 1 ms at T_c = 1 us (the cycle of
AWS's App. I4 estimates; idle rescaled to T_c, gates per cycle unchanged). eps_L = (eps_bit + eps_phase)/2 (AWS Fig. 5a).
nbar* = argmin of eps_L over nbar in [1, 16] for each d. Because the constant floor gives a plateau, the summary also
gives the band of nbar with eps_L within 10% of the minimum and the oversizing penalty eps_L(16) / min eps_L.

Usage: python examples/logical_projection_ocelot.py
Output: data/validation/logical_projection.csv, logical_projection_summary.csv, figures/logical_projection.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

from bias_ceiling_ocelot import idle_model  # same directory (data-derived idle, continued beyond |alpha|^2 = 4)
from logical_bitflip_ocelot import STAB, fig16_times, recipe_cx

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_REF = 2.8
DS = [3, 5, 7, 9, 11, 13, 15]
NB = np.round(np.arange(1.0, 16.001, 0.25), 2)
SCENARIOS = {"as built (T1 57 us; T_c 2.8 us)": (57.0, 2.8), "T1 300 us; T_c 2.8 us": (300.0, 2.8),
             "T1 1 ms; T_c 2.8 us": (1000.0, 2.8), "T1 1 ms; T_c 1 us": (1000.0, 1.0)}
ANC = ["A1", "A2", "A3", "A4"]
CATS = ["S1", "S2", "S3", "S4", "S5"]


def rows(name):
    return list(csv.DictReader(l for l in open(ROOT / "data" / "validation" / name, encoding="utf-8") if not l.startswith("#")))


def loglin(x, xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    return float(np.exp(np.interp(x, xs, np.log(np.clip(ys, 1e-300, None)))))


def gate_tables():
    tab = {}
    for r in rows("stabilizer_faults_large_nbar.csv"):
        anc = r["stabilizer"].split(":")[0]
        tab.setdefault((r["variant"], anc), []).append((float(r["nbar"]), float(r["p_odd"])))
    return {k: sorted(v) for k, v in tab.items()}


def idle_p(idle, cat, n, T_c):
    d = idle[cat]
    pts = sorted(d["points"].items())
    if n <= pts[-1][0]:
        p = loglin(n, [a for a, _ in pts], [b for _, b in pts])
    else:
        p = d["A"] * np.exp(-d["B"] * n)
    if T_c == T_REF:
        return p
    TX = -2 * T_REF / np.log(1 - 2 * p)
    return 0.5 * (1 - np.exp(-2 * T_c / TX))


P_TH = 0.2  # above this p_Z the code no longer suppresses phase flips (Lambda <= 1.1 in phase_flip_scaling.csv)


def phase_table():
    """{d: [(p_z, eps)]} below threshold; above it the sampled values are noise around 1/2 and are not used."""
    tab = {}
    for r in rows("phase_flip_scaling.csv"):
        e, pz = float(r["eps"]), float(r["p_z"])
        if np.isfinite(e) and pz <= P_TH:
            tab.setdefault(int(float(r["d"])), []).append((pz, e))
    return {d: sorted(v) for d, v in tab.items()}


def eps_phase(tab, d, pz):
    pts = tab[d]
    xs, ys = np.array([a for a, _ in pts]), np.array([b for _, b in pts])
    if pz >= xs[-1]:  # at/above threshold: rise linearly to 1/2 at p_Z = 1/2
        return float(min(0.5, ys[-1] + (0.5 - ys[-1]) * (pz - xs[-1]) / (0.5 - xs[-1])))
    if pz <= xs[0]:
        return float(ys[0] * (pz / xs[0]) ** ((d + 1) / 2))
    return float(np.exp(np.interp(np.log(pz), np.log(xs), np.log(np.clip(ys, 1e-300, None)))))


def main():
    idle = idle_model()
    gates = gate_tables()
    pcx = recipe_cx(fig16_times())
    phen = {}
    for anc, (a, b) in STAB.items():
        pa = 0.5 * (pcx[(f"{anc}-{a}", "g")] + pcx[(f"{anc}-{a}", "f")])
        pb = 0.5 * (pcx[(f"{anc}-{b}", "g")] + pcx[(f"{anc}-{b}", "f")])
        phen[anc] = 0.5 * (1 - (1 - 2 * pa) * (1 - 2 * pb))
    ptab = phase_table()
    out, summ = [], []
    for scen, (T1, T_c) in SCENARIOS.items():
        for model in ("zero-fit", "lambda3", "phenomenology"):
            for d in DS:
                best = None
                rows_d = []
                for n in NB:
                    prod = 1.0
                    for k in range(d - 1):
                        anc = ANC[k % 4]
                        if model == "phenomenology":
                            L = phen[anc]
                        else:
                            key = ("zero-fit" if model == "zero-fit" else "lam3", anc)
                            L = loglin(n, [a for a, _ in gates[key]], [b for _, b in gates[key]])
                        prod *= 1 - 2 * L
                    for k in range(d):
                        prod *= 1 - 2 * idle_p(idle, CATS[k % 5], n, T_c)
                    eb = 0.5 * (1 - prod)
                    pz = 0.5 * (1 - np.exp(-2 * n * T_c / T1))
                    ep = eps_phase(ptab, d, pz)
                    row = dict(scenario=scen, model=model, d=d, nbar=float(n), p_z=pz, eps_bit=eb, eps_phase=ep,
                               eps_L=0.5 * (eb + ep))
                    out.append(row)
                    rows_d.append(row)
                    if best is None or row["eps_L"] < best["eps_L"]:
                        best = row
                # the minimum can lie on a plateau (constant floor): report the band within 10% of it as well
                band = [r["nbar"] for r in rows_d if r["eps_L"] <= 1.1 * best["eps_L"]]
                summ.append(dict(scenario=scen, model=model, d=d, nbar_star=best["nbar"], eps_L_min=best["eps_L"],
                                 eps_bit_at_star=best["eps_bit"], eps_phase_at_star=best["eps_phase"],
                                 nbar_lo_10=min(band), nbar_hi_10=max(band),
                                 eps_ratio_nmax=rows_d[-1]["eps_L"] / best["eps_L"]))
    for name, data in (("logical_projection.csv", out), ("logical_projection_summary.csv", summ)):
        keys = list(data[0])
        with open(ROOT / "data" / "validation" / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# Repetition cat code (transmon ancillas) projection: eps_L = (eps_bit + eps_phase)/2 per cycle; bit "
                     "flips from our ULE fault tables (zero-fit, lambda3) or AWS's constant-CX phenomenology, idle part "
                     "data-derived (Fig. 16) in all; phase flips from the Stim table phase_flip_scaling.csv. "
                     "examples/logical_projection_ocelot.py\n")
            fh.write(",".join(keys) + "\n")
            for r in data:
                fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("scenario | model | d: nbar*, eps_L,min")
    for scen in SCENARIOS:
        for model in ("zero-fit", "lambda3", "phenomenology"):
            sel = [r for r in summ if r["scenario"] == scen and r["model"] == model]
            print(f"{scen:28s} {model:13s} " + "  ".join(f"{r['d']}: {r['nbar_star']:.2f}/{r['eps_L_min']:.2e}" for r in sel))
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.0), constrained_layout=True)
    styles = {"zero-fit": "-", "lambda3": "--", "phenomenology": ":"}
    model_label = {"zero-fit": "ours, zero fit", "lambda3": "ours, λ = 3", "phenomenology": "constant floor"}
    scen_label = {"as built (T1 57 us; T_c 2.8 us)": "as built: T₁ 57 µs, cycle 2.8 µs",
                  "T1 300 us; T_c 2.8 us": "T₁ 300 µs, cycle 2.8 µs", "T1 1 ms; T_c 2.8 us": "T₁ 1 ms, cycle 2.8 µs",
                  "T1 1 ms; T_c 1 us": "T₁ 1 ms, cycle 1 µs"}
    for i, scen in enumerate(SCENARIOS):
        for model, ls in styles.items():
            sel = [r for r in summ if r["scenario"] == scen and r["model"] == model]
            axs[0].plot([r["d"] for r in sel], [r["eps_L_min"] for r in sel], ls, marker="o", ms=3, color=SERIES[i],
                        label=scen_label[scen] if model == "zero-fit" else None)
            axs[2].plot([r["d"] for r in sel], [r["eps_ratio_nmax"] for r in sel], ls, marker="o", ms=3, color=SERIES[i])
    for model, ls in styles.items():
        axs[0].plot([], [], ls, color=TEXT_PRIMARY, label=model_label[model])
    axs[0].set_yscale("log")
    axs[0].set_xlabel("distance $d$")
    axs[0].set_ylabel(r"$\min_{\bar n}\,\varepsilon_L$ per cycle")
    axs[0].legend(fontsize=6.5)
    axs[0].set_title("(a) logical error at the optimal cat size", fontsize=9, loc="left")
    scen = "T1 1 ms; T_c 1 us"
    for j, (model, ls) in enumerate(styles.items()):
        sel = [r for r in summ if r["scenario"] == scen and r["model"] == model]
        d = np.array([r["d"] for r in sel]) + 0.25 * (j - 1)
        c = SERIES[3 + j]
        axs[1].fill_between(d, [r["nbar_lo_10"] for r in sel], [r["nbar_hi_10"] for r in sel], color=c, alpha=0.15,
                            lw=0)
        axs[1].plot(d, [r["nbar_star"] for r in sel], ls, marker="o", ms=3, color=c, label=model_label[model])
    axs[1].set_xlabel("distance $d$")
    axs[1].set_ylabel(r"optimal $\bar n$ (band: $\varepsilon_L$ within 10% of the minimum)")
    axs[1].set_ylim(0, 17)
    axs[1].legend(fontsize=6.5, loc="upper left")
    axs[1].set_title("(b) T₁ 1 ms, cycle 1 µs", fontsize=9, loc="left")
    axs[2].axhline(1.1, color=TEXT_PRIMARY, lw=0.6, ls=":")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("distance $d$")
    axs[2].set_ylabel(r"$\varepsilon_L(\bar n=16)\,/\,\min_{\bar n}\varepsilon_L$")
    axs[2].set_title("(c) penalty for an oversized cat", fontsize=9, loc="left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"logical_projection.{ext}", dpi=160)
    print("saved")


if __name__ == "__main__":
    main()
