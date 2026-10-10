"""Logical bit-flip error per cycle of the Ocelot repetition code (Fig. 4b, Fig. 31) from the stabilizer faults.

Z_L = Z_1 ... Z_d flips when an odd number of cats suffer an X error in a cycle; the correlated X_a X_b of
one stabilizer acts trivially. For independent stabilizers s and cat idling j,
    eps_L = 1/2 [1 - prod_s (1 - 2 L_s) prod_j (1 - 2 p_idle,j)],
with L_s = P(odd number of X on the two cats of stabilizer s), from the dressed-ladder ULE fault tables
(examples/stabilizer_faults_ocelot.py; ancilla in (g+f)/sqrt2, or in g for the control run of Fig. 31),
and p_idle,j the stabilized-idle bit flip of cat j per 2.8-us cycle, which is DATA-DERIVED: the static
protocol of Fig. 16 (geometric mean of the two pairs that contain the cat; T_X = inverse Pauli rate,
p = [1 - exp(-2 T_cycle / T_X)] / 2). The pump is off during the two CX layers, so the idle part is
a band: full cycle (upper edge, as in AWS's App. I 2) or scaled by the pumped fraction (lower edge).
Sections as run in the experiment: d3 S1-S3 (A1, A2), d3 S3-S5 (A3, A4), d5 (A1-A4).

Data-to-data check: AWS's phenomenological recipe (App. I 2, Eqs. I1-I2) rebuilt from Fig. 16 in Pauli
units, p^{CX|k} = [p^{CX^2|k} - p^idle] / 2 averaged over |alpha|^2 = 3-4, summed with the idle part.
It must reproduce AWS's dotted model lines of Fig. 4b if the conversion is right (paper App. H).
Data for the (g+f) runs: our fits to the raw shots of Zenodo 10.5281/zenodo.14257632 when available
(examples/zenodo_bitflip_ocelot.py; exact per-cycle value, with statistical errors), else Fig. 4b;
the ancilla-in-g control run is only in Fig. 31. All data are VALIDATION ONLY; nothing is fitted.

Usage: python examples/logical_bitflip_ocelot.py
Output: data/validation/logical_bitflip.csv, figures/logical_bitflip.png/.pdf
"""
from __future__ import annotations

import collections
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_REP_US, T_CX2_US = 2.8, 3.0  # cycle of the repetition code / of the Fig. 16 CX^2 experiments
SECTIONS = {"d3_S1-S3": (("A1", "A2"), ("S1", "S2", "S3")),
            "d3_S3-S5": (("A3", "A4"), ("S3", "S4", "S5")),
            "d5_S1-S5": (("A1", "A2", "A3", "A4"), ("S1", "S2", "S3", "S4", "S5"))}
STAB = {"A1": ("S1", "S2"), "A2": ("S2", "S3"), "A3": ("S3", "S4"), "A4": ("S4", "S5")}
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VARIANTS = ["zero-fit", "T50", "lam2", "lam3"]


def _rows(path):
    return list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))


def p_pauli(T_X_us: float, T_c_us: float) -> float:
    """Pauli flip probability per cycle for an inverse Pauli rate T_X (AWS's physical flip times)."""
    return 0.5 * (1.0 - np.exp(-2.0 * T_c_us / T_X_us))


def fig16_times():
    TX = collections.defaultdict(dict)
    for r in _rows(ROOT / "data" / "digitized" / "ocelot_fig16.csv"):
        TX[(r["pair"], r["protocol"])][float(r["alpha2"])] = 1e6 * float(r["T_X_s"])
    return TX


def time_at(TX, pair, prot, a2):
    """T_X (us), log-linearly extrapolated from the last two points where the marker is missing (off-scale)."""
    d = TX[(pair, prot)]
    if a2 in d:
        return d[a2], False
    xs = sorted(x for x in d if x < a2)[-2:]
    slope = (np.log(d[xs[1]]) - np.log(d[xs[0]])) / (xs[1] - xs[0])
    return float(np.exp(np.log(d[xs[1]]) + slope * (a2 - xs[1]))), True


def idle_p(TX, cat, a2, T_c_us=T_REP_US):
    pairs = [f"{a}-{cat}" for a, cats in STAB.items() if cat in cats]
    vals = [time_at(TX, p, "static", a2) for p in pairs]
    return p_pauli(float(np.exp(np.mean([np.log(v) for v, _ in vals]))), T_c_us), any(e for _, e in vals)


def recipe_cx(TX):
    """AWS Eq. I1 in Pauli units: p^{CX|k} = [p^{CX^2|k} - p^idle] / 2, averaged over |alpha|^2 = 3, 3.5, 4."""
    out = {}
    for anc, cats in STAB.items():
        for cat in cats:
            pair = f"{anc}-{cat}"
            for k in ("g", "f"):
                v = [(p_pauli(TX[(pair, k)][a2], T_CX2_US) - p_pauli(time_at(TX, pair, "static", a2)[0], T_CX2_US)) / 2
                     for a2 in (3.0, 3.5, 4.0)]
                out[(pair, k)] = float(np.mean(v))
    return out


def main():
    TX = fig16_times()
    cx = recipe_cx(TX)
    stab = {(r["stabilizer"].split(":")[0], r["variant"], float(r["nbar"])): r
            for r in _rows(ROOT / "data" / "validation" / "stabilizer_faults.csv")}
    fig4 = {(r["section"], r["series"], float(r["alpha2"])): float(r["eps_L_bitflip"])
            for r in _rows(ROOT / "data" / "digitized" / "ocelot_fig4b.csv")}
    fig31 = {(r["section"], r["protocol"], float(r["alpha2"])): float(r["eps_mean"])
             for r in _rows(ROOT / "data" / "digitized" / "ocelot_fig31.csv")}
    zen_path = ROOT / "data" / "validation" / "zenodo_logical_bitflip.csv"
    zen = {(r["section"], float(r["nbar"])): (float(r["data_eps_exact"]), float(r["data_eps_err"]))
           for r in _rows(zen_path)} if zen_path.exists() else {}
    rows = []
    for sec, (ancs, cats) in SECTIONS.items():
        # pumped fraction of the cycle: the pump is off for the two CX layers (padding unknown -> upper bound)
        L1 = max(float(stab[(a, "zero-fit", 1.0)]["gate_a_ns"]) for a in ancs)
        L2 = max(float(stab[(a, "zero-fit", 1.0)]["gate_b_ns"]) for a in ancs)
        duty = 1.0 - (L1 + L2) * 1e-3 / T_REP_US
        for prot in ("g+f", "g"):
            for nb in NBARS:
                idle = [idle_p(TX, c, nb) for c in cats]
                pid = np.array([v for v, _ in idle])
                extrap = any(e for _, e in idle)
                idle_hi = 0.5 * (1 - np.prod(1 - 2 * pid))
                idle_lo = 0.5 * (1 - np.prod(1 - 2 * duty * pid))
                pairs = [f"{a}-{c}" for a in ancs for c in STAB[a]]
                rec_cx = sum(0.5 * (cx[(p, "g")] + cx[(p, "f")]) if prot == "g+f" else cx[(p, "g")] for p in pairs)
                if prot == "g+f" and (sec, nb) in zen:
                    eps, eps_err, src = zen[(sec, nb)][0], zen[(sec, nb)][1], "zenodo_raw_fit"
                elif prot == "g+f":
                    eps, eps_err, src = fig4.get((sec, "data", nb), np.nan), np.nan, "fig4b_digitized"
                else:
                    eps, eps_err, src = fig31.get((sec, prot, nb), np.nan), np.nan, "fig31_digitized"
                row = dict(section=sec, protocol=prot, nbar=nb, data_source=src, duty=duty, idle_extrapolated=int(extrap),
                           data_eps=eps, data_eps_err=eps_err,
                           data_aws_model=fig4.get((sec, "aws_model", nb), np.nan) if prot == "g+f" else np.nan,
                           data_recipe_pauli=float(pid.sum() + rec_cx), data_recipe_gates=float(rec_cx),
                           data_idle_hi=float(idle_hi), data_idle_lo=float(idle_lo))
                for v in VARIANTS:
                    if (ancs[0], v, nb) not in stab:
                        continue
                    key = "p_odd" if prot == "g+f" else "p_odd_branch_g"
                    Ls = np.array([float(stab[(a, v, nb)][key]) for a in ancs])
                    gates = 0.5 * (1 - np.prod(1 - 2 * Ls))
                    row[f"gates_{v}"] = float(gates)
                    row[f"total_{v}_lo"] = float(0.5 * (1 - (1 - 2 * gates) * (1 - 2 * idle_lo)))
                    row[f"total_{v}_hi"] = float(0.5 * (1 - (1 - 2 * gates) * (1 - 2 * idle_hi)))
                rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = ROOT / "data" / "validation" / "logical_bitflip.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Logical bit-flip error per 2.8-us cycle. data_* columns come from, or are computed only from, AWS figures "
                 "(Fig. 4b/31 data, AWS model line, AWS recipe rebuilt from Fig. 16 in Pauli units, static idle band; VALIDATION "
                 "ONLY). gates_* = our stabilizer-fault model; total_* = gates_* combined with the data-derived idle band. "
                 "examples/logical_bitflip_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in keys) + "\n")
    print("section   prot  nbar | data | AWS model  recipe(Pauli) | idle lo-hi | gates: zero-fit T50 lam2 lam3 | "
          "(data - idle_hi)/gates zero-fit, lam3   [x1e-3]")
    for r in rows:
        if r["nbar"] in (1.0, 2.0, 3.0, 3.5, 4.0):
            g = [r.get(f"gates_{v}", np.nan) for v in VARIANTS]
            att = r["data_eps"] - r["data_idle_hi"]
            print(f"{r['section']} {r['protocol']:3s} {r['nbar']:.1f} | {r['data_eps'] * 1e3:6.2f} | {r['data_aws_model'] * 1e3:6.2f} "
                  f"{r['data_recipe_pauli'] * 1e3:6.2f} | {r['data_idle_lo'] * 1e3:5.2f}-{r['data_idle_hi'] * 1e3:5.2f} | "
                  + " ".join(f"{x * 1e3:5.2f}" for x in g)
                  + f" | {att / g[0]:5.1f} {att / g[3]:5.1f}" + ("  (idle extrapolated)" if r["idle_extrapolated"] else ""))
    use_style()
    fig, axs = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for j, sec in enumerate(SECTIONS):
        for i, prot in enumerate(("g+f", "g")):
            ax = axs[i, j]
            sel = [r for r in rows if r["section"] == sec and r["protocol"] == prot]
            x = [r["nbar"] for r in sel]
            src = sel[0]["data_source"]
            ax.errorbar(x, [r["data_eps"] for r in sel], yerr=[0.0 if np.isnan(r["data_eps_err"]) else r["data_eps_err"] for r in sel],
                        fmt="o", ms=5, mfc="none", color=TEXT_PRIMARY,
                        label={"zenodo_raw_fit": "data (raw shots, Zenodo; fit)", "fig4b_digitized": "data (Fig. 4b)",
                               "fig31_digitized": "data, ancilla in g (Fig. 31)"}[src])
            ax.set_yscale("log")
            if prot == "g+f":
                ax.semilogy(x, [r["data_aws_model"] for r in sel], ":", color=TEXT_PRIMARY, label="AWS phenomenological model")
            ax.semilogy(x, [r["data_recipe_pauli"] for r in sel], "--", color=SERIES[4], label="AWS recipe from Fig. 16 (Pauli)")
            ax.fill_between(x, [r["data_idle_lo"] for r in sel], [r["data_idle_hi"] for r in sel], color="#9a9893", alpha=0.35, lw=0,
                            label="idle part (static, data-derived)")
            for v, col in (("zero-fit", SERIES[0]), ("lam3", SERIES[1])):
                ax.fill_between(x, [r[f"total_{v}_lo"] for r in sel], [r[f"total_{v}_hi"] for r in sel], color=col, alpha=0.3,
                                lw=0, label=f"model gates ({v}) + idle band")
                ax.semilogy(x, [r[f"gates_{v}"] for r in sel], "-", lw=0.8, color=col, label=f"model gates only ({v})")
            if prot == "g":  # the control run is a heating test: ancilla at 50 mK during the gates
                ax.semilogy(x, [r["gates_T50"] for r in sel], "--", lw=0.8, color=SERIES[2], label="model gates only (T50)")
                ax.legend(fontsize=6, loc="upper right") if j == 0 else None
            ax.set_title(f"{sec}, ancilla {prot}", fontsize=9)
            ax.set_ylim(1e-4, 6e-2)
    for ax in axs[1]:
        ax.set_xlabel(r"$|\alpha|^2$")
    for ax in axs[:, 0]:
        ax.set_ylabel(r"logical bit-flip error per cycle $\epsilon_{L}$")
    axs[0, 0].legend(fontsize=6, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"logical_bitflip.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
