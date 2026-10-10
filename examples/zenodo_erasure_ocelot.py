"""Ancilla erasure statistics of the Ocelot repetition code from the raw syndromes (Zenodo 10.5281/zenodo.14257632).

In the phase-flip runs every stabilizer round reports 0 (even), 2 (odd) or 1 (erasure: the ancilla was read
in e). After unpreparation and the extra ef pi pulse (App. G5) the ideal outcomes are g (even) and f (odd);
e is reached by an f -> e decay while the ancilla is in (g+f)/sqrt2 (from preparation to unpreparation,
including the CX gates -- our model's herald), by an f -> e decay during the readout of an odd outcome, or
by assignment / preparation errors. The true syndrome rarely changes between rounds, so conditioning on the
previous round separates the readout part:
    E_even = P(erasure | previous round even)  ~ superposition-window decay + g-side errors,
    E_odd  = P(erasure | previous round odd)   ~ the same + f -> e during readout + f-side errors.
Also: the photon-number dependence, the persistence P(E_k | E_{k-1}) (leakage), and same-round
correlations between ancillas. Compared with the model herald over the two CX gates
(data/validation/stabilizer_faults.csv, lambda = 1 and 3) and with that herald extended to the whole
superposition window at the idle f -> e rate (pumped layers of max length; padding unknown, so a lower bound).
VALIDATION ONLY.

Usage: python examples/zenodo_erasure_ocelot.py
Output: data/validation/zenodo_erasures.csv, figures/zenodo_erasures.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZEN = ROOT / "literature" / "external_data" / "zenodo_14257632" / "data_upload"
SECTIONS = {"d3_S1-S3": ("first_d3_phase_flip", ("A1", "A2")), "d3_S3-S5": ("second_d3_phase_flip", ("A3", "A4")),
            "d5_S1-S5": ("d5_phase_flip", ("A1", "A2", "A3", "A4"))}
EVEN, ERASED, ODD = 0, 1, 2
MIN_ROUNDS = 5


def runs(folder: pathlib.Path):
    for p in sorted(folder.iterdir(), key=lambda q: int(q.name.split("_")[-1]) if q.name.startswith("num_cycles_") else -1):
        m = re.fullmatch(r"num_cycles_(\d+)", p.name)
        if m and int(m.group(1)) >= MIN_ROUNDS:
            yield int(m.group(1)), p


def erasure_stats(folder: pathlib.Path, ancs):
    """Pooled counts over all runs with >= MIN_ROUNDS rounds; interior rounds 1 .. N-2 only."""
    acc = {a: dict(n=0, e=0, n_even=0, e_even=0, n_odd=0, e_odd=0, n_even2=0, e_even2=0, n_odd2=0, e_odd2=0,
                   n_pairs=0, changes=0, n_after_e=0, e_after_e=0, first_n=0, first_e=0, last_n=0, last_e=0) for a in ancs}
    pair = {}
    for N, path in runs(folder):
        t = pq.read_table(path)
        syn = {a: np.stack([t.column(f"{a}_syndrome_{k}").to_numpy() for k in range(N)], axis=1) for a in ancs}
        for a in ancs:
            s = syn[a]
            er = s == ERASED
            mid = er[:, 1:N - 1]
            prev = s[:, 0:N - 2]
            c = acc[a]
            c["n"] += mid.size
            c["e"] += int(mid.sum())
            nxt = s[:, 2:N]
            for key, val in (("even", EVEN), ("odd", ODD)):
                m = prev == val
                c[f"n_{key}"] += int(m.sum())
                c[f"e_{key}"] += int((mid & m).sum())
                # both neighbours agree: the true syndrome of round k differs only after two changes / errors
                m2 = m & (nxt == val)
                c[f"n_{key}2"] += int(m2.sum())
                c[f"e_{key}2"] += int((mid & m2).sum())
            ok = (s[:, 1:] != ERASED) & (s[:, :-1] != ERASED)  # syndrome changes between consecutive reported rounds
            c["n_pairs"] += int(ok.sum())
            c["changes"] += int((ok & (s[:, 1:] != s[:, :-1])).sum())
            m = prev == ERASED
            c["n_after_e"] += int(m.sum())
            c["e_after_e"] += int((mid & m).sum())
            c["first_n"] += s.shape[0]
            c["first_e"] += int(er[:, 0].sum())
            c["last_n"] += s.shape[0]
            c["last_e"] += int(er[:, N - 1].sum())
        for i, a in enumerate(ancs):
            for b in ancs[i + 1:]:
                ea, eb = syn[a][:, 1:N - 1] == ERASED, syn[b][:, 1:N - 1] == ERASED
                q = pair.setdefault((a, b), dict(n=0, ea=0, eb=0, eab=0))
                q["n"] += ea.size
                q["ea"] += int(ea.sum())
                q["eb"] += int(eb.sum())
                q["eab"] += int((ea & eb).sum())
    return acc, pair


def frac(k, n):
    p = k / n if n else np.nan
    return p, (np.sqrt(p * (1 - p) / n) if n else np.nan)


def main():
    pre = load_preset("ocelot")
    stab = {(r["stabilizer"].split(":")[0], r["variant"], float(r["nbar"])): r
            for r in csv.DictReader(l for l in open(ROOT / "data" / "validation" / "stabilizer_faults.csv", encoding="utf-8")
                                    if not l.startswith("#"))}
    rows, pairs_out = [], []
    for sec, (name, ancs) in SECTIONS.items():
        # superposition window: the two CX layers of the section (front/back padding of the shorter gates)
        L1 = max(float(stab[(a, "zero-fit", 1.0)]["gate_a_ns"]) for a in ancs)
        L2 = max(float(stab[(a, "zero-fit", 1.0)]["gate_b_ns"]) for a in ancs)
        for nbdir in sorted((ZEN / name).iterdir()):
            nb = float(nbdir.name.split("_")[1])
            acc, pair = erasure_stats(nbdir, ancs)
            for a in ancs:
                c = acc[a]
                row = dict(section=sec, nbar=nb, ancilla=a, rounds=c["n"])
                for key, (k, n) in dict(all=(c["e"], c["n"]), even=(c["e_even"], c["n_even"]), odd=(c["e_odd"], c["n_odd"]),
                                        even2=(c["e_even2"], c["n_even2"]), odd2=(c["e_odd2"], c["n_odd2"]),
                                        after_erasure=(c["e_after_e"], c["n_after_e"]), first=(c["first_e"], c["first_n"]),
                                        last=(c["last_e"], c["last_n"])).items():
                    row[f"data_E_{key}"], row[f"data_E_{key}_err"] = frac(k, n)
                row["data_change_prob"], row["data_change_prob_err"] = frac(c["changes"], c["n_pairs"])
                # residual mixing of the two-sided classes: the true syndrome of round k differs from both reported
                # neighbours after two changes, m ~ q^2 / (q^2 + (1-q)^2) with q the observed change probability (which
                # also contains measurement errors, so m is approximate); unmix E_g (g side) and E_f (f side)
                q = row["data_change_prob"]
                m = q * q / (q * q + (1 - q) ** 2)
                ee, eo = row["data_E_even2"], row["data_E_odd2"]
                row["mixing_m"] = m
                row["data_E_g_unmixed"] = ((1 - m) * ee - m * eo) / (1 - 2 * m)
                row["data_E_f_unmixed"] = ((1 - m) * eo - m * ee) / (1 - 2 * m)
                own = float(stab[(a, "zero-fit", 1.0)]["gate_a_ns"]) + float(stab[(a, "zero-fit", 1.0)]["gate_b_ns"])
                idle_fe = 1.0 / (pre["ancilla"][a]["T1_ef_us"] * 1e3)  # 1/ns, coupler off
                pad = 0.5 * idle_fe * (L1 + L2 - own)  # f->e while waiting for the other layer, at the idle rate
                h = {}
                for v in ("zero-fit", "lam2", "lam3"):
                    if (a, v, nb) in stab:
                        h[v] = float(stab[(a, v, nb)]["p_herald"])
                        row[f"model_herald_gates_{v}"] = h[v]
                        row[f"model_herald_window_{v}"] = h[v] + pad
                # on-gate scaling that would account for all even-conditioned erasures (an upper bound: E_even also
                # contains assignment and preparation errors, and the unknown padding is not included)
                if len(h) == 3:
                    lams, hv = np.array([1.0, 2.0, 3.0]), np.array([h["zero-fit"], h["lam2"], h["lam3"]])
                    target = row["data_E_g_unmixed"] - pad
                    row["lambda_f_upper"] = float(np.interp(target, hv, lams, left=np.nan, right=np.inf)) \
                        if target <= hv[-1] else float(3.0 + (target - hv[-1]) / (hv[-1] - hv[-2]))
                rows.append(row)
            for (a, b), q in pair.items():
                pa, pb, pab = q["ea"] / q["n"], q["eb"] / q["n"], q["eab"] / q["n"]
                pairs_out.append(dict(section=sec, nbar=nb, pair=f"{a}-{b}", neighbours=int(int(b[1]) - int(a[1]) == 1),
                                      p_a=pa, p_b=pb, p_ab=pab, ratio=pab / (pa * pb), ratio_err=np.sqrt(q["eab"]) / q["n"] / (pa * pb)))
    path = ROOT / "data" / "validation" / "zenodo_erasures.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Ancilla erasure fractions per round (interior rounds of runs with >= 5 rounds) from the raw syndromes of "
                 "Zenodo 10.5281/zenodo.14257632 (Putterman et al., CC BY 4.0; attribution required); data_* = derived from "
                 "those data (VALIDATION ONLY); model_* = our f->e herald over the two CX gates, and over the superposition "
                 "window at the idle rate. examples/zenodo_erasure_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in keys) + "\n")
    print("section  nbar anc | E_all  E_even E_odd | E_even2 E_odd2 | change | E|E_prev first last (x1e-2) | "
          "model herald gates / window: zero-fit, lam3 (x1e-2) | lambda_f upper bound")
    for r in rows:
        print(f"{r['section']} {r['nbar']:.1f} {r['ancilla']} | {r['data_E_all'] * 100:5.2f} {r['data_E_even'] * 100:5.2f} "
              f"{r['data_E_odd'] * 100:5.2f} | {r['data_E_even2'] * 100:5.2f} {r['data_E_odd2'] * 100:5.2f} | "
              f"{r['data_change_prob'] * 100:5.2f} | {r['data_E_after_erasure'] * 100:6.2f} {r['data_E_first'] * 100:5.2f} "
              f"{r['data_E_last'] * 100:5.2f} | {r.get('model_herald_gates_zero-fit', np.nan) * 100:4.2f} / "
              f"{r.get('model_herald_window_zero-fit', np.nan) * 100:4.2f}, {r.get('model_herald_gates_lam3', np.nan) * 100:4.2f} / "
              f"{r.get('model_herald_window_lam3', np.nan) * 100:4.2f} | {r.get('lambda_f_upper', np.nan):.2f}")
    print("\nper ancilla (d5 and d3 runs pooled over |alpha|^2): median lambda_f upper bound, E_even range, E_odd - E_even")
    for a in ("A1", "A2", "A3", "A4"):
        sel = [r for r in rows if r["ancilla"] == a]
        slope = np.polyfit([r["nbar"] for r in sel if r["section"] == "d5_S1-S5"],
                           [r["data_E_g_unmixed"] for r in sel if r["section"] == "d5_S1-S5"], 1)[0]
        print(f"  {a}: lambda_f <= {np.nanmedian([r['lambda_f_upper'] for r in sel]):.2f}; E_g (unmixed) "
              f"{min(r['data_E_g_unmixed'] for r in sel) * 100:.2f}-{max(r['data_E_g_unmixed'] for r in sel) * 100:.2f}% "
              f"(d5 slope {slope * 100:+.2f}%/photon); E_f - E_g "
              f"{min(r['data_E_f_unmixed'] - r['data_E_g_unmixed'] for r in sel) * 100:.2f}-"
              f"{max(r['data_E_f_unmixed'] - r['data_E_g_unmixed'] for r in sel) * 100:.2f}%; "
              f"persistence E|E_prev - E_all {np.median([r['data_E_after_erasure'] - r['data_E_all'] for r in sel]) * 100:+.2f}%")
    print("\nsame-round erasure correlation P(E_a E_b) / (P(E_a) P(E_b)):")
    for q in pairs_out:
        if q["section"] == "d5_S1-S5":
            print(f"  {q['section']} {q['nbar']:.1f} {q['pair']} (neighbours {q['neighbours']}): {q['ratio']:.2f} +- {q['ratio_err']:.2f}")
    use_style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, a in enumerate(("A1", "A2", "A3", "A4")):
        sel = [r for r in rows if r["section"] == "d5_S1-S5" and r["ancilla"] == a]
        x = [r["nbar"] for r in sel]
        axs[0].errorbar(x, [r["data_E_even2"] for r in sel], yerr=[r["data_E_even2_err"] for r in sel], fmt="o-", ms=4,
                        color=SERIES[i], label=f"{a}: neighbours even")
        axs[0].errorbar(x, [r["data_E_odd2"] for r in sel], yerr=[r["data_E_odd2_err"] for r in sel], fmt="s--", ms=3,
                        mfc="none", color=SERIES[i], label=f"{a}: neighbours odd")
        axs[0].plot(x, [r["data_E_even"] for r in sel], ":", lw=0.7, color=SERIES[i])
        axs[1].plot(x, [r["data_E_g_unmixed"] for r in sel], "o", color=SERIES[i], label=f"{a} data (g side, unmixed)")
        axs[1].plot(x, [r.get("model_herald_window_zero-fit", np.nan) for r in sel], "-", lw=0.8, color=SERIES[i])
        axs[1].plot(x, [r.get("model_herald_window_lam3", np.nan) for r in sel], ":", lw=1.0, color=SERIES[i])
    axs[0].set_ylabel("erasure probability per round")
    axs[0].set_title("d5: erasures, both neighbouring rounds even (o) / odd (s)\n(dotted: previous round even only)", fontsize=9)
    axs[1].set_title("d5: g-side erasures (unmixed) vs model herald over the window\n(solid: zero-fit, dotted: λ = 3)", fontsize=9)
    for ax in axs:
        ax.set_xlabel(r"$|\alpha|^2$")
        ax.set_ylim(0, None)
    axs[0].legend(fontsize=6, ncol=2)
    axs[1].legend(fontsize=6)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"zenodo_erasures.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
