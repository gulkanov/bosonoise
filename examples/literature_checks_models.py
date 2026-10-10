"""Literature checks of the two literature-based reference noise models (dem/models.py).

A. allcat_model against Le Regent et al., Quantum 7, 1198 (2023) [LR23], Fig. 4b: circuit-level repetition cat code at
   T = 1/kappa2 with refreshing steps, |alpha|^2 = 8, MWPM. LR23 (App. A) prepare the +1 eigenstate of all stabilizers,
   run d noisy rounds and a final perfect round, and fit p_ZL = a d (eta/eta_th)^(c d) with a = 3.2e-2, c = 0.44,
   eta_th = 2.3e-3 (Fig. 4b caption). Same protocol here: X-basis memory, R = d, perfect initial and final readout.
   Reported: the ratio to their fit at every point, the crossing points of the d and d + 2 curves (the threshold),
   and our own fit of the same ansatz in the asymptotic regime eta <= 1e-3 (it fits the scaling ansatz to our
   simulation, not to error data).
A2. allcat_model bit flips against LR23 p. 8: p_XL <= 2 (d - 1) p_X^CNOT per round, p_X^CNOT = 0.5 exp(-2|alpha|^2).
B. kerrcat_model against Darmawan et al., PRX Quantum 2, 030345 (2021) [DA21], p. 14 and Fig. 9b: n = 9 repetition code,
   3 rounds, |alpha|^2 = 6.25, kappa/K = 2.08e-4, exact density-matrix simulation with optimal decoding, sequential checks,
   leakage neglected: total logical error 1.62 +- 0.02 %, "almost exclusively" bit flips. Here: d = 9, R = 3, perfect
   initial and final readout, logical bit flips (Z basis, no decoding) plus logical phase flips (X basis, MWPM), with
   and without our leakage.

Usage: python examples/literature_checks_models.py [--quick (smoke test, writes to runs/)] [--plot (redraw from the CSV)]
Output: data/validation/literature_checks_models.csv, figures/literature_checks_models.png/.pdf
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from bosonoise.dem.models import allcat_model, kerrcat_model
from bosonoise.dem.noise_models import Leakage, memory_circuit, sample_with_leakage

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "data" / "validation" / "literature_checks_models.csv"
OUT_FIG = ROOT / "figures" / "literature_checks_models"
LR23 = dict(a=3.2e-2, c=0.44, eta_th=2.3e-3)
ETAS = [5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
DS = [3, 5, 7, 9, 11]


def lr23_fit(d, eta, a, c, eta_th):
    return a * d * (eta / eta_th) ** (c * d)


def count_failures(circuit, basis, target, max_shots, batch, seed):
    """Logical failures of a leakage-free memory circuit (X basis: correlated MWPM; Z basis: no decoding)."""
    import pymatching

    sampler = circuit.compile_detector_sampler(seed=seed)
    m = None
    if basis == "X":
        dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    fails = shots = 0
    while shots < max_shots and fails < target:
        det, obs = sampler.sample(batch, separate_observables=True)
        pred = m.decode_batch(det, enable_correlations=True)[:, 0] if m is not None else np.zeros(batch, dtype=bool)
        fails += int(np.sum(pred != obs[:, 0]))
        shots += batch
    return fails, shots


def count_with_leakage(d, rounds, noise, basis, shots, batch, seed):
    import pymatching

    fails = done = 0
    m = None
    while done < shots:
        out = sample_with_leakage(d, rounds, noise, batch, basis=basis, seed=seed + done)
        if basis == "X":
            if m is None:
                dem = out["circuit"].detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
                m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
            pred = m.decode_batch(out["detectors"], enable_correlations=True)[:, 0]
        else:
            pred = np.zeros(batch, dtype=bool)
        fails += int(np.sum(pred != out["observables"][:, 0]))
        done += batch
    return fails, done


def crossings(rows):
    """eta at which the d and d + 2 curves cross (log-log interpolation of P_{d+2}/P_d on the eta grid)."""
    P = {(r["d"], r["eta"]): max(r["P"], 1e-12) for r in rows}
    out = {}
    for d1, d2 in zip(DS[:-1], DS[1:]):
        lr = [np.log(P[(d2, e)] / P[(d1, e)]) for e in ETAS]
        for k in range(len(ETAS) - 1):
            if lr[k] < 0 <= lr[k + 1]:
                f = -lr[k] / (lr[k + 1] - lr[k])
                out[(d1, d2)] = float(np.exp(np.log(ETAS[k]) + f * np.log(ETAS[k + 1] / ETAS[k])))
    return out


def perfect_ends(noise):
    noise.p_data_meas = 0.0
    noise.p_data_meas_z = 0.0
    return noise


def fit_ansatz(rows):
    """Least squares of log p = log a + log d + c d (log eta - log eta_th) over the points with >= 50 failures."""
    A, y, w = [], [], []
    for r in rows:
        if r["fails"] < 50:
            continue
        A.append([1.0, r["d"] * np.log(r["eta"]), -r["d"]])  # log a, c, c log eta_th
        y.append(np.log(r["P"]) - np.log(r["d"]))
        w.append(np.sqrt(r["fails"]))
    A, y, w = np.array(A), np.array(y), np.array(w)
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    la, c, clt = sol
    return dict(a=float(np.exp(la)), c=float(c), eta_th=float(np.exp(clt / c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer shots (smoke test)")
    ap.add_argument("--plot", action="store_true", help="redraw the figure from the CSV")
    a = ap.parse_args()
    if a.plot:
        return replot()
    target, max_shots, batch = (60, 100_000, 20_000) if a.quick else (300, 1_000_000, 20_000)
    out_csv, out_fig = OUT_CSV, OUT_FIG
    if a.quick:  # a smoke test must not overwrite the production outputs
        (ROOT / "runs").mkdir(exist_ok=True)
        out_csv, out_fig = ROOT / "runs" / "literature_checks_quick.csv", ROOT / "runs" / "literature_checks_quick"
    rows = []

    # A. all-cat phase flips versus LR23 Fig. 4b
    lr_rows = []
    for i, eta in enumerate(ETAS):
        for d in DS:
            noise = perfect_ends(allcat_model(8.0, d, eta=eta))
            fails, shots = count_failures(memory_circuit(d, d, noise, "X"), "X", target, max_shots, batch,
                                          seed=7000 + 100 * i + d)
            P = fails / shots
            ref = lr23_fit(d, eta, **LR23)
            r = dict(check="A_allcat_vs_LR23_fig4b", d=d, rounds=d, eta=eta, nbar=8.0, basis="X", shots=shots,
                     fails=fails, P=P, P_std=np.sqrt(P * (1 - P) / shots), reference=ref, ratio=P / ref if ref else np.nan)
            lr_rows.append(r)
            print(f"A  eta {eta:.1e} d {d:2d}: P {P:.3e} ({fails}/{shots}), LR23 fit {ref:.3e}, ratio {P / ref:.2f}", flush=True)
    rows += lr_rows
    cross = crossings(lr_rows)
    for (d1, d2), e in cross.items():
        print(f"A  curves d = {d1} and {d2} cross at eta = {e:.3e} (LR23 eta_th = 2.3e-3)", flush=True)
        rows.append(dict(check=f"A_crossing_d{d1}_d{d2}", d=d1, rounds=np.nan, eta=e, nbar=8.0, basis="X",
                         reference=LR23["eta_th"], ratio=e / LR23["eta_th"]))
    ours = fit_ansatz([r for r in lr_rows if r["eta"] <= 1e-3])
    print(f"A  our fit of the LR23 ansatz (eta <= 1e-3): a = {ours['a']:.3g}, c = {ours['c']:.3g}, "
          f"eta_th = {ours['eta_th']:.3g}  (LR23: a = 3.2e-2, c = 0.44, eta_th = 2.3e-3)", flush=True)
    rows.append(dict(check="A_fit_our_points_eta_le_1e-3", nbar=8.0, basis="X", **{f"fit_{k}": v for k, v in ours.items()}))

    # A2. all-cat bit flips versus the LR23 bound 2 (d - 1) p_X^CNOT per round (nbar = 3 so that they are visible)
    nb, d, R = 3.0, 5, 20
    noise = perfect_ends(allcat_model(nb, d, eta=1e-3))
    fails, shots = count_failures(memory_circuit(d, R, noise, "Z"), "Z", 10 ** 9, 200_000, batch, seed=7900)
    P = fails / shots
    eps = 0.5 * (1 - (1 - 2 * P) ** (1 / R))
    bound = 2 * (d - 1) * 0.5 * np.exp(-2 * nb)
    rows.append(dict(check="A2_allcat_bitflip_vs_LR23_bound", d=d, rounds=R, eta=1e-3, nbar=nb, basis="Z", shots=shots,
                     fails=fails, P=eps, P_std=np.sqrt(P * (1 - P) / shots) / R, reference=bound, ratio=eps / bound))
    print(f"A2 nbar {nb} d {d}: bit flips per round {eps:.3e}, LR23 bound {bound:.3e}, ratio {eps / bound:.3f} "
          f"(expected 2/3: 8 of the 12 bit-flip types change the parity)", flush=True)

    # B. Kerr cat, n = 9 repetition code, 3 rounds, versus DA21 (1.62 +- 0.02 %)
    d, R, kK = 9, 3, 2.08e-4
    res = {}
    for leak in (False, True):
        for basis in ("Z", "X"):
            noise = perfect_ends(kerrcat_model(6.25, d, kappa_over_K=kK))
            if not leak:
                noise.leak_data, noise.leak_anc = Leakage(), Leakage()
                fails, shots = count_failures(memory_circuit(d, R, noise, basis), basis, 10 ** 9, 400_000, batch,
                                              seed=8000 + (basis == "X"))
            else:
                fails, shots = count_with_leakage(d, R, noise, basis, 400_000, batch, seed=8100 + 10 ** 6 * (basis == "X"))
            P = fails / shots
            res[(leak, basis)] = P
            rows.append(dict(check=f"B_kerrcat_vs_DA21_rep9_{'leak' if leak else 'noleak'}_{'bitflip' if basis == 'Z' else 'phaseflip'}",
                             d=d, rounds=R, eta=kK, nbar=6.25, basis=basis, shots=shots, fails=fails, P=P,
                             P_std=np.sqrt(P * (1 - P) / shots), reference=0.0162, ratio=P / 0.0162))
            print(f"B  leakage {leak}, basis {basis}: P {P:.4f} ({fails}/{shots})", flush=True)
        tot = res[(leak, "Z")] + res[(leak, "X")]
        print(f"B  leakage {leak}: total {tot:.4f} vs DA21 0.0162 (ratio {tot / 0.0162:.2f})", flush=True)

    out = out_csv
    keys = ["check", "d", "rounds", "eta", "nbar", "basis", "shots", "fails", "P", "P_std", "reference", "ratio",
            "fit_a", "fit_c", "fit_eta_th"]
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Literature checks of the all-cat (LR23 Fig. 4b) and Kerr-cat (DA21 n = 9 repetition code) models; "
                 "examples/literature_checks_models.py; eta column = kappa/K for the B rows\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) if isinstance(r.get(k, ""), str) else
                              ("" if r.get(k) is None or (isinstance(r.get(k), float) and np.isnan(r.get(k)))
                               else f"{r.get(k):.6g}") for k in keys) + "\n")
    print("saved", out)
    plot(lr_rows, ours, cross, res, out_fig)


def replot():
    import csv

    rows = list(csv.DictReader(l for l in open(OUT_CSV, encoding="utf-8") if not l.startswith("#")))
    num = lambda r, k: float(r[k]) if r[k] != "" else np.nan
    lr_rows = [dict(d=int(num(r, "d")), eta=num(r, "eta"), P=num(r, "P"), P_std=num(r, "P_std"), fails=num(r, "fails"))
               for r in rows if r["check"] == "A_allcat_vs_LR23_fig4b"]
    fit = next(r for r in rows if r["check"].startswith("A_fit_our_points"))
    ours = dict(a=num(fit, "fit_a"), c=num(fit, "fit_c"), eta_th=num(fit, "fit_eta_th"))
    cross = {(int(num(r, "d")), 0): num(r, "eta") for r in rows if r["check"].startswith("A_crossing")}
    res = {}
    for r in rows:
        if r["check"].startswith("B_kerrcat"):
            res[("_leak_" in r["check"], "Z" if r["check"].endswith("bitflip") else "X")] = num(r, "P")
    plot(lr_rows, ours, cross, res, OUT_FIG)


def plot(lr_rows, ours, cross, res, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bosonoise.plotting import use_style

    use_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    eg = np.geomspace(4e-4, 3.5e-3, 50)
    for k, d in enumerate(DS):
        col = f"C{k}"
        pts = [r for r in lr_rows if r["d"] == d and r["fails"] > 0]
        ax.errorbar([r["eta"] for r in pts], [r["P"] for r in pts], yerr=[r["P_std"] for r in pts], fmt="o", ms=4,
                    color=col, label=f"d = {d}")
        ax.plot(eg, lr23_fit(d, eg, **LR23), "--", color=col, lw=1)
    ax.axvline(LR23["eta_th"], color="0.5", lw=0.8, ls=":")
    if cross:
        ax.axvspan(min(cross.values()), max(cross.values()), color="0.85", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\eta=\kappa_1/\kappa_2$")
    ax.set_ylabel(r"$p_{Z_L}$ (d rounds + perfect round)")
    ax.set_title(r"all-cat model, $|\alpha|^2=8$: points ours, dashed LR23 Fig. 4b fit", fontsize=9)
    ax.text(0.02, 0.97, r"our fit ($\eta\leq10^{-3}$): " + f"a = {ours['a']:.2g}, c = {ours['c']:.2f}, " + r"$\eta_{th}$"
            + f" = {ours['eta_th']:.2g}\n" + r"LR23: a = 3.2e-2, c = 0.44, $\eta_{th}$ = 2.3e-3 (dotted)" + "\n"
            + "grey band: crossings of our d, d + 2 curves", transform=ax.transAxes, va="top", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    labels = ["bit flips\n(no leakage)", "phase flips\n(no leakage)", "total\n(no leakage)", "total\n(with leakage)"]
    vals = [res[(False, "Z")], res[(False, "X")], res[(False, "Z")] + res[(False, "X")], res[(True, "Z")] + res[(True, "X")]]
    bx.bar(range(4), vals, color=["C0", "C1", "C2", "C3"])
    bx.axhline(0.0162, color="k", ls="--", lw=1)
    bx.text(3.4, 0.0165, "DA21 exact, optimal decoding: 1.62%", ha="right", va="bottom", fontsize=8)
    bx.set_xticks(range(4))
    bx.set_xticklabels(labels, fontsize=8)
    bx.set_ylabel("logical error, 3 rounds")
    bx.set_title("Kerr-cat model, n = 9 repetition code, 3 rounds\n" + r"$|\alpha|^2=6.25$, $\kappa/K=2.08\times10^{-4}$",
                 fontsize=9)
    bx.set_ylim(0, 0.019)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    print("saved", f"{stem}.png/.pdf")


if __name__ == "__main__":
    main()
