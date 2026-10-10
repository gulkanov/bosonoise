"""Logical bit-flip error per cycle from the raw Ocelot shots (Zenodo 10.5281/zenodo.14257632, CC BY 4.0).

The public bit-flip runs contain, per shot, the initial Z state of every cat (all |alpha> or all |-alpha>),
the number of syndrome-extraction cycles (2.8 us each) and the single-shot final Z state of every cat.
After the first stabilizer round the individual Z_i are randomized (X_i X_{i+1} projects |alpha, alpha>
onto (|alpha, alpha> +- |-alpha, -alpha>)/sqrt2), so only Z_L = Z_1 ... Z_d carries information: the data
give the logical bit-flip rate, not single-cat rates or X_i X_{i+1} correlations.

As in the paper (App. G/H): <Z_L(0) Z_L(n)> is fitted with A exp(-n T_cycle / T_Z) (no offset, points
weighted by their standard error), per initial state; the two states are combined by the mean T_Z;
eps = T_cycle / (2 T_Z). Our model comparison uses the exact per-cycle value, which coincides to < 1%.
This replaces the digitized Fig. 4b / Fig. 31 values by fits with statistical uncertainties.
VALIDATION ONLY (nothing here calibrates the model).

Usage: python examples/zenodo_bitflip_ocelot.py
Output: data/validation/zenodo_logical_bitflip.csv, figures/zenodo_logical_bitflip.png/.pdf
"""
from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import curve_fit

from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style

ROOT = pathlib.Path(__file__).resolve().parents[1]
ZEN = ROOT / "literature" / "external_data" / "zenodo_14257632" / "data_upload"  # unzipped data_upload.zip
T_CYCLE_US = 2.8
FILES = {"d3_S1-S3": ("first_d3_bit_flip", 3), "d3_S3-S5": ("second_d3_bit_flip", 3), "d5_S1-S5": ("d5_bit_flip", 5)}


def logical_correlator(name: str, d: int):
    """{(nbar, init): (n_cycles array, <Z_L(0) Z_L(n)> array, standard error array, shots per point)}."""
    t = pq.read_table(ZEN / name)
    cats = [c.split("_")[0] for c in t.column_names if c.endswith("_final_state")]  # S1..S3, S3..S5 or S1..S5
    assert len(cats) == d
    nbar = t.column("nbar").to_numpy()
    ncyc = t.column("num_cycles").to_numpy()
    init = np.array(t.column(f"{cats[0]}_initial_state").to_pylist())
    for c in cats[1:]:  # the runs use all-+Z or all--Z initial states
        assert np.all(np.array(t.column(f"{c}_initial_state").to_pylist()) == init)
    final = np.stack([t.column(f"{c}_final_state").to_numpy() for c in cats])
    parity = final.sum(axis=0) % 2
    assert np.all(parity == t.column("logical_state").to_numpy())
    zl0 = np.where(init == "-Z", (-1) ** d, 1)  # Z_L of the initial state
    corr = zl0 * (1 - 2 * parity)
    out = {}
    for nb in np.unique(nbar):
        for s in ("+Z", "-Z"):
            m = (nbar == nb) & (init == s)
            ns = np.unique(ncyc[m])
            mean, sem, cnt = [], [], []
            for n in ns:
                v = corr[m & (ncyc == n)]
                mu = v.mean()
                mean.append(mu)
                sem.append(np.sqrt(max(1.0 - mu ** 2, 1e-4) / len(v)))
                cnt.append(len(v))
            out[(float(nb), s)] = (ns, np.array(mean), np.array(sem), np.array(cnt))
    return out


def fit_tz(n, y, s):
    f = lambda n, A, T: A * np.exp(-n * T_CYCLE_US / T)
    T0 = T_CYCLE_US / max(1e-4, -np.log(max(y[len(y) // 3], 1e-3) / max(y[0], 1e-3)) / n[len(n) // 3])
    popt, pcov = curve_fit(f, n, y, p0=(y[0], T0), sigma=s, absolute_sigma=True, maxfev=20000)
    chi2 = float(np.sum(((y - f(n, *popt)) / s) ** 2) / (len(n) - 2))
    return popt[0], popt[1], float(np.sqrt(pcov[1, 1])), chi2


def main():
    fig4 = {(r["section"], float(r["alpha2"])): float(r["eps_L_bitflip"])
            for r in csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "ocelot_fig4b.csv", encoding="utf-8")
                                    if not l.startswith("#")) if r["series"] == "data"}
    rows, curves = [], {}
    for sec, (name, d) in FILES.items():
        data = logical_correlator(name, d)
        for nb in sorted({k[0] for k in data}):
            T, dT = [], []
            row = dict(section=sec, nbar=nb)
            for s in ("+Z", "-Z"):
                n, y, e, cnt = data[(nb, s)]
                A, Tz, dTz, chi2 = fit_tz(n, y, e)
                T.append(Tz)
                dT.append(dTz)
                tag = "p" if s == "+Z" else "m"
                row.update({f"data_A_{tag}": A, f"data_TZ_us_{tag}": Tz, f"data_TZ_err_{tag}": dTz, f"data_chi2red_{tag}": chi2,
                            f"shots_per_point_{tag}": float(np.median(cnt))})
                curves[(sec, nb, s)] = (n, y, e, A, Tz)
            Tm, dTm = float(np.mean(T)), float(0.5 * np.hypot(*dT))
            eps = T_CYCLE_US / (2 * Tm)
            row.update(data_TZ_us=Tm, data_TZ_err=dTm, data_eps=eps, data_eps_err=eps * dTm / Tm,
                       data_eps_exact=0.5 * (1 - np.exp(-T_CYCLE_US / Tm)), fig4b_digitized=fig4.get((sec, nb), np.nan))
            rows.append(row)
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / "zenodo_logical_bitflip.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Logical bit-flip error per 2.8-us cycle fitted from the raw shots of Zenodo 10.5281/zenodo.14257632 "
                 "(Putterman et al., CC BY 4.0; attribution required). data_* = derived from those data (VALIDATION ONLY). "
                 "examples/zenodo_bitflip_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("section  nbar | eps (x1e-3) +- stat | digitized Fig. 4b | ratio | T_Z +Z / -Z (us) | chi2_red +Z / -Z")
    for r in rows:
        print(f"{r['section']} {r['nbar']:.1f} | {r['data_eps'] * 1e3:6.3f} +- {r['data_eps_err'] * 1e3:5.3f} | "
              f"{r['fig4b_digitized'] * 1e3:6.3f} | {r['fig4b_digitized'] / r['data_eps']:5.3f} | "
              f"{r['data_TZ_us_p']:7.1f} / {r['data_TZ_us_m']:7.1f} | {r['data_chi2red_p']:4.2f} / {r['data_chi2red_m']:4.2f}")
    use_style()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, sec in zip(axs, FILES):
        for i, nb in enumerate((1.0, 2.0, 3.0, 4.0)):
            for s, mk in (("+Z", "o"), ("-Z", "s")):
                n, y, e, A, Tz = curves[(sec, nb, s)]
                ax.errorbar(n, y, yerr=e, fmt=mk, ms=2.5, mfc="none", color=SERIES[i], lw=0.6,
                            label=f"|α|² = {nb:g}" if s == "+Z" else None)
                nn = np.linspace(0, n.max(), 200)
                ax.plot(nn, A * np.exp(-nn * T_CYCLE_US / Tz), "-", lw=0.8, color=SERIES[i])
        ax.set_title(f"{sec}: <Z_L(0) Z_L(n)> (circles |α>, squares |-α>)", fontsize=8)
        ax.set_xlabel("number of cycles")
        ax.axhline(0, color=TEXT_PRIMARY, lw=0.5)
    axs[0].set_ylabel("logical Z correlator")
    axs[0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"zenodo_logical_bitflip.{ext}", dpi=160)
    print("saved", path)


if __name__ == "__main__":
    main()
