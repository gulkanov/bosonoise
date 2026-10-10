"""Detection-event statistics of the Ocelot repetition code from the raw syndromes (Zenodo 10.5281/zenodo.14257632).

Detection event D_k(A) = [s_k(A) != s_{k-1}(A)] for non-erased consecutive rounds. Pair correlations give the
probabilities of the error mechanisms that flip exactly those two detectors (standard estimator, e.g. Google's
"p_ij" analysis): p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4 <x_i x_j>)).
  time-like  (same ancilla, rounds k, k+1): a wrong syndrome in round k = measurement error p_m per round
             (readout assignment, preparation/unpreparation errors, and our model's silent syndrome faults: an ancilla
             that ends in the wrong branch gives a random bit, i.e. an error with probability 1/2);
  space-like (neighbouring ancillas, same round): a phase flip of the shared cat between two rounds.
Rounds with an erasure on any involved detector are excluded. Compared with the model's silent-syndrome
contribution p_silent/2 (data/validation/stabilizer_faults.csv) and with the phase-flip probability per cycle
kappa_1,eff |alpha|^2 T_cycle (T1,eff of Fig. 30). VALIDATION ONLY.

Usage: python examples/zenodo_detection_ocelot.py
Output: data/validation/zenodo_detection.csv
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import pyarrow.parquet as pq

from zenodo_erasure_ocelot import ERASED, SECTIONS, ZEN, runs  # same directory

ROOT = pathlib.Path(__file__).resolve().parents[1]
T_REP_US = 2.8
T1_EFF_US = {"S1": 60.0, "S2": 55.0, "S3": 51.0, "S4": 61.0, "S5": 59.0}  # Fig. 30 (d = 5 run)
STAB = {"A1": ("S1", "S2"), "A2": ("S2", "S3"), "A3": ("S3", "S4"), "A4": ("S4", "S5")}


def pij(xi, xj, xij):
    den = 1 - 2 * xi - 2 * xj + 4 * xij
    arg = 1 - 4 * (xij - xi * xj) / den
    return 0.5 - 0.5 * np.sqrt(max(arg, 0.0))


def detection_stats(folder, ancs):
    acc = {("t", a): np.zeros(4) for a in ancs}  # n, sum xi, sum xj, sum xixj
    for i in range(len(ancs) - 1):
        acc[("s", ancs[i], ancs[i + 1])] = np.zeros(4)
    for N, path in runs(folder):
        t = pq.read_table(path)
        s = {a: np.stack([t.column(f"{a}_syndrome_{k}").to_numpy() for k in range(N)], axis=1) for a in ancs}
        D, ok = {}, {}
        for a in ancs:
            sa = s[a]
            ok[a] = (sa[:, 1:] != ERASED) & (sa[:, :-1] != ERASED)  # detector k (between rounds k-1, k), k = 1..N-1
            D[a] = (sa[:, 1:] != sa[:, :-1])
        for a in ancs:  # time-like: detectors k and k+1 of the same ancilla, interior (k = 1..N-3 -> rounds up to N-2)
            m = ok[a][:, :-2] & ok[a][:, 1:-1]
            xi, xj = D[a][:, :-2][m], D[a][:, 1:-1][m]
            acc[("t", a)] += [m.sum(), xi.sum(), xj.sum(), (xi & xj).sum()]
        for i in range(len(ancs) - 1):  # space-like: same detector index, neighbouring ancillas
            a, b = ancs[i], ancs[i + 1]
            m = ok[a][:, :-1] & ok[b][:, :-1]
            xi, xj = D[a][:, :-1][m], D[b][:, :-1][m]
            acc[("s", a, b)] += [m.sum(), xi.sum(), xj.sum(), (xi & xj).sum()]
    return acc


def main():
    stab = {(r["stabilizer"].split(":")[0], r["variant"], float(r["nbar"])): r
            for r in csv.DictReader(l for l in open(ROOT / "data" / "validation" / "stabilizer_faults.csv", encoding="utf-8")
                                    if not l.startswith("#"))}
    rows = []
    for sec, (name, ancs) in SECTIONS.items():
        for nbdir in sorted((ZEN / name).iterdir()):
            nb = float(nbdir.name.split("_")[1])
            acc = detection_stats(nbdir, ancs)
            for key, (n, si, sj, sij) in acc.items():
                xi, xj, xij = si / n, sj / n, sij / n
                p = pij(xi, xj, xij)
                row = dict(section=sec, nbar=nb, kind="time" if key[0] == "t" else "space", detectors="-".join(key[1:]),
                           samples=int(n), data_detection_rate=0.5 * (xi + xj), data_p_pair=p)
                if key[0] == "t":
                    a = key[1]
                    for v in ("zero-fit", "lam3"):
                        if (a, v, nb) in stab:
                            row[f"model_silent_half_{v}"] = 0.5 * float(stab[(a, v, nb)]["p_silent_syndrome"])
                else:
                    cat = STAB[key[1]][1]  # the shared cat
                    row["shared_cat"] = cat
                    row["phase_flip_expected"] = 0.5 * (1 - np.exp(-2 * nb * T_REP_US / T1_EFF_US[cat]))
                rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = ROOT / "data" / "validation" / "zenodo_detection.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Detection-event pair probabilities from the raw syndromes of Zenodo 10.5281/zenodo.14257632 (Putterman et al., "
                 "CC BY 4.0; attribution required): time-like = measurement error per round, space-like = phase flip of the "
                 "shared cat per round (data_*); model_* = our silent-syndrome contribution; phase_flip_expected = "
                 "kappa_1,eff |alpha|^2 T_cycle with Fig. 30 T1,eff. examples/zenodo_detection_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r.get(k, np.nan):.6g}" if not isinstance(r.get(k), str) else r[k] for k in keys) + "\n")
    print("section  nbar kind  detectors | detection rate | p_pair (x1e-2) | model silent/2 zero-fit, lam3 (x1e-2) | expected phase flip")
    for r in rows:
        print(f"{r['section']} {r['nbar']:.1f} {r['kind']:5s} {r['detectors']:6s} | {r['data_detection_rate'] * 100:6.2f} | "
              f"{r['data_p_pair'] * 100:6.3f} | {r.get('model_silent_half_zero-fit', np.nan) * 100:6.3f} "
              f"{r.get('model_silent_half_lam3', np.nan) * 100:6.3f} | {r.get('phase_flip_expected', np.nan) * 100:6.2f}")
    print("saved", path)


if __name__ == "__main__":
    main()
