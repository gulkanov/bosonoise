"""Syndrome datasets of the three reference noise models (paper, Data availability): repetition cat code memories,
d = 3, 5, 7, both bases, on a grid of mean photon numbers, in the benchmark format 0.1 (benchmark/schema.py):
circuit.stim, detectors.b8, observables.b8, heralds.b8, leakage.b8, metadata.json (noise parameters with provenance,
generator and seed, baseline decoding).

Sampling: dem/noise_models.sample_with_leakage (stim.FlipSimulator + leakage in the incoherent approximation).
Baseline decoder: X basis, PyMatching on the circuit's DEM (leakage not in the DEM), correlated matching with the
heralds, and without them; Z basis, nothing to decode (no Z stabilizers), a heralded bit flip of a cat is known and
undone. Logical error per round from two memory lengths (R = d and R = 3d; the boundary factor cancels), with its
binomial standard error.

Points per model (the first is the reference point; knobs and provenance in each metadata.json):
  ocelot   nbar 2 (reference), 1, 3, 4; lambda3 variant, T1,eff 57 us, cycle 2.8 us (dem/models.py). The grid stays
           inside the range of the Ocelot data (|alpha|^2 = 1 ... 3.5; the idle table is data-derived up to 4).
  allcat   nbar 8 (reference: the operating point of Le Regent et al., Quantum 7, 1198 (2023), eta = kappa1/kappa2
           = 1e-3, no heralds; bit flips ~1e-7 per CNOT), and 3, 4, 6 with the same eta and buffer-emission heralds
           of the cat bit flips at efficiency 0.5 (an assumption, see the metadata), where the bit flips are visible
           in the Z basis at 3 and 4 (data/syndromes/README.md)
  kerrcat  nbar 6.25 (reference: the design point of Darmawan et al., PRX Quantum 2, 030345 (2021), kappa/K 1.67e-4,
           n_th 8%, K/2pi 10 MHz, LRU on), and 1, 2, 4 inside the range of the DA21 tables (presets/kerrcat.yaml)

Seeds: 1000 d + (1 for X, 2 for Z) + 10000 (point index) + 100000 (model index); the long memory uses seed + 7.

Usage: python examples/noise_model_datasets.py [--models ocelot,allcat,kerrcat] [--points all|ref] [--shots 20000]
Output: data/syndromes/<model>_nbar<nbar>_d<d>_<basis>/ and data/syndromes/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib

import numpy as np

from bosonoise.benchmark.schema import write_dataset
from bosonoise.dem import models
from bosonoise.dem.noise_models import decode_memory, herald_mask, sample_with_leakage

ROOT = pathlib.Path(__file__).resolve().parents[1]
DS = [3, 5, 7]
HERALD_ASSUMPTION = ("herald efficiency 0.5: of the order of our estimate for gate-induced errors in the hybrid cell "
                     "(36-53% heralded at a useful false-herald rate, paper App. I); every bit flip of a stabilized "
                     "dissipative cat comes with a buffer-emission burst (Ferrari et al., arXiv:2607.01375); no "
                     "calculation for the all-cat CNOT; false heralds are not modelled (a herald is always a real flip)")
MODELS = {  # factory, fixed knobs, points (nbar, extra knobs); the first point is the reference point
    "ocelot": dict(factory="ocelot_model", kwargs=dict(variant="lambda3", T1_eff_us=57.0, T_cycle_us=2.8),
                   points=[(2.0, {}), (1.0, {}), (3.0, {}), (4.0, {})]),
    "allcat": dict(factory="allcat_model", kwargs=dict(eta=1e-3, n_th=0.0),
                   points=[(8.0, dict(herald_efficiency=0.0)), (3.0, dict(herald_efficiency=0.5)),
                           (4.0, dict(herald_efficiency=0.5)), (6.0, dict(herald_efficiency=0.5))]),
    "kerrcat": dict(factory="kerrcat_model", kwargs=dict(kappa_over_K=1.67e-4, n_th=0.08, K_MHz=10.0,
                                                         kappa2_over_K=0.1, lru=True),
                    points=[(6.25, {}), (1.0, {}), (2.0, {}), (4.0, {})]),
}


def dataset_name(model, nbar, d, basis):
    return f"{model}_nbar{nbar:g}_d{d}_{basis}"


def to_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if hasattr(x, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(x, k)) for k in x.__dataclass_fields__}
    return x


def per_round(P1, R1, P2, R2, shots):
    """Logical error per round from P_fail at two memory lengths, and its binomial standard error."""
    y1, y2 = 1 - 2 * P1, 1 - 2 * P2
    if y1 <= 0 or y2 <= 0:
        return float("nan"), float("nan")
    k = 1.0 / (R2 - R1)
    r = (y2 / y1) ** k
    dy1, dy2 = (2 * np.sqrt(P * (1 - P) / shots) for P in (P1, P2))
    return float(0.5 * (1 - r)), float(0.5 * k * r * np.hypot(dy1 / y1, dy2 / y2))


def failures(circ, dets, obs, basis, use_heralds):
    """Baseline logical failure rate. X basis: PyMatching; Z basis: no decoding, heralded cat bit flips undone."""
    if basis == "X":
        pred = decode_memory(circ, dets, use_heralds=use_heralds)[:, 0]
    else:
        pred = np.zeros(len(dets), dtype=bool)
        kind2 = herald_mask(circ, kinds=(2,))
        if use_heralds and kind2.any():
            pred = dets[:, kind2].sum(axis=1) % 2 == 1
    return float(np.mean(pred != obs[:, 0]))


def make_dataset(name, spec, k, nbar, extra, d, basis, shots):
    knobs = {**spec["kwargs"], **extra}
    noise = getattr(models, spec["factory"])(nbar, d, **knobs)
    seed = 1000 * d + (1 if basis == "X" else 2) + 10000 * k + 100000 * list(MODELS).index(name)
    out = sample_with_leakage(d, d, noise, shots, basis=basis, seed=seed)
    circ = out["circuit"]
    res = {use_h: failures(circ, out["detectors"], out["observables"], basis, use_h) for use_h in (True, False)}
    # per-round logical error from a second, longer memory (not stored)
    long = sample_with_leakage(d, 3 * d, noise, shots, basis=basis, seed=seed + 7)
    P3 = failures(long["circuit"], long["detectors"], long["observables"], basis, True)
    eps, eps_err = per_round(res[True], d, P3, 3 * d, shots)
    ref_nbar, ref_extra = spec["points"][0]
    prov = dict(noise.provenance)
    if extra.get("herald_efficiency", 0) > 0:
        prov["herald_efficiency"] = HERALD_ASSUMPTION
    meta = {
        "dataset": dataset_name(name, nbar, d, basis),
        "code": {"family": "repetition cat code", "distance": d, "rounds": d, "basis": basis,
                 "layout": "cats 0..d-1, ancillas d..2d-2; stabilizer j = X_j X_{j+1}; CX(ancilla j -> cat j) then (-> cat j+1)",
                 "observable": "parity change of cat 0 (X_L = X_1)" if basis == "X" else "Z_1 ... Z_d"},
        "noise_model": {"name": noise.name, "model": name, "nbar": nbar, "knobs": knobs,
                        "is_reference_point": k == 0,
                        "reference_point": {"nbar": ref_nbar, **spec["kwargs"], **ref_extra},
                        "grid_nbar": [p[0] for p in spec["points"]],
                        "parameters": to_jsonable({f: getattr(noise, f) for f in noise.__dataclass_fields__
                                                   if f not in ("provenance", "name")}),
                        "provenance": prov},
        "generator": {"script": "examples/noise_model_datasets.py", "seed": seed,
                      "sampler": "stim.FlipSimulator + leakage in the incoherent approximation (dem/noise_models.py)"},
        "layout": {"heralds": "round-major; per round the d-1 syndrome-erasure heralds of ancillas 0..d-2 "
                              "(kind 1: ancilla found outside the qubit subspace, outcome random), then, if "
                              "the model has them, the heralded bit flips of cats 0..d-1 (kind 2); the same "
                              "bits are the herald detectors of detectors.b8, coordinates (x, round, kind); "
                              "kind 0 = syndrome detectors",
                   "leakage": "rounds x (cats then ancillas) (round-major); 1 = leaked during that round"},
        "baseline_decoding": {"decoder": "X basis: PyMatching 2 on the circuit DEM (leakage not in the DEM), "
                                         "correlated matching with the heralds; Z basis: none (no Z "
                                         "stabilizers), heralded cat bit flips undone",
                              "logical_failure_with_heralds": res[True],
                              "logical_failure_without_heralds": res[False],
                              "logical_error_per_round_from_R_d_and_3d": eps,
                              "logical_error_per_round_std": eps_err,
                              "note": "Z basis: the failure is the odd-parity probability of all cats, "
                                      "including the initial and final well readout"},
    }
    path = ROOT / "data" / "syndromes" / meta["dataset"]
    write_dataset(path, circ, out["detectors"], out["observables"], metadata=meta, heralds=out["heralds"],
                  leakage=out["leakage"])
    variant = ";".join(f"{key}={val:g}" for key, val in extra.items() if isinstance(val, (int, float)))
    row = dict(dataset=meta["dataset"], model=name, nbar=nbar, reference=int(k == 0), variant=variant, d=d,
               basis=basis, shots=shots, P_fail_heralds=res[True], P_fail_no_heralds=res[False], eps_per_round=eps,
               eps_std=eps_err, herald_rate=float(out["heralds"].mean()), leak_rate=float(out["leakage"].mean()))
    print(f"{meta['dataset']}: P_fail {res[True]:.4f} (no heralds {res[False]:.4f}), eps/round {eps:.3e} +- "
          f"{eps_err:.1e}, heralds {out['heralds'].mean():.4f}, leakage {out['leakage'].mean():.2e}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--points", choices=["all", "ref"], default="all")
    ap.add_argument("--shots", type=int, default=20000)
    a = ap.parse_args()
    names = a.models.split(",")
    summary = []
    for name in names:
        spec = MODELS[name]
        pts = spec["points"] if a.points == "all" else spec["points"][:1]
        for k, (nbar, extra) in enumerate(pts):
            for d in DS:
                for basis in ("X", "Z"):
                    summary.append(make_dataset(name, spec, k, nbar, extra, d, basis, a.shots))
    path = ROOT / "data" / "syndromes" / "summary.csv"
    keys = list(summary[0])
    new = {r["dataset"] for r in summary}
    old = []
    if path.exists():  # keep the rows of datasets not regenerated now (same column layout only)
        old = [r for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#"))
               if r.get("dataset") not in new and set(keys) <= set(r)]
    rows = sorted(old + summary, key=lambda r: (list(MODELS).index(r["model"]), -int(r["reference"]),
                                                float(r["nbar"]), int(r["d"]), r["basis"]))
    text = ("model", "basis", "dataset", "variant")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Syndrome datasets of the reference noise models (examples/noise_model_datasets.py); CC BY 4.0\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) if k in text else f"{float(r[k]):.6g}" for k in keys) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
