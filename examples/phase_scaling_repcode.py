"""Logical phase-flip error per cycle of the repetition cat code vs distance and cat phase-flip probability (Stim +
PyMatching with heralds; dem/repetition.py). A generic table for the projections (examples/logical_projection_ocelot.py).

Uniform parameters: p_z per cat per round (grid), syndrome measurement error p_meas = 1.6% (SPAM 1.5% + the retuned
dispersive-phase error, section 6.16.2), heralded erasure p_erase = 5% (our f->e herald at lambda = 3 plus f->e during
readout), share of phase flips between a cat's two CX gates 0.2; noiseless initial and final parity readout (the
per-cycle rate in the bulk). Memory of R = 3d rounds; eps per cycle from P_fail(R) = [1 - (1 - 2 eps)^R]/2.
Adaptive shots: until MIN_FAILS failures or MAX_SHOTS; points with fewer than 20 failures are flagged and replaced by a
fit eps = A Lambda^-((d+1)/2) over the reliable distances at the same p_z (not a rare-event estimator: flagged
'extrapolated').

Usage: python examples/phase_scaling_repcode.py
Output: data/validation/phase_flip_scaling.csv
"""
from __future__ import annotations

import pathlib
import time

import numpy as np

from bosonoise.dem.repetition import PhaseFlipNoise, logical_failures, phase_flip_circuit

ROOT = pathlib.Path(__file__).resolve().parents[1]
DS = [3, 5, 7, 9, 11, 13, 15]
PZ = np.round(np.geomspace(0.001, 0.45, 16), 6)
P_MEAS, P_ERASE, F_MID = 0.016, 0.05, 0.2
MIN_FAILS, MAX_SHOTS = 300, 400_000


def p_fail(d, R, pz, seed):
    circ = phase_flip_circuit(d, R, PhaseFlipNoise(p_z=pz, p_meas=P_MEAS, p_erase=P_ERASE, f_mid=F_MID))
    fails, shots, batch = 0, 0, 20_000
    while fails < MIN_FAILS and shots < MAX_SHOTS:
        f, n = logical_failures(circ, batch, seed=seed + shots)
        fails, shots = fails + f, shots + n
        batch = min(4 * batch, MAX_SHOTS - shots) if fails < MIN_FAILS / 10 else batch
        if batch <= 0:
            break
    return fails, shots


def main():
    rows = []
    for pz in PZ:
        for d in DS:
            t0 = time.time()
            R = 3 * d
            f, n = p_fail(d, R, pz, seed=1000 * d + 7)
            P = f / n
            eps = float(0.5 * (1 - (1 - 2 * P) ** (1.0 / R))) if f > 0 and P < 0.5 else (0.5 if P >= 0.5 else np.nan)
            se = float(np.sqrt(P * (1 - P) / n) / R * (1 - 2 * P) ** (1.0 / R - 1)) if f > 0 and P < 0.5 else np.nan
            rows.append(dict(p_z=pz, d=d, rounds=R, fails=f, shots=n, P_fail=P, eps=eps, eps_err=se, reliable=int(f >= 20)))
            print(f"p_z {pz:.4f} d {d:2d}: P(R={R}) {P:.3e} ({f}/{n}) -> eps {eps:.3e} ({time.time() - t0:.0f} s)", flush=True)
    # extrapolate unreliable points per p_z: log eps linear in (d + 1)/2 over reliable points
    for pz in PZ:
        sel = [r for r in rows if r["p_z"] == pz]
        good = [r for r in sel if r["reliable"] and r["eps"] > 0 and r["eps"] < 0.45]
        fit = None
        if len(good) >= 2:
            x = np.array([(r["d"] + 1) / 2 for r in good[-3:]])
            y = np.log([r["eps"] for r in good[-3:]])
            fit = np.polyfit(x, y, 1)
        for r in sel:
            r["Lambda"] = float(np.exp(-fit[0])) if fit is not None else np.nan
            if not r["reliable"]:
                r["eps"] = float(np.exp(np.polyval(fit, (r["d"] + 1) / 2))) if fit is not None else np.nan
                r["eps_err"] = np.nan
            r["extrapolated"] = int(not r["reliable"])
    path = ROOT / "data" / "validation" / "phase_flip_scaling.csv"
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# Repetition cat code, logical phase-flip error per cycle vs p_z and d (Stim + PyMatching with heralds); "
                 f"p_meas {P_MEAS}, p_erase {P_ERASE}, f_mid {F_MID}; noiseless boundaries; eps from an R = 3d memory; "
                 f"'extrapolated' = fewer than 20 failures, replaced by eps = A Lambda^-((d+1)/2). "
                 f"examples/phase_scaling_repcode.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" for k in keys) + "\n")
    print("saved", path)


if __name__ == "__main__":
    main()
