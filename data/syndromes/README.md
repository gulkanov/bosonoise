# Syndrome datasets of the three reference noise models

Simulated syndrome data of repetition cat code memories under the three reference circuit-level noise models of
`bosonoise.dem.models` (Ocelot-type hybrid cat–transmon, all-cat in the style of Alice & Bob, Kerr cats). Every
number of every noise model carries its provenance in the dataset's `metadata.json`. The models are described in
`src/bosonoise/dem/models.py` and checked against the literature by `examples/literature_checks_models.py`.

Licence: CC BY 4.0 (see `data/LICENSE.md`). Regenerate with `python examples/noise_model_datasets.py` (~2 min on
one core; the arrays are reproduced bit for bit from the seeds in the metadata).

## Contents

72 datasets = 3 models × 4 mean photon numbers n̄ × distance d ∈ {3, 5, 7} × memory basis {X, Z}. Each dataset
has 20 000 shots of R = d rounds.

- **X memory:** logical phase flips. The cats are prepared and read out in the parity (X) basis, and the stabilizers
  X_jX_{j+1} are decoded.
- **Z memory:** logical bit flips. The cats are prepared and read out in the well (Z) basis. There are no Z
  stabilizers, so the logical failure is the odd parity of the cats' bit flips.

Directory names: `<model>_nbar<n̄>_d<d>_<basis>`, e.g. `kerrcat_nbar6.25_d5_X`. `summary.csv` lists every dataset
with its baseline decoding.

| model | n̄ | point | ε_phase per round, % (X; d = 3 / 5 / 7) | ε_bit per round, % (Z; d = 3 / 5 / 7) | herald rate | leakage flags |
|---|---|---|---|---|---|---|
| ocelot | 2 | reference | 3.62 / 1.72 / 1.02 | 0.68 / 1.20 / 1.85 | 0.066 | 3.5e-4 |
| ocelot | 1 | grid | 1.63 / 0.49 / 0.15 | 2.01 / 3.79 / 4.96 | 0.066 | 3.5e-4 |
| ocelot | 3 | grid | 6.13 / 3.85 / 2.61 | 0.28 / 0.90 / 1.07 | 0.066 | 3.2e-4 |
| ocelot | 4 | grid | 8.73 / 6.12 / 5.12 | 0.27 / 0.61 / 0.95 | 0.065 | 3.4e-4 |
| allcat | 8 | reference (no heralds) | 1.52 / 0.61 / 0.29 | ≈ 0 (\|ε\| < 2e-4) | 0 | 0 |
| allcat | 3 | grid (herald efficiency 0.5) | 1.53 / 0.23 / 0.06 | 0.22 / 0.31 / 0.47 | 4.8e-4 | 0 |
| allcat | 4 | grid (herald efficiency 0.5) | 1.87 / 0.32 / 0.09 | 0.04 ± 0.03 / 0.02 ± 0.03 / 0.11 ± 0.03 | 7.7e-5 | 0 |
| allcat | 6 | grid (herald efficiency 0.5) | 1.12 / 0.38 / 0.18 | ≈ 0 (\|ε\| < 3e-4) | 6.7e-6 | 0 |
| kerrcat | 6.25 | reference | 2.30 / 1.04 / 0.57 | 0.03 ± 0.03 / 0.15 / 0.16 | 0 | 6.2e-4 |
| kerrcat | 1 | grid | 2.83 / 1.36 / 0.78 | 0.92 / 1.95 / 2.53 | 0 | 4.2e-3 |
| kerrcat | 2 | grid | 3.12 / 1.44 / 0.84 | 0.53 / 1.02 / 1.46 | 0 | 2.2e-3 |
| kerrcat | 4 | grid | 2.91 / 1.28 / 0.67 | 0.24 / 0.36 / 0.46 | 0 | 1.1e-3 |

- **ε per round.** From the failure probabilities of the stored memory (R = d) and of a longer one (R = 3d, not
  stored), so that the preparation and readout boundaries cancel. The baseline decoder is described below; ± is one
  binomial standard error, shown where it matters.
- **Herald rate.** Fraction of set herald bits.
- **Leakage flags.** Fraction of set (qubit, round) flags.

## Operating points

- **ocelot.** Reference n̄ = 2; grid n̄ = 1, 3, 4. Fixed knobs: on-gate decay variant λ = 3, storage T1,eff = 57 µs,
  cycle 2.8 µs.
  - The grid stays inside the range of the Ocelot data (|α|² = 1 … 3.5); the idle part is data-derived up to n̄ = 4.
  - Every dataset has syndrome-erasure heralds (kind 1) and heralded ancilla leakage.
- **allcat.** η = κ₁/κ₂ = 1e-3, no thermal population.
  - Reference n̄ = 8: the operating point of Le Régent et al., Quantum 7, 1198 (2023), below their threshold. Bit flips
    there are ~1e-7 per CNOT, so the Z memory holds only the readout errors of the cats.
  - Grid n̄ = 3, 4, 6, with buffer-emission heralds of the cats' bit flips (kind 2) at efficiency 0.5. This efficiency
    is an assumption, stated in the metadata. False heralds are not modelled: every herald is a real flip.
  - At n̄ = 3 and 4 the Z-memory failure of these short memories is dominated by the well readout, e^{−1.5−0.9n̄}
    per cat (Chamberland et al. 2022). The bit flips show up in ε per round.
- **kerrcat.** Fixed knobs: κ/K = 1.67e-4, n_th = 8%, K/2π = 10 MHz, the κ₂ step of each CX on (LRU).
  - Reference n̄ = 6.25: the design point of Darmawan et al., PRX Quantum 2, 030345 (2021).
  - Grid n̄ = 1, 2, 4, inside the range of their operation tables.
  - Leakage of cats and ancillas, from single-photon gain.

## Files (format v0.1, `bosonoise/benchmark/schema.py`)

| file | content |
|---|---|
| `circuit.stim` | the Stim circuit without leakage. It defines the detectors (coordinates (x, round, kind); kind 0 = syndrome, 1 = syndrome-erasure herald, 2 = heralded cat bit flip) and the observable. Its DEM is the baseline decoder's graph. |
| `detectors.b8` | detection events, shots × detectors, Stim b8 (bits of a shot little-endian, each shot padded to whole bytes) |
| `observables.b8` | logical observable flip, shots × 1 |
| `heralds.b8` | the herald detectors again, round-major: per round the d − 1 ancilla heralds (kind 1), then the d cat heralds (kind 2) if the model has them |
| `leakage.b8` | leakage flags, round-major: per round the d cats, then the d − 1 ancillas; 1 = leaked during that round |
| `metadata.json` | schema version; code and layout; noise model (name, knobs, reference point, grid, every parameter and its provenance); generator (package version, commit of the development repository that generated the data, script, seed); baseline decoding; shot and bit counts |

The leakage is sampled in the incoherent approximation (stim.FlipSimulator between the round segments). It is not
part of `circuit.stim`, so decoders that use the circuit's DEM do not know about it; the flags are the ground truth.

## Loading

```python
from bosonoise.benchmark.schema import read_dataset

ds = read_dataset("data/syndromes/kerrcat_nbar6.25_d5_X")
ds["detectors"].shape, ds["observables"].shape, ds["heralds"].shape, ds["leakage"].shape
ds["metadata"]["noise_model"]["provenance"]
```

Without bosonoise, with numpy only:

```python
import json, pathlib
import numpy as np

p = pathlib.Path("data/syndromes/kerrcat_nbar6.25_d5_X")
meta = json.loads((p / "metadata.json").read_text(encoding="utf-8"))
def b8(name, width):
    raw = np.frombuffer((p / name).read_bytes(), dtype=np.uint8).reshape(meta["shots"], -1)
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :width].astype(bool)
det = b8("detectors.b8", meta["num_detectors"])
obs = b8("observables.b8", meta["num_observables"])
```

## Baseline decoding (in `metadata.json` and `summary.csv`)

- **X memory.** PyMatching 2 on the circuit's DEM with correlated matching, with the heralds and with the herald
  detectors zeroed.
- **Z memory.** No decoding; the heralded cat bit flips (kind 2) are undone.

These are reference numbers for the datasets, not optimized decoders.
