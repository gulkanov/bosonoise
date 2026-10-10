# Changelog

## 0.1.0 — first public version

The code and data products of the paper
"Photon-stimulated decay of χ-matched ancillas causes unheralded bit flips in bosonic codes": a package that
reproduces every figure and table of the paper with one command each, three reference noise models of repetition cat
codes in Stim format, and 72 syndrome datasets.

### Package (`src/bosonoise`)

- `model/`: charge-basis transmons with the full cosine, and the storage–coupler–ancilla cell in the lab frame
  (no RWA). Calibrations of the Ocelot cells are in `data/calibration/`.
- `spectra/`: dressed states and dispersive observables, calibration to static spectroscopy only, branch analysis
  versus storage photon number, dressed ancilla transitions, the diabatic-crossing atlas along the flux trajectory.
- `dynamics/`: flux pulses in the irrotational gauge, the time-dependent cell, the universal Lindblad equation (ULE),
  adiabatic perturbation theory (APT) for ramps, and Floquet branches with Floquet–Markov rates of a transmon driven by
  a displaced mode (`floquet.py`, for the GKP-ECD cell).
- `channels/`:
  - the CX² fault model in the dressed ladder with ULE families;
  - the syndrome-cycle model (stabilizer fault tables);
  - the conversion map by fault sequence and buffer-emission heralds;
  - stabilization windows (exact odd-sector cycle map);
  - TLS ensembles;
  - phenomenological CX bit-flip baselines.
- `dem/`:
  - Stim circuits of the repetition cat code memories, with heralds and leakage (incoherent approximation via
    stim.FlipSimulator);
  - PyMatching decoding;
  - the three reference noise models: `ocelot_model`, `allcat_model`, `kerrcat_model`.
- `benchmark/`: dataset format 0.1 (Stim b8 + JSON metadata with provenance, generator commit and seed).
- `presets/`:
  - Ocelot and companion device parameters with per-number provenance;
  - our Ocelot fault tables;
  - the literature operation models of the all-cat (LR23) and Kerr-cat (DA21) noise models.

### Reproducibility

- `tools/reproduce.py`: one command per figure and data product (103 targets), dependency order, redraw modes, logs,
  measured runtimes, one BLAS thread per target. The target `prx_figures` draws the fifteen figures of the paper from
  the stored data products. `docs/reproduction.md` lists every target with its runtime and external inputs, and
  summarizes the reproduction checks.
- `tools/compare_to_git.py`: regenerated outputs against the committed ones (CSV cells, YAML/JSON leaves, binaries).
- Tests: pytest suite; CI workflow (GitHub Actions, Python 3.10–3.12).

### Data

- `data/syndromes/`: 72 datasets (3 noise models × 4 n̄ × d = 3, 5, 7 × X/Z memories, 20 000 shots each), CC BY 4.0.
- `data/validation/`: every model-versus-data and model-versus-literature comparison of the paper, among them the CX²
  gate errors of the eight Ocelot pairs, the logical memory from the public raw data, the GKP-ECD ancilla under the
  cavity's photons, and the universal map and law of the dressing-opened f → g channel.
- `data/digitized/`: numbers extracted from figures of the Ocelot and companion papers (not re-licensed; see
  `data/LICENSE.md`).

### Not in this version

- A GKP-ECD preset file: the two GKP devices are defined in `examples/gkp_ecd_ancilla_photons.py` (parameters and
  sources in paper App. G).
- A separate archive of the datasets (Zenodo / Hugging Face): the format is ready.
