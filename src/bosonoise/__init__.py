"""bosonoise -- first-principles noise channels of bosonic-qubit cells.

Pipeline: circuit -> spectra -> dynamics -> channels -> detector error models (Stim),
plus reference noise models and decoder-benchmark datasets.

Conventions used throughout the package
---------------------------------------
* Energies/frequencies are ordinary frequencies f = omega / 2pi in GHz (h = 1).
* Times are in ns, so a phase is 2*pi*f*t with f in GHz and t in ns.
* Transmon-like elements are always treated in the charge basis with the full
  cosine potential (no Duffing truncation, no two-level truncation).

Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
"""

__version__ = "0.1.0"
