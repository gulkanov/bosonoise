"""Presets with per-parameter provenance (YAML).

Device presets (ocelot, companion; ocelot_rep_noise holds our Ocelot fault tables) record, for each
number, where it comes from (paper section / figure / page) and what kind of number it is (measured,
AWS model, design value, AWS-fitted, derived). Only `use: calibration` entries may be used to
calibrate Hamiltonians. The literature presets of the reference noise models (allcat, kerrcat) record
the published operation models (kind: literature / assumption / derived; use: model / validation).
"""
from __future__ import annotations

from importlib import resources

import yaml


def load_preset(name: str) -> dict:
    """Load a preset by name, e.g. ``load_preset("ocelot")``."""
    text = resources.files(__package__).joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def available() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files(__package__).iterdir() if p.name.endswith(".yaml"))
