"""Build src/bosonoise/presets/ocelot_rep_noise.yaml: the tabulated ingredients of the Ocelot-type reference noise model
(dem/models.py) from our computed validation tables, with provenance.

Per stabilizer A1..A4 (cats S_j, S_j+1; CX with S_j first) and nbar = 1 .. 16, variants zero-fit and lambda3:
  p_x_a, p_x_b, p_xx   marginal X probabilities on the first/second cat and the joint X X per stabilizer measurement
                       (dressed-ladder ULE fault tables; data/validation/stabilizer_faults_large_nbar.csv)
  p_herald_gates       f -> e herald over the two CX gates (same source)
  p_disp               syndrome error from the photon-number-dependent dispersive phase, gate fixed and retuned
                       (data/validation/syndrome_phase.csv)
Per ancilla: herald from waiting for the other CX layer (zenodo_erasures.csv model columns, window - gates, nbar = 1).
Per storage: idle bit flip per 2.8-us cycle, DATA-DERIVED (Ocelot Fig. 16 static, geometric mean of the two pairs,
Pauli convention) at nbar <= 4, and its continuation A exp(-B nbar) fitted on 2..4 (examples/bias_ceiling_ocelot.py).

Usage: python tools/build_ocelot_noise_table.py
"""
import csv
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from bias_ceiling_ocelot import idle_model  # noqa: E402


def rows(name):
    return list(csv.DictReader(l for l in open(ROOT / "data" / "validation" / name, encoding="utf-8") if not l.startswith("#")))


def main():
    out = {"meta": {
        "description": "Tabulated ingredients of the Ocelot-type reference noise model of the repetition cat code (dem/models.py)",
        "built_by": "tools/build_ocelot_noise_table.py",
        "sources": {
            "stabilizers": "our dressed-ladder ULE fault tables, data/validation/stabilizer_faults_large_nbar.csv (paper App. D); zero-fit or lambda = 3 on-gate ancilla decay (lambda derived from Ocelot CX^2, logical and erasure data, paper Sec. 6, App. H)",
            "dispersive": "our dressed-ladder dispersive phase, data/validation/syndrome_phase.csv (paper App. H)",
            "herald_window": "our f->e herald while waiting for the other CX layer, data/validation/zenodo_erasures.csv model columns (paper App. H)",
            "idle": "DATA-DERIVED: Ocelot Fig. 16 static protocol (Putterman et al., Nature 638, 927 (2025)), Pauli per 2.8-us cycle, continued as A exp(-B nbar) fitted on 2 <= nbar <= 4 (examples/bias_ceiling_ocelot.py)",
        }},
        "stabilizers": {}, "dispersive": {}, "herald_window": {}, "idle": {}}
    for r in rows("stabilizer_faults_large_nbar.csv"):
        anc = r["stabilizer"].split(":")[0]
        var = {"zero-fit": "zero-fit", "lam3": "lambda3"}.get(r["variant"])
        if var is None:
            continue
        out["stabilizers"].setdefault(anc, {}).setdefault(var, {})[float(r["nbar"])] = {
            k: float(r[k]) for k in ("p_x_a", "p_x_b", "p_xx", "p_herald")}
    for r in rows("syndrome_phase.csv"):
        out["dispersive"].setdefault(r["ancilla"], {})[float(r["nbar"])] = {
            "fixed_gate": float(r["model_syndrome_error_fixed_gate"]), "retuned": float(r["model_syndrome_error_retuned"])}
    for r in rows("zenodo_erasures.csv"):
        if float(r["nbar"]) == 1.0 and r["section"] == "d5_S1-S5":
            for var, col in (("zero-fit", "zero-fit"), ("lambda3", "lam3")):
                out["herald_window"].setdefault(r["ancilla"], {})[var] = max(
                    float(r[f"model_herald_window_{col}"]) - float(r[f"model_herald_gates_{col}"]), 0.0)
    for cat, v in idle_model().items():
        out["idle"][cat] = {"points": {float(k): float(p) for k, p in v["points"].items()}, "A": v["A"], "B": v["B"]}
    path = ROOT / "src" / "bosonoise" / "presets" / "ocelot_rep_noise.yaml"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(out, fh, sort_keys=False, allow_unicode=True, width=120)
    print("saved", path)


if __name__ == "__main__":
    main()
