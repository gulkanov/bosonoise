"""Static dispersive observables of the storage-coupler-ancilla cell.

From dressed energies E_i(n) (ancilla i in {g,e,f}, coupler in its ground state, storage
Fock n) we define, exactly as in Putterman et al. (arXiv:2409.13025, App. D):

    f_s|i = E_i(1) - E_i(0)                  conditional storage frequency
    chi_ge = f_s|e - f_s|g,  chi_gf = f_s|f - f_s|g
    K_i   = E_i(2) - 2 E_i(1) + E_i(0)        conditional storage self-Kerr
            (H = f a^dag a + (K/2) a^dag a^dag a a  =>  E(2) - 2E(1) + E(0) = K)

All quantities are in GHz (f = omega/2pi).
"""
from __future__ import annotations

import numpy as np

from ..model.cell import StorageCouplerAncilla

ANCILLA = {"g": 0, "e": 1, "f": 2}


def static_observables(model: StorageCouplerAncilla, flux: float) -> dict:
    E, V = model.eigh(flux)
    idx = model.dressed_index
    En = {i: [E[idx(V, n, 0, q)] for n in range(3)] for i, q in ANCILLA.items()}
    out = {
        "flux": flux,
        "f_ge": En["e"][0] - En["g"][0],
        "f_ef": En["f"][0] - En["e"][0],
        "f_c": E[idx(V, 0, 1, 0)] - En["g"][0],
    }
    for i in ANCILLA:
        out[f"f_s|{i}"] = En[i][1] - En[i][0]
        out[f"K_{i}"] = En[i][2] - 2 * En[i][1] + En[i][0]
    out["f_s"] = out["f_s|g"]
    out["chi_ge"] = out["f_s|e"] - out["f_s|g"]
    out["chi_gf"] = out["f_s|f"] - out["f_s|g"]
    out["alpha_q"] = out["f_ef"] - out["f_ge"]
    return out


def sweep(model: StorageCouplerAncilla, fluxes) -> dict[str, np.ndarray]:
    rows = [static_observables(model, float(x)) for x in fluxes]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}
