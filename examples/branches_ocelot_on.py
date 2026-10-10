"""Branch analysis of the Ocelot A2-S3 cell versus storage photon number, coupler "on".

For the calibrated model (data/calibration/ocelot_A2S3_fit{A,B}.yaml) at Phi_c = Phi0/2,
track the dressed branches that start in |n=0, coupler 0, ancilla g/e/f> up to n = 20,
report every change of branch character (multiphoton hybridization), and check
convergence against all truncations (Fock, coupler levels, ancilla levels, charge
cutoff, energy cutoff).

Usage:  python examples/branches_ocelot_on.py [--fit A|B] [--flux 0.5] [--ng 0.0]
Outputs: figures/branches_ocelot_<label>_flux<flux>_fit<fit>_ng<ng>.png/.pdf (label A2S3, or A4<cell> with --calib),
         data/branches/ocelot_<same tag>.csv, stdout table (convergence check unless --no-convergence)
"""
from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bosonoise.model import CellParams, StorageCouplerAncilla, Truncation
from bosonoise.model.cell import load_cell_params
from bosonoise.plotting import SERIES, use_style
from bosonoise.spectra import branch_analysis, find_resonances

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = {(0, 0): "g", (0, 1): "e", (0, 2): "f"}

def _tr(e_cut=160.0, n_c=8, n_q=14, ncut=25, f_s=5.31):
    # Fock cutoff chosen so that the bare-energy cutoff (not n_s) is the binding constraint
    return Truncation(n_s=int(e_cut / f_s) + 3, n_c=n_c, n_q=n_q, ncut=ncut, e_cut=e_cut)


BASE_TRUNC = _tr()
VARIANTS = {
    "base": BASE_TRUNC,
    "e_cut+25": _tr(e_cut=185.0),
    "n_c+2": _tr(n_c=10),
    "n_q+4": _tr(n_q=18),
    "ncut+10": _tr(ncut=35),
}


def load_params(fit: str, ng: float) -> CellParams:
    rec = yaml.safe_load(open(ROOT / "data" / "calibration" / f"ocelot_A2S3_fit{fit}.yaml", encoding="utf-8"))
    return CellParams(**rec["params"]).with_(ng_q=ng)


def run(params, trunc, flux, nmax=20):
    m = StorageCouplerAncilla(params, trunc)
    t0 = time.time()
    br = branch_analysis(m, flux, starts=tuple(NAMES), nmax=nmax)
    return m, br, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="A")
    ap.add_argument("--flux", type=float, default=0.5)
    ap.add_argument("--ng", type=float, default=0.0)
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--no-convergence", action="store_true")
    ap.add_argument("--calib", default=None, help="calibration YAML (default: A2-S3 fit --fit)")
    ap.add_argument("--cell", default=None, help="cell key inside a multi-cell calibration YAML")
    ap.add_argument("--label", default=None, help="cell label for file names (default A2S3)")
    a = ap.parse_args()
    if a.calib is None:
        params = load_params(a.fit, a.ng)
        label = "A2S3"
    else:
        params = load_cell_params(a.calib, a.cell).with_(ng_q=a.ng)
        label = a.label or f"A4{a.cell}"
    tag = f"ocelot_{label}_flux{a.flux:.3f}_fit{a.fit}_ng{a.ng:.2f}"

    m, br, dt = run(params, BASE_TRUNC, a.flux, a.nmax)
    print(f"[{tag}] dim={m.basis.dim}, diag+branches {dt:.1f} s")
    f_s = br.energy[0, 1] - br.energy[0, 0]

    # ------------------------------------------------------------- resonances
    res = find_resonances(br)
    print(f"branch-character changes (|dN_q|>0.02 or |dN_c|>0.02 or overlap<0.98): {len(res)}")
    for r in res:
        top = ", ".join(f"{w:.3f}|s={l[0]},c={l[1]},q={l[2]}>" for w, l in r["top"])
        print(f"  branch {NAMES[r['branch']]} n={r['n']:2d} dNq={r['dNq']:+.4f} dNc={r['dNc']:+.4f} "
              f"ov={r['overlap']:.4f}  top: {top}")

    # ------------------------------------------------------------- convergence
    conv_lines = []
    if not a.no_convergence:
        ref = br
        for name, tr in VARIANTS.items():
            if name == "base":
                continue
            mv, bv, dtv = run(params, tr, a.flux, a.nmax)
            dE = np.max(np.abs((bv.energy - bv.energy[:, :1]) - (ref.energy - ref.energy[:, :1])))
            dq = np.max(np.abs(bv.mean_q - ref.mean_q))
            dc = np.max(np.abs(bv.mean_c - ref.mean_c))
            line = (f"  {name:9s} dim={mv.basis.dim:5d} ({dtv:4.1f}s): max|dE_branch|={dE * 1e6:9.3f} kHz, "
                    f"max|d<N_q>|={dq:.2e}, max|d<N_c>|={dc:.2e}")
            print(line)
            conv_lines.append(line)

    # ------------------------------------------------------------- save table
    out = ROOT / "data" / "branches"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# branch analysis, params fit {a.fit} (data/calibration), flux={a.flux}, ng_q={a.ng}\n")
        fh.write(f"# truncation {vars(BASE_TRUNC)}\n")
        fh.write("branch,n,E_GHz,dev_from_linear_MHz,mean_q,mean_c,mean_s,overlap\n")
        for b, st in enumerate(br.starts):
            E0 = br.energy[b, 0]
            for n in br.n:
                dev = (br.energy[b, n] - E0 - n * f_s) * 1e3
                fh.write(f"{NAMES[st]},{n},{br.energy[b, n]:.9f},{dev:.6f},{br.mean_q[b, n]:.6e},"
                         f"{br.mean_c[b, n]:.6e},{br.mean_s[b, n]:.6f},{br.overlap[b, n]:.8f}\n")

    # ------------------------------------------------------------- figure
    use_style()
    fig, ax = plt.subplots(4, 1, figsize=(6.4, 9.0), sharex=True, constrained_layout=True)
    cols = {"g": SERIES[0], "e": SERIES[1], "f": SERIES[2]}
    for b, st in enumerate(br.starts):
        nm = NAMES[st]
        E0 = br.energy[b, 0]
        dev = (br.energy[b] - E0 - br.n * f_s) * 1e3
        ax[0].plot(br.n, dev, "o-", ms=3, color=cols[nm], label=f"branch {nm}")
        ax[1].semilogy(br.n, np.abs(br.mean_q[b] - st[1]) + 1e-12, "o-", ms=3, color=cols[nm])
        ax[2].semilogy(br.n, br.mean_c[b] + 1e-12, "o-", ms=3, color=cols[nm])
        ax[3].semilogy(br.n, 1 - br.overlap[b] + 1e-16, "o-", ms=3, color=cols[nm])
    ax[0].set_ylabel(r"$E_i(n)-E_i(0)-n f_s$ (MHz)")
    ax[0].legend(frameon=False, ncol=3)
    ax[1].set_ylabel(r"$|\langle N_q\rangle - q_i|$")
    ax[2].set_ylabel(r"$\langle N_c\rangle$")
    ax[3].set_ylabel(r"$1-$ladder overlap")
    ax[3].set_xlabel("storage photon number n")
    ax[0].set_title(f"Ocelot {label} model (fit {a.fit}), $\\Phi_c={a.flux}\\,\\Phi_0$, $n_g={a.ng}$", fontsize=10)
    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"branches_{tag}.{ext}", dpi=150)
    print("saved", figdir / f"branches_{tag}.png")


if __name__ == "__main__":
    main()
