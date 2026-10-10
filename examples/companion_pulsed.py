"""Pulsed stabilization of the AWS companion device (PRX 15, 011070 (2025), Fig. 7b; paper App. J).

The window model (channels/window.py, exact odd-sector cycle map) with the buffer eliminated exactly, including the
storage-buffer cross-Kerr chi_sb (stabilization_superop), and ZERO-FIT inputs of the companion device
(presets/companion.yaml): kappa2 = 2pi 124 kHz (g2 from the pure two-photon dissipation dynamics), kappa_b = 2pi 10.7 MHz,
chi_sb = 2pi 156 kHz (AWS circuit model), K_s = 2pi 0.97 kHz (measured), storage T1, T2 with the pump on (79, 116 us)
and off (76, 139 us), buffer detuning 0 (+-1 MHz band), n_th = 0.5% (prior). Cycle 3 us: T_on = 3 (static), 1, 0.2 us;
square pulses. Phase tracking (the paper: the storage phase is tracked so that the dissipation is aligned with the
coherent states when it turns back on): the state is rotated back by f times the mean Kerr rotation K nbar T_off of the
coherent state; baseline f = 1 ('mean', the literal reading); 'optimal' = the f that maximizes T_X (a control choice, not
a fit; it comes out f ~ 0.1 because the flip-prone low-photon tail rotates less than the mean). Compared with the
Fig. 7b markers (VALIDATION ONLY, data/digitized/companion_fig7b.csv).

Variants (sensitivity): no chi_sb; buffer detuning +-1 MHz; n_th 0 and 1%; pumped rates also while off; no / optimal
Kerr tracking; K_s = 5 kHz at T_on = 1 us (AWS's dotted curve); pulsed windows 30 ns shorter (ramps). Cross-check against AWS's own simulations: 'aws_like' = their inputs
(K_s 1.1 kHz, the white-noise dephasing halved as they did to match their static data -- an aws_fitted parameter, used
here ONLY to test that our effective model reproduces their dashed curves, never as our prediction).
--validate: the effective single-mode model against the full two-mode storage + buffer model (no adiabatic
elimination), |alpha|^2 = 1 ... 4: static T_X (sparse shift-invert) and the pulsed cycle maps (Arnoldi on
expm_multiply) with the mean and the single-mode-optimal tracking; buffer cutoff 4 (6 as a check).

Usage: python examples/companion_pulsed.py [--validate] [--only VARIANTS] [--plot]
Output: data/validation/companion_pulsed.csv, figures/companion_pulsed.png/.pdf; --validate:
        data/validation/companion_pulsed_two_mode.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from bosonoise.channels.window import CatWindow
from bosonoise.plotting import SERIES, TEXT_PRIMARY, use_style
from bosonoise.presets import load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO_PI = 2 * np.pi
T_CYCLE = 3.0
T_ONS = (3.0, 1.0, 0.2)
SERIES_OF = {3.0: "static", 1.0: "Ton_1us", 0.2: "Ton_200ns"}
NBARS = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
NTH = 0.005


def params(pre, **over):
    b, s = pre["buffer"], pre["storage"]
    T1, T2, T1b, T2b = (s[k]["value"] for k in ("T1_under_2ph_diss_us", "T2_under_2ph_diss_us", "T1_us", "T2_us"))
    p = dict(kappa2=TWO_PI * b["kappa2_kHz"]["value"] * 1e-3, kappa_b=TWO_PI * b["kappa_b_MHz"]["value"],
             chi_sb=TWO_PI * b["chi_sb_kHz"]["value"] * 1e-3, delta_b=0.0, K=TWO_PI * s["K_s_kHz"]["value"] * 1e-3,
             kappa1=1 / T1, kphi=2 / T2 - 1 / T1, kappa1_off=1 / T1b, kphi_off=2 / T2b - 1 / T1b, nth=NTH, track="mean", ton_loss=0.0)
    p.update(over)
    return p


VARIANTS = {
    "zero-fit": {},
    "no chi_sb": dict(chi_sb=0.0),
    "detuning +1 MHz": dict(delta_b=TWO_PI * 1.0),
    "detuning -1 MHz": dict(delta_b=-TWO_PI * 1.0),
    "n_th 0": dict(nth=0.0),
    "n_th 1%": dict(nth=0.01),
    "pumped rates while off": "pumped_off",
    "no Kerr tracking": dict(track="none"),
    "optimal tracking": dict(track="optimal"),
    "K_s 5 kHz": dict(K=TWO_PI * 5e-3),
    "aws_like": "aws_like",
    "aws_like optimal tracking": "aws_like_opt",
    "T_on - 30 ns (ramps)": dict(ton_loss=0.03),
    "pump-off dephasing": "offdeph",
    "pump-off dephasing optimal tracking": "offdeph_opt",
}


def resolve(pre, name):
    v = VARIANTS[name]
    if v in ("offdeph", "offdeph_opt"):  # hypothesis: the flip-relevant dephasing is the pump-off (white) value, on and off
        p = params(pre, track="optimal" if v == "offdeph_opt" else "mean")
        return dict(p, kphi=p["kphi_off"])
    if v == "pumped_off":
        p = params(pre)
        return params(pre, kappa1_off=p["kappa1"], kphi_off=p["kphi"])
    if v in ("aws_like", "aws_like_opt"):  # AWS's inputs (App. G4): K_s 1.1 kHz, dephasing halved (aws_fitted), no thermal
        p = params(pre, K=TWO_PI * 1.1e-3, nth=0.0, track="optimal" if v == "aws_like_opt" else "mean")
        return dict(p, kphi=0.5 * p["kphi"], kappa1_off=p["kappa1"], kphi_off=0.5 * p["kphi"])
    return params(pre, **v)


def window(nbar, p, N=None):
    return CatWindow(nbar, N=N, kappa2=p["kappa2"], kappa1=p["kappa1"], nth=p["nth"], kphi=p["kphi"], K=p["K"],
                     kappa1_off=p["kappa1_off"], kphi_off=p["kphi_off"], kappa_b=p["kappa_b"], chi_sb=p["chi_sb"],
                     delta_b=p["delta_b"])


def cycle_TX(w, nbar, p, T_on):
    """Bit-flip time 2 T_Z (s) of the repeated cycle: stabilization T_on, off T_cycle - T_on; returns (T_X, p, f).
    Phase tracking before the stabilization returns (CatWindow.pulsed_bitflip): the state is rotated back by f times the
    mean Kerr rotation of the coherent state, K nbar T_off; p['track'] = 'mean' (f = 1: aligned with the coherent state,
    the paper's description), 'none' (f = 0) or 'optimal' (a control choice, not a fit). p['ton_loss'] shortens each
    pulsed window (ramps counted as off)."""
    if T_on < T_CYCLE:
        T_on = T_on - p["ton_loss"]
    T_off = T_CYCLE - T_on
    if T_off <= 0:
        pc, f = w.bitflip_per_cycle(w.on(T_CYCLE)), np.nan
    else:
        pc, f = w.pulsed_bitflip(T_on, T_off, p["K"] * nbar * T_off, p["track"])
    return -2 * T_CYCLE / np.log(1 - 2 * pc) * 1e-6, pc, f


def load_data():
    rows = list(csv.DictReader(l for l in open(ROOT / "data" / "digitized" / "companion_fig7b.csv", encoding="utf-8")
                               if not l.startswith("#")))
    dat = {(r["series"], float(r["alpha2"])): (float(r["T_bitflip_s"]), float(r["T_low_s"]), float(r["T_high_s"]))
           for r in rows if r["kind"] == "data"}
    sim = {}
    for r in rows:
        if r["kind"] == "aws_sim":
            sim.setdefault(r["series"], []).append((float(r["alpha2"]), float(r["T_bitflip_s"])))
    return dat, {k: sorted(v) for k, v in sim.items()}


# ---------------------------------------------------------------------------------------------- two-mode validation
def two_mode_odd(nbar, p, Ns, Nb, pump):
    """Odd-sector (storage m - n odd) Liouvillian of storage + buffer, column stacking; storage index = i // Nb."""
    import qutip as qt
    a = qt.tensor(qt.destroy(Ns), qt.qeye(Nb))
    b = qt.tensor(qt.qeye(Ns), qt.destroy(Nb))
    g2 = np.sqrt(p["kappa2"] * p["kappa_b"] / 4)
    H = p["delta_b"] * b.dag() * b + p["chi_sb"] * a.dag() * a * b.dag() * b + 0.5 * p["K"] * a.dag() ** 2 * a ** 2
    if pump:
        H = H + g2 * (a ** 2 * b.dag() + a.dag() ** 2 * b) - g2 * nbar * (b + b.dag())
    k1, kp = (p["kappa1"], p["kphi"]) if pump else (p["kappa1_off"], p["kphi_off"])
    c_ops = [np.sqrt(p["kappa_b"]) * b, np.sqrt(k1 * (1 + p["nth"])) * a, np.sqrt(kp) * a.dag() * a]
    if p["nth"]:
        c_ops.append(np.sqrt(k1 * p["nth"]) * a.dag())
    Lq = qt.liouvillian(H, c_ops).to("csr")
    L = sp.csr_matrix(Lq.data.as_scipy())
    D = Ns * Nb
    i, j = np.divmod(np.arange(D * D), D)[::-1]  # K = i + D j
    idx = np.flatnonzero((i // Nb - j // Nb) % 2 != 0)
    return L.tocsr()[idx][:, idx].tocsc(), (i[idx] // Nb) - (j[idx] // Nb)


def two_mode_static_TX(nbar, p, Ns, Nb=4):
    L, _ = two_mode_odd(nbar, p, Ns, Nb, True)
    ev = sla.eigs(L, k=4, sigma=0, which="LM", return_eigenvectors=False)
    lam = ev[np.argmax(ev.real)].real
    return -2 / lam * 1e-6


def two_mode_cycle_TX(nbar, p, T_on, Ns, f, Nb=4):
    """Pulsed cycle map of the two-mode model (square pulses), tracking by f times the mean Kerr rotation K nbar T_off."""
    Lon, dmn = two_mode_odd(nbar, p, Ns, Nb, True)
    Loff, _ = two_mode_odd(nbar, p, Ns, Nb, False)
    T_off = T_CYCLE - T_on
    ph = np.exp(1j * f * p["K"] * nbar * dmn * T_off)

    def mv(v):
        return sla.expm_multiply(Lon * T_on, ph * sla.expm_multiply(Loff * T_off, v))

    op = sla.LinearOperator(Lon.shape, matvec=mv, dtype=complex)
    ev = sla.eigs(op, k=2, which="LM", return_eigenvectors=False, tol=1e-10)
    mu = ev[np.argmax(np.abs(ev))]
    return -2 * T_CYCLE / np.log(mu.real) * 1e-6


def validate(pre):
    """Two-mode storage + buffer model (no adiabatic elimination) on the unsaturated range |alpha|^2 <= 4, zero-fit
    inputs; pulsed protocols with the mean tracking and with the single-mode optimal tracking fraction."""
    p = resolve(pre, "zero-fit")
    rows = []

    def add(nb, proto, track, one, two, f, Nb):
        rows.append(dict(nbar=nb, protocol=proto, tracking=track, tracking_f=f, Nb=Nb, single_mode_s=one, two_mode_s=two,
                         ratio=two / one))
        print(f"nbar {nb:.1f} {proto:9s} {track:7s} Nb {Nb} | single-mode {one:.4g} s  two-mode {two:.4g} s  "
              f"ratio {two / one:.3f}", flush=True)

    for nb in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        Ns = int(2 * nb + 8 * np.sqrt(nb) + 14)
        add(nb, "static", "-", cycle_TX(window(nb, p), nb, p, T_CYCLE)[0], two_mode_static_TX(nb, p, Ns), np.nan, 4)
        for T_on in (1.0, 0.2):
            for track in ("mean", "optimal"):
                q = dict(p, track=track)
                one, _, f = cycle_TX(window(nb, q), nb, q, T_on)
                add(nb, SERIES_OF[T_on], track, one, two_mode_cycle_TX(nb, q, T_on, Ns, f), f, 4)
    # buffer-cutoff convergence
    nb, Ns = 4.0, int(2 * 4 + 8 * 2 + 14)
    add(nb, "static", "-", cycle_TX(window(nb, p), nb, p, T_CYCLE)[0], two_mode_static_TX(nb, p, Ns, Nb=6), np.nan, 6)
    q = dict(p, track="mean")
    add(nb, "Ton_200ns", "mean", cycle_TX(window(nb, q), nb, q, 0.2)[0], two_mode_cycle_TX(nb, q, 0.2, Ns, 1.0, Nb=6), 1.0, 6)
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / "companion_pulsed_two_mode.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Companion device, zero-fit inputs: bit-flip time 2T_Z (s), effective single-mode model (exact buffer "
                 "elimination with chi_sb) vs the full two-mode storage + buffer model (buffer cutoff Nb); pulsed with the "
                 "mean or the single-mode-optimal Kerr tracking. examples/companion_pulsed.py --validate\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("saved", path)


# ---------------------------------------------------------------------------------------------------------- main scan
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated variants to (re)compute; other rows kept from the CSV")
    ap.add_argument("--plot", action="store_true", help="summary and figure from the saved CSVs")
    a = ap.parse_args()
    pre = load_preset("companion")
    if a.validate:
        validate(pre)
        return
    dat, sim = load_data()
    if a.plot:
        rows = read_rows("companion_pulsed.csv")
        summary(rows, sim)
        plot(rows, dat, sim)
        return
    path = ROOT / "data" / "validation" / "companion_pulsed.csv"
    only = [v.strip() for v in a.only.split(",") if v.strip()]
    assert all(v in VARIANTS for v in only), only
    rows = []
    if only and path.exists():
        for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")):
            if r["variant"] not in only:
                rows.append({k: (v if k in ("variant", "series") else float(v)) for k, v in r.items()})
    for name in (only or VARIANTS):
        p = resolve(pre, name)
        for nb in NBARS:
            w = window(nb, p)
            for T_on in T_ONS:
                if name == "K_s 5 kHz" and T_on != 1.0:
                    continue
                TX, pc, f = cycle_TX(w, nb, p, T_on)
                d = dat.get((SERIES_OF[T_on], nb), (np.nan, np.nan, np.nan))
                rows.append(dict(variant=name, series=SERIES_OF[T_on], nbar=nb, tracking_f=f, p_cycle=pc, model_T_s=TX,
                                 data_T_s=d[0],
                                 data_T_low_s=d[1], data_T_high_s=d[2]))
        print(f"{name}: done", flush=True)
    # Fock-cutoff check at the largest cat
    p = resolve(pre, "zero-fit")
    for T_on in (() if only else T_ONS):
        a1 = cycle_TX(window(8.0, p), 8.0, p, T_on)[0]
        a2 = cycle_TX(window(8.0, p, N=int(2 * 8 + 8 * np.sqrt(8) + 14) + 12), 8.0, p, T_on)[0]
        print(f"cutoff check |alpha|^2 = 8, {SERIES_OF[T_on]}: {a1:.5g} s vs {a2:.5g} s (N + 12)")
    keys = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Companion device (PRX 15, 011070), pulsed stabilization, T_cycle = 3 us: bit-flip time 2T_Z (s) of the "
                 "window model (exact buffer elimination with chi_sb) per variant vs Fig. 7b (data_*: VALIDATION ONLY). "
                 "'aws_like' uses AWS's fitted dephasing to cross-check their simulations, not a prediction. "
                 "examples/companion_pulsed.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("saved", path)
    summary(rows, sim)
    plot(rows, dat, sim)


def summary(rows, sim):
    two = read_rows("companion_pulsed_two_mode.csv")
    get = lambda v, s, nb: next((r for r in rows if r["variant"] == v and r["series"] == s and r["nbar"] == nb), None)
    print("\ndata / model (bit-flip time), |alpha|^2 <= 4 (the static data saturate above):")
    for v in ("zero-fit", "optimal tracking", "pump-off dephasing", "pump-off dephasing optimal tracking"):
        for s in ("static", "Ton_1us", "Ton_200ns"):
            sel = [r for r in rows if r["variant"] == v and r["series"] == s and r["nbar"] <= 4 and not np.isnan(r["data_T_s"])]
            if v in ("optimal tracking", "pump-off dephasing optimal tracking") and s == "static":
                continue
            print(f"  {v:17s} {s:9s} " + "  ".join(f"{r['nbar']:g}: {r['data_T_s'] / r['model_T_s']:.2f}" for r in sel))
    if two:
        for tr in ("mean", "optimal"):
            for s in ("static", "Ton_1us", "Ton_200ns"):
                if s == "static" and tr == "optimal":
                    continue
                sel = [r for r in two if r["protocol"] == s and r["Nb"] == 4 and (s == "static" or r["tracking"] == tr)]
                out = []
                for r in sel:
                    d = get("zero-fit", s, r["nbar"])
                    if d is not None and not np.isnan(d["data_T_s"]):
                        out.append(f"{r['nbar']:g}: {d['data_T_s'] / r['two_mode_s']:.2f}")
                print(f"  two-mode {tr:8s} {s:9s} " + "  ".join(out))
    print("pulsed / static (model mean tracking | optimal | two-mode mean | data):")
    for s in ("Ton_1us", "Ton_200ns"):
        out = []
        for nb in range(1, 9):
            m, o, st = get("zero-fit", s, nb), get("optimal tracking", s, nb), get("zero-fit", "static", nb)
            if m is None or np.isnan(m["data_T_s"]):
                continue
            t2 = [r["two_mode_s"] for r in two if r["protocol"] == s and r["nbar"] == nb and r["Nb"] == 4 and r["tracking"] == "mean"]
            t2s = [r["two_mode_s"] for r in two if r["protocol"] == "static" and r["nbar"] == nb and r["Nb"] == 4]
            tm = f"{t2[0] / t2s[0]:.2f}" if t2 and t2s else "  - "
            out.append(f"{nb}: {m['model_T_s'] / st['model_T_s']:.2f} | {o['model_T_s'] / st['model_T_s']:.2f} | {tm} | "
                       f"{m['data_T_s'] / st['data_T_s']:.2f}")
        print(f"  {s:9s} " + ";  ".join(out))
    print("pulsed / static, hypothesis 'flip-relevant dephasing = pump-off value' (mean | optimal tracking | data):")
    for s in ("Ton_1us", "Ton_200ns"):
        out = []
        for nb in range(1, 9):
            m, o = get("pump-off dephasing", s, nb), get("pump-off dephasing optimal tracking", s, nb)
            sm, so = get("pump-off dephasing", "static", nb), get("pump-off dephasing optimal tracking", "static", nb)
            if m is None or np.isnan(m["data_T_s"]):
                continue
            out.append(f"{nb}: {m['model_T_s'] / sm['model_T_s']:.2f} | {o['model_T_s'] / so['model_T_s']:.2f} | "
                       f"{m['data_T_s'] / sm['data_T_s']:.2f}")
        print(f"  {s:9s} " + ";  ".join(out))
    print("optimal tracking fraction f (of the mean Kerr rotation), 1 us / 200 ns: " + "  ".join(
        f"{nb:g}: {get('optimal tracking', 'Ton_1us', nb)['tracking_f']:.2f}/{get('optimal tracking', 'Ton_200ns', nb)['tracking_f']:.2f}"
        for nb in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)))
    print("AWS simulation / ours with AWS's inputs (aws_like: mean tracking | aws_like optimal tracking):")
    for s in ("static", "Ton_1us", "Ton_200ns"):
        xs, ys = np.array(sim[s]).T
        out = []
        for nb in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
            m, o = get("aws_like", s, nb), get("aws_like optimal tracking", s, nb)
            if xs.min() <= nb <= xs.max() and m is not None:
                out.append(f"{nb:g}: {np.interp(nb, xs, ys) / m['model_T_s']:.2f} | {np.interp(nb, xs, ys) / o['model_T_s']:.2f}")
        print(f"  {s:9s} " + ";  ".join(out))
    xs, ys = np.array(sim["Ton_1us_Ks_5kHz"]).T
    sel = [r for r in rows if r["variant"] == "K_s 5 kHz" and r["series"] == "Ton_1us" and r["nbar"] in (2.0, 4.0, 6.0, 8.0)]
    print("K_s 5 kHz, T_on 1 us (our zero-fit, mean tracking) vs AWS's dotted curve (s): " +
          "  ".join(f"{r['nbar']:g}: {r['model_T_s']:.3g}/{np.interp(r['nbar'], xs, ys):.3g}" for r in sel))


def read_rows(name, strings=("variant", "series", "protocol", "tracking")):
    path = ROOT / "data" / "validation" / name
    if not path.exists():
        return []
    return [{k: (v if k in strings else float(v)) for k, v in r.items()}
            for r in csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#"))]


def plot(rows, dat, sim):
    """Left: bit-flip times, data vs the zero-fit model (single mode: mean tracking solid, optimal tracking dashed;
    two-mode storage + buffer for |alpha|^2 <= 4: squares) and AWS's simulations (dotted). Right: pulsed / static."""
    two = read_rows("companion_pulsed_two_mode.csv")
    use_style()
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    cols = {"static": TEXT_PRIMARY, "Ton_1us": SERIES[0], "Ton_200ns": SERIES[1]}
    lab = {"static": "static", "Ton_1us": "T_on = 1 µs", "Ton_200ns": "T_on = 200 ns"}
    model = lambda v, s: sorted(((r["nbar"], r["model_T_s"]) for r in rows if r["variant"] == v and r["series"] == s))
    tm = lambda s, tr: sorted((r["nbar"], r["two_mode_s"]) for r in two if r["protocol"] == s and r["Nb"] == 4
                              and (s == "static" or r["tracking"] == tr))
    ax = axs[0]
    for s in ("static", "Ton_1us", "Ton_200ns"):
        pts = sorted((k[1], v) for k, v in dat.items() if k[0] == s)
        y = np.array([q[1][0] for q in pts])
        err = np.array([[q[1][0] - q[1][1] for q in pts], [q[1][2] - q[1][0] for q in pts]])
        ax.errorbar([q[0] for q in pts], y, yerr=err, fmt="o", ms=4, mfc="none", color=cols[s], label=f"data {lab[s]}")
        x, yv = np.array(model("zero-fit", s)).T
        ax.plot(x, yv, "-", color=cols[s], label=f"model {lab[s]}")
        if s != "static":
            x, yv = np.array(model("optimal tracking", s)).T
            ax.plot(x, yv, "--", color=cols[s], lw=1.0)
        else:
            x, yv = np.array(model("pump-off dephasing", s)).T
            ax.plot(x, yv, "-.", color=cols[s], lw=1.0, label="model static, pump-off dephasing")
        if tm(s, "mean"):
            x, yv = np.array(tm(s, "mean")).T
            ax.plot(x, yv, "s", ms=3.5, color=cols[s])
        xs, ys = np.array(sim[s]).T
        ax.plot(xs, ys, ":", lw=1.0, color=cols[s], alpha=0.6)
    ax.plot([], [], "--", color="0.4", lw=1.0, label="model, optimal Kerr tracking")
    ax.plot([], [], "s", ms=3.5, color="0.4", label="two-mode model (no elimination)")
    ax.plot([], [], ":", color="0.4", label="AWS simulation (fitted dephasing)")
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 5)
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_ylabel("bit-flip time 2T_Z (s)")
    ax.set_title("companion device (PRX 15, 011070, Fig. 7b) vs zero-fit window model", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax = axs[1]
    for s in ("Ton_1us", "Ton_200ns"):
        for v, ls, tag in (("zero-fit", "-", "mean tracking"), ("optimal tracking", "--", "optimal tracking"),
                           ("pump-off dephasing optimal tracking", "-.", "pump-off dephasing, optimal tracking")):
            a, b = dict(model(v, s)), dict(model(v, "static"))
            x = [nb for nb in NBARS if nb in a]
            ax.plot(x, [a[nb] / b[nb] for nb in x], ls, color=cols[s], lw=1.5 if ls == "-" else 1.0,
                    label=f"model {lab[s]}, {tag}")
        if tm(s, "mean"):
            a, b = dict(tm(s, "mean")), dict(tm("static", None))
            x = sorted(a)
            ax.plot(x, [a[nb] / b[nb] for nb in x], "s", ms=3.5, color=cols[s])
        x = [nb for nb in NBARS if (s, nb) in dat and ("static", nb) in dat]
        r = np.array([dat[(s, nb)][0] / dat[("static", nb)][0] for nb in x])
        rel = np.array([np.hypot((dat[(s, nb)][2] - dat[(s, nb)][1]) / 2 / dat[(s, nb)][0],
                                 (dat[("static", nb)][2] - dat[("static", nb)][1]) / 2 / dat[("static", nb)][0]) for nb in x])
        ax.errorbar(x, r, yerr=r * rel, fmt="o", ms=4, mfc="none", color=cols[s], label=f"data {lab[s]}")
    ax.axhline(1, color="0.6", lw=0.6)
    ax.axvspan(4.25, 8.3, color="0.9", zorder=0)
    ax.text(6.25, 0.97, "static data saturated\n(0.3–0.6 s, unexplained)", fontsize=7, ha="center", va="top", color="0.4",
            transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\alpha|^2$")
    ax.set_ylabel("T_X(pulsed) / T_X(static)")
    ax.set_title("cost of pulsing: model vs data", fontsize=9)
    ax.legend(fontsize=6, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"companion_pulsed.{ext}", dpi=160)
    print("saved figures/companion_pulsed.png/.pdf")


if __name__ == "__main__":
    main()
