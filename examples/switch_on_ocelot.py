"""The switch-on transient of the two-photon stabilization (storage + buffer, no adiabatic elimination; paper App. J).

Two-mode model in the frame of the pump and drive (App. D1, F8):
    H = Delta_b b^dag b + chi_sb a^dag a b^dag b + (K_s/2) a^dag^2 a^2
        + g2 s_p(t) (a^2 b^dag + h.c.) - g2 nbar s_d(t) (b + b^dag),
    kappa_b D[b] (buffer), kappa1 D[a] and kphi D[a^dag a] (storage, T1 and T2 under dissipation, Fig. 9g,h),
with flat-top Gaussian envelopes s_p (pump) and s_d (drive), ramp time tau (3 sigma) and a delay between them
(delay > 0: the pump starts first). The storage starts in |0_L> rotated by theta (a post-gate distortion) and the
buffer in vacuum; after the ramp and a hold the storage's flip probability is taken with the adiabatic projector.
References with the same hold and the same idle noise:
  'adiabatic'      single-mode kappa2 D[a^2 - nbar] (+ Kerr-cat term of Delta_b) switched on instantly;
  'instantaneous'  the noise-free asymptotic map of the CX model.
  'always on'      the two-mode model with the stabilization on from t = 0 (--separation).
two-mode - adiabatic = the steady non-adiabatic (buffer, cross-Kerr) correction + the switch-on transient;
two-mode - always on = the switch-on transient itself (and the timing error). Switching would add bit flips if
the transient exceeded the idle contribution over the same time. Buffer and storage parameters:
Fig. 9, Fig. 13 (measured), AWS model nonlinearities (App. D1: chi_sb/2pi < 120 kHz, 390 kHz for the S4 cell).

Usage: python examples/switch_on_ocelot.py [--separation] [--precision] [--conversion] [--plot]
Output: data/validation/switch_on.csv, figures/switch_on.png/.pdf; the options write data/validation/switch_on_{separation,precision,conversion}.csv
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
import qutip as qt

from bosonoise.channels.conversion import AsymptoticProjector, logical_states
from bosonoise.plotting import SERIES, use_style
from bosonoise.presets import load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
TWO_PI = 2 * np.pi
HOLD = 2.0  # us after the ramps
OPTS = {"atol": 1e-12, "rtol": 1e-10, "nsteps": 500000}


def envelope(t, t0, ramp):
    if t < t0:
        return 0.0
    if t < t0 + ramp:
        return float(np.exp(-0.5 * ((t - t0 - ramp) / (ramp / 3.0)) ** 2))
    return 1.0


def cat(nbar, theta, N):
    z, _ = logical_states(nbar, N)
    return ((-1j * theta * qt.num(N)).expm() * qt.Qobj(z)).unit()


def two_mode(nbar, theta, delay, ramp, p, Nb=4, always_on=False, Ns=None, opts=None):
    """always_on: pump and drive at full strength from t = 0 (a step instead of the ramps), same total time.
    Ns: storage cutoff (default: the projector's); opts: solver options (default OPTS)."""
    P = AsymptoticProjector(nbar, N=Ns, kerr_ratio=-p["Db"] / p["kb"])
    Ns = P.N
    a = qt.tensor(qt.destroy(Ns), qt.qeye(Nb))
    b = qt.tensor(qt.qeye(Ns), qt.destroy(Nb))
    H0 = p["Db"] * b.dag() * b + p["chi"] * a.dag() * a * b.dag() * b + 0.5 * p["Ks"] * a.dag() ** 2 * a ** 2
    Hp = p["g2"] * (a ** 2 * b.dag() + a.dag() ** 2 * b)
    Hd = -p["g2"] * nbar * (b + b.dag())
    tp, td = max(0.0, -delay), max(0.0, delay)
    env = (lambda t, t0, r: 1.0) if always_on else envelope
    H = [H0, [Hp, lambda t: env(t, tp, ramp)], [Hd, lambda t: env(t, td, ramp)]]
    c_ops = [np.sqrt(p["kb"]) * b, np.sqrt(p["k1"]) * a, np.sqrt(p["kphi"]) * a.dag() * a]
    rho0 = qt.tensor(qt.ket2dm(cat(nbar, theta, Ns)), qt.fock_dm(Nb, 0))
    T = abs(delay) + ramp + HOLD
    res = qt.mesolve(H, rho0, [0, T], c_ops, options=opts or OPTS)
    return P.p_flip(res.states[-1].ptrace(0).full()), T


def adiabatic(nbar, theta, T, p):
    P = AsymptoticProjector(nbar, kerr_ratio=-p["Db"] / p["kb"])
    a = qt.destroy(P.N)
    k2 = 4 * p["g2"] ** 2 / p["kb"]
    L2 = a ** 2 - nbar
    H = k2 * (-p["Db"] / p["kb"]) * L2.dag() * L2 + 0.5 * p["Ks"] * a.dag() ** 2 * a ** 2
    c_ops = [np.sqrt(k2) * L2, np.sqrt(p["k1"]) * a, np.sqrt(p["kphi"]) * a.dag() * a]
    res = qt.mesolve(H, qt.ket2dm(cat(nbar, theta, P.N)), [0, T], c_ops, options=OPTS)
    return P.p_flip(res.states[-1].full()), P.p_flip(qt.ket2dm(cat(nbar, theta, P.N)).full())


def cell_params(pre, st, chi):
    s = pre["storage"][st]
    return dict(g2=TWO_PI * pre["two_photon_dissipation"][st]["g2_kHz"] * 1e-3, kb=TWO_PI * pre["buffer"][f"B{st[1]}"]["kappa_b_MHz"],
                Db=TWO_PI * -2.0, chi=TWO_PI * chi, Ks=TWO_PI * (2.7e-3 if st == "S4" else 1e-3), k1=1 / s["T1_under_2ph_diss_us"],
                kphi=max(2 / s["T2_under_2ph_diss_us"] - 1 / s["T1_under_2ph_diss_us"], 0.0))


def separation(pre, cells):
    """Switch-on transient vs steady two-mode correction at zero delay (30-ns ramps)."""
    rows = []
    for st, chi in cells.items():
        p = cell_params(pre, st, chi)
        for nbar in (2.0, 4.0):
            for theta in (0.0, 0.5):
                p_ramp, T = two_mode(nbar, theta, 0.0, 0.03, p)
                p_on, _ = two_mode(nbar, theta, 0.0, 0.03, p, always_on=True)
                pa, pinst = adiabatic(nbar, theta, T, p)
                rows.append(dict(cell=st, nbar=nbar, theta=theta, two_mode_ramp=p_ramp, two_mode_always_on=p_on, adiabatic=pa,
                                 instantaneous=pinst, switch_on_transient=p_ramp - p_on, steady_two_mode_correction=p_on - pa,
                                 idle_during_relaxation=pa - pinst))
                print(f"{st} nbar {nbar:.0f} theta {theta:.1f} | transient {p_ramp - p_on:+.2e} | steady correction {p_on - pa:+.2e} "
                      f"| adiabatic {pa:.3e} | instantaneous {pinst:.3e}", flush=True)
    write_csv(rows, "switch_on_separation.csv", "Switch-on transient (ramped - always on) vs steady non-adiabatic "
              "correction (always on - adiabatic) vs idle noise during relaxation (adiabatic - noise-free map); zero delay, "
              "30-ns ramps, 2-us hold. Model only. examples/switch_on_ocelot.py --separation")


def write_csv(rows, name, header):
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / name
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {header}\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("saved", path)


def precision(pre):
    """Spot checks of the ~1e-8 switch-on transient (S1, |alpha|^2 = 4, zero delay, 30-ns ramps): tighter solver
    tolerances, a larger buffer cutoff, a larger storage cutoff."""
    p = cell_params(pre, "S1", 0.12)
    N0 = AsymptoticProjector(4.0).N
    tight = {"atol": 1e-14, "rtol": 1e-12, "nsteps": 5000000}
    rows = []
    for label, kw in [("base", {}), ("tolerances 1e-14/1e-12", {"opts": tight}), ("buffer cutoff 6", {"Nb": 6}),
                      (f"storage cutoff {N0 + 10}", {"Ns": N0 + 10})]:
        for theta in (0.0, 0.5):
            t0 = time.time()
            p_ramp, _ = two_mode(4.0, theta, 0.0, 0.03, p, **kw)
            p_on, _ = two_mode(4.0, theta, 0.0, 0.03, p, always_on=True, **kw)
            rows.append(dict(check=label, theta=theta, two_mode_ramp=p_ramp, two_mode_always_on=p_on,
                             switch_on_transient=p_ramp - p_on))
            print(f"{label:24s} theta {theta:.1f} | ramped {p_ramp:.8e} always on {p_on:.8e} | transient {p_ramp - p_on:+.4e} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    write_csv(rows, "switch_on_precision.csv", "Precision spot checks of the switch-on transient, S1, |alpha|^2 = 4, "
              "zero delay, 30-ns ramps, 2-us hold. Model only. examples/switch_on_ocelot.py --precision")


def conversion(pre, cells):
    """Idle noise during the re-stabilization and the flip conversion of a rotated cat (single-mode adiabatic model).
    A decay at a uniformly random time of a CX stops the storage rotation at a uniform theta in [0, pi]. Compared: the
    flip probability after the window (noisy) and the noise-free map composed with the ideal-cat idle flip,
    P_free (1 - 2 p_idle) + p_idle; their difference is the noise-assisted conversion (excess)."""
    thetas = np.unique(np.concatenate([np.linspace(0, np.pi, 25), [0.25, 0.5, 1.0]]))
    rows = []
    for st, chi in cells.items():
        p = cell_params(pre, st, chi)
        for nbar in (2.0, 4.0, 6.0):
            p_idle = adiabatic(nbar, 0.0, HOLD, p)[0]
            sel = []
            for th in thetas:
                pn, pf = adiabatic(nbar, th, HOLD, p)
                sel.append(dict(cell=st, nbar=nbar, theta=th, noisy=pn, noise_free=pf, idle=p_idle,
                                excess=pn - (pf * (1 - 2 * p_idle) + p_idle)))
            rows += sel
            avg = lambda k: float(np.sum([(u[k] + v[k]) / 2 * (v["theta"] - u["theta"]) for u, v in zip(sel, sel[1:])]) / np.pi)
            at = {round(r["theta"], 2): r for r in sel}
            print(f"{st} nbar {nbar:.0f} | idle {p_idle:.2e} | uniform-theta mean: noise-free {avg('noise_free'):.4f} "
                  f"noisy {avg('noisy'):.4f} excess {avg('excess'):+.2e} | excess/noise-free at theta 0.25, 0.5, 1.0: "
                  + ", ".join(f"{at[t]['excess']:+.2e}/{at[t]['noise_free']:.2e}" for t in (0.25, 0.5, 1.0)), flush=True)
    write_csv(rows, "switch_on_conversion.csv", "Flip conversion of a cat rotated by theta after a 2-us stabilization "
              "window with idle noise (noisy) vs the noise-free map composed with the ideal-cat idle flip; single-mode "
              "adiabatic model. Model only. examples/switch_on_ocelot.py --conversion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--separation", action="store_true")
    ap.add_argument("--precision", action="store_true")
    ap.add_argument("--conversion", action="store_true")
    ap.add_argument("--plot", action="store_true", help="redraw figures/switch_on.* from the saved CSVs")
    a = ap.parse_args()
    pre = load_preset("ocelot")
    rows = []
    cells = {"S1": 0.12, "S4": 0.39}  # chi_sb/2pi in MHz: AWS model bound for most cells / the S4 cell (App. D1)
    if a.plot:
        plot(read_csv("switch_on.csv"), list(cells))
        return
    if a.separation or a.precision or a.conversion:
        if a.separation:
            separation(pre, cells)
        if a.precision:
            precision(pre)
        if a.conversion:
            conversion(pre, cells)
        return
    for st, chi in cells.items():
        p = cell_params(pre, st, chi)
        for nbar in (2.0, 4.0):
            for theta in (0.0, 0.5):
                for ramp, delay in [(0.03, 0.0), (0.01, 0.0), (0.1, 0.0), (0.03, 0.02), (0.03, 0.05), (0.03, 0.1),
                                    (0.03, 0.2), (0.03, -0.05)]:
                    t0 = time.time()
                    p2, T = two_mode(nbar, theta, delay, ramp, p)
                    pa, pinst = adiabatic(nbar, theta, T, p)
                    rows.append(dict(cell=st, chi_sb_MHz=chi, nbar=nbar, theta=theta, ramp_ns=ramp * 1e3, delay_ns=delay * 1e3,
                                     two_mode=p2, adiabatic=pa, instantaneous=pinst, excess=p2 - pa))
                    print(f"{st} nbar {nbar:.0f} theta {theta:.1f} ramp {ramp * 1e3:4.0f} delay {delay * 1e3:+5.0f} ns | "
                          f"two-mode {p2:.3e} adiabatic {pa:.3e} instantaneous {pinst:.3e} | excess {p2 - pa:+.2e} "
                          f"({time.time() - t0:.0f} s)", flush=True)
    keys = list(rows[0])
    path = ROOT / "data" / "validation" / "switch_on.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Switch-on transient: flip probability after ramp + 2-us hold, two-mode storage+buffer vs adiabatic single "
                 "mode with the same idle noise vs the noise-free instantaneous map. Model only. examples/switch_on_ocelot.py\n")
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if not isinstance(r[k], str) else r[k] for k in keys) + "\n")
    print("saved", path)
    plot(rows, list(cells))


def read_csv(name):
    path = ROOT / "data" / "validation" / name
    if not path.exists():
        return []
    rows = list(csv.DictReader(l for l in open(path, encoding="utf-8") if not l.startswith("#")))
    num = lambda v: float(v) if v.replace(".", "", 1).replace("e", "", 1).replace("-", "").replace("+", "").isdigit() else v
    return [{k: num(v) for k, v in r.items()} for r in rows]


def plot(rows, cells):
    """Two-mode minus adiabatic vs delay (switch-on + steady two-mode correction + timing error), with the isolated
    switch-on transient at zero delay (ramped minus always on, from --separation) as crosses."""
    sep = read_csv("switch_on_separation.csv")
    use_style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for i, (st, nb) in enumerate([(s_, n_) for s_ in cells for n_ in (2.0, 4.0)]):
        for ax, th in zip(axs, (0.0, 0.5)):
            sel = sorted((r for r in rows if r["cell"] == st and r["nbar"] == nb and r["theta"] == th and r["ramp_ns"] == 30),
                         key=lambda r: r["delay_ns"])
            ax.plot([r["delay_ns"] for r in sel], [max(r["excess"], 1e-9) for r in sel], "o-", ms=3, color=SERIES[i],
                    label=f"{st}, |α|²={nb:g}")
            tr = [r["switch_on_transient"] for r in sep if r["cell"] == st and r["nbar"] == nb and r["theta"] == th]
            if tr:
                ax.plot([0.0], [max(tr[0], 1e-10)], "x", ms=6, mew=1.5, color=SERIES[i],
                        label="switch-on transient alone (zero delay)" if i == 0 else None)
    for ax, th in zip(axs, (0.0, 0.5)):
        ax.set_yscale("log")
        ax.set_xlabel("pump-before-drive delay (ns)")
        ax.set_title(f"two-mode − adiabatic after ramps + 2-µs hold, start rotated by θ = {th}", fontsize=9)
    axs[0].set_ylabel("extra flip probability")
    axs[0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"switch_on.{ext}", dpi=160)
    print("saved figures/switch_on.png/.pdf")


if __name__ == "__main__":
    main()
