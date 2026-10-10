"""The three reference noise models of repetition cat codes (paper, Data availability), as RoundNoise factories
(dem/noise_models.py), parametrized by the mean photon number nbar and a few physical knobs. Every parameter carries
its provenance: our calculation (with the section of the paper), a dataset, or a literature reference.

  ocelot_model   hybrid cat-transmon cell (AWS Ocelot): our first-principles stabilizer fault tables (X, X X pairs,
                 f -> e heralds), our dispersive-phase syndrome error, data-derived idle bit flips and storage T1,eff,
                 ancilla leakage from the erasure persistence of the public data.
  allcat_model   dissipative cats as data and ancillas (Alice & Bob style)       [literature model: presets/allcat.yaml]
  kerrcat_model  Kerr-cat qubits (Hamiltonian stabilization, leakage)            [literature model: presets/kerrcat.yaml]
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import yaml

from ..presets import load_preset
from .noise_models import Leakage, RoundNoise

_ANC = ["A1", "A2", "A3", "A4"]
_CATS = ["S1", "S2", "S3", "S4", "S5"]


def _ocelot_tables():
    text = resources.files("bosonoise.presets").joinpath("ocelot_rep_noise.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _loglin(x, pts):
    xs = np.array(sorted(pts))
    ys = np.array([pts[k] for k in xs])
    return float(np.exp(np.interp(x, xs, np.log(np.clip(ys, 1e-300, None)))))


def ocelot_model(nbar: float, d: int, variant: str = "lambda3", T1_eff_us: float = 57.0, T_cycle_us: float = 2.8,
                 spam: float = 0.015, dispersive: str = "fixed_gate", n_th: float = 0.005, f_mid: float = 0.2,
                 ancilla_leakage: bool = True) -> RoundNoise:
    """Ocelot-type hybrid repetition cat code, the d = 5 device repeated periodically (stabilizers A1..A4, cats S1..S5).
    Knobs: nbar (1 .. 16), variant ('zero-fit' or 'lambda3' on-gate ancilla decay), storage T1,eff, cycle time, SPAM,
    'fixed_gate' or 'retuned' dispersive phase, thermal population, ancilla leakage on/off."""
    tab = _ocelot_tables()
    pre = yaml.safe_load(resources.files("bosonoise.presets").joinpath("ocelot.yaml").read_text(encoding="utf-8"))
    p_data = np.zeros((d, 3))
    for i in range(d):
        idle = tab["idle"][_CATS[i % 5]]
        pts = {float(k): float(v) for k, v in idle["points"].items()}
        px = _loglin(nbar, pts) if nbar <= max(pts) else idle["A"] * np.exp(-idle["B"] * nbar)
        TX = -2 * 2.8 / np.log(1 - 2 * px)  # the table is per 2.8-us cycle
        p_data[i, 0] = 0.5 * (1 - np.exp(-2 * T_cycle_us / TX))
        p_data[i, 2] = 0.5 * (1 - np.exp(-2 * (nbar * (1 + 2 * n_th) + n_th) * T_cycle_us / T1_eff_us))
    p_stab = np.zeros((d - 1, 3))
    p_meas, p_herald = np.zeros(d - 1), np.zeros(d - 1)
    for j in range(d - 1):
        anc = _ANC[j % 4]
        st = {float(k): v for k, v in tab["stabilizers"][anc][variant].items()}
        g = lambda key: _loglin(nbar, {k: v[key] for k, v in st.items()})
        pxa, pxb, pxx = g("p_x_a"), g("p_x_b"), g("p_xx")
        p_stab[j] = (max(pxa - pxx, 0.0), max(pxb - pxx, 0.0), pxx)
        disp = _loglin(nbar, {float(k): v[dispersive] for k, v in tab["dispersive"][anc].items()})
        p_meas[j] = spam + disp
        a = pre["ancilla"][anc]
        ro = 0.5 * (1 - np.exp(-(a["readout_ns"] + 80.0) * 1e-3 / a["T1_ef_us"]))
        h = g("p_herald") + tab["herald_window"][anc][variant]
        p_herald[j] = h + ro - h * ro
    leak = Leakage(p_leak=4e-4, p_return=0.5, heralded=True, randomize_measurement=True) if ancilla_leakage else Leakage()
    prov = {
        "p_data[X]": "DATA-DERIVED idle bit flip, Ocelot Fig. 16 static (Putterman et al., Nature 638, 927 (2025)); continuation A exp(-B nbar) beyond |alpha|^2 = 4",
        "p_data[Z]": f"kappa_1,eff (nbar (1 + 2 n_th) + n_th) T_cycle with T1,eff = {T1_eff_us} us (default: Ocelot Fig. 30 mean, rep-code run) and n_th = {n_th} (prior)",
        "f_mid": "share of phase flips between a cat's two CX gates; 0.2 from the space-like/diagonal detector-pair split of the raw data (paper App. H) and the pulse timing",
        "p_stab_x": f"our dressed-ladder ULE fault tables, variant {variant} (paper Sec. 4, App. D)",
        "p_meas": f"SPAM {spam} (Ocelot App. F3) + our dispersive-phase syndrome error, {dispersive} (paper App. H); the measured growth with |alpha|^2 is only partly explained",
        "p_herald": "our f->e herald over the gates and the other layer's gate (paper App. H) + f->e during the readout of odd outcomes (readout + 80 ns / T1_ef, Ocelot Fig. 9d, App. F3)",
        "p_data_meas": f"SPAM {spam} of the g/e ancilla readout used for both the parity and the well (Z) readout (Ocelot App. F3; the Z readout fidelity vs |alpha|^2 of Fig. 19b is not digitized)",
        "leak_anc": "DATA-DERIVED: excess persistence of erasures between rounds, 0.2-0.6% (examples/zenodo_erasure_ocelot.py; paper App. H) -> p_leak 4e-4 per round with mean duration 2 rounds; heralded, outcome random; effect on the cats unknown (none assumed)",
        "leak_data": "none: two-photon stabilization returns the cats to the code space; excursions are converted into bit/phase flips (our conversion map, paper Sec. 4, App. D)",
    }
    return RoundNoise(p_data=p_data, f_mid=f_mid, p_stab_x=p_stab, p_meas=p_meas, p_herald=p_herald,
                      p_data_meas=spam, leak_anc=leak, provenance=prov,
                      name=f"ocelot(nbar={nbar:g}, {variant}, T1eff={T1_eff_us:g} us, Tc={T_cycle_us:g} us)")


@lru_cache(maxsize=None)
def _preset(name: str) -> dict:
    return load_preset(name)


def _v(entry):
    """The number of a preset entry ({value: x, source: ...} or a bare value)."""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


def _boundary_idle(p_data, extra):
    """The two boundary cats take part in one CX per round and idle during the other layer: add that idle (X, Y, Z)."""
    p = np.array(p_data, float)
    p[0] += extra
    p[-1] += extra
    return p


def allcat_model(nbar: float, d: int, eta: float = 1e-3, n_th: float = 0.0, herald_efficiency: float = 0.0,
                 kappa2_MHz: float | None = None) -> RoundNoise:
    """Dissipative cats as data and ancillas (Alice & Bob style), in the convention of Le Regent et al., Quantum 7, 1198
    (2023) [LR23]: operations of duration T = 1/kappa2 and a refreshing step between the two CNOTs, the cycle of LR23
    and Gouzien et al., PRL 131, 040602 (2023) [GO23]: prep | CNOT | refresh | CNOT | X measurement = 5/kappa2.
    Knobs: nbar = |alpha|^2, eta = kappa1/kappa2, thermal population (kappa1 -> kappa1 (1 + 2 n_th), Chamberland et al.,
    PRX Quantum 3, 010329 (2022) [CH22] App. D), the efficiency of buffer-emission heralds of data bit flips (an
    assumption, off by default), and kappa2 (absolute times only; default from the preset).
    Every literature number is read from presets/allcat.yaml, with its source."""
    P = _preset("allcat")
    c, idl, anc, ro = P["cnot"], P["idle"], P["ancilla"], P["readout"]
    if kappa2_MHz is None:
        kappa2_MHz = _v(P["timing"]["kappa2_MHz"])
    paulis = tuple(_v(c["bitflip_paulis"]))
    e = eta * (1 + 2 * n_th)
    za = nbar * e + _v(c["Z_ancilla_nonadiabatic"]) / nbar     # LR23 Eq. (3) at T = 1/kappa2
    zd = _v(c["Z_data_fraction"]) * nbar * e                  # LR23 Eq. (4): Z_d = Z_a Z_d
    px_cnot = _v(c["bitflip_prefactor"]) * np.exp(-_v(c["bitflip_exponent"]) * nbar)  # summed over the 12 Paulis
    idle_z = _v(idl["Z_per_step"]) * nbar * e                  # per step of 1/kappa2
    idle_x = _v(idl["XY_prefactor"]) * nbar * e * np.exp(-_v(idl["XY_exponent"]) * nbar)  # X and Y each; negligible
    herald_frac = min(max(herald_efficiency, 0.0), 1.0)
    cx = {"ZI": za, "IZ": zd, "ZZ": zd}
    for k in paulis:
        cx[k] = cx.get(k, 0.0) + (1 - herald_frac) * px_cnot / len(paulis)
    # data cats idle during prep, refresh (between their two CNOTs: the 'mid' window) and measurement: 3 steps;
    # the boundary cats also during the CNOT layer that does not involve them
    n = _v(idl["data_idle_steps"])
    p_data = _boundary_idle(np.tile([n * idle_x, n * idle_x, n * idle_z], (d, 1)), [idle_x, idle_x, idle_z])
    # heralded part of the data bit flips from the two CNOTs acting on each cat (the types that flip the target)
    p_hx = herald_frac * 2 * (sum(k[1] in "XY" for k in paulis) / len(paulis)) * px_cnot
    prov = {
        "preset": "presets/allcat.yaml (every literature number with its page and equation)",
        "cycle": f"LR23 Sec. 3 / GO23 App. D: prep, CNOT, refresh, CNOT, M_X, each 1/kappa2 = {1e3 / (2 * np.pi * kappa2_MHz):.0f} ns (kappa2/2pi = {kappa2_MHz} MHz, GO23 p. 4)",
        "cx_channel[Z]": "LR23 Eqs. (3)-(4): Z_a = |a|^2 kappa1 T + 0.159/(|a|^2 kappa2 T), Z_d = Z_aZ_d = |a|^2 kappa1 T/2, T = 1/kappa2",
        "cx_channel[bit flips]": "LR23 p. 8 and GO23 Eq. (D8): 0.5 exp(-2|a|^2) summed over 12 bit-flip-type Paulis; split equally among them (assumption: no published split at T = 1/kappa2)",
        "p_data": "idle Z |a|^2 kappa1 T per step (LR23 pp. 7-8, CH22 Eq. (D4)) for 3 steps (4 for the two boundary cats, idle during the other CNOT layer), so that every cat has 5 |a|^2 eta per round (LR23 p. 8); idle X = Y = 0.5 kappa1 |a|^2 T exp(-4|a|^2) per step (CH22 Table II)",
        "f_mid": "1/3: the refreshing step lies between a data cat's two CNOTs (LR23 Fig. 4 inset)",
        "p_prep/p_meas": "ancilla preparation, refresh and X measurement each |a|^2 kappa1 T (LR23 pp. 7-8); data readout the same",
        "p_data_meas_z": "well (Z) readout exp(-1.5 - 0.9 |a|^2) (CH22 Eq. (38), T_measure ~ 850 ns)",
        "thermal": f"kappa1 -> kappa1 (1 + 2 n_th) (CH22 App. D), n_th = {n_th}",
        "leakage": "none: with T = 1/kappa2 and refreshing steps the leakage stays below the bit-flip probability (LR23 Sec. 2.3)",
        "heralds": f"data bit-flip heralds with efficiency {herald_frac} (buffer emission, Ferrari et al., arXiv:2607.01375; shown for idle flips, assumed here for gate flips)",
        "validation": "reproduces the circuit-level logical phase-flip fit of LR23 Fig. 4b (examples/literature_checks_models.py)",
    }
    return RoundNoise(p_data=p_data, f_mid=_v(idl["f_mid"]), cx_channel=cx, p_meas=_v(anc["meas_steps"]) * idle_z,
                      p_prep=_v(anc["prep_steps"]) * idle_z, p_data_meas=_v(ro["parity_steps"]) * idle_z,
                      p_data_meas_z=float(np.exp(-_v(ro["well_a"]) - _v(ro["well_b"]) * nbar)), p_data_herald_x=p_hx,
                      provenance=prov, name=f"allcat(nbar={nbar:g}, eta={eta:g}, n_th={n_th:g}, herald_eff={herald_frac:g})")


@lru_cache(maxsize=None)
def _da21_tables() -> dict:
    """The DA21 operation tables of presets/kerrcat.yaml, keyed by |alpha|^2; times in 1/K, T_m = T_P + T_CD."""
    out = {}
    for t in _preset("kerrcat")["operations"].values():
        out[float(t["nbar"])] = dict(kappa=tuple(t["kappa_over_K"]), cx=tuple(t["cx_infidelity"]), bias=tuple(t["cx_bias"]),
                                     prep=tuple(t["prep_infidelity"]), meas=tuple(t["meas_infidelity"]),
                                     T_cx=t["T_cx"], T_p=t["T_p"], T_m=t["T_p"] + t["T_cd"])
    return out


def _powerlaw(x, xs, ys):
    """Piecewise power-law interpolation in x, continued with the end segments outside the table."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    i = int(np.clip(np.searchsorted(lx, np.log(x)) - 1, 0, len(lx) - 2))
    return float(np.exp(ly[i] + (ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i]) * (np.log(x) - lx[i])))


def _da21(nbar, kappa_over_K, key):
    """A DA21 quantity at (nbar, kappa/K): power law in kappa/K within each table, log-linear in nbar between
    |alpha|^2 = 1 and 6.25, the 6.25 value beyond. key: 'z' (CX Z part), 'nonz' (CX non-Z part), 'prep', 'meas',
    or a time 'T_p', 'T_m'."""
    tables = _da21_tables()

    def at(t):
        if key.startswith("T_"):
            return t[key]
        cx, b = np.asarray(t["cx"]), np.asarray(t["bias"])
        ys = {"z": cx * b / (1 + b), "nonz": cx / (1 + b)}.get(key)
        return _powerlaw(kappa_over_K, t["kappa"], ys if ys is not None else t[key])
    (n_lo, lo_t), (n_hi, hi_t) = sorted(tables.items())
    lo, hi = at(lo_t), at(hi_t)
    s = min((nbar - n_lo) / (n_hi - n_lo), 1.0)
    return float(lo * (hi / lo) ** s)


def kerrcat_model(nbar: float, d: int, kappa_over_K: float | None = None, n_th: float | None = None,
                  K_MHz: float | None = None, kappa2_over_K: float | None = None, lru: bool = True,
                  ancilla_leakage: bool = True) -> RoundNoise:
    """Kerr-cat qubits (Hamiltonian stabilization) with the bias-preserving CX of Puri et al., Sci. Adv. 6, eaay5901
    (2020) [PU20], built on the operations of Darmawan et al., PRX Quantum 2, 030345 (2021) [DA21] (Tables I and II,
    |alpha|^2 = 1 and 6.25; n_th = 8%, kappa2 = K/10 after each CX as autonomous leakage correction). Per round: ancilla
    preparation T_P, two CX (T_CX = 115/(|alpha|^2 K): DA21 112.6/K at 1, 18.47/K at 6.25), measurement T_M = T_P + T_CD.
    Knobs: nbar >= 1, kappa/K, n_th, K (absolute times), kappa2/K, the leakage-removing kappa2 step (LRU) on or off,
    ancilla leakage on or off. Unset knobs take the DA21 design point of presets/kerrcat.yaml (nbar 6.25, kappa/K
    1.67e-4, n_th 8%, K/2pi 10 MHz, kappa2 = K/10)."""
    P = _preset("kerrcat")
    kappa_over_K = _v(P["design_point"]["kappa_over_K"]) if kappa_over_K is None else kappa_over_K
    n_th = _v(P["noise"]["n_th"]) if n_th is None else n_th
    K_MHz = _v(P["design_point"]["K_MHz"]) if K_MHz is None else K_MHz
    kappa2_over_K = _v(P["noise"]["kappa2_over_K"]) if kappa2_over_K is None else kappa2_over_K
    n_lo = min(_da21_tables())
    if nbar < n_lo:
        raise ValueError(f"kerrcat_model: the DA21 operations are published for |alpha|^2 = 1 and 6.25; nbar < {n_lo:g} is not covered")
    n_hi = max(_da21_tables())
    paulis = tuple(_v(P["cx"]["bitflip_paulis"]))
    split = _v(P["cx"]["Z_split"])
    K = 2 * np.pi * K_MHz                            # rad/us
    kappa = kappa_over_K * K
    T_cx = _v(P["cx"]["time_law"]) / (nbar * K)
    T_p, T_m = _da21(nbar, kappa_over_K, "T_p") / K, _da21(nbar, kappa_over_K, "T_m") / K
    T_round = T_p + 2 * T_cx + T_m
    thermal = (1 + 2 * n_th) / (1 + 2 * _v(P["noise"]["n_th"]))
    z_cx = _da21(nbar, kappa_over_K, "z") * thermal  # Z part of the CX (loss and gain)
    nonz = _da21(nbar, kappa_over_K, "nonz")
    if nbar > n_hi:
        nonz *= n_hi / nbar                          # heating-limited beyond Table II: kappa n_th T_CX ~ 1/nbar
    cx = {"ZI": z_cx * split[0], "IZ": z_cx * split[1], "ZZ": z_cx * split[2], **{k: nonz / len(paulis) for k in paulis}}
    z = lambda T: kappa * (1 + 2 * n_th) * nbar * T  # idle phase flips from loss and gain (PU20 Table I, thermal bath)
    TX = 1.0 / (kappa * n_th)                        # heating-limited plateau T_X = (kappa n_th)^-1 (Frattini et al.)
    idle_x = lambda T: 0.5 * (1 - np.exp(-T / TX))
    p_data = _boundary_idle(np.tile([idle_x(T_p + T_m), 0.0, z(T_p + T_m)], (d, 1)), [idle_x(T_cx), 0.0, z(T_cx)])
    p_prep, p_meas = _da21(nbar, kappa_over_K, "prep"), _da21(nbar, kappa_over_K, "meas")
    p_leak = kappa * n_th * T_round                  # single-photon gain excites the first excited manifold (PU20 p. 7)
    # return by loss kappa (1 + n_th) over the round and, with the LRU, by two-photon loss 4 kappa2 |a|^2 during the
    # kappa2 half of each CX (T_CX / 2 per gate, two gates per round)
    ret = kappa * (1 + n_th) * T_round + (4 * kappa2_over_K * K * nbar * T_cx if lru else 0.0)
    p_return = 1 - np.exp(-ret)
    leak = Leakage(p_leak=p_leak, p_return=p_return, partner={"Z": 0.5}, dephase_self=True,
                   randomize_measurement=False, return_paulis="Z")
    leak_a = (Leakage(p_leak=p_leak, p_return=p_return, randomize_measurement=True, return_paulis="Z")
              if ancilla_leakage else Leakage())
    prov = {
        "preset": "presets/kerrcat.yaml (the DA21 tables and every other literature number with its source)",
        "cycle": f"DA21 Tables I-II operation times (units of 1/K): T_P 16.5 / 2.6, T_M = T_P + T_CD 58.5 / 5.1 at |a|^2 = 1 / 6.25 (log-linear in |a|^2 between, 6.25 values beyond), T_CX = 115/|a|^2 (112.6 / 18.47); K/2pi = {K_MHz} MHz -> round {T_round * 1e3:.0f} ns",
        "cx_channel[Z]": "DA21 Tables I-II: CX infidelity x bias/(1 + bias) (power law in kappa/K, log-linear in |a|^2 between 1 and 6.25, constant beyond since T_CX ~ 1/|a|^2), times (1 + 2 n_th)/(1 + 2 x 0.08) for other thermal populations; split Z_aI : IZ_d : Z_aZ_d = 1 : 1/2 : 1/2 (DA21 generic bias-preserving CX, pp. 8-9). Our loss-and-gain estimate 2 kappa (1 + 2 n_th) |a|^2 T_CX agrees within 5% at |a|^2 = 6.25 and 5-12% at 1",
        "cx_channel[non-Z]": "DA21 Tables I-II: CX infidelity/(1 + bias), split equally over the 12 bit-flip-type Paulis (no published split); beyond |a|^2 = 6.25 heating-limited, scaled by 6.25/|a|^2 (our estimate kappa n_th T_CX/2 = 1.2e-4 accounts for 84% of the DA21 value 1.5e-4 at the design point)",
        "p_data[Z]": "kappa (1 + 2 n_th) |a|^2 per unit time during T_P + T_M, and during the other CX layer for the two boundary cats (PU20 Table I, thermal bath)",
        "p_data[X]": "heating-limited bit-flip plateau T_X = (kappa n_th)^-1 over the same idle times (Frattini et al., PRX 14, 031040 (2024), plateau model; measured T_X ~ 0.6-1.2 ms: Hajr et al., PRX 14, 041049 (2024); Qing et al., PNAS 123, e2520479123 (2026))",
        "p_prep/p_meas": "DA21 Tables I-II preparation (Z + Y) and measurement infidelity (same interpolation as the CX); at the design point 0.003 and 0.008",
        "p_data_meas_z": "well (Z) readout: single-shot infidelity 0.46% (Frattini et al., PRX 14, 031040 (2024))",
        "leak_data": "single-photon gain excites the first excited manifold at kappa n_th (PU20 p. 7); per round kappa n_th T_round (DA21: 1e-4 - 1e-3 per qubit and round); return by loss kappa (1 + n_th) and, with the LRU, by two-photon loss 4 kappa2 |a|^2 during the kappa2 half of each CX (DA21 two-step CX, kappa2 = K/10); return by two-photon loss gives a phase flip (PU20), so the return re-randomizes Z only; while leaked the neighbouring syndromes are random (incoherent approximation, Marshall & Kafri 2025)",
        "leak_anc": "ancillas are Kerr cats too: the same rates; a leaked ancilla gives a random outcome; its effect on the CX partners is not modelled",
        "note": "DA21 publish only infidelity and average bias, and neglect the residual leakage in their code simulations; current experiments (K/2pi = 1.2 MHz, T1 ~ 40 us, n_th 4-5%: Hajr et al., Qing et al.) are far above the repetition-code threshold with this CX",
        "validation": "reproduces the DA21 CX, preparation and measurement numbers by construction; compared with the DA21 exact n = 9 repetition-code simulation in examples/literature_checks_models.py",
    }
    return RoundNoise(p_data=p_data, f_mid=0.0, cx_channel=cx, p_meas=p_meas, p_prep=p_prep, p_data_meas=p_meas,
                      p_data_meas_z=_v(P["readout"]["well_infidelity"]), leak_data=leak, leak_anc=leak_a, provenance=prov,
                      name=f"kerrcat(nbar={nbar:g}, kappa/K={kappa_over_K:g}, n_th={n_th:g}, K={K_MHz:g} MHz, LRU={lru})")
