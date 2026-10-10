"""Calibration of cell circuit parameters to *static spectroscopy only*.

Rule (paper Sec. 6, App. A): no parameter is ever tuned to error-rate data. Allowed targets are
static quantities: mode frequencies, dispersive shifts chi_ge/chi_gf versus coupler flux,
conditional storage Kerr. Every target carries a `source` string recorded with the fit.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.optimize import least_squares

from ..model.cell import CellParams, StorageCouplerAncilla, Truncation
from .dispersive import static_observables

FIT_FIELDS = ("f_s", "EJ_q", "EC_q", "EJmax_c", "EC_c", "d_c", "g_sc", "g_sq", "g_cq", "K_s_extra")

# physical bounds (GHz; d_c dimensionless)
BOUNDS = {
    "f_s": (4.5, 6.5), "EJ_q": (5.0, 60.0), "EC_q": (0.08, 0.5), "EJmax_c": (10.0, 150.0),
    "EC_c": (0.05, 0.8), "d_c": (0.0, 0.99), "g_sc": (0.0, 0.5), "g_sq": (-0.1, 0.1),
    "g_cq": (0.0, 0.8), "K_s_extra": (-2e-5, 2e-5),
}


@dataclass(frozen=True)
class Target:
    quantity: str  # key of static_observables output
    flux: float
    value: float  # GHz
    sigma: float  # GHz
    source: str


@dataclass
class FitResult:
    params: CellParams
    targets: list
    residuals: np.ndarray  # (model - target)/sigma
    cost: float
    success: bool
    message: str
    free: tuple
    cov: np.ndarray | None = None
    stderr: dict = field(default_factory=dict)

    def table(self) -> str:
        lines = [f"{'quantity':10s} {'flux':>6s} {'target':>12s} {'model':>12s} {'(m-t)/sig':>9s}  source"]
        for t, r in zip(self.targets, self.residuals):
            lines.append(f"{t.quantity:10s} {t.flux:6.3f} {t.value:12.7f} {t.value + r * t.sigma:12.7f} {r:9.2f}  {t.source}")
        return "\n".join(lines)


def targets_from_fig7_csv(path: str, fluxes, sig_f=5e-4, sig_chi=5e-6, sig_K=1e-7) -> list[Target]:
    """Targets from the vector-extracted Fig. 7 of arXiv:2409.13025 (AWS model, A2-S3)."""
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.reader(line for line in fh if not line.startswith("#"))]
    head, data = rows[0], np.array(rows[1:], dtype=float)
    col = {h: data[:, i] for i, h in enumerate(head)}
    x = col["flux_Phi0"]
    spec = [  # (csv column, quantity, scale to GHz, sigma)
        ("omega_s_GHz", "f_s", 1.0, sig_f), ("omega_a_ge_GHz", "f_ge", 1.0, sig_f),
        ("omega_a_ef_GHz", "f_ef", 1.0, sig_f),
        ("chi_ge_MHz", "chi_ge", 1e-3, sig_chi), ("chi_gf_MHz", "chi_gf", 1e-3, sig_chi),
        ("K_s_g_kHz", "K_g", 1e-6, sig_K), ("K_s_e_kHz", "K_e", 1e-6, sig_K), ("K_s_f_kHz", "K_f", 1e-6, sig_K),
    ]
    out = []
    for fl in fluxes:
        for c, q, scale, sig in spec:
            out.append(Target(q, float(fl), float(np.interp(fl, x, col[c])) * scale, sig,
                              f"Ocelot Fig.7 (AWS model, A2-S3) {c}"))
    return out


def _vec_to_params(x, base: CellParams, free) -> CellParams:
    return base.with_(**{k: float(v) for k, v in zip(free, x)})


def residuals(model_factory, params: CellParams, targets) -> np.ndarray:
    model = model_factory(params)
    cache: dict = {}
    r = np.empty(len(targets))
    for i, t in enumerate(targets):
        if t.flux not in cache:
            cache[t.flux] = static_observables(model, t.flux)
        r[i] = (cache[t.flux][t.quantity] - t.value) / t.sigma
    return r


def fit(base: CellParams, targets, trunc: Truncation = Truncation(n_s=7, n_c=5, n_q=6),
        free=FIT_FIELDS, verbose: int = 0, diff_step: float = 1e-4, max_nfev: int = 400) -> FitResult:
    free = tuple(free)
    x0 = np.array([getattr(base, k) for k in free])
    lb = np.array([BOUNDS[k][0] for k in free])
    ub = np.array([BOUNDS[k][1] for k in free])

    def factory(p):
        return StorageCouplerAncilla(p, trunc)

    def fun(x):
        return residuals(factory, _vec_to_params(x, base, free), targets)

    # K_s_extra is ~1e-6 GHz: give it a sensible scale for the finite-difference step
    x_scale = np.array([max(abs(v), 1e-6) for v in x0])
    res = least_squares(fun, x0, bounds=(lb, ub), x_scale=x_scale, diff_step=diff_step,
                        verbose=verbose, max_nfev=max_nfev)
    p = _vec_to_params(res.x, base, free)
    cov, stderr = None, {}
    try:
        J = res.jac
        dof = max(1, len(res.fun) - len(free))
        s2 = 2 * res.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        stderr = {k: float(np.sqrt(max(cov[i, i], 0))) for i, k in enumerate(free)}
    except Exception:  # pragma: no cover
        pass
    return FitResult(params=p, targets=list(targets), residuals=res.fun, cost=float(res.cost),
                     success=bool(res.success), message=str(res.message), free=free, cov=cov, stderr=stderr)


def params_to_yaml_dict(p: CellParams) -> dict:
    return {f.name: float(getattr(p, f.name)) for f in fields(p)}
