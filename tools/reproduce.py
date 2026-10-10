"""One command per figure and data product.

    python tools/reproduce.py --list                     targets: outputs, runtime, external inputs, fast mode
    python tools/reproduce.py <target> [<target> ...]    regenerate these targets (dependencies are not rerun)
    python tools/reproduce.py --all [--max-minutes M]    every target, in dependency order (optionally only cheap ones)
    python tools/reproduce.py --fast <target> ...        use the target's redraw mode (figure from the stored CSV)
    python tools/reproduce.py --missing                  targets whose external inputs are absent on this machine

A target is a figure or a data product; the registry below names the command that makes it, what it writes, what it
reads that is not produced by another target (the arXiv PDF of the Ocelot paper, the public Zenodo archive, both
local only, see docs/reproduction.md), which targets must run first, and the runtime measured on the development laptop
(4 cores, CPU only). Targets run as subprocesses from the repository root with SOURCE_DATE_EPOCH set (reproducible
PDF metadata) and one BLAS thread each unless OMP_NUM_THREADS, OPENBLAS_NUM_THREADS or MKL_NUM_THREADS is set;
logs go to runs/reproduce/<target>.log.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
OCELOT_PDF = "literature/ocelot_2409.13025.pdf"
COMPANION_PDF = "literature/2409.17556.pdf"
ZENODO = "literature/external_data/zenodo_14257632/data_upload"


@dataclasses.dataclass
class Target:
    name: str
    script: str                    # path relative to the repository root
    args: tuple = ()
    outputs: tuple = ()            # files written (relative paths; figures without extension mean .png and .pdf)
    reads: tuple = ()              # external inputs (not produced by another target)
    after: tuple = ()              # targets whose outputs this one reads
    minutes: float = 1.0           # measured runtime of the full command on the development laptop
    fast: tuple | None = None      # arguments of the redraw mode, if the script has one
    notes: str = ""

    def command(self, fast=False):
        return [sys.executable, self.script, *(self.fast if fast and self.fast is not None else self.args)]


REGISTRY: list[Target] = []


def fig(stem):
    """A figure written as figures/<stem>.png and .pdf."""
    return f"figures/{stem}"


def T(name, script, *args, out=(), reads=(), after=(), minutes=1.0, fast=None, notes=""):
    REGISTRY.append(Target(name, script, tuple(args), tuple(out), tuple(reads), tuple(after), float(minutes),
                           None if fast is None else tuple(fast), notes))


CAL_A4 = "data/calibration/ocelot_A4_fcmin6.569.yaml"
CALS = ("calibration_A1", "calibration_A2", "calibration_A3", "calibration_A4")

# 1. Extraction of published figures (vector paths of the arXiv PDFs; needs pdfminer.six and the local PDFs)
for stem, csv_name in [("fig7", "ocelot_fig7"), ("fig12_gates", "ocelot_fig12_gates"), ("fig16", "ocelot_fig16"),
                       ("fig18b", "ocelot_fig18b"), ("fig18c", "ocelot_fig18c"), ("fig36", "ocelot_fig36"),
                       ("fig37d", "ocelot_fig37d")]:
    T(f"digitized_{stem}", f"tools/extract_ocelot_{stem}.py", OCELOT_PDF, f"data/digitized/{csv_name}.csv",
      out=(f"data/digitized/{csv_name}.csv",), reads=(OCELOT_PDF,), minutes=0.2)
T("digitized_fig4b_fig31", "tools/extract_ocelot_fig4b.py", OCELOT_PDF, "data/digitized",
  out=("data/digitized/ocelot_fig4b.csv", "data/digitized/ocelot_fig31.csv"), reads=(OCELOT_PDF,), minutes=0.2)
T("digitized_fig2c_fig15d", "tools/check_ocelot_flip_time_convention.py", OCELOT_PDF,
  out=("data/digitized/ocelot_fig2c.csv", "data/digitized/ocelot_fig15d_fits.csv"), reads=(OCELOT_PDF,), minutes=0.2,
  notes="the flip-time convention check (paper App. C.2)")
T("digitized_companion_fig7b", "tools/extract_companion_fig7b.py", COMPANION_PDF, "data/digitized",
  out=("data/digitized/companion_fig7b.csv",), reads=(COMPANION_PDF,), minutes=0.2)

# 2. Static calibrations (spectroscopy only)
T("calibration_A2S3", "examples/calibrate_ocelot_A2S3.py",
  out=("data/calibration/ocelot_A2S3_fitA.yaml", "data/calibration/ocelot_A2S3_fitB.yaml"), after=("digitized_fig7",),
  minutes=4)
for k in (1, 2, 3, 4):
    T(f"calibration_A{k}", "examples/calibrate_ocelot_ancilla.py", "--anc", f"A{k}",
      out=(f"data/calibration/ocelot_A{k}_fcmin6.569.yaml",), after=("calibration_A2S3",), minutes=2)
for fc in ("6.25", "6.75"):
    T(f"calibration_A4_fcmin{fc}", "examples/calibrate_ocelot_A4.py", "--fcmin", fc,
      out=(f"data/calibration/ocelot_A4_fcmin{float(fc):.3f}.yaml",), after=("calibration_A2S3",), minutes=2,
      notes="prior scan of the unpublished coupler frequency")

# 3. Spectra: Fig. 7, branch analysis, dressed charge elements, crossing atlas
T("fig7_ocelot_reproduction", "examples/fig7_ocelot.py", out=(fig("fig7_ocelot_reproduction"), "data/validation/fig7_ocelot_model.csv"),
  after=("calibration_A2S3", "digitized_fig7"), minutes=1.1)
for tag, args, after in [("A2S3_flux0.500_fitA_ng0.00", (), "calibration_A2S3"),
                         ("A2S3_flux0.500_fitA_ng0.25", ("--ng", "0.25"), "calibration_A2S3"),
                         ("A2S3_flux0.500_fitA_ng0.50", ("--ng", "0.5"), "calibration_A2S3"),
                         ("A2S3_flux0.500_fitB_ng0.00", ("--fit", "B"), "calibration_A2S3"),
                         ("A4S4_flux0.500_fitA_ng0.00", ("--calib", CAL_A4, "--cell", "S4"), "calibration_A4"),
                         ("A4S5_flux0.500_fitA_ng0.00", ("--calib", CAL_A4, "--cell", "S5"), "calibration_A4")]:
    T(f"branches_{tag}", "examples/branches_ocelot_on.py", *args,
      out=(fig(f"branches_ocelot_{tag}"), f"data/branches/ocelot_{tag}.csv"), after=(after,), minutes=0.5,
      fast=(*args, "--no-convergence"), notes="'fitA' in the A4 names is the unused --fit default")
for tag, args, after in [("A2S3_fitA_flux0.500", (), "calibration_A2S3"),
                         ("A4S4_fitA_flux0.500", ("--calib", CAL_A4, "--cell", "S4"), "calibration_A4"),
                         ("A4S4_fitA_flux0.500_nmax40", ("--calib", CAL_A4, "--cell", "S4", "--nmax", "40", "--ecut",
                                                         "250", "--nq", "12"), "calibration_A4"),
                         ("A4S5_fitA_flux0.500", ("--calib", CAL_A4, "--cell", "S5"), "calibration_A4")]:
    T(f"dressed_fg_{tag}", "examples/dressed_charge_elements_ocelot.py", *args,
      out=(fig(f"dressed_fg_ocelot_{tag}"), f"data/branches/dressed_charge_elements_ocelot_{tag}.csv"), after=(after,),
      minutes=0.3)
T("f9_dressing_vs_data", "examples/compare_f9_dressing.py", out=(fig("f9_dressing_vs_data"),),
  after=("dressed_fg_A4S4_fitA_flux0.500_nmax40", "digitized_fig18b"), minutes=0.2)
T("atlas_preview", "examples/atlas_preview_ocelot.py",
  out=(fig("atlas_preview_ocelot_A2S3_fitA_dflux0.01"), "data/atlas/crossings_ocelot_A2S3_fitA_dflux0.01.csv"),
  after=("calibration_A2S3",), minutes=3.2)
T("atlas_gaps_f_-3_1_2", "examples/refine_gaps_ocelot.py", "--family=-3,1,2", "--branch", "f",
  out=("data/atlas/gaps_ocelot_A2S3_fitA_dflux0.01_f_-3_1_2.csv",), after=("atlas_preview",), minutes=2)
T("atlas_gap_survey_n10", "examples/survey_gaps_ocelot.py", "--n", "10",
  out=("data/atlas/gap_survey_n10_ocelot_A2S3_fitA_dflux0.01.csv",), after=("atlas_preview",), minutes=5)
T("apt_ramp_scan", "examples/apt_ramp_scan_ocelot.py", out=(fig("apt_ramp_scan"), "data/apt/ramp_scan.csv"),
  after=("calibration_A2S3", "calibration_A4"), minutes=2)
T("fig36_beta", "examples/fig36_beta_ocelot.py", out=(fig("fig36_beta_ocelot"),), after=("digitized_fig36",), minutes=0.1)
T("fig18c_coupler_loss", "examples/fig18c_coupler_loss_ocelot.py", out=(fig("fig18c_coupler_loss"),),
  after=("digitized_fig18c", "calibration_A3"), minutes=0.15)
T("spectator_screening", "examples/spectator_screening_ocelot.py", out=("data/validation/spectator_screening.csv",),
  after=CALS, minutes=2)

# 4. TLS
T("tls_ensemble_literature_prior", "examples/tls_ensemble_ocelot.py", "--prior", "literature", "--area-um2", "0.02",
  "0.05", "0.15", "--nreal", "2000", "--tag", "literature_prior",
  out=(fig("tls_ensemble_literature_prior"), "data/tls/ensemble_summary_literature_prior.json"),
  after=("calibration_A2S3", "calibration_A4"), minutes=15)
T("tls_ensemble_sensitivity_broadprior", "examples/tls_ensemble_ocelot.py", "--nreal", "2000", "--tag",
  "sensitivity_broadprior",
  out=(fig("tls_ensemble_sensitivity_broadprior"), "data/tls/ensemble_summary_sensitivity_broadprior.json"),
  after=("calibration_A2S3", "calibration_A4"), minutes=8)
T("tls_density_fig18c", "examples/tls_density_fig18c_ocelot.py", out=("data/tls/density_fig18c.csv",),
  after=CALS + ("digitized_fig18c",), minutes=3,
  notes="TLS density of Ocelot's ancillas from the T1 dips of Fig. 18c (broad prior), paper App. E.4 and Table 6")
T("tls_ensemble_fig18c_density", "examples/tls_ensemble_ocelot.py", "--cells", "A4S4", "A4S5", "--density", "20", "40",
  "80", "--nreal", "2000", "--tag", "fig18c_density",
  out=(fig("tls_ensemble_fig18c_density"), "data/tls/ensemble_summary_fig18c_density.json"),
  after=("calibration_A4", "tls_density_fig18c"), minutes=12,
  notes="F9 and CX f->g statistics at the TLS density inferred from Fig. 18c, paper Table 6")
T("tls_ensemble_fig18c_literature_x3", "examples/tls_ensemble_ocelot.py", "--prior", "literature", "--area-um2", "0.05",
  "0.15", "--surf-T1-us", "33.3", "--cells", "A4S4", "A4S5", "--nreal", "2000", "--tag", "fig18c_literature_x3",
  out=(fig("tls_ensemble_fig18c_literature_x3"), "data/tls/ensemble_summary_fig18c_literature_x3.json"),
  after=("calibration_A4", "tls_density_fig18c"), minutes=9,
  notes="the same with the literature prior at the upper Fig. 18c density (x3 surface), paper Table 6")
T("tls_single_map", "examples/tls_single_map_ocelot.py",
  out=(fig("tls_single_map_A4S4_g0.5MHz_T1tls1us"), "data/tls/single_map_A4S4_g0.5MHz_T1tls1us.csv"),
  after=("calibration_A4",), minutes=2)
T("tls_f9_family", "examples/tls_f9_family_ocelot.py",
  out=(fig("tls_f9_family_A4S4"), "data/tls/f9_family_A4S4.csv", "data/tls/f9_family_peaks_A4S4.csv"),
  after=("calibration_A4", "digitized_fig18b"), minutes=0.55)
T("tls_spectral_diffusion", "examples/tls_spectral_diffusion_ocelot.py",
  out=(fig("tls_spectral_diffusion"), "data/tls/spectral_diffusion_summary.json"), after=("calibration_A4",), minutes=5)
T("stark_tls_t1", "examples/stark_tls_t1_ocelot.py", out=(fig("stark_tls_t1"), "data/tls/stark_tls_summary.json"),
  after=("calibration_A4",), minutes=5)
T("pulse_tls_lz", "examples/pulse_tls_lz_ocelot.py", out=(fig("pulse_tls_lz"), "data/tls/pulse_lz_summary.json"),
  after=("calibration_A4",), minutes=5)

# 5. CX^2 in the dressed ladder (ULE); ladders cached in runs/cx2/
T("cx2_ule", "examples/predict_cx2_ule_ocelot.py",
  out=(fig("cx2_ule_fexcess"), fig("cx2_ule_gexcess"), "data/validation/cx2_ule_excess.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16"), minutes=25, fast=("--recompare",),
  notes="--recompare reuses the stored model columns and redoes the data columns and the figures")
T("cx2_lambda_needed", "examples/plot_cx2_lambda_needed.py", out=(fig("cx2_lambda_needed"),), after=("cx2_ule",),
  minutes=0.1)
T("g_floor", "examples/g_floor_ocelot.py", out=(fig("g_floor"), "data/validation/g_floor.csv"),
  after=("cx2_ule", "digitized_fig16"), minutes=0.12)
T("cx2_ramp_exchange", "examples/ramp_exchange_cx2_ocelot.py", out=("data/validation/cx2_ramp_exchange.csv",),
  after=CALS + ("cx2_ule",), minutes=2)
T("cx2_ge_protocol", "examples/predict_cx2_ge_ocelot.py", out=(fig("cx2_ge_protocol"), "data/validation/cx2_ge_protocol.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16"), minutes=10, fast=("--recompare",))
T("cx2_coupler_hypothesis", "examples/coupler_hypothesis_ocelot.py",
  out=(fig("cx2_coupler_hypothesis"), "data/validation/cx2_coupler_hypothesis.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16", "cx2_ule"), minutes=15, fast=("--plot-only",))
T("cx2_fexcess_flat_top", "examples/predict_cx2_fexcess_ocelot.py",
  out=(fig("cx2_fexcess_prediction"), "data/validation/cx2_fexcess.csv"), after=CALS + ("digitized_fig16",), minutes=2,
  notes="superseded flat-top rate model, kept as a cross-check")
T("cx2_kerr_calibration", "examples/kerr_calibration_ocelot.py", out=("data/validation/cx2_kerr_calibration.csv",),
  after=CALS + ("digitized_fig12_gates",), minutes=5)
for pair in ("A4-S5", "A1-S1"):
    T(f"cx2_convergence_{pair}", "examples/convergence_cx2_ule.py", "--pair", pair, after=CALS, minutes=10,
      notes="convergence check, stdout only (paper App. C.1)")

# 6. Syndrome cycle, public data, logical errors
T("stabilizer_faults", "examples/stabilizer_faults_ocelot.py",
  out=(fig("stabilizer_faults"), "data/validation/stabilizer_faults.csv"), after=CALS + ("digitized_fig12_gates",),
  minutes=10)
T("stabilizer_faults_large_nbar", "examples/stabilizer_faults_ocelot.py", "--nl", "40", "--nmax", "38", "--nbars",
  "1,2,3,4,5,6,8,10,12,14,16", "--variants", "zero-fit,lam3,T50", "--tag", "large_nbar",
  out=(fig("stabilizer_faults_large_nbar"), "data/validation/stabilizer_faults_large_nbar.csv"),
  after=CALS + ("digitized_fig12_gates",), minutes=30)
T("zenodo_erasures", "examples/zenodo_erasure_ocelot.py", out=(fig("zenodo_erasures"), "data/validation/zenodo_erasures.csv"),
  reads=(ZENODO,), after=("stabilizer_faults",), minutes=0.7)
T("zenodo_detection", "examples/zenodo_detection_ocelot.py", out=("data/validation/zenodo_detection.csv",),
  reads=(ZENODO,), after=("stabilizer_faults",), minutes=0.5)
T("syndrome_phase", "examples/syndrome_phase_ocelot.py", out=("data/validation/syndrome_phase.csv",),
  after=CALS + ("digitized_fig12_gates", "zenodo_detection"), minutes=0.15,
  notes="without zenodo_detection.csv the data columns are NaN (no error)")
T("zenodo_logical_bitflip", "examples/zenodo_bitflip_ocelot.py",
  out=(fig("zenodo_logical_bitflip"), "data/validation/zenodo_logical_bitflip.csv"), reads=(ZENODO,),
  after=("digitized_fig4b_fig31",), minutes=0.55)
T("logical_bitflip", "examples/logical_bitflip_ocelot.py", out=(fig("logical_bitflip"), "data/validation/logical_bitflip.csv"),
  after=("stabilizer_faults", "zenodo_logical_bitflip", "digitized_fig16", "digitized_fig4b_fig31"), minutes=0.1,
  notes="without zenodo_logical_bitflip.csv it falls back to the digitized Fig. 4b (no error)")
T("zenodo_phase_logical", "examples/zenodo_phase_decode_ocelot.py",
  out=(fig("zenodo_phase_logical"), "data/validation/zenodo_phase_logical.csv", "data/validation/zenodo_phase_runs.csv"),
  reads=(ZENODO,), after=("zenodo_detection", "zenodo_erasures"), minutes=18)
T("phase_logical_model", "examples/phase_logical_model_ocelot.py",
  out=(fig("phase_logical_model"), "data/validation/phase_logical_model.csv"), reads=(ZENODO,),
  after=("zenodo_detection", "zenodo_erasures", "syndrome_phase", "zenodo_phase_logical"), minutes=25)
T("bias_ceiling", "examples/bias_ceiling_ocelot.py",
  out=(fig("bias_ceiling"), "data/validation/bias_ceiling.csv", "data/validation/bias_ceiling_summary.csv"),
  after=("stabilizer_faults_large_nbar", "digitized_fig16"), minutes=0.2)
T("phase_flip_scaling", "examples/phase_scaling_repcode.py", out=("data/validation/phase_flip_scaling.csv",), minutes=30)
T("logical_projection", "examples/logical_projection_ocelot.py",
  out=(fig("logical_projection"), "data/validation/logical_projection.csv",
       "data/validation/logical_projection_summary.csv"),
  after=("stabilizer_faults_large_nbar", "phase_flip_scaling", "logical_bitflip", "digitized_fig16"), minutes=0.1)
T("fault_conversion", "examples/fault_conversion_ocelot.py",
  out=(fig("fault_conversion"), "data/validation/fault_conversion.csv"), after=CALS + ("digitized_fig12_gates",),
  minutes=15)
T("fault_conversion_better_ancilla", "examples/fault_conversion_ocelot.py", "--variants", "lam0.333",
  "--tag", "better_ancilla",
  out=(fig("fault_conversion_better_ancilla"), "data/validation/fault_conversion_better_ancilla.csv"),
  after=CALS + ("digitized_fig12_gates",), minutes=5,
  notes="the same map with the idle-referenced decay rates divided by three (paper Sec. 4 and Fig. 2: the opened "
        "channel and the double decay as the ancilla improves)")
T("herald_conversion", "examples/herald_conversion_ocelot.py",
  out=(fig("herald_conversion"), "data/validation/herald_conversion.csv"), after=CALS + ("digitized_fig12_gates",),
  minutes=15)

# 7. Stabilization windows, switch-on, companion device (paper App. J)
T("window_pulsed", "examples/window_ocelot.py", out=(fig("window_pulsed"), "data/validation/window_pulsed.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16"), minutes=15)
T("window_duty", "examples/window_ocelot.py", "--duty", out=("data/validation/window_duty.csv",), minutes=10)
T("switch_on_separation", "examples/switch_on_ocelot.py", "--separation",
  out=("data/validation/switch_on_separation.csv",), minutes=10)
T("switch_on_precision_conversion", "examples/switch_on_ocelot.py", "--precision", "--conversion",
  out=("data/validation/switch_on_precision.csv", "data/validation/switch_on_conversion.csv"), minutes=10)
T("switch_on", "examples/switch_on_ocelot.py", out=(fig("switch_on"), "data/validation/switch_on.csv"),
  after=("switch_on_separation",), minutes=10, fast=("--plot",),
  notes="the crosses of the figure come from switch_on_separation.csv; omitted without it (no error)")
T("companion_pulsed_two_mode", "examples/companion_pulsed.py", "--validate",
  out=("data/validation/companion_pulsed_two_mode.csv",), after=("digitized_companion_fig7b",), minutes=10)
T("companion_pulsed", "examples/companion_pulsed.py", out=(fig("companion_pulsed"), "data/validation/companion_pulsed.csv"),
  after=("companion_pulsed_two_mode", "digitized_companion_fig7b"), minutes=10, fast=("--plot",),
  notes="the two-mode points come from companion_pulsed_two_mode.csv; omitted without it (no error)")
T("logical_total", "examples/logical_total_ocelot.py", out=(fig("logical_total"), "data/validation/logical_total.csv"),
  after=("zenodo_logical_bitflip", "zenodo_phase_logical", "phase_logical_model", "logical_bitflip", "window_pulsed"),
  minutes=0.1)

# 8. Reference noise models and syndrome datasets
T("ocelot_rep_noise_table", "tools/build_ocelot_noise_table.py", out=("src/bosonoise/presets/ocelot_rep_noise.yaml",),
  after=("stabilizer_faults_large_nbar", "syndrome_phase", "zenodo_erasures", "digitized_fig16"), minutes=0.2)
T("syndrome_datasets", "examples/noise_model_datasets.py", out=("data/syndromes/summary.csv",),
  after=("ocelot_rep_noise_table",), minutes=2.2, notes="72 datasets under data/syndromes/")
T("literature_checks_models", "examples/literature_checks_models.py",
  out=(fig("literature_checks_models"), "data/validation/literature_checks_models.csv"), minutes=10, fast=("--plot",))

# 9. The opened channel: measured on-gate coherence, the GKP cell, the universal map and law, chi-matching, the F9 fits
T("cx2_measured_bath", "examples/measured_bath_cx2_ocelot.py",
  out=(fig("cx2_measured_bath"), "data/validation/cx2_measured_bath.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16", "digitized_fig18c", "digitized_fig37d"), minutes=15,
  fast=("--plot",),
  notes="the ancilla decay at 'on' from measured coherence (Fig. 37d, Fig. 18c), no fit; paper App. C.2")
T("gkp_ecd_ancilla", "examples/gkp_ecd_ancilla_photons.py",
  out=(fig("gkp_ecd_ancilla"), "data/validation/gkp_ecd_ancilla.csv"), minutes=40, fast=("--plot",),
  notes="Floquet branches and Floquet-Markov rates of the transmon driven by the displaced cavity; paper App. G")
T("universal_map", "examples/universal_map.py", out=(fig("universal_map"), "data/validation/universal_map.csv"),
  after=CALS, minutes=0.5, fast=("--plot",),
  notes="dressing-opened one-photon f -> g branching, Ocelot and GKP; paper Fig. 1(d)")
T("universal_law", "examples/universal_law.py", out=(fig("universal_law"), "data/validation/universal_law.csv"),
  after=CALS, minutes=0.5, fast=("--plot",),
  notes="the opened branching against the dispersive law |chi_ge| alpha / (2 |Delta (Delta + alpha)|); paper Table 5")
T("chi_matching", "examples/chi_matching.py",
  out=("data/validation/chi_matching.csv", "data/validation/chi_matching_scan.csv"), after=CALS, minutes=0.5,
  notes="chi-matching fixes Delta/alpha and the opened cost per gate (transmon, Ocelot and the drive-matched ancillas "
        "of Reinhold et al. 2020 and Rosenblum et al. 2018), and the cost against Delta/alpha; paper Eq. (5), Table 4, "
        "Fig. 1(c)")
T("f9_bayes_channels", "examples/f9_bayes_ocelot.py", "prepare", out=("data/tls/f9_bayes_channels_A4.csv",),
  after=("calibration_A4", "calibration_A4_fcmin6.25", "calibration_A4_fcmin6.75"), minutes=2,
  notes="F9 channels of A4 (S4, S5; three coupler priors): opened charge channel and the cos(phi) channel; "
        "paper App. E.2")
for _h in ("A", "B"):
    T(f"f9_bayes_{_h}", "examples/f9_bayes_ocelot.py", "fit", "--model", _h,
      out=tuple(f"data/tls/f9_bayes_{_h}{s}" for s in (".json", "_marginals.csv", "_curves.csv")),
      after=("f9_bayes_channels", "digitized_fig18b"), minutes=35,
      notes=f"Bayesian one-defect fit of F9, hypothesis {_h} (A: opened channel; B: critical current), with the "
            "evidences of thinned data, paper App. E.2")
    T(f"f9_bayes_sensitivity_{_h}", "examples/f9_bayes_ocelot.py", "sensitivity", "--model", _h,
      out=(f"data/tls/f9_bayes_sensitivity_{_h}.json",), after=("f9_bayes_channels", "digitized_fig18b"), minutes=12,
      notes=f"evidences of hypothesis {_h} under prior, noise and coupler variants (coarse grid), paper App. E.2")
    T(f"f9_bayes_smooth_{_h}", "examples/f9_bayes_ocelot.py", "smooth", "--model", _h,
      out=(f"data/tls/f9_bayes_smooth_{_h}.json", f"data/tls/f9_bayes_smooth_{_h}_curves.csv"),
      after=("f9_bayes_channels", "digitized_fig18b"), minutes=15,
      notes=f"hypothesis {_h} with the smooth opened channel free: its strength s (an indication of c1); "
            "paper App. E.2")
for _h in ("A", "B"):
    T(f"f9_two_defects_{_h}", "examples/f9_two_defects_ocelot.py", "--model", _h,
      out=(f"data/tls/f9_two_defects_{_h}.json",), after=("f9_bayes_channels", "digitized_fig18b"), minutes=30,
      notes="F9: a second defect with the smooth channel at s = 1, against one defect with s free, paper App. E.2")
    T(f"f9_two_defects_{_h}_wide", "examples/f9_two_defects_ocelot.py", "--model", _h, "--wide",
      out=(f"data/tls/f9_two_defects_{_h}_wide.json",), after=("f9_bayes_channels", "digitized_fig18b"), minutes=65,
      notes="the same on a 9 x 9 grid in T1_TLS and sigma_sd with priors widened to 30 us and 10 MHz, paper App. E.2")
T("f9_strength_cx2", "examples/f9_strength_cx2_ocelot.py", out=("data/validation/f9_strength_cx2.csv",),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16", "digitized_fig37d"), minutes=7,
  notes="the f -> g strength favoured by F9 (s = 5-11) against the CX^2 f-excess of the A4 pairs, paper App. E.3")
T("f9_floor", "examples/f9_floor_ocelot.py",
  out=("data/validation/f9_floor.csv", "data/validation/f9_floor_summary.csv"),
  after=CALS + ("digitized_fig12_gates", "digitized_fig16", "digitized_fig18c", "digitized_fig37d"), minutes=12,
  notes="floor, bias ceiling and bias of the cats served by A4 with the f -> g strength favoured by F9, paper App. E.3")
T("f9_bayes", "examples/f9_bayes_ocelot.py", "summary", out=(fig("f9_bayes"), "data/tls/f9_bayes_summary.json"),
  after=("f9_bayes_A", "f9_bayes_B", "f9_bayes_sensitivity_A", "f9_bayes_sensitivity_B", "f9_bayes_smooth_A",
         "f9_bayes_smooth_B"), minutes=0.2,
  notes="Bayes factors A/B (smooth channel fixed and free), the smooth-channel strength, the bands of both storages, "
        "paper App. E.2, Fig. 13")
T("coupler_from_storage", "examples/coupler_from_storage_ocelot.py", out=("data/validation/coupler_from_storage.csv",),
  after=CALS + ("digitized_fig12_gates",), minutes=0.5,
  notes="coupler T1 at 'on' implied by the storage decay during the gates (negative); paper App. C.2")
T("prx_figures", "examples/prx_figures.py",
  out=tuple(fig(f"prx_{n}") for n in ("channel", "rfg", "gkp", "calibration", "branches", "atlas", "cx2_model",
                                      "cx2_measured", "faults", "tls_family", "f9_bayes", "projection", "conversion",
                                      "consequences", "cx2_ratios")),
  after=("digitized_fig7", "digitized_fig18b", "dressed_fg_A4S4_fitA_flux0.500_nmax40", "universal_map", "gkp_ecd_ancilla",
         "fig7_ocelot_reproduction", "branches_A4S4_flux0.500_fitA_ng0.00", "atlas_preview", "cx2_ule", "cx2_measured_bath",
         "f9_strength_cx2", "fault_conversion", "fault_conversion_better_ancilla", "tls_f9_family", "f9_bayes_smooth_A",
         "f9_bayes_smooth_B", "logical_projection", "bias_ceiling", "chi_matching", "universal_law"),
  minutes=0.3,
  notes="the fifteen figures of the paper (Figs. 1-15) at print size, drawn from stored data products only")


# Runtimes measured in the v0.1 reproduction check (2026-09-30, development laptop, default BLAS threading,
# minutes; the largest of repeated runs). They replace the estimates above.
MEASURED_MINUTES = {
    "apt_ramp_scan": 0.9, "atlas_gap_survey_n10": 5.3, "atlas_gaps_f_-3_1_2": 2.8,
    "atlas_preview": 4.4, "bias_ceiling": 0.2, "branches_A2S3_flux0.500_fitA_ng0.00": 0.8,
    "branches_A2S3_flux0.500_fitA_ng0.25": 0.7, "branches_A2S3_flux0.500_fitA_ng0.50": 0.7, "branches_A2S3_flux0.500_fitB_ng0.00": 0.7,
    "branches_A4S4_flux0.500_fitA_ng0.00": 0.6, "branches_A4S5_flux0.500_fitA_ng0.00": 0.7, "calibration_A1": 1.7,
    "calibration_A2": 1.1, "calibration_A2S3": 4.1, "calibration_A3": 1.2,
    "calibration_A4": 0.7, "calibration_A4_fcmin6.25": 1.7, "calibration_A4_fcmin6.75": 1.2,
    "companion_pulsed": 73.9, "companion_pulsed_two_mode": 18.9, "cx2_convergence_A1-S1": 2.1,
    "cx2_convergence_A4-S5": 2.6, "cx2_coupler_hypothesis": 69.0, "cx2_fexcess_flat_top": 0.3,
    "cx2_ge_protocol": 18.1, "cx2_kerr_calibration": 1.0, "cx2_lambda_needed": 0.1,
    "cx2_measured_bath": 7.7, "cx2_ramp_exchange": 6.9, "cx2_ule": 23.0, "digitized_companion_fig7b": 0.1,
    "digitized_fig12_gates": 0.1, "digitized_fig16": 0.1, "digitized_fig18b": 0.1,
    "digitized_fig18c": 0.1, "digitized_fig2c_fig15d": 0.1, "digitized_fig36": 0.1,
    "digitized_fig37d": 0.1, "digitized_fig4b_fig31": 0.1, "digitized_fig7": 0.1,
    "dressed_fg_A2S3_fitA_flux0.500": 0.2, "dressed_fg_A4S4_fitA_flux0.500": 0.3, "dressed_fg_A4S4_fitA_flux0.500_nmax40": 0.4,
    "dressed_fg_A4S5_fitA_flux0.500": 0.2, "f9_dressing_vs_data": 0.2, "fault_conversion": 44.4,
    "fault_conversion_better_ancilla": 6.8,
    "fig18c_coupler_loss": 0.1, "fig36_beta": 0.1, "fig7_ocelot_reproduction": 1.1,
    "g_floor": 0.1, "gkp_ecd_ancilla": 32.0, "herald_conversion": 10.7,
    "literature_checks_models": 3.5, "logical_bitflip": 0.1, "logical_projection": 0.1,
    "logical_total": 0.1, "ocelot_rep_noise_table": 0.1, "phase_flip_scaling": 35.2,
    "phase_logical_model": 33.4, "prx_figures": 0.3, "pulse_tls_lz": 1.2, "spectator_screening": 0.1,
    "stabilizer_faults": 7.0, "stabilizer_faults_large_nbar": 18.4, "stark_tls_t1": 2.5,
    "switch_on": 10.4, "switch_on_precision_conversion": 4.3, "switch_on_separation": 1.8,
    "syndrome_datasets": 2.8, "syndrome_phase": 0.1, "tls_ensemble_literature_prior": 7.8,
    "coupler_from_storage": 0.3, "tls_density_fig18c": 5.0, "tls_ensemble_fig18c_density": 10.6,
    "tls_ensemble_fig18c_literature_x3": 8.5,
    "tls_ensemble_sensitivity_broadprior": 5.1, "tls_f9_family": 0.5, "tls_single_map": 0.6,
    "tls_spectral_diffusion": 26.4, "universal_law": 0.1, "window_duty": 9.4,
    "window_pulsed": 9.3, "zenodo_detection": 0.5, "zenodo_erasures": 0.7,
    "zenodo_logical_bitflip": 0.6, "zenodo_phase_logical": 17.9,
}
for _t in REGISTRY:
    _t.minutes = MEASURED_MINUTES.get(_t.name, _t.minutes)


def _by_name():
    return {t.name: t for t in REGISTRY}


def _order(names):
    """Dependency order (targets of `after` first, when they are in the selection)."""
    reg, seen, out = _by_name(), set(), []

    def visit(n):
        if n in seen:
            return
        seen.add(n)
        for m in reg[n].after:
            if m in names:
                visit(m)
        out.append(n)
    for n in names:
        visit(n)
    return out


def _missing_inputs(t):
    return [r for r in t.reads if not (ROOT / r).exists()]


def run(t: Target, fast=False, timeout_min=None):
    log_dir = ROOT / "runs" / "reproduce"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH="1790812800", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    for var in THREAD_VARS:  # one BLAS thread per target (the laptop is shared); override from the environment
        env.setdefault(var, "1")
    start = time.time()
    with open(log_dir / f"{t.name}.log", "w", encoding="utf-8") as log:
        log.write(" ".join(t.command(fast)) + "\n")
        log.flush()
        try:
            rc = subprocess.run(t.command(fast), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=None if timeout_min is None else 60 * timeout_min).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    minutes = (time.time() - start) / 60
    expected = _expand(t.outputs)
    if fast and t.fast is not None:  # a redraw mode rewrites the figures; the data may be left as they are
        expected = [o for o in expected if o.startswith("figures/")] or expected
    stale = [o for o in expected if not (ROOT / o).exists() or (ROOT / o).stat().st_mtime < start - 1]
    return rc, minutes, stale


def _expand(outputs):
    out = []
    for o in outputs:
        if o.startswith("figures/") and not o.endswith((".png", ".pdf")):  # a stem (names may contain dots)
            out += [o + ".png", o + ".pdf"]
        else:
            out.append(o)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true", help="use the redraw mode where a target has one")
    ap.add_argument("--max-minutes", type=float, default=None, help="with --all: skip targets slower than this")
    ap.add_argument("--timeout", type=float, default=None, help="minutes per target")
    ap.add_argument("--missing", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="print the registry as the table of docs/reproduction.md")
    a = ap.parse_args()
    reg = _by_name()
    if a.list:
        for t in REGISTRY:
            print(f"{t.name:44s} {t.minutes:7.1f} min  {'fast' if t.fast is not None else '    '}  "
                  f"{' '.join(t.command()[1:])}")
        return 0
    if a.markdown:
        print("| target | command | writes | external inputs | runs after | runtime | redraw |")
        print("|---|---|---|---|---|---|---|")
        for t in REGISTRY:
            cmd = "`python " + " ".join([t.script, *t.args]) + "`"
            fast = "`" + " ".join(t.fast) + "`" if t.fast else ("—" if t.fast is None else "(no args)")
            mins = f"{t.minutes:g} min" if t.minutes < 60 else f"{t.minutes / 60:.1f} h"
            outs = [o + ".{png,pdf}" if o.startswith("figures/") and not o.endswith((".png", ".pdf")) else o
                    for o in t.outputs]
            print(f"| {t.name} | {cmd} | {'<br>'.join(outs) or '(stdout)'} | {'<br>'.join(t.reads) or '—'} | "
                  f"{', '.join(t.after) or '—'} | {mins} | {fast} |")
        return 0
    if a.missing:
        for t in REGISTRY:
            if _missing_inputs(t):
                print(t.name, "->", ", ".join(_missing_inputs(t)))
        return 0
    names = [t.name for t in REGISTRY] if a.all else a.targets
    unknown = [n for n in names if n not in reg]
    if unknown or not names:
        ap.error(f"unknown targets {unknown}; see --list" if unknown else "no target given")
    if a.all and a.max_minutes is not None:
        names = [n for n in names if reg[n].minutes <= a.max_minutes or (a.fast and reg[n].fast is not None)]
    failed = []
    for n in _order(names):
        t = reg[n]
        miss = _missing_inputs(t)
        if miss:
            print(f"SKIP {n}: missing external inputs {miss}", flush=True)
            continue
        rc, minutes, stale = run(t, fast=a.fast, timeout_min=a.timeout)
        ok = rc == 0 and not stale
        print(f"{'OK  ' if ok else 'FAIL'} {n}: {minutes:.1f} min" + ("" if rc == 0 else f", exit {rc}")
              + ("" if not stale else f", not written: {stale}"), flush=True)
        if not ok:
            failed.append(n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
