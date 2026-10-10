# Reproducing every figure and data product

`tools/reproduce.py` names one command for every figure in `figures/` and every data product in `data/`. Run it
from the repository root after `pip install -e ".[dev,qec,data]"`.

```bash
python tools/reproduce.py --list                    # all targets, runtime, redraw mode
python tools/reproduce.py fig7_ocelot_reproduction  # one target
python tools/reproduce.py --all --max-minutes 15    # everything that takes at most 15 min
python tools/reproduce.py --fast switch_on          # redraw a figure from its stored CSV
python tools/reproduce.py --missing                 # targets whose external inputs are absent
```

- Each target runs as a subprocess from the repository root with `SOURCE_DATE_EPOCH` set, so the PDF metadata do not
  change between runs. The log goes to `runs/reproduce/<target>.log`.
- A target is reported as failed if the script exits with an error or does not rewrite one of its outputs.
- The dependencies of a target are not rerun automatically. `--all` runs the targets in dependency order.

## External inputs (not in the repository)

Three sets of inputs are not versioned: the papers (copyright), the public data archive (its own licence) and the
simulation caches. Targets that need a missing input are skipped with a message.

| input | used by | how to get it |
|---|---|---|
| `literature/ocelot_2409.13025.pdf` | the extraction tools `tools/extract_ocelot_*.py` (digitized figures of the Ocelot paper) | arXiv:2409.13025v2 (Putterman et al., Nature 638, 927 (2025)), https://arxiv.org/pdf/2409.13025v2 |
| `literature/2409.17556.pdf` | `tools/extract_companion_fig7b.py` | arXiv:2409.17556v2 (Putterman et al., PRX 15, 011070 (2025)) |
| `literature/external_data/zenodo_14257632/data_upload/` | `examples/zenodo_*_ocelot.py` | Zenodo 10.5281/zenodo.14257632 (CC BY 4.0): `data_upload.zip`, 145,469,232 bytes, md5 `d4f051ba40bf3d1940f90f9da4e9953c`; unzip into `literature/external_data/zenodo_14257632/` |

The digitized figures (`data/digitized/`) are in the repository. The PDFs are needed only to redo the extraction.

## Caches

- The dressed-ladder CX² simulations cache their ladders in `runs/cx2/` (pickles keyed by a hash of every
  parameter). A first run without the cache takes longer than the runtimes below.
- Delete `runs/cx2/` to force a full recomputation.

## Targets

Runtimes are for the full command on the development laptop (4 cores, CPU only, one core per target).

| target | command | writes | external inputs | runs after | runtime | redraw |
|---|---|---|---|---|---|---|
| digitized_fig7 | `python tools/extract_ocelot_fig7.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig7.csv` | data/digitized/ocelot_fig7.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig12_gates | `python tools/extract_ocelot_fig12_gates.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig12_gates.csv` | data/digitized/ocelot_fig12_gates.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig16 | `python tools/extract_ocelot_fig16.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig16.csv` | data/digitized/ocelot_fig16.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig18b | `python tools/extract_ocelot_fig18b.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig18b.csv` | data/digitized/ocelot_fig18b.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig18c | `python tools/extract_ocelot_fig18c.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig18c.csv` | data/digitized/ocelot_fig18c.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig36 | `python tools/extract_ocelot_fig36.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig36.csv` | data/digitized/ocelot_fig36.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig37d | `python tools/extract_ocelot_fig37d.py literature/ocelot_2409.13025.pdf data/digitized/ocelot_fig37d.csv` | data/digitized/ocelot_fig37d.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig4b_fig31 | `python tools/extract_ocelot_fig4b.py literature/ocelot_2409.13025.pdf data/digitized` | data/digitized/ocelot_fig4b.csv<br>data/digitized/ocelot_fig31.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_fig2c_fig15d | `python tools/check_ocelot_flip_time_convention.py literature/ocelot_2409.13025.pdf` | data/digitized/ocelot_fig2c.csv<br>data/digitized/ocelot_fig15d_fits.csv | literature/ocelot_2409.13025.pdf | — | 0.1 min | — |
| digitized_companion_fig7b | `python tools/extract_companion_fig7b.py literature/2409.17556.pdf data/digitized` | data/digitized/companion_fig7b.csv | literature/2409.17556.pdf | — | 0.1 min | — |
| calibration_A2S3 | `python examples/calibrate_ocelot_A2S3.py` | data/calibration/ocelot_A2S3_fitA.yaml<br>data/calibration/ocelot_A2S3_fitB.yaml | — | digitized_fig7 | 4.1 min | — |
| calibration_A1 | `python examples/calibrate_ocelot_ancilla.py --anc A1` | data/calibration/ocelot_A1_fcmin6.569.yaml | — | calibration_A2S3 | 1.7 min | — |
| calibration_A2 | `python examples/calibrate_ocelot_ancilla.py --anc A2` | data/calibration/ocelot_A2_fcmin6.569.yaml | — | calibration_A2S3 | 1.1 min | — |
| calibration_A3 | `python examples/calibrate_ocelot_ancilla.py --anc A3` | data/calibration/ocelot_A3_fcmin6.569.yaml | — | calibration_A2S3 | 1.2 min | — |
| calibration_A4 | `python examples/calibrate_ocelot_ancilla.py --anc A4` | data/calibration/ocelot_A4_fcmin6.569.yaml | — | calibration_A2S3 | 0.7 min | — |
| calibration_A4_fcmin6.25 | `python examples/calibrate_ocelot_A4.py --fcmin 6.25` | data/calibration/ocelot_A4_fcmin6.250.yaml | — | calibration_A2S3 | 1.7 min | — |
| calibration_A4_fcmin6.75 | `python examples/calibrate_ocelot_A4.py --fcmin 6.75` | data/calibration/ocelot_A4_fcmin6.750.yaml | — | calibration_A2S3 | 1.2 min | — |
| fig7_ocelot_reproduction | `python examples/fig7_ocelot.py` | figures/fig7_ocelot_reproduction.{png,pdf}<br>data/validation/fig7_ocelot_model.csv | — | calibration_A2S3, digitized_fig7 | 1.1 min | — |
| branches_A2S3_flux0.500_fitA_ng0.00 | `python examples/branches_ocelot_on.py` | figures/branches_ocelot_A2S3_flux0.500_fitA_ng0.00.{png,pdf}<br>data/branches/ocelot_A2S3_flux0.500_fitA_ng0.00.csv | — | calibration_A2S3 | 0.8 min | `--no-convergence` |
| branches_A2S3_flux0.500_fitA_ng0.25 | `python examples/branches_ocelot_on.py --ng 0.25` | figures/branches_ocelot_A2S3_flux0.500_fitA_ng0.25.{png,pdf}<br>data/branches/ocelot_A2S3_flux0.500_fitA_ng0.25.csv | — | calibration_A2S3 | 0.7 min | `--ng 0.25 --no-convergence` |
| branches_A2S3_flux0.500_fitA_ng0.50 | `python examples/branches_ocelot_on.py --ng 0.5` | figures/branches_ocelot_A2S3_flux0.500_fitA_ng0.50.{png,pdf}<br>data/branches/ocelot_A2S3_flux0.500_fitA_ng0.50.csv | — | calibration_A2S3 | 0.7 min | `--ng 0.5 --no-convergence` |
| branches_A2S3_flux0.500_fitB_ng0.00 | `python examples/branches_ocelot_on.py --fit B` | figures/branches_ocelot_A2S3_flux0.500_fitB_ng0.00.{png,pdf}<br>data/branches/ocelot_A2S3_flux0.500_fitB_ng0.00.csv | — | calibration_A2S3 | 0.7 min | `--fit B --no-convergence` |
| branches_A4S4_flux0.500_fitA_ng0.00 | `python examples/branches_ocelot_on.py --calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S4` | figures/branches_ocelot_A4S4_flux0.500_fitA_ng0.00.{png,pdf}<br>data/branches/ocelot_A4S4_flux0.500_fitA_ng0.00.csv | — | calibration_A4 | 0.6 min | `--calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S4 --no-convergence` |
| branches_A4S5_flux0.500_fitA_ng0.00 | `python examples/branches_ocelot_on.py --calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S5` | figures/branches_ocelot_A4S5_flux0.500_fitA_ng0.00.{png,pdf}<br>data/branches/ocelot_A4S5_flux0.500_fitA_ng0.00.csv | — | calibration_A4 | 0.7 min | `--calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S5 --no-convergence` |
| dressed_fg_A2S3_fitA_flux0.500 | `python examples/dressed_charge_elements_ocelot.py` | figures/dressed_fg_ocelot_A2S3_fitA_flux0.500.{png,pdf}<br>data/branches/dressed_charge_elements_ocelot_A2S3_fitA_flux0.500.csv | — | calibration_A2S3 | 0.2 min | — |
| dressed_fg_A4S4_fitA_flux0.500 | `python examples/dressed_charge_elements_ocelot.py --calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S4` | figures/dressed_fg_ocelot_A4S4_fitA_flux0.500.{png,pdf}<br>data/branches/dressed_charge_elements_ocelot_A4S4_fitA_flux0.500.csv | — | calibration_A4 | 0.3 min | — |
| dressed_fg_A4S4_fitA_flux0.500_nmax40 | `python examples/dressed_charge_elements_ocelot.py --calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S4 --nmax 40 --ecut 250 --nq 12` | figures/dressed_fg_ocelot_A4S4_fitA_flux0.500_nmax40.{png,pdf}<br>data/branches/dressed_charge_elements_ocelot_A4S4_fitA_flux0.500_nmax40.csv | — | calibration_A4 | 0.4 min | — |
| dressed_fg_A4S5_fitA_flux0.500 | `python examples/dressed_charge_elements_ocelot.py --calib data/calibration/ocelot_A4_fcmin6.569.yaml --cell S5` | figures/dressed_fg_ocelot_A4S5_fitA_flux0.500.{png,pdf}<br>data/branches/dressed_charge_elements_ocelot_A4S5_fitA_flux0.500.csv | — | calibration_A4 | 0.2 min | — |
| f9_dressing_vs_data | `python examples/compare_f9_dressing.py` | figures/f9_dressing_vs_data.{png,pdf} | — | dressed_fg_A4S4_fitA_flux0.500_nmax40, digitized_fig18b | 0.2 min | — |
| atlas_preview | `python examples/atlas_preview_ocelot.py` | figures/atlas_preview_ocelot_A2S3_fitA_dflux0.01.{png,pdf}<br>data/atlas/crossings_ocelot_A2S3_fitA_dflux0.01.csv | — | calibration_A2S3 | 4.4 min | — |
| atlas_gaps_f_-3_1_2 | `python examples/refine_gaps_ocelot.py --family=-3,1,2 --branch f` | data/atlas/gaps_ocelot_A2S3_fitA_dflux0.01_f_-3_1_2.csv | — | atlas_preview | 2.8 min | — |
| atlas_gap_survey_n10 | `python examples/survey_gaps_ocelot.py --n 10` | data/atlas/gap_survey_n10_ocelot_A2S3_fitA_dflux0.01.csv | — | atlas_preview | 5.3 min | — |
| apt_ramp_scan | `python examples/apt_ramp_scan_ocelot.py` | figures/apt_ramp_scan.{png,pdf}<br>data/apt/ramp_scan.csv | — | calibration_A2S3, calibration_A4 | 0.9 min | — |
| fig36_beta | `python examples/fig36_beta_ocelot.py` | figures/fig36_beta_ocelot.{png,pdf} | — | digitized_fig36 | 0.1 min | — |
| fig18c_coupler_loss | `python examples/fig18c_coupler_loss_ocelot.py` | figures/fig18c_coupler_loss.{png,pdf} | — | digitized_fig18c, calibration_A3 | 0.1 min | — |
| spectator_screening | `python examples/spectator_screening_ocelot.py` | data/validation/spectator_screening.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 0.1 min | — |
| tls_ensemble_literature_prior | `python examples/tls_ensemble_ocelot.py --prior literature --area-um2 0.02 0.05 0.15 --nreal 2000 --tag literature_prior` | figures/tls_ensemble_literature_prior.{png,pdf}<br>data/tls/ensemble_summary_literature_prior.json | — | calibration_A2S3, calibration_A4 | 7.8 min | — |
| tls_ensemble_sensitivity_broadprior | `python examples/tls_ensemble_ocelot.py --nreal 2000 --tag sensitivity_broadprior` | figures/tls_ensemble_sensitivity_broadprior.{png,pdf}<br>data/tls/ensemble_summary_sensitivity_broadprior.json | — | calibration_A2S3, calibration_A4 | 5.1 min | — |
| tls_density_fig18c | `python examples/tls_density_fig18c_ocelot.py` | data/tls/density_fig18c.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig18c | 5 min | — |
| tls_ensemble_fig18c_density | `python examples/tls_ensemble_ocelot.py --cells A4S4 A4S5 --density 20 40 80 --nreal 2000 --tag fig18c_density` | figures/tls_ensemble_fig18c_density.{png,pdf}<br>data/tls/ensemble_summary_fig18c_density.json | — | calibration_A4, tls_density_fig18c | 10.6 min | — |
| tls_ensemble_fig18c_literature_x3 | `python examples/tls_ensemble_ocelot.py --prior literature --area-um2 0.05 0.15 --surf-T1-us 33.3 --cells A4S4 A4S5 --nreal 2000 --tag fig18c_literature_x3` | figures/tls_ensemble_fig18c_literature_x3.{png,pdf}<br>data/tls/ensemble_summary_fig18c_literature_x3.json | — | calibration_A4, tls_density_fig18c | 8.5 min | — |
| tls_single_map | `python examples/tls_single_map_ocelot.py` | figures/tls_single_map_A4S4_g0.5MHz_T1tls1us.{png,pdf}<br>data/tls/single_map_A4S4_g0.5MHz_T1tls1us.csv | — | calibration_A4 | 0.6 min | — |
| tls_f9_family | `python examples/tls_f9_family_ocelot.py` | figures/tls_f9_family_A4S4.{png,pdf}<br>data/tls/f9_family_A4S4.csv<br>data/tls/f9_family_peaks_A4S4.csv | — | calibration_A4, digitized_fig18b | 0.5 min | — |
| tls_spectral_diffusion | `python examples/tls_spectral_diffusion_ocelot.py` | figures/tls_spectral_diffusion.{png,pdf}<br>data/tls/spectral_diffusion_summary.json | — | calibration_A4 | 26.4 min | — |
| stark_tls_t1 | `python examples/stark_tls_t1_ocelot.py` | figures/stark_tls_t1.{png,pdf}<br>data/tls/stark_tls_summary.json | — | calibration_A4 | 2.5 min | — |
| pulse_tls_lz | `python examples/pulse_tls_lz_ocelot.py` | figures/pulse_tls_lz.{png,pdf}<br>data/tls/pulse_lz_summary.json | — | calibration_A4 | 1.2 min | — |
| cx2_ule | `python examples/predict_cx2_ule_ocelot.py` | figures/cx2_ule_fexcess.{png,pdf}<br>figures/cx2_ule_gexcess.{png,pdf}<br>data/validation/cx2_ule_excess.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16 | 23 min | `--recompare` |
| cx2_lambda_needed | `python examples/plot_cx2_lambda_needed.py` | figures/cx2_lambda_needed.{png,pdf} | — | cx2_ule | 0.1 min | — |
| g_floor | `python examples/g_floor_ocelot.py` | figures/g_floor.{png,pdf}<br>data/validation/g_floor.csv | — | cx2_ule, digitized_fig16 | 0.1 min | — |
| cx2_ramp_exchange | `python examples/ramp_exchange_cx2_ocelot.py` | data/validation/cx2_ramp_exchange.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, cx2_ule | 6.9 min | — |
| cx2_ge_protocol | `python examples/predict_cx2_ge_ocelot.py` | figures/cx2_ge_protocol.{png,pdf}<br>data/validation/cx2_ge_protocol.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16 | 18.1 min | `--recompare` |
| cx2_coupler_hypothesis | `python examples/coupler_hypothesis_ocelot.py` | figures/cx2_coupler_hypothesis.{png,pdf}<br>data/validation/cx2_coupler_hypothesis.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16, cx2_ule | 1.1 h | `--plot-only` |
| cx2_fexcess_flat_top | `python examples/predict_cx2_fexcess_ocelot.py` | figures/cx2_fexcess_prediction.{png,pdf}<br>data/validation/cx2_fexcess.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig16 | 0.3 min | — |
| cx2_kerr_calibration | `python examples/kerr_calibration_ocelot.py` | data/validation/cx2_kerr_calibration.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 1 min | — |
| cx2_convergence_A4-S5 | `python examples/convergence_cx2_ule.py --pair A4-S5` | (stdout) | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 2.6 min | — |
| cx2_convergence_A1-S1 | `python examples/convergence_cx2_ule.py --pair A1-S1` | (stdout) | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 2.1 min | — |
| stabilizer_faults | `python examples/stabilizer_faults_ocelot.py` | figures/stabilizer_faults.{png,pdf}<br>data/validation/stabilizer_faults.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 7 min | — |
| stabilizer_faults_large_nbar | `python examples/stabilizer_faults_ocelot.py --nl 40 --nmax 38 --nbars 1,2,3,4,5,6,8,10,12,14,16 --variants zero-fit,lam3,T50 --tag large_nbar` | figures/stabilizer_faults_large_nbar.{png,pdf}<br>data/validation/stabilizer_faults_large_nbar.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 18.4 min | — |
| zenodo_erasures | `python examples/zenodo_erasure_ocelot.py` | figures/zenodo_erasures.{png,pdf}<br>data/validation/zenodo_erasures.csv | literature/external_data/zenodo_14257632/data_upload | stabilizer_faults | 0.7 min | — |
| zenodo_detection | `python examples/zenodo_detection_ocelot.py` | data/validation/zenodo_detection.csv | literature/external_data/zenodo_14257632/data_upload | stabilizer_faults | 0.5 min | — |
| syndrome_phase | `python examples/syndrome_phase_ocelot.py` | data/validation/syndrome_phase.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, zenodo_detection | 0.1 min | — |
| zenodo_logical_bitflip | `python examples/zenodo_bitflip_ocelot.py` | figures/zenodo_logical_bitflip.{png,pdf}<br>data/validation/zenodo_logical_bitflip.csv | literature/external_data/zenodo_14257632/data_upload | digitized_fig4b_fig31 | 0.6 min | — |
| logical_bitflip | `python examples/logical_bitflip_ocelot.py` | figures/logical_bitflip.{png,pdf}<br>data/validation/logical_bitflip.csv | — | stabilizer_faults, zenodo_logical_bitflip, digitized_fig16, digitized_fig4b_fig31 | 0.1 min | — |
| zenodo_phase_logical | `python examples/zenodo_phase_decode_ocelot.py` | figures/zenodo_phase_logical.{png,pdf}<br>data/validation/zenodo_phase_logical.csv<br>data/validation/zenodo_phase_runs.csv | literature/external_data/zenodo_14257632/data_upload | zenodo_detection, zenodo_erasures | 17.9 min | — |
| phase_logical_model | `python examples/phase_logical_model_ocelot.py` | figures/phase_logical_model.{png,pdf}<br>data/validation/phase_logical_model.csv | literature/external_data/zenodo_14257632/data_upload | zenodo_detection, zenodo_erasures, syndrome_phase, zenodo_phase_logical | 33.4 min | — |
| bias_ceiling | `python examples/bias_ceiling_ocelot.py` | figures/bias_ceiling.{png,pdf}<br>data/validation/bias_ceiling.csv<br>data/validation/bias_ceiling_summary.csv | — | stabilizer_faults_large_nbar, digitized_fig16 | 0.2 min | — |
| phase_flip_scaling | `python examples/phase_scaling_repcode.py` | data/validation/phase_flip_scaling.csv | — | — | 35.2 min | — |
| logical_projection | `python examples/logical_projection_ocelot.py` | figures/logical_projection.{png,pdf}<br>data/validation/logical_projection.csv<br>data/validation/logical_projection_summary.csv | — | stabilizer_faults_large_nbar, phase_flip_scaling, logical_bitflip, digitized_fig16 | 0.1 min | — |
| fault_conversion | `python examples/fault_conversion_ocelot.py` | figures/fault_conversion.{png,pdf}<br>data/validation/fault_conversion.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 44.4 min | — |
| fault_conversion_better_ancilla | `python examples/fault_conversion_ocelot.py --variants lam0.333 --tag better_ancilla` | figures/fault_conversion_better_ancilla.{png,pdf}<br>data/validation/fault_conversion_better_ancilla.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 6.8 min | — |
| herald_conversion | `python examples/herald_conversion_ocelot.py` | figures/herald_conversion.{png,pdf}<br>data/validation/herald_conversion.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 10.7 min | — |
| window_pulsed | `python examples/window_ocelot.py` | figures/window_pulsed.{png,pdf}<br>data/validation/window_pulsed.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16 | 9.3 min | — |
| window_duty | `python examples/window_ocelot.py --duty` | data/validation/window_duty.csv | — | — | 9.4 min | — |
| switch_on_separation | `python examples/switch_on_ocelot.py --separation` | data/validation/switch_on_separation.csv | — | — | 1.8 min | — |
| switch_on_precision_conversion | `python examples/switch_on_ocelot.py --precision --conversion` | data/validation/switch_on_precision.csv<br>data/validation/switch_on_conversion.csv | — | — | 4.3 min | — |
| switch_on | `python examples/switch_on_ocelot.py` | figures/switch_on.{png,pdf}<br>data/validation/switch_on.csv | — | switch_on_separation | 10.4 min | `--plot` |
| companion_pulsed_two_mode | `python examples/companion_pulsed.py --validate` | data/validation/companion_pulsed_two_mode.csv | — | digitized_companion_fig7b | 18.9 min | — |
| companion_pulsed | `python examples/companion_pulsed.py` | figures/companion_pulsed.{png,pdf}<br>data/validation/companion_pulsed.csv | — | companion_pulsed_two_mode, digitized_companion_fig7b | 1.2 h | `--plot` |
| logical_total | `python examples/logical_total_ocelot.py` | figures/logical_total.{png,pdf}<br>data/validation/logical_total.csv | — | zenodo_logical_bitflip, zenodo_phase_logical, phase_logical_model, logical_bitflip, window_pulsed | 0.1 min | — |
| ocelot_rep_noise_table | `python tools/build_ocelot_noise_table.py` | src/bosonoise/presets/ocelot_rep_noise.yaml | — | stabilizer_faults_large_nbar, syndrome_phase, zenodo_erasures, digitized_fig16 | 0.1 min | — |
| syndrome_datasets | `python examples/noise_model_datasets.py` | data/syndromes/summary.csv | — | ocelot_rep_noise_table | 2.8 min | — |
| literature_checks_models | `python examples/literature_checks_models.py` | figures/literature_checks_models.{png,pdf}<br>data/validation/literature_checks_models.csv | — | — | 3.5 min | `--plot` |
| cx2_measured_bath | `python examples/measured_bath_cx2_ocelot.py` | figures/cx2_measured_bath.{png,pdf}<br>data/validation/cx2_measured_bath.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16, digitized_fig18c, digitized_fig37d | 7.7 min | `--plot` |
| gkp_ecd_ancilla | `python examples/gkp_ecd_ancilla_photons.py` | figures/gkp_ecd_ancilla.{png,pdf}<br>data/validation/gkp_ecd_ancilla.csv | — | — | 32 min | `--plot` |
| universal_map | `python examples/universal_map.py` | figures/universal_map.{png,pdf}<br>data/validation/universal_map.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 0.5 min | `--plot` |
| universal_law | `python examples/universal_law.py` | figures/universal_law.{png,pdf}<br>data/validation/universal_law.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 0.1 min | `--plot` |
| chi_matching | `python examples/chi_matching.py` | data/validation/chi_matching.csv<br>data/validation/chi_matching_scan.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4 | 0.5 min | — |
| f9_bayes_channels | `python examples/f9_bayes_ocelot.py prepare` | data/tls/f9_bayes_channels_A4.csv | — | calibration_A4, calibration_A4_fcmin6.25, calibration_A4_fcmin6.75 | 2 min | — |
| f9_bayes_A | `python examples/f9_bayes_ocelot.py fit --model A` | data/tls/f9_bayes_A.json<br>data/tls/f9_bayes_A_marginals.csv<br>data/tls/f9_bayes_A_curves.csv | — | f9_bayes_channels, digitized_fig18b | 35 min | — |
| f9_bayes_sensitivity_A | `python examples/f9_bayes_ocelot.py sensitivity --model A` | data/tls/f9_bayes_sensitivity_A.json | — | f9_bayes_channels, digitized_fig18b | 12 min | — |
| f9_bayes_smooth_A | `python examples/f9_bayes_ocelot.py smooth --model A` | data/tls/f9_bayes_smooth_A.json<br>data/tls/f9_bayes_smooth_A_curves.csv | — | f9_bayes_channels, digitized_fig18b | 15 min | — |
| f9_bayes_B | `python examples/f9_bayes_ocelot.py fit --model B` | data/tls/f9_bayes_B.json<br>data/tls/f9_bayes_B_marginals.csv<br>data/tls/f9_bayes_B_curves.csv | — | f9_bayes_channels, digitized_fig18b | 35 min | — |
| f9_bayes_sensitivity_B | `python examples/f9_bayes_ocelot.py sensitivity --model B` | data/tls/f9_bayes_sensitivity_B.json | — | f9_bayes_channels, digitized_fig18b | 12 min | — |
| f9_bayes_smooth_B | `python examples/f9_bayes_ocelot.py smooth --model B` | data/tls/f9_bayes_smooth_B.json<br>data/tls/f9_bayes_smooth_B_curves.csv | — | f9_bayes_channels, digitized_fig18b | 15 min | — |
| f9_two_defects_A | `python examples/f9_two_defects_ocelot.py --model A` | data/tls/f9_two_defects_A.json | — | f9_bayes_channels, digitized_fig18b | 30 min | — |
| f9_two_defects_A_wide | `python examples/f9_two_defects_ocelot.py --model A --wide` | data/tls/f9_two_defects_A_wide.json | — | f9_bayes_channels, digitized_fig18b | 1.1 h | — |
| f9_two_defects_B | `python examples/f9_two_defects_ocelot.py --model B` | data/tls/f9_two_defects_B.json | — | f9_bayes_channels, digitized_fig18b | 30 min | — |
| f9_two_defects_B_wide | `python examples/f9_two_defects_ocelot.py --model B --wide` | data/tls/f9_two_defects_B_wide.json | — | f9_bayes_channels, digitized_fig18b | 1.1 h | — |
| f9_strength_cx2 | `python examples/f9_strength_cx2_ocelot.py` | data/validation/f9_strength_cx2.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16, digitized_fig37d | 7 min | — |
| f9_floor | `python examples/f9_floor_ocelot.py` | data/validation/f9_floor.csv<br>data/validation/f9_floor_summary.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates, digitized_fig16, digitized_fig18c, digitized_fig37d | 12 min | — |
| f9_bayes | `python examples/f9_bayes_ocelot.py summary` | figures/f9_bayes.{png,pdf}<br>data/tls/f9_bayes_summary.json | — | f9_bayes_A, f9_bayes_B, f9_bayes_sensitivity_A, f9_bayes_sensitivity_B, f9_bayes_smooth_A, f9_bayes_smooth_B | 0.2 min | — |
| coupler_from_storage | `python examples/coupler_from_storage_ocelot.py` | data/validation/coupler_from_storage.csv | — | calibration_A1, calibration_A2, calibration_A3, calibration_A4, digitized_fig12_gates | 0.3 min | — |
| prx_figures | `python examples/prx_figures.py` | figures/prx_channel.{png,pdf}<br>figures/prx_rfg.{png,pdf}<br>figures/prx_gkp.{png,pdf}<br>figures/prx_calibration.{png,pdf}<br>figures/prx_branches.{png,pdf}<br>figures/prx_atlas.{png,pdf}<br>figures/prx_cx2_model.{png,pdf}<br>figures/prx_cx2_measured.{png,pdf}<br>figures/prx_faults.{png,pdf}<br>figures/prx_tls_family.{png,pdf}<br>figures/prx_f9_bayes.{png,pdf}<br>figures/prx_projection.{png,pdf}<br>figures/prx_conversion.{png,pdf}<br>figures/prx_consequences.{png,pdf}<br>figures/prx_cx2_ratios.{png,pdf} | — | digitized_fig7, digitized_fig18b, dressed_fg_A4S4_fitA_flux0.500_nmax40, universal_map, gkp_ecd_ancilla, fig7_ocelot_reproduction, branches_A4S4_flux0.500_fitA_ng0.00, atlas_preview, cx2_ule, cx2_measured_bath, f9_strength_cx2, fault_conversion, fault_conversion_better_ancilla, tls_f9_family, f9_bayes_smooth_A, f9_bayes_smooth_B, logical_projection, bias_ceiling, chi_matching, universal_law | 0.3 min | — |

## Reproduction checks

### Full check (2026-09-30)

All 82 targets of the registry at that time were regenerated on the development laptop with `tools/reproduce.py` (a
few cheap scripts, checked before the registry was complete, by equivalent batch runs of the same command) and compared
with the committed outputs by `tools/compare_to_git.py`: CSV cell by cell (rtol 1e-9, atol 1e-12), YAML and JSON leaf
by leaf ignoring the date, creation and commit fields, PNG byte by byte. Committed PDFs carry creation dates and were
not compared. Summary:

| outcome | targets |
|---|---|
| identical or equal | 60 (all digitized data, calibrations, atlas, TLS, Zenodo, window, cycle and dataset targets, …) |
| data differ at rounding level (≤ 1e-4) | branches A2S3 fitA ng 0.25 / 0.50 and fitB (≤ 8.3e-5, last digit of a small number); herald_conversion 3.5e-6; cx2_coupler_hypothesis 6.1e-6; companion_pulsed 8.1e-6 |
| data differ within the optimizer tolerance | cx2_kerr_calibration 1.5e-4 (Nelder–Mead calibration) |
| data differ in a cancellation | cx2_ule: 2.4% of 3.3e-8 in one derived column (absolute 8e-10, the difference of two ~1e-4 numbers) |
| Monte Carlo (the seed has been deterministic since) | phase_logical_model and logical_total: max 5.7%, rms 1.08σ of the MC error |
| only the figure differs | branches A2S3 fitA ng 0.00 and dressed_fg A4S4/A4S5 (corrected titles; the dressed A4 figures were titled "A2-S3"), literature_checks_models (shared style), cx2_lambda_needed and switch_on (antialiasing only, ≤ 1% intensity) |
| computed for the first time (nothing to compare with) | gkp_ecd_ancilla, universal_map, universal_law, cx2_measured_bath |
| stdout only | cx2_convergence_A4-S5, cx2_convergence_A1-S1 (convergence printouts; paper App. C.1) |

- The PNGs of the four A2S3 branch figures differ in 17% of their pixels because the committed ones predate the shared
  plotting style. The plotted data are the same (CSVs within 1e-4; checked by eye for n_g = 0.25).
- No target failed; one run of cx2_ule was stopped by hand and passed when rerun. The targets added to the registry
  after this check were not part of it.

### Check of the public version (2026-10-09)

The public copy was exported into an empty folder and checked there, with `bosonoise` imported from that copy:

- `pytest`: 95 passed in 45 s.
- 15 targets regenerated with `tools/reproduce.py` (6 min, one core per target; their dependencies were not rerun):
  `prx_figures` (the fifteen figures of the paper); the data behind Figs. 1, 3, 5, 6, 7, 12 and 14 and Tables 4, 5
  and 7 (`universal_law`, `chi_matching`, `universal_map`, `fig7_ocelot_reproduction`,
  `dressed_fg_A4S4_fitA_flux0.500_nmax40`, `branches_A4S4_flux0.500_fitA_ng0.00`, `bias_ceiling`,
  `logical_projection`, `tls_f9_family`); `f9_strength_cx2` (Figs. 4 and 10); `f9_bayes`, `coupler_from_storage`,
  `ocelot_rep_noise_table` and `syndrome_datasets`.
- Result: the fifteen figures of the paper are identical byte for byte (PNG and PDF), as are the arrays and circuits of
  the 72 syndrome datasets and the Ocelot noise table. The dataset metadata agree leaf for leaf apart from the
  generator commit and the creation date. CSV values agree to 1e-12 (last-digit noise in numbers below 1e-6). Three
  diagnostic PDFs that are not in the paper differ by at most 6e-9 pt in vector coordinates; their PNGs are identical.
