# 78 Energy-Guided Diffusion Policy Limits

Submission-hardening version: v5 expanded support-gap rebuild

Terminal decision: KILL_ARCHIVE for ICLR main conference.

Paper 78 was rebuilt into a larger hostile-review diagnostic study of whether energy guidance during reverse diffusion can safely create behavior outside a diffusion-policy prior. The v5 rebuild adds five support-gap splits, eight seeds, thirteen methods, strong trajectory-optimization and graph-search baselines, aggregate hard-regime statistics, component ablations, stress sweeps, fixed-risk budgets, curated negative cases, and a 54-page ICLR-style manuscript generated from frozen CSV artifacts.

The evidence is useful but not submission-ready as an ICLR-main claim. The reference `support_aware_energy_bridge_v5` solves the decisive off-support corridor at `1.000 +/- 0.000`, but so do `mode_diverse_diffusion`, `diffusion_cem_hybrid`, `chomp_like_optimizer`, `cem_trajectory_optimizer`, `graph_search_planner`, and the oracle. The bridge therefore does not establish a method-specific advantage over hostile baselines. Worse for the claim, ablations such as `no_support_awareness`, `no_energy_gradient`, `no_prior_score`, and `optimizer_handoff` also reach `1.000 +/- 0.000`, so the named mechanism is not necessary under the frozen local protocol. The honest decision remains archive/kill rather than polishing the result into a false acceptance story.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

This writes:

- `results/rollouts.csv` with 6,240 main rollout rows.
- `results/training_summary.csv` with 480 support-diagnostic rows.
- `results/raw_seed_metrics.csv` with 520 seed-level metric rows.
- `results/metrics.csv` and `results/pairwise_stats.csv`.
- `results/aggregate_seed_metrics.csv`, `results/aggregate_metrics.csv`, and `results/aggregate_pairwise_stats.csv`.
- `results/ablation_rollouts.csv`, `results/ablation_seed_metrics.csv`, and `results/ablation_metrics.csv`.
- `results/stress_sweep_raw.csv`, `results/stress_sweep.csv`, and `figures/stress_curve_data.csv`.
- `results/fixed_risk_raw.csv`, `results/fixed_risk_seed_metrics.csv`, `results/fixed_risk_metrics.csv`, and `results/fixed_risk_pairwise.csv`.
- `results/negative_cases.csv`.
- Figures under `figures/`.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/78.pdf`

Final PDF audit: 54 pages, SHA256 `2FD1529E9EB44BFC3BB2FC8B18FFA2ECD8CD61D06A47D746B53E56D388D64F91`.

No visible Desktop PDF is required or produced.

## Validate Artifacts

```powershell
python scripts\validate_submission_artifacts.py
```

The validator checks frozen row counts, LaTeX hard-warning patterns, bright citation-box configuration, the `Downloads/78.pdf` page count, and Desktop exclusion.
