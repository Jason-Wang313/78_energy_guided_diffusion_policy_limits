# 78 Energy-Guided Diffusion Policy Limits

Submission-hardening version: v4

Terminal decision: KILL_ARCHIVE for ICLR main conference.

Paper 78 was rebuilt from a v3 synthetic archive into a concrete trajectory-diffusion limit study. The rebuild implements a data-driven kernel reverse-diffusion trajectory sampler, energy reranking, energy guidance during denoising, a stronger safety-projected guided sampler, CEM trajectory optimization, and a grid-oracle upper bound on a planar obstacle-navigation benchmark with support gaps.

The evidence is negative. On the decisive `off_support_corridor` split, the safe upper homotopy is absent from the diffusion prior while the demonstrated lower homotopy is blocked. `energy_guided_diffusion` reaches `0.000 +/- 0.000` success, exactly matching `energy_rerank_unguided`, while `cem_trajectory_optimizer` and `grid_oracle` both reach `1.000 +/- 0.000`. The paired success difference is `0.00000` versus reranking and `-1.00000` versus CEM.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

This writes:

- `results/rollouts.csv` with 980 main rollout rows.
- `results/raw_seed_metrics.csv` with 245 seed-level metric rows.
- `results/metrics.csv` with aggregate metrics and confidence intervals.
- `results/pairwise_stats.csv` with paired comparisons.
- `results/ablation_rollouts.csv` and `results/ablation_metrics.csv`.
- `results/stress_sweep_raw.csv` and `results/stress_sweep.csv`.
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

No visible Desktop PDF is required or produced.
