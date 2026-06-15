# Paper 78 Terminal Audit

Date: 2026-06-15 07:45:34 +0100
Paper: 78 - `energy_guided_diffusion_policy_limits`
Decision: `KILL_ARCHIVE`

## Verification Performed

- Compiled `src/run_experiment.py`.
- Verified required CSV artifacts and schemas.
- Confirmed evidence scale: 980 main rollout rows, 245 seed metric rows, 210 aggregate metric rows, 100 pairwise rows, 392 ablation rollout rows, 14 ablation aggregate rows, 420 stress-sweep raw rows, 30 stress-sweep aggregate rows, 140 training-summary rows, and 9 negative cases.
- Confirmed seven seeds: 0 through 6.
- Confirmed required baselines: `behavior_clone_nearest`, `unguided_diffusion_prior`, `energy_rerank_unguided`, `energy_guided_diffusion`, `strong_guided_diffusion`, `cem_trajectory_optimizer`, and `grid_oracle`.
- Rebuilt the LaTeX/BibTeX PDF after fixing bibliography author warnings and fragile float placement warnings.
- Copied only `78.pdf` to Downloads.
- Confirmed no `C:/Users/wangz/Desktop/78.pdf` exists.

## Decisive Evidence

On `off_support_corridor`:

- `behavior_clone_nearest`: 0.000 success, 1.000 collision, 0.000 mode escape.
- `energy_rerank_unguided`: 0.000 success, 1.000 collision, 0.000 mode escape.
- `energy_guided_diffusion`: 0.000 success, 1.000 collision, 0.000 mode escape.
- `strong_guided_diffusion`: 0.000 success, 0.000 collision, 0.000 mode escape.
- `cem_trajectory_optimizer`: 1.000 success, 0.000 collision, 1.000 mode escape.
- `grid_oracle`: 1.000 success, 0.000 collision, 1.000 mode escape.

Paired comparisons:

- Guided minus rerank success: 0.000 +/- 0.000, 0/7 better seeds.
- Guided minus CEM success: -1.000 +/- 0.000, 0/7 better seeds.

Energy guidance does not create the missing safe homotopy. It either stays in the blocked prior-supported mode or leaves support unsafely.

## Ablation Gate

On `off_support_corridor`:

- `full_energy_guided`: 0.000 success, 1.000 collision, 0.000 mode escape.
- `rerank_only`: 0.000 success, 1.000 collision, 0.000 mode escape.
- `strong_projection`: 0.000 success, 0.036 collision, 0.000 mode escape.
- `no_prior_score`: 0.000 success, 0.929 collision, 0.607 mode escape.
- `overguidance`: 0.000 success, 1.000 collision, 0.000 mode escape.

The ablation evidence rejects the submission claim: safe support escape is absent, and unsafe support escape is not useful task behavior.

## Stress Gate

At stress level 1.0, support frequency is 0.000:

- `energy_rerank_unguided`: 0.000 success.
- `energy_guided_diffusion`: 0.000 success.
- `strong_guided_diffusion`: 0.000 success.
- `cem_trajectory_optimizer`: 1.000 success.
- `grid_oracle`: 1.000 success.

The hardest stress condition preserves the negative result.

## Artifact Result

- PDF: `C:/Users/wangz/Downloads/78.pdf`
- SHA256: `3AEDE33B8D93C4D1F1C280AB86BB39595E2BA6EB0B92DFA30FAE1E82E327ACF0`
- Public GitHub: `https://github.com/Jason-Wang313/78_energy_guided_diffusion_policy_limits`

## Final Recommendation

Keep `KILL_ARCHIVE`. A future revival would require a diffusion mechanism that safely generates absent homotopies and competes with trajectory optimization under support gaps.
