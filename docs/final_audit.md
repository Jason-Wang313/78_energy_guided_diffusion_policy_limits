# Final Audit

1. Chosen thesis: energy guidance should change reachable trajectory behavior rather than merely re-rank samples from a diffusion prior.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Evidence added: a data-driven kernel reverse-diffusion trajectory sampler, energy reranking, guided diffusion, safety-projected guidance, CEM optimization, and grid oracle.
5. Main evidence rows: 980 main rollouts, 245 seed-level metric rows, 392 ablation rollouts, 420 stress-sweep rollouts.
6. Decisive result: on `off_support_corridor`, `energy_guided_diffusion` success is 0.000 +/- 0.000, `energy_rerank_unguided` is 0.000 +/- 0.000, and `cem_trajectory_optimizer` is 1.000 +/- 0.000.
7. Paired result: guided-minus-rerank success difference is 0.00000; guided-minus-CEM success difference is -1.00000.
8. Ablation result: guidance without prior has 0.607 mode escape but 0.000 success and 0.929 collision, showing unsafe off-support escape rather than useful behavior.
9. Reproducibility: `python src/run_experiment.py` regenerates CSVs and figures.
10. Exact Downloads PDF path: `C:/Users/wangz/Downloads/78.pdf`
11. GitHub URL: https://github.com/Jason-Wang313/78_energy_guided_diffusion_policy_limits
12. Confirmation: no visible Desktop copy was requested or made.

## Continuation Audit 2026-06-15

Rechecked gates:

- `python -m py_compile src/run_experiment.py` passed.
- CSV integrity passed with the expected evidence scale: 980 main rollout rows, 245 seed-level metric rows, 210 aggregate metric rows, 100 pairwise rows, 392 ablation rollout rows, 14 ablation aggregate rows, 420 stress-sweep raw rows, 30 stress-sweep aggregate rows, 140 training-summary rows, and 9 negative cases.
- Required baselines were present: `behavior_clone_nearest`, `unguided_diffusion_prior`, `energy_rerank_unguided`, `energy_guided_diffusion`, `strong_guided_diffusion`, `cem_trajectory_optimizer`, and `grid_oracle`.
- LaTeX/BibTeX rebuilt a 4-page PDF after repairing missing bibliography authors and fragile float placement warnings.
- `C:/Users/wangz/Downloads/78.pdf` SHA256 is `3AEDE33B8D93C4D1F1C280AB86BB39595E2BA6EB0B92DFA30FAE1E82E327ACF0`.
- `C:/Users/wangz/Desktop/78.pdf` does not exist.

The decisive negative result was reproduced. On `off_support_corridor`, `energy_guided_diffusion` scores 0.000 +/- 0.000 success with 1.000 collision, exactly matching `energy_rerank_unguided` at 0.000 +/- 0.000 success. `cem_trajectory_optimizer` and `grid_oracle` both score 1.000 +/- 0.000 success with 0.000 collision.

The paired tests remain terminal:

- Guided minus rerank success difference: 0.000 +/- 0.000 with 0/7 better seeds.
- Guided minus CEM success difference: -1.000 +/- 0.000 with 0/7 better seeds.

Ablations confirm the mechanism failure. `full_energy_guided`, `rerank_only`, `no_task_gradient`, `no_safety_projection`, `strong_projection`, and `overguidance` all have 0.000 off-support success. `no_prior_score` reaches 0.607 mode escape but still has 0.000 success and 0.929 collision, so leaving support without a safe optimizer is not a solution.

Stress evidence is unfavorable at the hardest support gap. At stress level 1.0, `energy_guided_diffusion`, `energy_rerank_unguided`, and `strong_guided_diffusion` all have 0.000 success, while CEM and oracle remain at 1.000.

Continuation decision: keep `KILL_ARCHIVE`. Revival would require a diffusion method that safely generates absent homotopies and competes with trajectory optimization, not only reranks or perturbs prior-supported samples.
