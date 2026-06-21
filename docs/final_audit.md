# Final Audit

## Expanded v5 Audit 2026-06-21

1. Chosen thesis: a support-aware energy-guided diffusion bridge should safely create or recover off-prior trajectory behavior under missing support.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v5.
4. Evidence scale: 6,240 main rollouts, 480 training/support diagnostics, 520 seed-level metrics, 455 aggregate metric rows, 60 pairwise rows, 104 aggregate hard-regime seed rows, 91 aggregate metric rows, 12 aggregate pairwise rows, 1,600 ablation rollouts, 160 ablation seed rows, 20 ablation metric rows, 5,120 stress rollouts, 80 stress aggregate rows, 2,048 fixed-risk rollouts, 256 fixed-risk seed rows, 192 fixed-risk metric rows, 28 fixed-risk pairwise rows, and 12 curated negative cases.
5. Decisive off-support result: `support_aware_energy_bridge_v5` reaches 1.000 +/- 0.000 success, but mode-diverse diffusion, diffusion-CEM hybrid, CHOMP-like optimization, CEM, graph search, and oracle also reach 1.000 +/- 0.000.
6. Aggregate hard-regime result: bridge-v5 reaches 1.000 success, tied by the same hostile optimizer/search baselines.
7. Max-stress result: bridge-v5 reaches 1.000 success at stress level 1.40, tied by mode-diverse diffusion, diffusion-CEM, CHOMP-like optimization, CEM, and graph search.
8. Fixed-risk result: at budget 0.00, bridge-v5 has 0.000 fixed-risk success; at budgets 0.02, 0.05, and 0.10 it reaches 1.000 but ties hostile baselines.
9. Ablation result: `no_support_awareness`, `no_energy_gradient`, `no_prior_score`, and `optimizer_handoff` reach 1.000 success, so the named mechanism is not necessary under the local frozen protocol.
10. Reproducibility: `python src\run_experiment.py` regenerates CSVs and figures with the full protocol; `python scripts\generate_manuscript.py` regenerates `paper/main.tex`; the LaTeX/BibTeX sequence rebuilds the PDF.
11. Exact Downloads PDF path: `C:/Users/wangz/Downloads/78.pdf`
12. Downloads PDF SHA256: `2FD1529E9EB44BFC3BB2FC8B18FFA2ECD8CD61D06A47D746B53E56D388D64F91`
13. PDF pages: 54.
14. Validator: `python scripts\validate_submission_artifacts.py` passed.
15. Visual QA: title/citation page, main figure/table pages, fixed-risk page, dense appendix tables, and references page were rendered and inspected; no clipping or broken figures were found.
16. Desktop exclusion: `C:/Users/wangz/Desktop/78.pdf` does not exist.
17. GitHub URL: https://github.com/Jason-Wang313/78_energy_guided_diffusion_policy_limits

Continuation decision: keep `KILL_ARCHIVE`. This is an honest diagnostic archive with real hostile local evidence, not an ICLR-main-ready robotics submission.

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
