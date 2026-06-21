# Paper 78 Expanded Submission Plan

Date: 2026-06-21

Goal: rebuild Paper 78 into a serious 25+ page ICLR-style submission artifact while preserving hostile-review honesty. The current v4 artifact is `KILL_ARCHIVE`: energy-guided diffusion does not safely create the absent off-support homotopy and exactly matches reranking at zero success on the decisive split. The v5 rebuild may only improve if a stronger, still-diffusion-grounded method survives frozen local gates against strong optimizers and graph planners. It remains `KILL_ARCHIVE` if the mechanism fails, if success comes from unsafe support escape, or if the method is merely trajectory optimization wearing diffusion clothes.

## Current State

- Current version: v4.
- Current terminal decision: `KILL_ARCHIVE`.
- Current PDF: 4 pages.
- Current decisive split: `off_support_corridor`.
- Current proposed method: `energy_guided_diffusion`.
- Current proposed success on decisive split: 0.000 +/- 0.000.
- Current rerank baseline success on decisive split: 0.000 +/- 0.000.
- Current CEM and grid-oracle success on decisive split: 1.000 +/- 0.000.
- Current paired success difference versus rerank: 0.000 +/- 0.000.
- Current paired success difference versus CEM: -1.000 +/- 0.000.
- Current failure mode: guidance either stays inside the blocked prior homotopy or escapes support unsafely without producing a feasible trajectory.

## Plan-First Freeze

Before interpreting final results, freeze:

- Seeds: 0 through 7.
- Main splits: `supported_single_mode`, `supported_multimodal_shift`, `rare_mode_recovery`, `off_support_corridor`, `deceptive_energy_basin`.
- Hard aggregate splits: `rare_mode_recovery`, `off_support_corridor`, `deceptive_energy_basin`.
- Main tasks per split and seed: 12.
- Demonstrations per task: 48.
- Diffusion trajectory horizon: 31 waypoints.
- Candidate samples per diffusion method: CPU-light defaults only; no GPU, no neural training loop, no large tensor caches.
- Ablation splits: `rare_mode_recovery` and `off_support_corridor`.
- Ablation tasks per split and seed: 10.
- Stress scenarios per seed: 8.
- Stress levels: 0.00 through 1.40 in eight steps.
- Fixed-risk scenarios per seed: 8.
- Risk budgets: 0.00, 0.02, 0.05, 0.10.
- Canonical v5 reference method: `support_aware_energy_bridge_v5`.

## Planned Evidence Scale

The frozen default run should produce at least:

- Main rollouts: 8 seeds x 5 splits x 12 tasks x 13 methods = 6,240 rows.
- Training/support diagnostics: 8 seeds x 5 splits x 12 tasks = 480 rows.
- Raw seed metrics: 8 seeds x 5 splits x 13 methods = 520 rows.
- Aggregate hard-regime seed metrics: 8 seeds x 13 methods = 104 rows.
- Main aggregate metrics: 455 long-format metric rows.
- Pairwise comparisons against the v5 reference: 60 rows.
- Ablation rollouts: 8 seeds x 2 splits x 10 tasks x 10 variants = 1,600 rows.
- Ablation seed metrics: 8 seeds x 2 splits x 10 variants = 160 rows.
- Stress raw rows: 8 stress levels x 8 seeds x 8 tasks x 10 methods = 5,120 rows.
- Aggregate hard-regime metrics: 91 long-format metric rows.
- Aggregate hard-regime pairwise comparisons: 12 rows.
- Fixed-risk raw rows: 4 budgets x 8 seeds x 8 tasks x 8 methods = 2,048 rows.
- Fixed-risk metric rows: 192 long-format metric rows.
- Curated negative cases: at least 12.

If implementation changes the method count for a substantive reason, update this plan before the full run and before interpreting outcomes.

## Method Upgrade

Add `support_aware_energy_bridge_v5` as the only new proposed reference method.

It may combine:

- prior-score guidance from the kernel diffusion sampler,
- energy gradients with conservative clipping,
- support-distance awareness instead of blind support rejection,
- homotopy bridge proposals from low-resolution waypoint search,
- safety projection after every denoising block,
- velocity and acceleration repair,
- mode-diversity resampling,
- final local smoothing that is still bounded by the same compute budget.

Strict boundary: if the method delegates the hard part entirely to CEM, A*, RRT, or oracle grid search, it must be labeled as a hybrid baseline or optimizer baseline, not as the proposed diffusion method.

## Strong Hostile Baselines

Keep all v4 baselines and add stronger competitors:

- `behavior_clone_nearest`
- `unguided_diffusion_prior`
- `energy_rerank_unguided`
- `energy_guided_diffusion`
- `strong_guided_diffusion`
- `support_projected_guidance`
- `mode_diverse_diffusion`
- `diffusion_cem_hybrid`
- `chomp_like_optimizer`
- `cem_trajectory_optimizer`
- `graph_search_planner`
- `grid_oracle`

The main gate compares v5 against the strongest non-oracle method, with separate reporting against rerank, strong guided diffusion, CEM, and graph/search baselines.

## Evaluation Gates

The paper can only remain alive if all local gates pass:

- v5 beats `energy_rerank_unguided` on `off_support_corridor` success with a positive paired lower bound.
- v5 beats `strong_guided_diffusion` on `off_support_corridor` success with no worse collision or dynamics violations.
- v5 is competitive with the strongest non-oracle trajectory optimizer on `off_support_corridor`; a large gap to CEM/search is terminal unless the paper is explicitly framed as a diagnostic limit rather than a method claim.
- v5 improves aggregate hard-regime success against the strongest non-oracle diffusion baseline and does not lose safety.
- v5 passes every fixed-risk budget without trading collision risk for success.
- v5 wins or is non-inferior at maximum stress.
- v5 full variant beats all component-removal ablations on the decisive split.
- v5 does not win merely by increasing sample count or by using an optimizer oracle outside the diffusion mechanism.

If any gate fails, the terminal decision remains `KILL_ARCHIVE`.

## Theory Additions

The manuscript must add non-trash theory:

- A support-barrier proposition showing why energy gradients cannot reliably create a missing homotopy when the reverse diffusion score points back into the demonstrated mode.
- A safety-support tradeoff lemma: guidance without a support or projection term can increase mode escape while increasing collision/dynamics risk.
- A boundary condition for revival: a diffusion method needs an explicit safe bridge generator or topology proposal to alter reachable homotopy classes.
- A negative-result theorem sketch explaining why reranking, weak gradient guidance, and overguidance fail under the blocked-prior corridor.

These theory sections must be tied directly to measured diagnostics: support distance, mode escape, collision, dynamics violation, and optimizer gap.

## Manuscript And Citation Requirements

- Generate a 25+ page ICLR-style archive manuscript from CSV artifacts only.
- Use bright boxed citation links with `hyperref` so in-text citations visibly route to references.
- Include main tables, aggregate hard-regime tables, ablation tables, stress tables, fixed-risk tables, negative cases, and seed-level appendices.
- Include explicit limitations: local benchmark only, no real robot, no external benchmark, kernel diffusion rather than a trained large-scale policy checkpoint.
- Do not hide negative results behind polished prose.

## Validation Requirements

Before closure:

- `python -m py_compile src/run_experiment.py scripts/generate_manuscript.py scripts/validate_submission_artifacts.py` passes.
- Full frozen protocol runs with `PAPER78_QUICK` unset.
- `python scripts/validate_submission_artifacts.py` checks row counts, summary decision, LaTeX log, page count, bright citation settings, Downloads-only PDF, Desktop exclusion, and SHA256.
- Render the final PDF with Poppler and inspect title/citations, main figures, fixed-risk/stress plots, dense appendix tables, and references.
- Copy only `C:/Users/wangz/Downloads/78.pdf`.
- Do not copy any PDF to the visible Desktop.
- Commit and push the public GitHub repo.
- Update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, `MASTER_SUBMISSION_REPORT.md`, and `SUBMISSION_AUDIT_MATRIX.csv`.
