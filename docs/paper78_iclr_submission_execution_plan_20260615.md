# Paper 78 ICLR-Main Submission-Readiness Execution Plan

Date: 2026-06-15
Paper: 78 - `energy_guided_diffusion_policy_limits`
Target venue posture: ICLR main only if energy guidance safely changes reachable behavior beyond reranking
Current terminal label entering audit: `KILL_ARCHIVE`

## Goal

Audit Paper 78 as a real submission candidate rather than a polished diffusion-policy negative. The core question is whether energy guidance can create safe off-support trajectories, or whether it merely reranks/support-perturbs samples and fails when the needed homotopy is absent from the prior.

## Decision Rule

Upgrade from `KILL_ARCHIVE` only if all of the following are true:

1. `energy_guided_diffusion` decisively beats `energy_rerank_unguided` on `off_support_corridor`.
2. It reaches nonzero off-support success without unsafe collision shortcuts.
3. It is competitive with stronger trajectory-optimization baselines such as `cem_trajectory_optimizer`.
4. Ablations show guidance, safety projection, and prior score are necessary for safe support escape.
5. Stress-sweep evidence remains favorable at the hardest support-gap setting.
6. The evidence is reproducible from checked-in code, raw CSVs, a clean PDF, and a public GitHub repository.

If any decisive gate fails, preserve `KILL_ARCHIVE` and document the exact failure mode.

## Evidence Gates

Run these checks before changing the decision:

1. Code integrity: compile `src/run_experiment.py`.
2. Result integrity: verify all required CSVs exist, are nonempty, finite, and schema-valid.
3. Scale check: confirm 980 main rollout rows, 245 seed metric rows, 392 ablation rollout rows, and 420 stress-sweep raw rows.
4. Baseline check: confirm `behavior_clone_nearest`, `unguided_diffusion_prior`, `energy_rerank_unguided`, `energy_guided_diffusion`, `strong_guided_diffusion`, `cem_trajectory_optimizer`, and `grid_oracle` are present.
5. Central comparison check: verify `off_support_corridor` success, collision, support distance, and paired success differences.
6. Ablation check: verify whether guidance components create safe success or only unsafe mode escape.
7. Stress check: verify hardest support-gap behavior against rerank, CEM, and oracle.
8. Documentation consistency check: correct stale counts or wording if direct artifacts disagree.
9. Paper build: run LaTeX/BibTeX to produce a clean PDF and copy only `78.pdf` to Downloads.
10. Artifact hygiene: confirm no numbered PDF is copied to the visible Desktop.
11. GitHub hygiene: confirm the matching public GitHub repository exists and the local commit is pushed.
12. Root-report hygiene: update `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md`.

## Expected Risk

The existing summary reports a decisive negative result: on `off_support_corridor`, `energy_guided_diffusion` and `energy_rerank_unguided` both achieve 0.000 success, while `cem_trajectory_optimizer` and `grid_oracle` achieve 1.000. The `no_prior_score` ablation escapes the prior but remains unsafe, with 0.000 success and 0.929 collision. Unless direct verification contradicts that result, Paper 78 remains a reproducible negative-result archive.

## Execution Order

1. Re-check repository cleanliness and result inventory.
2. Run code and CSV integrity gates.
3. Extract central, pairwise, ablation, and stress evidence.
4. Rebuild the paper PDF and repair recoverable build warnings.
5. Update child status and audit docs with exact verified evidence.
6. Update root reports through Paper 78.
7. Commit and push the Paper 78 repository.
8. Verify `Downloads/78.pdf`, no Desktop copy, public GitHub visibility, clean git state, and root report consistency.
