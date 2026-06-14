# Paper 78 Rebuild Plan: Energy-Guided Diffusion Policy Limits

## Goal

Rebuild Paper 78 into an honest ICLR-main gate study of the claim:

> Energy guidance changes reachable robot behavior rather than merely re-ranking trajectories already supported by a diffusion policy prior.

The rebuild will not treat prettier text or larger synthetic tables as submission evidence. It must implement a concrete data-driven diffusion-style trajectory sampler, strong baselines, ablations, stress tests, and a terminal decision. If the proposed mechanism does not clearly beat reranking and strong trajectory-optimization baselines, the correct decision is `KILL_ARCHIVE`.

## Benchmark

Implement a local continuous-control trajectory benchmark for a planar robot end-effector navigating from start to goal through obstacle and dynamics constraints. Each task has:

- A 2D start and goal.
- Circular obstacles and narrow passages.
- Velocity and smoothness limits.
- Offline demonstration trajectories that define the support of the diffusion policy prior.
- Hidden support gaps where the feasible test homotopy is absent or rare in the offline data.

The decisive question is whether energy-guided reverse diffusion can escape unsupported demonstration modes without becoming unsafe.

## Splits

Evaluate five splits over seven seeds:

- `supported_single_mode`: the feasible homotopy exists in demonstrations.
- `supported_multimodal_shift`: multiple homotopies exist, with shifted goal/obstacle geometry.
- `rare_mode_recovery`: the correct homotopy is present but very rare.
- `off_support_corridor`: the demonstrated homotopy is blocked at test time and the safe homotopy is absent from the prior.
- `deceptive_energy_basin`: the task energy prefers a shortcut that violates collision or dynamics constraints unless the sampler respects support and safety.

## Methods

Implement all methods against the same task generator:

- `behavior_clone_nearest`: nearest demonstration trajectory for the task goal.
- `unguided_diffusion_prior`: kernel denoising reverse-diffusion sampler over demonstration trajectories.
- `energy_rerank_unguided`: draw many unguided diffusion samples and pick the lowest task energy.
- `energy_guided_diffusion`: add differentiable task-energy gradients during every reverse diffusion step.
- `strong_guided_diffusion`: more samples, guidance clipping, and safety projection.
- `cem_trajectory_optimizer`: model-free cross-entropy trajectory optimizer with the same energy and constraint penalties.
- `grid_oracle`: coarse grid planner upper bound for geometric feasibility.

## Metrics

Primary metrics:

- `success`: reaches the goal, avoids collisions, obeys velocity limit, and has bounded smoothness.
- `collision_rate`.
- `dynamic_violation_rate`.
- `mean_task_energy`.
- `support_distance`: nearest-neighbor distance from the selected trajectory to the offline demonstrations.
- `mode_escape_rate`: selected trajectory uses a homotopy absent or rare in the prior.
- `paired_success_diff_vs_rerank`.
- `paired_success_diff_vs_cem`.

Secondary diagnostics:

- Energy improvement before and after guidance.
- Diversity of sampled homotopies.
- Failure case labels: prior trap, collision shortcut, overguidance, dynamics violation, optimizer failure.

## Ablations

Run ablations on the decisive off-support and rare-mode settings:

- No task gradient, only diffusion prior.
- Rerank only, no gradient guidance.
- Guidance without safety projection.
- Guidance without prior score.
- Guidance weight sweep.
- Sample-count sweep.
- Support-gap sweep where the target homotopy frequency moves from common to absent.

## Submission Gate

The paper may reach only `STRONG_REVISE`, not full ICLR-ready, unless there is real robot or recognized high-fidelity benchmark validation. It must be `KILL_ARCHIVE` if any of the following hold:

- Energy-guided diffusion does not beat `energy_rerank_unguided` on the decisive off-support or rare-mode splits.
- Energy-guided diffusion does not beat or meaningfully match `cem_trajectory_optimizer` while preserving safety.
- Gains come only from increasing sample count, not from gradient guidance.
- The method reaches goals by increasing collisions or dynamics violations.
- Ablations show guidance cannot create safe off-support behavior.

## Deliverables

- Replace `src/run_experiment.py` with the trajectory benchmark and runner.
- Save raw rollout rows, per-seed metrics, pairwise statistics, ablations, stress sweeps, training/support diagnostics, and negative cases under `results/`.
- Save figures under `figures/`.
- Rewrite the README, audit docs, reproducibility checklist, readiness decision, hostile reviewer response, and ICLR gate.
- Rewrite `paper/main.tex` as a real evidence report with the terminal decision.
- Compile `paper/main.pdf` and copy only `C:/Users/wangz/Downloads/78.pdf`.
- Commit and push to the public GitHub repo.
- Update root pool reports after Paper 78 reaches a terminal state.
