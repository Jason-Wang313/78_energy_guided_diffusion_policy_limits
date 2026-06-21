# Submission Version Log

## v1 - Generated Draft

- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening

- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed synthetic metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive

- Applied the stricter ICLR-main-conference standard.
- Determined that missing real-robot/high-fidelity evidence and template-generated experiments were not recoverable by paper polishing.
- Terminal decision: KILL_ARCHIVE.

## v4 - Trajectory Diffusion-Limit Rebuild

- Added `docs/paper78_rebuild_plan.md` before execution.
- Replaced the synthetic probability-table script with a data-driven trajectory-diffusion support-gap benchmark.
- Added kernel reverse diffusion, energy reranking, guided diffusion, safety-projected guidance, CEM, and oracle baselines.
- Generated 980 main rollout rows, 392 ablation rows, and 420 stress rows.
- Rewrote docs and manuscript around the negative result.
- Terminal decision: KILL_ARCHIVE.

## 2026-06-15 Continuation Audit

- Rechecked code, CSV, ablation, stress, BibTeX/PDF, artifact-location, public-GitHub, and stale-documentation gates.
- Rebuilt the PDF after adding bibliography authors and replacing fragile `[h]` float specifiers.
- Confirmed the negative result: energy guidance equals reranking at 0.000 off-support success, while CEM and oracle solve the task.
- Terminal decision remains: KILL_ARCHIVE.

## v5 - Expanded Support-Gap Submission-Hardening

- Added `docs/paper78_expanded_submission_plan_20260621.md` before full execution.
- Expanded to eight seeds, five support-gap splits, thirteen methods, 48 demonstrations per task, and a 31-step horizon.
- Added `support_aware_energy_bridge_v5`, mode-diverse diffusion, diffusion-CEM hybrid, CHOMP-like optimization, CEM, graph search, and oracle comparisons.
- Added hard-regime aggregate seed metrics and pairwise statistics.
- Added ten ablations, stress levels from 0.00 through 1.40, and fixed-risk budgets 0.00, 0.02, 0.05, and 0.10.
- Generated 6,240 main rollouts, 480 support diagnostics, 520 seed metrics, 1,600 ablation rollouts, 5,120 stress rollouts, 2,048 fixed-risk rollouts, and 12 curated negative cases.
- Rebuilt the manuscript into a 54-page ICLR-style PDF with bright citation boxes and appendices generated only from frozen CSV artifacts.
- `python scripts\validate_submission_artifacts.py` passed with SHA256 `2FD1529E9EB44BFC3BB2FC8B18FFA2ECD8CD61D06A47D746B53E56D388D64F91`.
- Terminal decision remains: KILL_ARCHIVE.
