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
