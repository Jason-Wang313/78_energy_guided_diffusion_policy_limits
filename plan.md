# Plan

Paper 78 v5 expanded rebuild is complete and terminal.

The paper was rebuilt around a stricter falsifiable claim: a support-aware energy-guided diffusion bridge should change reachable safe trajectory behavior under missing-prior-support regimes and survive optimizer, graph-search, mode-diverse, ablation, stress, and fixed-risk attacks. The frozen v5 evidence does not support an ICLR-main submission claim. Terminal action: `KILL_ARCHIVE`.

## 2026-06-15 Continuation Plan

- Re-run code integrity and result-schema gates without rerunning expensive experiments.
- Verify the full evidence scale: 980 main rollout rows, 245 seed metric rows, 392 ablation rollout rows, 420 stress-sweep raw rows, and 9 negative cases.
- Re-evaluate the decisive `off_support_corridor` comparison against reranking, strong guidance, CEM, and oracle baselines.
- Check collision, support distance, mode escape, and paired success differences.
- Re-check ablations and support-gap stress sweep.
- Rebuild the LaTeX/BibTeX PDF, copy only `78.pdf` to Downloads, and confirm no Desktop PDF exists.
- Update child and root status artifacts, then commit and push the public GitHub repository.

## 2026-06-15 Continuation Result

The continuation audit preserved `KILL_ARCHIVE`. On `off_support_corridor`, `energy_guided_diffusion` and `energy_rerank_unguided` both reach 0.000 +/- 0.000 success, while `cem_trajectory_optimizer` and `grid_oracle` both reach 1.000 +/- 0.000. Paired guided-minus-rerank success difference is 0.000, and guided-minus-CEM is -1.000. The `no_prior_score` ablation reaches 0.607 mode escape but 0.000 success and 0.929 collision, showing unsafe support escape rather than a viable policy.

## 2026-06-21 v5 Expanded Submission-Hardening Plan

- Freeze a CPU-only, RAM-light protocol before execution.
- Use eight seeds, five support-gap splits, thirteen methods, and 48 demonstrations per task.
- Compare reranking, vanilla/strong/projected guidance, mode-diverse diffusion, diffusion-CEM, CHOMP-like optimization, CEM, graph search, the bridge reference, and an oracle.
- Add hard-regime aggregate metrics over rare-mode, off-support, and deceptive-energy splits.
- Add component ablations for bridge proposals, support awareness, safety projection, mode diversity, energy gradient, prior score, reranking only, sample-count-only, and optimizer handoff.
- Add stress sweep levels 0.00 through 1.40.
- Add fixed-risk budgets 0.00, 0.02, 0.05, and 0.10.
- Generate the manuscript only from frozen CSV artifacts with bright citation boxes and no Desktop PDF.
- Validate exact counts, LaTeX hard-warning patterns, PDF page count, and artifact locations before publishing.

## 2026-06-21 v5 Expanded Result

The v5 execution completed cleanly and preserved `KILL_ARCHIVE`. Counts are exact: 6,240 main rollouts, 480 training/support diagnostics, 520 seed metrics, 104 aggregate hard-regime seed rows, 1,600 ablation rollouts, 5,120 stress rollouts, 2,048 fixed-risk rollouts, and 12 curated negative cases.

The reference bridge reaches 1.000 +/- 0.000 on the decisive off-support corridor, but this is not a decisive win: mode-diverse diffusion, diffusion-CEM hybrid, CHOMP-like optimization, CEM, graph search, and the oracle also reach 1.000 +/- 0.000. Hard-regime aggregate and max-stress results show the same tie. Fixed-risk budget 0.00 gives bridge-v5 0.000 fixed-risk success, and ablations show the named mechanism is not necessary because `no_support_awareness`, `no_energy_gradient`, `no_prior_score`, and `optimizer_handoff` also reach 1.000.

PDF result: `C:/Users/wangz/Downloads/78.pdf`, 54 pages, SHA256 `2FD1529E9EB44BFC3BB2FC8B18FFA2ECD8CD61D06A47D746B53E56D388D64F91`. Validator and visual QA passed; no Desktop PDF exists.
