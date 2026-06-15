# Plan

Paper 78 v4 rebuild is complete and terminal.

The paper was rebuilt around the falsifiable claim that energy guidance should change reachable behavior rather than merely rerank samples from a diffusion prior. The implemented benchmark shows the claim fails when the safe homotopy is absent from the prior. Terminal action: `KILL_ARCHIVE`.

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
