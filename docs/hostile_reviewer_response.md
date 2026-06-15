# Hostile Reviewer Response

Paper: 78 Energy-Guided Diffusion Policy Limits

## Strongest Technical Threats

- Energy guidance for diffusion policies and offline reinforcement learning can be interpreted as sample reranking unless it changes support.
- Planning and trajectory-optimization baselines such as CEM can create new feasible homotopies without relying on demonstration support.
- Diffusion-policy robustness papers already study guidance, feasibility filters, tactile feasibility, and behavior discovery.
- A local kernel sampler is not enough to claim a trained neural diffusion-policy advance.

## v4 Response

The hostile reviewer would still reject this as an ICLR main submission, but for a stronger reason than v3. The v4 rebuild now includes implemented evidence. That evidence shows the central mechanism fails on the decisive support-gap split:

- Guided diffusion: 0.000 off-support success.
- Reranking: 0.000 off-support success.
- CEM: 1.000 off-support success.
- Oracle: 1.000 off-support success.

The method does not safely create absent homotopies. It either stays near the blocked prior mode or, without prior score, leaves support unsafely.

## Honest Action

The paper remains `KILL_ARCHIVE`. The repository should be retained as a negative diagnostic, not reframed as an ICLR-main algorithm paper.

## Continuation Response 2026-06-15

The hostile reviewer remains correct after re-audit.

- `energy_guided_diffusion` and `energy_rerank_unguided` both score 0.000 success on the off-support split.
- CEM and oracle both score 1.000 success, so the task is solvable but not by the diffusion-guidance mechanism.
- `strong_guided_diffusion` avoids collision but still has 0.000 success and 0.000 mode escape.
- `no_prior_score` escapes the prior, but unsafely: 0.607 mode escape, 0.000 success, and 0.929 collision.
- At stress level 1.0, guided diffusion remains 0.000 while CEM and oracle are 1.000.

Updated response: keep `KILL_ARCHIVE`.
