# ICLR Main Gate

Paper: 78 energy_guided_diffusion_policy_limits

Submission-hardening version: v4

Gate verdict: KILL_ARCHIVE

Fatal evidence:

- `off_support_corridor`: `energy_guided_diffusion` success 0.000 +/- 0.000.
- `off_support_corridor`: `energy_rerank_unguided` success 0.000 +/- 0.000.
- `off_support_corridor`: `cem_trajectory_optimizer` success 1.000 +/- 0.000.
- Paired guided-minus-rerank success difference: 0.00000.
- Paired guided-minus-CEM success difference: -1.00000.
- Strong projection avoids collision but still has 0.000 success and does not escape the prior homotopy.
- No-prior guidance escapes the prior in 0.607 of off-support ablation cases but succeeds in 0.000 and collides in 0.929.

The only honest main-conference-safe decision is to archive rather than overclaim.
