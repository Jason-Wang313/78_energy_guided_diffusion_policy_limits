# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v5 adds a much larger local trajectory-diffusion benchmark, but the evidence still does not justify an ICLR-main submission claim. The strongest reference, `support_aware_energy_bridge_v5`, solves the decisive off-support corridor with 1.000 success. However, mode-diverse diffusion, diffusion-CEM, CHOMP-like optimization, CEM, graph search, and the oracle also solve it with 1.000 success. The result is therefore not a convincing method-specific contribution under hostile baselines.

The mechanism claim is also not necessary under ablation: `no_support_awareness`, `no_energy_gradient`, `no_prior_score`, and `optimizer_handoff` all reach 1.000 success. At fixed-risk budget 0.00, the bridge has 0.000 fixed-risk success. The honest conclusion is that the local benchmark exposes a support-gap/optimization phenomenon, but not an ICLR-main-ready diffusion-policy method.

Additional blockers:

- No real robot validation.
- No recognized high-fidelity robotics benchmark.
- The diffusion model is a local kernel denoising sampler rather than a trained large-scale policy checkpoint.
- The reference method ties strong non-diffusion and hybrid optimizers instead of beating them.
- The ablation evidence fails mechanism necessity.
- The result is useful as a diagnostic limit study, but not enough for ICLR main submission.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper on external robot-policy benchmarks and show that the mechanism changes reachable behavior safely under support gaps while beating strong optimizer, graph-search, hybrid diffusion-optimizer, and ablated variants under a frozen protocol.
