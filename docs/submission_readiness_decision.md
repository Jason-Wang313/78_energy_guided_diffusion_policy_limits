# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v4 adds real local trajectory-diffusion evidence, but the evidence refutes the central claim. Energy-guided reverse diffusion does not create safe off-support behavior beyond reranking. On the decisive off-support corridor split, guided diffusion and reranking both achieve 0.000 success, while CEM and the oracle reach 1.000.

Additional blockers:

- No real robot validation.
- No recognized high-fidelity robotics benchmark.
- The diffusion model is a local kernel denoising sampler rather than a trained large-scale policy checkpoint.
- The result is useful as a diagnostic limit study, but not enough for ICLR main submission.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: rebuild as a real empirical robotics paper on external robot-policy benchmarks and show that the mechanism changes reachable behavior safely under support gaps.
