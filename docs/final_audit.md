# Final Audit

1. Chosen thesis: energy guidance should change reachable trajectory behavior rather than merely re-rank samples from a diffusion prior.
2. ICLR-main decision: KILL_ARCHIVE.
3. Submission-hardening version: v4.
4. Evidence added: a data-driven kernel reverse-diffusion trajectory sampler, energy reranking, guided diffusion, safety-projected guidance, CEM optimization, and grid oracle.
5. Main evidence rows: 980 main rollouts, 245 seed-level metric rows, 392 ablation rollouts, 420 stress-sweep rollouts.
6. Decisive result: on `off_support_corridor`, `energy_guided_diffusion` success is 0.000 +/- 0.000, `energy_rerank_unguided` is 0.000 +/- 0.000, and `cem_trajectory_optimizer` is 1.000 +/- 0.000.
7. Paired result: guided-minus-rerank success difference is 0.00000; guided-minus-CEM success difference is -1.00000.
8. Ablation result: guidance without prior has 0.607 mode escape but 0.000 success and 0.929 collision, showing unsafe off-support escape rather than useful behavior.
9. Reproducibility: `python src/run_experiment.py` regenerates CSVs and figures.
10. Exact Downloads PDF path: `C:/Users/wangz/Downloads/78.pdf`
11. GitHub URL: https://github.com/Jason-Wang313/78_energy_guided_diffusion_policy_limits
12. Confirmation: no visible Desktop copy was requested or made.
