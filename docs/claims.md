# Claims

- Tested mechanism claim: energy guidance during reverse diffusion should create safe reachable behavior beyond energy reranking of prior-supported samples.
- Evidence claim: the v4 runner implements a trajectory-diffusion support-gap benchmark with seven seeds, five splits, strong baselines, ablations, stress sweeps, and negative cases.
- Empirical result: the mechanism fails on the decisive off-support split; `energy_guided_diffusion` has 0.000 success, matching reranking and losing to CEM/oracle at 1.000.
- Scope claim: the repository is useful as a negative diagnostic for diffusion-policy support limits, not as an ICLR-main submission.
- Unsupported claim explicitly avoided: no claim that energy guidance enables out-of-support robot behavior.
