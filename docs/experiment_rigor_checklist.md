# Experiment Rigor Checklist

## v4 Local Rigor

- [x] Seven random seeds.
- [x] Implemented trajectory generator and continuous obstacle/dynamics checks.
- [x] Data-driven kernel reverse-diffusion sampler over offline demonstrations.
- [x] Energy reranking baseline.
- [x] Energy-guided diffusion baseline.
- [x] Safety-projected guided diffusion.
- [x] CEM trajectory optimizer baseline.
- [x] Grid-oracle upper bound.
- [x] Paired comparisons against reranking and CEM.
- [x] Ablations over prior score, gradient guidance, projection, and overguidance.
- [x] Support-gap stress sweep.
- [x] Negative cases.
- [x] Paper-specific figures.

## Still Missing For ICLR Main

- [ ] Real-robot validation.
- [ ] Recognized high-fidelity simulator benchmark.
- [ ] Trained neural diffusion policy checkpoint.
- [ ] External competing learned baselines.
- [ ] Full manual related-work synthesis.

Decision: fail ICLR main empirical-rigor gate because the implemented evidence is negative and external validation is absent.
