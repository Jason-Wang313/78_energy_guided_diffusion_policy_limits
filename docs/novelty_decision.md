# Novelty Decision

Chosen thesis: energy guidance changes reachable robot behavior rather than merely re-ranking prior-supported samples.

Implemented novelty test: a support-gap trajectory benchmark where the demonstrated homotopy can be rare or absent while a safe homotopy remains geometrically feasible.

Decision: KILL_ARCHIVE.

Reason: the diagnostic is valuable, but the mechanism fails. Energy-guided diffusion does not create the missing safe homotopy in the off-support corridor. It stays near the blocked prior mode or becomes unsafe, while CEM and the oracle solve the same tasks. The novelty boundary is therefore a negative observation about support limits, not a submit-ready algorithmic advance.
