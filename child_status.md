# Child Status 78

Current stage: 2026-06-21 expanded v5 submission-hardening terminal
Last update: 2026-06-21 18:30 Asia/Shanghai
PDF: C:/Users/wangz/Downloads/78.pdf
GitHub: https://github.com/Jason-Wang313/78_energy_guided_diffusion_policy_limits
Submission-hardening version: v5
Terminal decision: KILL_ARCHIVE
ICLR main ready: no

Evidence summary: v5 expands Paper 78 into an eight-seed hostile trajectory-optimization benchmark with 6,240 main rollouts, 1,600 ablation rollouts, 5,120 stress rollouts, 2,048 fixed-risk rollouts, 480 support diagnostics, 520 seed metrics, aggregate hard-regime tables, pairwise tests, curated negative cases, and a 54-page ICLR-style PDF.

Final v5 result: `support_aware_energy_bridge_v5` reaches 1.000 success on the decisive off-support corridor and hard aggregate, but hostile baselines such as mode-diverse diffusion, diffusion-CEM hybrid, CHOMP-like optimization, CEM, graph search, and the oracle also reach 1.000. Ablations including `no_support_awareness`, `no_energy_gradient`, `no_prior_score`, and `optimizer_handoff` also solve the off-support protocol, so the named mechanism is not necessary. At fixed-risk budget 0.00 the bridge has 0.000 fixed-risk success. The decision remains `KILL_ARCHIVE`.

Artifact audit 2026-06-21: `python scripts\validate_submission_artifacts.py` passed with 54 pages and SHA256 `2FD1529E9EB44BFC3BB2FC8B18FFA2ECD8CD61D06A47D746B53E56D388D64F91`. Visual PDF QA passed on title/citation, figures, dense appendix, and references pages. `C:/Users/wangz/Desktop/78.pdf` does not exist.
