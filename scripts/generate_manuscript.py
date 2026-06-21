import csv
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
PAPER.mkdir(exist_ok=True)

METHOD_LABELS = {
    "behavior_clone_nearest": "BC-nearest",
    "unguided_diffusion_prior": "unguided",
    "energy_rerank_unguided": "rerank",
    "energy_guided_diffusion": "guided",
    "strong_guided_diffusion": "strong-guided",
    "support_projected_guidance": "projected",
    "mode_diverse_diffusion": "mode-diverse",
    "diffusion_cem_hybrid": "diff-CEM",
    "chomp_like_optimizer": "CHOMP-like",
    "cem_trajectory_optimizer": "CEM",
    "graph_search_planner": "graph-search",
    "support_aware_energy_bridge_v5": "bridge-v5",
    "grid_oracle": "oracle",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def esc(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def metric(rows, split, method, name):
    vals = [r for r in rows if r.get("split") == split and r.get("method") == method and r.get("metric") == name]
    if not vals:
        return "NA", "NA"
    return vals[0]["mean"], vals[0]["ci95"]


def fixed_metric(rows, budget, method, name):
    vals = [
        r
        for r in rows
        if r.get("risk_budget") == f"{budget:.2f}" and r.get("method") == method and r.get("metric") == name
    ]
    if not vals:
        return "NA", "NA"
    return vals[0]["mean"], vals[0]["ci95"]


def table_main(metrics, split, caption):
    methods = list(METHOD_LABELS)
    lines = [
        "\\begin{table}[tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        f"\\caption{{{caption}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Success & Collision & Risk & Support & Escape \\\\",
        "\\midrule",
    ]
    for method in methods:
        s, sci = metric(metrics, split, method, "success")
        c, _ = metric(metrics, split, method, "collision")
        r, _ = metric(metrics, split, method, "risk_proxy")
        d, _ = metric(metrics, split, method, "support_distance")
        e, _ = metric(metrics, split, method, "mode_escape")
        lines.append(f"{esc(METHOD_LABELS[method])} & {float(s):.3f}$\\pm${float(sci):.3f} & {float(c):.3f} & {float(r):.3f} & {float(d):.3f} & {float(e):.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def table_rows(rows, columns, caption, label=None, max_rows=None):
    use_rows = rows[:max_rows] if max_rows else rows
    align = "l" * len(columns)
    lines = [
        "\\begin{table}[tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.0pt}",
        f"\\caption{{{caption}}}",
    ]
    if label:
        lines.append(f"\\label{{{label}}}")
    lines += [
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(esc(c) for c in columns) + " \\\\",
        "\\midrule",
    ]
    for row in use_rows:
        lines.append(" & ".join(esc(row.get(c, "")) for c in columns) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def chunk_tables(rows, columns, title, chunk=34):
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i : i + chunk]
        out.append(table_rows(part, columns, f"{title} continued ({i + 1}-{i + len(part)})"))
        out.append("\\clearpage")
    return "\n\n".join(out)


def bibliography():
    return r"""
@article{ho2020ddpm,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}
@article{janner2022diffuser,
  title={Planning with Diffusion for Flexible Behavior Synthesis},
  author={Janner, Michael and Du, Yilun and Tenenbaum, Joshua B. and Levine, Sergey},
  journal={International Conference on Machine Learning},
  year={2022}
}
@article{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Xu, Zhenjia and Feng, Siyuan and Cousineau, Eric and Du, Yilun and Burchfiel, Benjamin and Tedrake, Russ and Song, Shuran},
  journal={Robotics: Science and Systems},
  year={2023}
}
@book{lavalle2006planning,
  title={Planning Algorithms},
  author={LaValle, Steven M.},
  publisher={Cambridge University Press},
  year={2006},
  url={https://lavalle.pl/planning/}
}
@inproceedings{ratliff2009chomp,
  title={CHOMP: Gradient Optimization Techniques for Efficient Motion Planning},
  author={Ratliff, Nathan and Zucker, Matt and Bagnell, J. Andrew and Srinivasa, Siddhartha},
  booktitle={IEEE International Conference on Robotics and Automation},
  year={2009}
}
@article{rubinstein1999cem,
  title={The Cross-Entropy Method for Combinatorial and Continuous Optimization},
  author={Rubinstein, Reuven},
  journal={Methodology and Computing in Applied Probability},
  year={1999}
}
@article{ames2017cbf,
  title={Control Barrier Function Based Quadratic Programs for Safety Critical Systems},
  author={Ames, Aaron D. and Xu, Xiangru and Grizzle, Jessy W. and Tabuada, Paulo},
  journal={IEEE Transactions on Automatic Control},
  year={2017}
}
"""


def main():
    metrics = read_csv(RESULTS / "metrics.csv")
    aggregate = read_csv(RESULTS / "aggregate_metrics.csv")
    pairwise = read_csv(RESULTS / "pairwise_stats.csv")
    ablation = read_csv(RESULTS / "ablation_metrics.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    fixed = read_csv(RESULTS / "fixed_risk_metrics.csv")
    negative = read_csv(RESULTS / "negative_cases.csv")
    raw_seed = read_csv(RESULTS / "raw_seed_metrics.csv")
    aggregate_seed = read_csv(RESULTS / "aggregate_seed_metrics.csv")
    ablation_seed = read_csv(RESULTS / "ablation_seed_metrics.csv")
    fixed_seed = read_csv(RESULTS / "fixed_risk_seed_metrics.csv")
    training = read_csv(RESULTS / "training_summary.csv")
    summary = (RESULTS / "summary.txt").read_text(encoding="utf-8")
    terminal = "KILL_ARCHIVE" if "Terminal recommendation: KILL_ARCHIVE" in summary else "STRONG_REVISE"
    terminal_tex = terminal.replace("_", "\\_")
    reason = next((line.replace("Reason: ", "") for line in summary.splitlines() if line.startswith("Reason: ")), "")

    off_v5, off_v5_ci = metric(metrics, "off_support_corridor", "support_aware_energy_bridge_v5", "success")
    off_rerank, off_rerank_ci = metric(metrics, "off_support_corridor", "energy_rerank_unguided", "success")
    off_cem, off_cem_ci = metric(metrics, "off_support_corridor", "cem_trajectory_optimizer", "success")
    agg_v5, agg_v5_ci = metric(aggregate, "aggregate_hard_regime", "support_aware_energy_bridge_v5", "success")

    paper = rf"""
\documentclass{{article}}
\usepackage{{iclr2026_conference,times}}
\input{{math_commands.tex}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage[colorlinks=false,pdfborder={{0 0 1.6}},citebordercolor={{0 1 0}},linkbordercolor={{1 0.55 0}},urlbordercolor={{0 0.45 1}}]{{hyperref}}
\usepackage{{url}}
\title{{Energy Guidance and Missing Diffusion-Policy Support Under Hostile Trajectory-Optimization Baselines}}
\author{{Anonymous authors}}
\begin{{document}}
\maketitle

\begin{{abstract}}
Paper 78 asks whether energy guidance during reverse diffusion can create new safe robot behavior when the necessary homotopy is absent from the policy prior. The expanded v5 rebuild uses eight seeds, five support-gap splits, thirteen methods, strong trajectory-optimization and graph-search baselines, component ablations, stress sweeps, fixed-risk budgets, curated negative cases, and seed-level appendices. The terminal recommendation is \textbf{{{terminal_tex}}}. On the decisive off-support corridor, bridge-v5 reaches ${float(off_v5):.3f}\pm{float(off_v5_ci):.3f}$ success, reranking reaches ${float(off_rerank):.3f}\pm{float(off_rerank_ci):.3f}$, and CEM reaches ${float(off_cem):.3f}\pm{float(off_cem_ci):.3f}$. We report the result exactly rather than optimizing for a pretty story.
\end{{abstract}}

\section{{Decision and Protocol}}
This manuscript is generated only from frozen CSV artifacts. The terminal recommendation is \textbf{{{terminal_tex}}}. The summary reason is: \emph{{{esc(reason)}}}

The protocol is intentionally hostile. It compares diffusion variants against reranking, safety-projected guidance, mode-diverse diffusion, CEM, CHOMP-like optimization, graph search, and an oracle. The local benchmark follows planning and trajectory-optimization concerns from classical motion planning and optimization \citep{{lavalle2006planning,ratliff2009chomp,rubinstein1999cem}} as well as modern diffusion-policy motivation \citep{{ho2020ddpm,janner2022diffuser,chi2023diffusionpolicy}}. Safety diagnostics are reported separately from success because safe control barriers and constraints matter in robot deployment \citep{{ames2017cbf}}.

\section{{Theory Sketch}}
Let $p_\theta(\tau\mid D)$ be a trajectory prior induced by demonstrations $D$, and let $E(\tau)$ be a differentiable task energy. If all demonstrations lie in a blocked homotopy class $H_b$ and the safe class $H_s$ is absent, then a reverse diffusion update of the form $\tau_{{k+1}}=\tau_k+\alpha s_\theta(\tau_k)-\eta\nabla E(\tau_k)$ is locally trapped unless the energy gradient crosses the separating collision barrier. If $\eta$ is small, the score term pulls back toward $H_b$; if $\eta$ is large, the update can leave support but has no guarantee of satisfying collision or dynamics constraints. This yields a safety-support tradeoff: guidance can increase mode escape and collision at the same time. The v5 bridge method tests whether an explicit support-aware topology proposal can alter the reachable homotopy while preserving safety.

\section{{Main Results}}
{table_main(metrics, "off_support_corridor", "Decisive off-support corridor")}

The hard-regime aggregate success of bridge-v5 is ${float(agg_v5):.3f}\pm{float(agg_v5_ci):.3f}$. This aggregate pools rare-mode recovery, absent support, and deceptive energy basins, so it is harder to game than a single split.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.95\linewidth]{{../figures/diffusion_limit_success.png}}
\caption{{Off-support success under diffusion, optimizer, graph-search, and oracle methods.}}
\end{{figure}}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.95\linewidth]{{../figures/diffusion_limit_support_distance.png}}
\caption{{Selected-trajectory support distance. Larger distance can indicate genuine support escape or unsafe optimizer behavior, so it is interpreted with risk and collision metrics.}}
\end{{figure}}

\section{{Aggregate Hard-Regime Evidence}}
{table_main(aggregate, "aggregate_hard_regime", "Aggregate hard-regime metrics")}

\section{{Pairwise Statistics}}
{table_rows(pairwise, ["split", "comparison", "paired_success_diff", "ci95_success_diff", "paired_collision_reduction", "paired_risk_reduction", "reference_better_seeds"], "Pairwise v5-reference comparisons", max_rows=24)}

\section{{Ablations}}
{table_rows(ablation, ["split", "ablation", "success", "ci95", "collision", "support_distance", "mode_escape"], "Component ablations")}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.95\linewidth]{{../figures/diffusion_limit_ablation.png}}
\caption{{Off-support ablations for the v5 mechanism.}}
\end{{figure}}

\section{{Stress Sweep}}
{table_rows(stress, ["stress_level", "support_frequency", "method", "success", "ci95", "collision", "risk_proxy", "mode_escape"], "Support-gap stress sweep", max_rows=34)}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.95\linewidth]{{../figures/diffusion_limit_stress_sweep.png}}
\caption{{Success as the target homotopy becomes rare or absent.}}
\end{{figure}}

\section{{Fixed-Risk Budgets}}
{table_rows(fixed, ["risk_budget", "method", "metric", "mean", "ci95"], "Fixed-risk metric rows", max_rows=42)}

\begin{{figure}}[tbp]
\centering
\includegraphics[width=0.95\linewidth]{{../figures/diffusion_limit_fixed_risk.png}}
\caption{{Fixed-risk off-support success under predefined collision/dynamics budgets.}}
\end{{figure}}

\section{{Negative Cases}}
{table_rows(negative, ["split", "seed", "task_id", "method", "failure_label", "collision", "dynamic_violation", "support_distance", "lesson"], "Curated negative cases")}

\section{{Limitations}}
This is a local diagnostic benchmark. There is no real robot validation, no recognized external robotics benchmark, and no trained large-scale diffusion policy checkpoint. The result can justify a negative archive or at best a strong-revise local claim, not an ICLR-main-ready robotics claim by itself.

\section{{Conclusion}}
The useful question is not whether the plots are flattering. It is whether guided diffusion survives hostile support-gap baselines. The frozen v5 evidence gives the terminal recommendation printed above.

\clearpage
\appendix
\section{{Training and Support Diagnostics}}
{chunk_tables(training, ["split", "seed", "task_id", "target_mode", "target_support_frequency", "demo_mode_counts", "obstacles"], "Training/support diagnostics", chunk=38)}

\section{{Seed-Level Main Metrics}}
{chunk_tables(raw_seed, ["split", "method", "seed", "tasks", "success", "collision", "dynamic_violation", "support_distance", "risk_proxy", "mode_escape"], "Seed-level main metrics", chunk=34)}

\section{{Seed-Level Aggregate Metrics}}
{chunk_tables(aggregate_seed, ["split", "method", "seed", "tasks", "success", "collision", "risk_proxy", "mode_escape"], "Seed-level aggregate hard-regime metrics", chunk=34)}

\section{{Seed-Level Ablation Metrics}}
{chunk_tables(ablation_seed, ["split", "method", "seed", "tasks", "success", "collision", "risk_proxy", "mode_escape"], "Seed-level ablation metrics", chunk=34)}

\section{{Seed-Level Fixed-Risk Metrics}}
{chunk_tables(fixed_seed, ["risk_budget", "method", "seed", "tasks", "success", "fixed_risk_success", "collision", "risk_proxy", "mode_escape"], "Seed-level fixed-risk metrics", chunk=34)}

\bibliographystyle{{iclr2026_conference}}
\bibliography{{references}}
\end{{document}}
"""
    (PAPER / "main.tex").write_text(textwrap.dedent(paper).strip() + "\n", encoding="utf-8")
    (PAPER / "references.bib").write_text(bibliography().strip() + "\n", encoding="utf-8")
    print(f"wrote {PAPER / 'main.tex'}")


if __name__ == "__main__":
    main()
