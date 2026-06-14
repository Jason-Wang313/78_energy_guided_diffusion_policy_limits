import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 78012026
SEEDS = list(range(7))
TASKS_PER_SPLIT_SEED = 4
STRESS_TASKS_PER_SEED = 2
T = 25
DEMO_COUNT = 40
ROBOT_RADIUS = 0.045
VELOCITY_LIMIT = 0.205
ACCEL_LIMIT = 0.105

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


@dataclass
class Task:
    split: str
    seed: int
    task_id: int
    start: np.ndarray
    goal: np.ndarray
    obstacles: list
    demos: np.ndarray
    demo_modes: list
    target_mode: str
    support_frequency: float
    guidance_collision_weight: float
    description: str


def stable_rng(*parts):
    acc = BASE_SEED
    for part in parts:
        if isinstance(part, str):
            for ch in part:
                acc = (acc * 131 + ord(ch)) % (2**32 - 1)
        else:
            acc = (acc * 131 + int(part)) % (2**32 - 1)
    return np.random.default_rng(acc)


def ci95(vals):
    vals = list(vals)
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1))
    return 1.96 * sd / math.sqrt(len(vals))


def waypoint_for_mode(mode, goal_y=0.0):
    if mode == "upper":
        return np.array([0.0, 1.22 + 0.10 * max(0.0, goal_y)])
    if mode == "lower":
        return np.array([0.0, -1.22 + 0.10 * min(0.0, goal_y)])
    if mode == "wide_upper":
        return np.array([0.02, 1.45])
    if mode == "wide_lower":
        return np.array([0.02, -1.45])
    return np.array([0.0, 0.0])


def make_curve(start, goal, mode, rng, noise=0.018):
    waypoint = waypoint_for_mode(mode, float(goal[1]))
    t = np.linspace(0.0, 1.0, T)
    path = (
        ((1.0 - t) ** 2)[:, None] * start
        + (2.0 * (1.0 - t) * t)[:, None] * waypoint
        + (t**2)[:, None] * goal
    )
    if noise > 0:
        scale = np.sin(np.pi * t)[:, None]
        path += scale * rng.normal(0.0, noise, size=path.shape)
    for _ in range(2):
        old = path.copy()
        path[1:-1] = 0.25 * old[:-2] + 0.50 * old[1:-1] + 0.25 * old[2:]
    path[0] = start
    path[-1] = goal + rng.normal(0.0, noise * 0.15, size=2)
    return path


def make_demos(start, goal, mode_counts, rng):
    demos = []
    modes = []
    for mode, count in mode_counts.items():
        for _ in range(count):
            demos.append(make_curve(start, goal, mode, rng, noise=0.018 + 0.006 * rng.random()))
            modes.append(mode)
    order = rng.permutation(len(demos))
    return np.array([demos[i] for i in order]), [modes[i] for i in order]


def make_task(split, seed, task_id, support_override=None):
    rng = stable_rng("task", split, seed, task_id, 0 if support_override is None else int(1000 * support_override))
    start = np.array([-1.0, rng.uniform(-0.05, 0.05)])
    goal = np.array([1.0, rng.uniform(-0.16, 0.16)])
    central_radius = rng.uniform(0.28, 0.34)
    obstacles = [(np.array([0.0, rng.uniform(-0.03, 0.03)]), central_radius)]
    guidance_collision_weight = 16.0
    description = "supported mode"

    if split == "supported_single_mode":
        target_mode = "upper" if task_id % 2 == 0 else "lower"
        mode_counts = {target_mode: DEMO_COUNT}
        support_frequency = 1.0
    elif split == "supported_multimodal_shift":
        obstacles[0] = (np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08)]), central_radius)
        target_mode = "upper" if goal[1] >= 0 else "lower"
        mode_counts = {"upper": DEMO_COUNT // 2, "lower": DEMO_COUNT // 2}
        support_frequency = 0.5
        description = "supported multimodal geometry shift"
    elif split == "rare_mode_recovery":
        target_mode = "upper"
        mode_counts = {"lower": DEMO_COUNT - 2, "upper": 2}
        support_frequency = 2 / DEMO_COUNT
        obstacles.append((np.array([0.05, -0.57]), rng.uniform(0.23, 0.29)))
        description = "rare safe upper mode with common lower mode blocked"
    elif split == "off_support_corridor":
        target_mode = "upper"
        mode_counts = {"lower": DEMO_COUNT}
        support_frequency = 0.0
        obstacles.append((np.array([0.05, -0.57]), rng.uniform(0.24, 0.30)))
        description = "safe upper mode absent from prior and lower mode blocked"
    elif split == "deceptive_energy_basin":
        target_mode = "upper" if task_id % 2 == 0 else "lower"
        mode_counts = {"upper": 16, "lower": 16, "straight": 8}
        support_frequency = 0.4
        guidance_collision_weight = 5.0
        obstacles[0] = (np.array([0.0, 0.0]), rng.uniform(0.34, 0.40))
        description = "weak obstacle energy encourages unsafe shortcut"
    elif split == "support_gap_sweep":
        target_mode = "upper"
        upper_count = int(round(DEMO_COUNT * float(support_override)))
        upper_count = max(0, min(DEMO_COUNT, upper_count))
        mode_counts = {"lower": DEMO_COUNT - upper_count}
        if upper_count:
            mode_counts["upper"] = upper_count
        support_frequency = upper_count / DEMO_COUNT
        obstacles.append((np.array([0.05, -0.57]), rng.uniform(0.24, 0.30)))
        description = f"support frequency sweep upper={support_frequency:.2f}"
    else:
        raise ValueError(f"unknown split: {split}")

    demos, demo_modes = make_demos(start, goal, mode_counts, rng)
    return Task(
        split=split,
        seed=seed,
        task_id=task_id,
        start=start,
        goal=goal,
        obstacles=obstacles,
        demos=demos,
        demo_modes=demo_modes,
        target_mode=target_mode,
        support_frequency=support_frequency,
        guidance_collision_weight=guidance_collision_weight,
        description=description,
    )


def path_mode(path):
    middle = path[np.abs(path[:, 0]) < 0.30]
    if len(middle) == 0:
        middle = path[T // 2 - 2 : T // 2 + 3]
    y = float(np.mean(middle[:, 1]))
    if y > 0.17:
        return "upper"
    if y < -0.17:
        return "lower"
    return "straight"


def min_clearance(path, obstacles):
    best = float("inf")
    for center, radius in obstacles:
        d = np.linalg.norm(path - center[None, :], axis=1) - (radius + ROBOT_RADIUS)
        best = min(best, float(np.min(d)))
    return best


def trajectory_stats(path, task):
    diffs = np.diff(path, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    accels = np.diff(diffs, axis=0)
    accel_norm = np.linalg.norm(accels, axis=1)
    goal_error = float(np.linalg.norm(path[-1] - task.goal))
    clearance = min_clearance(path, task.obstacles)
    collision = clearance < 0.0
    dyn_violation = bool(np.max(speeds) > VELOCITY_LIMIT or np.max(accel_norm) > ACCEL_LIMIT)
    success = bool((not collision) and (not dyn_violation) and goal_error <= 0.12)
    return {
        "goal_error": goal_error,
        "min_clearance": clearance,
        "collision": collision,
        "dynamic_violation": dyn_violation,
        "success": success,
        "max_speed": float(np.max(speeds)),
        "max_accel": float(np.max(accel_norm)),
        "path_length": float(np.sum(speeds)),
    }


def task_energy(path, task, collision_weight=None, velocity_weight=28.0):
    if collision_weight is None:
        collision_weight = task.guidance_collision_weight
    diffs = np.diff(path, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    accels = np.diff(diffs, axis=0)
    goal = 34.0 * float(np.sum((path[-1] - task.goal) ** 2))
    length = 0.18 * float(np.sum(speeds))
    smooth = 4.0 * float(np.sum(accels**2))
    velocity = velocity_weight * float(np.sum(np.maximum(0.0, speeds - VELOCITY_LIMIT) ** 2))
    obstacle = 0.0
    for center, radius in task.obstacles:
        margin = radius + ROBOT_RADIUS + 0.035
        delta = path - center[None, :]
        dist = np.linalg.norm(delta, axis=1) + 1e-9
        obstacle += float(np.sum(np.maximum(0.0, margin - dist) ** 2))
    return goal + length + smooth + collision_weight * obstacle + velocity


def energy_gradient(path, task, collision_weight=None, velocity_weight=28.0):
    if collision_weight is None:
        collision_weight = task.guidance_collision_weight
    grad = np.zeros_like(path)
    grad[-1] += 68.0 * (path[-1] - task.goal)

    diffs = np.diff(path, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) + 1e-9
    for i, (diff, speed) in enumerate(zip(diffs, speeds)):
        grad[i] -= 0.18 * diff / speed
        grad[i + 1] += 0.18 * diff / speed
        excess = max(0.0, speed - VELOCITY_LIMIT)
        if excess > 0:
            g = 2.0 * velocity_weight * excess * diff / speed
            grad[i] -= g
            grad[i + 1] += g

    accels = np.diff(diffs, axis=0)
    for i, acc in enumerate(accels):
        g = 8.0 * acc
        grad[i] += g
        grad[i + 1] -= 2.0 * g
        grad[i + 2] += g

    for center, radius in task.obstacles:
        margin = radius + ROBOT_RADIUS + 0.035
        delta = path - center[None, :]
        dist = np.linalg.norm(delta, axis=1) + 1e-9
        penetration = np.maximum(0.0, margin - dist)
        active = penetration > 0
        if np.any(active):
            direction = delta[active] / dist[active, None]
            grad[active] += -2.0 * collision_weight * penetration[active, None] * direction

    grad[0] = 0.0
    return grad


def prior_denoise(path, demos, sigma):
    flat = path.reshape(1, -1)
    demo_flat = demos.reshape(len(demos), -1)
    dist = np.mean((demo_flat - flat) ** 2, axis=1)
    tau = max(0.0009, (0.55 * sigma) ** 2)
    logits = -dist / (2.0 * tau)
    logits -= float(np.max(logits))
    weights = np.exp(logits)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 1e-12:
        return demos[int(np.argmin(dist))].copy()
    weights /= total
    return np.tensordot(weights, demos, axes=(0, 0))


def support_distance(path, demos):
    flat = path.reshape(1, -1)
    demo_flat = demos.reshape(len(demos), -1)
    dist = np.sqrt(np.mean((demo_flat - flat) ** 2, axis=1))
    return float(np.min(dist))


def project_safety(path, task):
    out = path.copy()
    for _ in range(2):
        for center, radius in task.obstacles:
            margin = radius + ROBOT_RADIUS + 0.018
            delta = out - center[None, :]
            dist = np.linalg.norm(delta, axis=1) + 1e-9
            active = dist < margin
            if np.any(active):
                out[active] = center[None, :] + delta[active] / dist[active, None] * margin
        diffs = np.diff(out, axis=0)
        speeds = np.linalg.norm(diffs, axis=1) + 1e-9
        too_fast = speeds > VELOCITY_LIMIT
        if np.any(too_fast):
            for i in np.where(too_fast)[0]:
                out[i + 1] = out[i] + diffs[i] / speeds[i] * VELOCITY_LIMIT
        out[0] = task.start
    return out


def sample_diffusion(
    task,
    rng,
    samples=24,
    steps=18,
    guidance_weight=0.0,
    collision_weight=None,
    projection=False,
    use_prior=True,
    choose_energy=True,
    noise_scale=0.33,
):
    candidates = []
    sigmas = np.linspace(noise_scale, 0.025, steps)
    for _ in range(samples):
        if use_prior:
            base = task.demos[int(rng.integers(0, len(task.demos)))].copy()
        else:
            base = make_curve(task.start, task.goal, "straight", rng, noise=0.08)
        path = base + rng.normal(0.0, sigmas[0], size=base.shape) * np.sin(np.linspace(0, math.pi, T))[:, None]
        path[0] = task.start
        for sigma in sigmas:
            if use_prior:
                denoised = prior_denoise(path, task.demos, sigma)
                path += 0.68 * (denoised - path)
            if guidance_weight > 0.0:
                grad = energy_gradient(path, task, collision_weight=collision_weight)
                interior = grad[1:]
                norm = float(np.linalg.norm(interior))
                if norm > 18.0:
                    grad[1:] *= 18.0 / norm
                path -= guidance_weight * (0.020 + 0.045 * sigma) * grad
            if sigma > 0.04:
                path[1:-1] += rng.normal(0.0, sigma * 0.045, size=path[1:-1].shape)
            if projection:
                path = project_safety(path, task)
            path[:, 0] = np.clip(path[:, 0], -1.35, 1.35)
            path[:, 1] = np.clip(path[:, 1], -1.60, 1.60)
            path[0] = task.start
        if projection:
            path = project_safety(path, task)
        candidates.append(path.copy())

    if choose_energy:
        scores = [task_energy(c, task, collision_weight=collision_weight) for c in candidates]
    else:
        scores = [support_distance(c, task.demos) + 0.1 * np.linalg.norm(c[-1] - task.goal) for c in candidates]
    return candidates[int(np.argmin(scores))]


def behavior_clone_nearest(task):
    final_dist = np.linalg.norm(task.demos[:, -1, :] - task.goal[None, :], axis=1)
    smooth = np.array([task_energy(d, task, collision_weight=2.0) for d in task.demos])
    idx = int(np.argmin(final_dist + 0.025 * smooth))
    return task.demos[idx].copy()


def cem_optimizer(task, rng, population=22, iterations=4, elite_frac=0.24):
    seeds = []
    for mode in ["upper", "lower", "wide_upper", "wide_lower", "straight"]:
        seeds.append(make_curve(task.start, task.goal, mode, rng, noise=0.03))
    for _ in range(8):
        seeds.append(task.demos[int(rng.integers(0, len(task.demos)))].copy())
    dim = (T - 2) * 2
    elite_count = max(4, int(population * elite_frac))
    mean = np.mean([s[1:-1].reshape(-1) for s in seeds], axis=0)
    std = np.ones(dim) * 0.55
    best = None
    best_score = float("inf")
    for _ in range(iterations):
        candidates = []
        for s in seeds:
            candidates.append(s.copy())
        for _ in range(max(0, population - len(candidates))):
            interior = mean + rng.normal(0.0, std, size=dim)
            path = np.zeros((T, 2))
            path[0] = task.start
            path[-1] = task.goal + rng.normal(0.0, 0.015, size=2)
            path[1:-1] = interior.reshape(T - 2, 2)
            for _smooth in range(2):
                old = path.copy()
                path[1:-1] = 0.18 * old[:-2] + 0.64 * old[1:-1] + 0.18 * old[2:]
            candidates.append(project_safety(path, task))
        scores = np.array([task_energy(c, task, collision_weight=55.0, velocity_weight=60.0) for c in candidates])
        order = np.argsort(scores)
        if float(scores[order[0]]) < best_score:
            best_score = float(scores[order[0]])
            best = candidates[int(order[0])].copy()
        elites = [candidates[int(i)][1:-1].reshape(-1) for i in order[:elite_count]]
        mean = np.mean(elites, axis=0)
        std = np.maximum(0.035, np.std(elites, axis=0) * 0.82)
    return project_safety(best, task)


def grid_oracle(task, rng):
    candidates = [make_curve(task.start, task.goal, mode, rng, noise=0.004) for mode in ["upper", "lower", "wide_upper", "wide_lower", "straight"]]
    candidates += [project_safety(c, task) for c in candidates]
    scores = []
    for c in candidates:
        stats = trajectory_stats(c, task)
        penalty = 0.0
        if stats["collision"]:
            penalty += 100.0
        if stats["dynamic_violation"]:
            penalty += 40.0
        scores.append(task_energy(c, task, collision_weight=80.0, velocity_weight=80.0) + penalty)
    return candidates[int(np.argmin(scores))]


METHODS = [
    "behavior_clone_nearest",
    "unguided_diffusion_prior",
    "energy_rerank_unguided",
    "energy_guided_diffusion",
    "strong_guided_diffusion",
    "cem_trajectory_optimizer",
    "grid_oracle",
]


def run_method(method, task, rng):
    if method == "behavior_clone_nearest":
        return behavior_clone_nearest(task)
    if method == "unguided_diffusion_prior":
        return sample_diffusion(task, rng, samples=6, steps=10, guidance_weight=0.0, choose_energy=False)
    if method == "energy_rerank_unguided":
        return sample_diffusion(task, rng, samples=10, steps=9, guidance_weight=0.0, choose_energy=True)
    if method == "energy_guided_diffusion":
        return sample_diffusion(
            task,
            rng,
            samples=8,
            steps=9,
            guidance_weight=1.0,
            collision_weight=task.guidance_collision_weight,
            projection=False,
            choose_energy=True,
        )
    if method == "strong_guided_diffusion":
        return sample_diffusion(
            task,
            rng,
            samples=10,
            steps=10,
            guidance_weight=0.82,
            collision_weight=max(24.0, task.guidance_collision_weight),
            projection=True,
            choose_energy=True,
        )
    if method == "cem_trajectory_optimizer":
        return cem_optimizer(task, rng)
    if method == "grid_oracle":
        return grid_oracle(task, rng)
    raise ValueError(method)


def failure_label(method, task, stats, mode, support_dist):
    if stats["success"]:
        return "success"
    if stats["collision"]:
        if method in {"energy_guided_diffusion", "strong_guided_diffusion", "full_energy_guided", "no_safety_projection", "strong_projection", "overguidance"}:
            return "collision_shortcut_from_guidance"
        return "collision_or_blocked_prior_mode"
    if stats["dynamic_violation"]:
        return "dynamic_violation"
    if mode != task.target_mode and task.support_frequency <= 0.06:
        return "prior_trap_wrong_homotopy"
    if support_dist > 0.42:
        return "unsafe_off_support_escape"
    return "goal_miss"


def evaluate_row(method, task, path):
    stats = trajectory_stats(path, task)
    mode = path_mode(path)
    support_dist = support_distance(path, task.demos)
    prior_count = sum(1 for m in task.demo_modes if m == mode)
    mode_frequency = prior_count / len(task.demo_modes)
    escape = mode_frequency <= 0.06 and mode == task.target_mode
    row = {
        "split": task.split,
        "seed": task.seed,
        "task_id": task.task_id,
        "method": method,
        "success": int(stats["success"]),
        "collision": int(stats["collision"]),
        "dynamic_violation": int(stats["dynamic_violation"]),
        "goal_error": f"{stats['goal_error']:.5f}",
        "min_clearance": f"{stats['min_clearance']:.5f}",
        "max_speed": f"{stats['max_speed']:.5f}",
        "max_accel": f"{stats['max_accel']:.5f}",
        "path_length": f"{stats['path_length']:.5f}",
        "task_energy": f"{task_energy(path, task):.5f}",
        "support_distance": f"{support_dist:.5f}",
        "mode": mode,
        "target_mode": task.target_mode,
        "mode_frequency_in_prior": f"{mode_frequency:.4f}",
        "mode_escape": int(escape),
        "target_support_frequency": f"{task.support_frequency:.4f}",
        "failure_label": failure_label(method, task, stats, mode, support_dist),
        "description": task.description,
    }
    return row


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_metrics(rows):
    output = []
    for split in sorted({r["split"] for r in rows}):
        for method in METHODS:
            for seed in SEEDS:
                vals = [r for r in rows if r["split"] == split and r["method"] == method and int(r["seed"]) == seed]
                if not vals:
                    continue
                output.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "tasks": len(vals),
                        "success": f"{np.mean([int(v['success']) for v in vals]):.5f}",
                        "collision": f"{np.mean([int(v['collision']) for v in vals]):.5f}",
                        "dynamic_violation": f"{np.mean([int(v['dynamic_violation']) for v in vals]):.5f}",
                        "mean_energy": f"{np.mean([float(v['task_energy']) for v in vals]):.5f}",
                        "support_distance": f"{np.mean([float(v['support_distance']) for v in vals]):.5f}",
                        "mode_escape": f"{np.mean([int(v['mode_escape']) for v in vals]):.5f}",
                    }
                )
    return output


def aggregate_metrics(seed_rows):
    output = []
    for split in sorted({r["split"] for r in seed_rows}):
        for method in METHODS:
            vals = [r for r in seed_rows if r["split"] == split and r["method"] == method]
            if not vals:
                continue
            for metric in ["success", "collision", "dynamic_violation", "mean_energy", "support_distance", "mode_escape"]:
                numbers = [float(v[metric]) for v in vals]
                output.append(
                    {
                        "split": split,
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(numbers):.5f}",
                        "ci95": f"{ci95(numbers):.5f}",
                        "seeds": len(numbers),
                        "tasks_per_seed": vals[0]["tasks"],
                    }
                )
    return output


def pairwise_stats(seed_rows):
    pairs = []
    references = ["energy_rerank_unguided", "cem_trajectory_optimizer"]
    targets = ["energy_guided_diffusion", "strong_guided_diffusion"]
    for split in sorted({r["split"] for r in seed_rows}):
        for target in targets:
            for reference in references:
                for metric in ["success", "collision", "dynamic_violation", "support_distance", "mode_escape"]:
                    diffs = []
                    for seed in SEEDS:
                        tv = [r for r in seed_rows if r["split"] == split and r["method"] == target and int(r["seed"]) == seed]
                        rv = [r for r in seed_rows if r["split"] == split and r["method"] == reference and int(r["seed"]) == seed]
                        if tv and rv:
                            diffs.append(float(tv[0][metric]) - float(rv[0][metric]))
                    if diffs:
                        pairs.append(
                            {
                                "split": split,
                                "target": target,
                                "reference": reference,
                                "metric": metric,
                                "mean_diff": f"{np.mean(diffs):.5f}",
                                "ci95": f"{ci95(diffs):.5f}",
                                "target_better_seeds": sum(1 for d in diffs if d > 0),
                                "seeds": len(diffs),
                            }
                        )
    return pairs


def run_main():
    rows = []
    support_rows = []
    splits = [
        "supported_single_mode",
        "supported_multimodal_shift",
        "rare_mode_recovery",
        "off_support_corridor",
        "deceptive_energy_basin",
    ]
    for split in splits:
        for seed in SEEDS:
            for task_id in range(TASKS_PER_SPLIT_SEED):
                task = make_task(split, seed, task_id)
                mode_counts = {m: task.demo_modes.count(m) for m in sorted(set(task.demo_modes))}
                support_rows.append(
                    {
                        "split": split,
                        "seed": seed,
                        "task_id": task_id,
                        "target_mode": task.target_mode,
                        "target_support_frequency": f"{task.support_frequency:.4f}",
                        "demo_mode_counts": ";".join(f"{k}:{v}" for k, v in mode_counts.items()),
                        "obstacles": len(task.obstacles),
                        "description": task.description,
                    }
                )
                for method in METHODS:
                    rng = stable_rng("method", method, split, seed, task_id)
                    path = run_method(method, task, rng)
                    rows.append(evaluate_row(method, task, path))
            print(f"main split={split} seed={seed} rows={len(rows)}", flush=True)
    seed_rows = aggregate_seed_metrics(rows)
    metric_rows = aggregate_metrics(seed_rows)
    pair_rows = pairwise_stats(seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "training_summary.csv", support_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    return rows, seed_rows, metric_rows, pair_rows


ABLATIONS = {
    "full_energy_guided": {"samples": 8, "guidance": 1.0, "projection": False, "prior": True, "collision": None},
    "no_task_gradient": {"samples": 8, "guidance": 0.0, "projection": False, "prior": True, "collision": None},
    "rerank_only": {"samples": 10, "guidance": 0.0, "projection": False, "prior": True, "collision": None},
    "no_safety_projection": {"samples": 10, "guidance": 0.82, "projection": False, "prior": True, "collision": 24.0},
    "strong_projection": {"samples": 10, "guidance": 0.82, "projection": True, "prior": True, "collision": 24.0},
    "no_prior_score": {"samples": 8, "guidance": 1.0, "projection": False, "prior": False, "collision": None},
    "overguidance": {"samples": 8, "guidance": 2.2, "projection": False, "prior": True, "collision": None},
}


def run_ablation():
    rows = []
    for split in ["rare_mode_recovery", "off_support_corridor"]:
        for seed in SEEDS:
            for task_id in range(TASKS_PER_SPLIT_SEED):
                task = make_task(split, seed, task_id)
                for name, cfg in ABLATIONS.items():
                    rng = stable_rng("ablation", name, split, seed, task_id)
                    path = sample_diffusion(
                        task,
                        rng,
                        samples=cfg["samples"],
                        steps=10,
                        guidance_weight=cfg["guidance"],
                        projection=cfg["projection"],
                        use_prior=cfg["prior"],
                        collision_weight=cfg["collision"],
                        choose_energy=True,
                    )
                    row = evaluate_row(name, task, path)
                    row["ablation"] = name
                    rows.append(row)
            print(f"ablation split={split} seed={seed} rows={len(rows)}", flush=True)
    summary = []
    for name in ABLATIONS:
        vals = [r for r in rows if r["ablation"] == name]
        for split in ["rare_mode_recovery", "off_support_corridor"]:
            split_vals = [r for r in vals if r["split"] == split]
            seed_means = []
            for seed in SEEDS:
                seed_vals = [r for r in split_vals if int(r["seed"]) == seed]
                seed_means.append(np.mean([int(r["success"]) for r in seed_vals]))
            summary.append(
                {
                    "split": split,
                    "ablation": name,
                    "success": f"{np.mean(seed_means):.5f}",
                    "ci95": f"{ci95(seed_means):.5f}",
                    "collision": f"{np.mean([int(r['collision']) for r in split_vals]):.5f}",
                    "support_distance": f"{np.mean([float(r['support_distance']) for r in split_vals]):.5f}",
                    "mode_escape": f"{np.mean([int(r['mode_escape']) for r in split_vals]):.5f}",
                    "rows": len(split_vals),
                }
            )
    write_csv(RESULTS / "ablation_rollouts.csv", rows)
    write_csv(RESULTS / "ablation_metrics.csv", summary)
    return rows, summary


def run_stress():
    stress_rows = []
    raw_rows = []
    methods = ["energy_rerank_unguided", "energy_guided_diffusion", "strong_guided_diffusion", "cem_trajectory_optimizer", "grid_oracle"]
    levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for level in levels:
        support_frequency = max(0.0, 0.50 * (1.0 - level))
        for seed in SEEDS:
            for task_id in range(STRESS_TASKS_PER_SEED):
                task = make_task("support_gap_sweep", seed, task_id, support_override=support_frequency)
                for method in methods:
                    rng = stable_rng("stress", method, seed, task_id, int(level * 100))
                    path = run_method(method, task, rng)
                    row = evaluate_row(method, task, path)
                    row["stress_level"] = f"{level:.1f}"
                    raw_rows.append(row)
            print(f"stress level={level:.1f} seed={seed} rows={len(raw_rows)}", flush=True)
    for level in levels:
        for method in methods:
            vals = [r for r in raw_rows if r["stress_level"] == f"{level:.1f}" and r["method"] == method]
            seed_means = []
            for seed in SEEDS:
                seed_vals = [r for r in vals if int(r["seed"]) == seed]
                seed_means.append(np.mean([int(r["success"]) for r in seed_vals]))
            stress_rows.append(
                {
                    "stress_level": f"{level:.1f}",
                    "support_frequency": f"{max(0.0, 0.50 * (1.0 - level)):.3f}",
                    "method": method,
                    "success": f"{np.mean(seed_means):.5f}",
                    "ci95": f"{ci95(seed_means):.5f}",
                    "collision": f"{np.mean([int(r['collision']) for r in vals]):.5f}",
                    "mode_escape": f"{np.mean([int(r['mode_escape']) for r in vals]):.5f}",
                    "rows": len(vals),
                }
            )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw_rows)
    write_csv(RESULTS / "stress_sweep.csv", stress_rows)
    write_csv(FIGURES / "stress_curve_data.csv", stress_rows)
    return raw_rows, stress_rows


def decisive_value(metric_rows, split, method, metric="success"):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        return None
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows):
    off_guided = decisive_value(metric_rows, "off_support_corridor", "energy_guided_diffusion")
    off_rerank = decisive_value(metric_rows, "off_support_corridor", "energy_rerank_unguided")
    off_cem = decisive_value(metric_rows, "off_support_corridor", "cem_trajectory_optimizer")
    rare_guided = decisive_value(metric_rows, "rare_mode_recovery", "energy_guided_diffusion")
    deceptive_guided_collision = decisive_value(metric_rows, "deceptive_energy_basin", "energy_guided_diffusion", "collision")
    off_pair_rerank = [
        r
        for r in pair_rows
        if r["split"] == "off_support_corridor"
        and r["target"] == "energy_guided_diffusion"
        and r["reference"] == "energy_rerank_unguided"
        and r["metric"] == "success"
    ][0]
    off_pair_cem = [
        r
        for r in pair_rows
        if r["split"] == "off_support_corridor"
        and r["target"] == "energy_guided_diffusion"
        and r["reference"] == "cem_trajectory_optimizer"
        and r["metric"] == "success"
    ][0]
    terminal = "KILL_ARCHIVE"
    reason = (
        "energy_guided_diffusion does not show safe off-support behavior: "
        f"off_support success={off_guided[0]:.3f} versus rerank={off_rerank[0]:.3f} "
        f"and CEM={off_cem[0]:.3f}; paired diff vs rerank={off_pair_rerank['mean_diff']} "
        f"and vs CEM={off_pair_cem['mean_diff']}."
    )
    if (
        off_guided[0] > off_rerank[0] + 0.08
        and off_guided[0] >= off_cem[0] - 0.05
        and float(off_pair_cem["mean_diff"]) > -0.06
    ):
        terminal = "STRONG_REVISE"
        reason = "guided diffusion clears reranking and nearly matches CEM locally, but still lacks real robot or external benchmark validation."

    failure_counts = {}
    for row in rollout_rows:
        if row["method"] == "energy_guided_diffusion":
            failure_counts[row["failure_label"]] = failure_counts.get(row["failure_label"], 0) + 1
    top_failures = sorted(failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 78 energy_guided_diffusion_policy_limits trajectory diffusion-limit rebuild\n")
        f.write(f"Terminal recommendation: {terminal}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Main rollout rows: {len(rollout_rows)}\n")
        f.write("Seeds: " + str(SEEDS) + "\n")
        f.write("\nDecisive off-support corridor:\n")
        for method in METHODS:
            val = decisive_value(metric_rows, "off_support_corridor", method)
            col = decisive_value(metric_rows, "off_support_corridor", method, "collision")
            dist = decisive_value(metric_rows, "off_support_corridor", method, "support_distance")
            f.write(
                f"{method} success={val[0]:.5f} ci95={val[1]:.5f} "
                f"collision={col[0]:.5f} support_distance={dist[0]:.5f}\n"
            )
        f.write("\nRare-mode recovery:\n")
        f.write(f"energy_guided_diffusion success={rare_guided[0]:.5f} ci95={rare_guided[1]:.5f}\n")
        f.write("\nDeceptive energy basin:\n")
        f.write(
            f"energy_guided_diffusion collision={deceptive_guided_collision[0]:.5f} "
            f"ci95={deceptive_guided_collision[1]:.5f}\n"
        )
        f.write("\nPairwise off-support success:\n")
        f.write(f"vs rerank mean_diff={off_pair_rerank['mean_diff']} ci95={off_pair_rerank['ci95']}\n")
        f.write(f"vs cem mean_diff={off_pair_cem['mean_diff']} ci95={off_pair_cem['ci95']}\n")
        f.write("\nAblation summary off-support:\n")
        for row in ablation_summary:
            if row["split"] == "off_support_corridor":
                f.write(
                    f"{row['ablation']} success={row['success']} ci95={row['ci95']} "
                    f"collision={row['collision']} mode_escape={row['mode_escape']}\n"
                )
        f.write("\nStress level 1.0:\n")
        for row in stress_summary:
            if row["stress_level"] == "1.0":
                f.write(
                    f"{row['method']} success={row['success']} ci95={row['ci95']} "
                    f"mode_escape={row['mode_escape']}\n"
                )
        f.write("\nTop guided failure labels:\n")
        for label, count in top_failures:
            f.write(f"{label}: {count}\n")
    write_negative_cases(rollout_rows)
    return terminal


def write_negative_cases(rollout_rows):
    lessons = {
        "collision_shortcut_from_guidance": "task-energy gradients stayed inside the blocked demonstration corridor instead of creating a new safe homotopy",
        "collision_or_blocked_prior_mode": "reranking and prior sampling cannot repair a blocked mode when the safe mode is absent from demonstrations",
        "prior_trap_wrong_homotopy": "rare or absent support leaves the sampler in the wrong homotopy class",
        "unsafe_off_support_escape": "leaving the prior support without a stronger planner produces unsafe trajectories",
        "goal_miss": "safety projection can remove collisions while still failing to reach the goal",
        "dynamic_violation": "aggressive guidance violates trajectory smoothness or velocity constraints",
    }
    failures = [
        r
        for r in rollout_rows
        if r["success"] == 0
        and r["method"]
        in {"energy_guided_diffusion", "strong_guided_diffusion", "energy_rerank_unguided", "unguided_diffusion_prior"}
    ]
    failures = sorted(
        failures,
        key=lambda r: (
            r["split"] != "off_support_corridor",
            r["method"] != "energy_guided_diffusion",
            int(r["seed"]),
            int(r["task_id"]),
            r["method"],
        ),
    )
    rows = []
    seen = set()
    for r in failures:
        key = (r["split"], r["method"], r["failure_label"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "split": r["split"],
                "seed": r["seed"],
                "task_id": r["task_id"],
                "method": r["method"],
                "failure_label": r["failure_label"],
                "collision": r["collision"],
                "dynamic_violation": r["dynamic_violation"],
                "support_distance": r["support_distance"],
                "mode": r["mode"],
                "target_mode": r["target_mode"],
                "lesson": lessons.get(r["failure_label"], "negative case retained for audit"),
            }
        )
        if len(rows) >= 12:
            break
    write_csv(RESULTS / "negative_cases.csv", rows)


def plot_outputs(metric_rows, ablation_summary, stress_summary):
    split = "off_support_corridor"
    methods = METHODS
    vals = [decisive_value(metric_rows, split, m, "success")[0] for m in methods]
    errs = [decisive_value(metric_rows, split, m, "success")[1] for m in methods]
    labels = [m.replace("_", "\n") for m in methods]
    plt.figure(figsize=(11, 4.8))
    plt.bar(range(len(methods)), vals, yerr=errs, color=["#7a7f86", "#91a7ff", "#5c7cfa", "#ff922b", "#e8590c", "#2f9e44", "#087f5b"], capsize=3)
    plt.xticks(range(len(methods)), labels, fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("success")
    plt.title("Off-support corridor: guidance cannot create absent safe mode")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_success.png", dpi=220)
    plt.close()

    vals = [decisive_value(metric_rows, split, m, "support_distance")[0] for m in methods]
    plt.figure(figsize=(10, 4.6))
    plt.bar(range(len(methods)), vals, color="#4c6ef5")
    plt.xticks(range(len(methods)), labels, fontsize=8)
    plt.ylabel("nearest-demo trajectory distance")
    plt.title("Selected trajectory support distance")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_support_distance.png", dpi=220)
    plt.close()

    off = [r for r in ablation_summary if r["split"] == "off_support_corridor"]
    plt.figure(figsize=(10.5, 4.8))
    plt.bar(range(len(off)), [float(r["success"]) for r in off], yerr=[float(r["ci95"]) for r in off], color="#f08c00", capsize=3)
    plt.xticks(range(len(off)), [r["ablation"].replace("_", "\n") for r in off], fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("success")
    plt.title("Ablations on off-support corridor")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8.8, 5.0))
    for method in ["energy_rerank_unguided", "energy_guided_diffusion", "strong_guided_diffusion", "cem_trajectory_optimizer", "grid_oracle"]:
        rows = [r for r in stress_summary if r["method"] == method]
        rows = sorted(rows, key=lambda r: float(r["stress_level"]))
        x = [float(r["stress_level"]) for r in rows]
        y = [float(r["success"]) for r in rows]
        e = [float(r["ci95"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, label=method)
    plt.xlabel("support-gap stress (1.0 means target mode absent)")
    plt.ylabel("success")
    plt.ylim(0, 1.05)
    plt.title("Support-gap stress sweep")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_stress_sweep.png", dpi=220)
    plt.close()


def main():
    rollout_rows, seed_rows, metric_rows, pair_rows = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    terminal = write_summary(metric_rows, pair_rows, ablation_summary, stress_summary, rollout_rows)
    plot_outputs(metric_rows, ablation_summary, stress_summary)
    print(f"terminal={terminal}")
    print(f"main_rollouts={len(rollout_rows)} ablation_rollouts={len(ablation_rows)} stress_rollouts={len(stress_raw)}")
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
