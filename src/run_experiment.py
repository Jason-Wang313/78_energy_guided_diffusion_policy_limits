import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_SEED = 78012026
QUICK_MODE = os.getenv("PAPER78_QUICK", "0") == "1"


def int_env(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def float_list_env(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


SEED_COUNT = 1 if QUICK_MODE else int_env("PAPER78_SEED_COUNT", 8)
SEEDS = list(range(SEED_COUNT))
TASKS_PER_SPLIT_SEED = 3 if QUICK_MODE else int_env("PAPER78_TASKS_PER_SPLIT_SEED", 12)
ABLATION_TASKS_PER_SPLIT_SEED = 2 if QUICK_MODE else int_env("PAPER78_ABLATION_TASKS", 10)
STRESS_TASKS_PER_SEED = 2 if QUICK_MODE else int_env("PAPER78_STRESS_TASKS", 8)
FIXED_RISK_TASKS_PER_SEED = 2 if QUICK_MODE else int_env("PAPER78_FIXED_RISK_TASKS", 8)
T = 21 if QUICK_MODE else int_env("PAPER78_HORIZON", 31)
DEMO_COUNT = 24 if QUICK_MODE else int_env("PAPER78_DEMO_COUNT", 48)
ROBOT_RADIUS = 0.045
VELOCITY_LIMIT = 0.205
ACCEL_LIMIT = 0.105
REFERENCE_METHOD = "support_aware_energy_bridge_v5"
MAIN_SPLITS = [
    "supported_single_mode",
    "supported_multimodal_shift",
    "rare_mode_recovery",
    "off_support_corridor",
    "deceptive_energy_basin",
]
HARD_SPLITS = ["rare_mode_recovery", "off_support_corridor", "deceptive_energy_basin"]
STRESS_LEVELS = [round(x, 2) for x in (np.linspace(0.0, 1.40, 3 if QUICK_MODE else 8))]
RISK_BUDGETS = float_list_env("PAPER78_RISK_BUDGETS", [0.0, 0.02, 0.05, 0.10])

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


def risk_proxy_from_stats(stats, support_dist):
    clearance_risk = max(0.0, 0.040 - stats["min_clearance"]) * 1.8
    speed_risk = max(0.0, stats["max_speed"] - 0.90 * VELOCITY_LIMIT) * 2.2
    accel_risk = max(0.0, stats["max_accel"] - 0.90 * ACCEL_LIMIT) * 1.4
    support_risk = max(0.0, support_dist - 0.52) * 0.035
    hard_failure = 1.0 if stats["collision"] or stats["dynamic_violation"] else 0.0
    return float(max(hard_failure, clearance_risk + speed_risk + accel_risk + support_risk))


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


def smooth_path(path, passes=2):
    out = path.copy()
    for _ in range(passes):
        old = out.copy()
        out[1:-1] = 0.20 * old[:-2] + 0.60 * old[1:-1] + 0.20 * old[2:]
    return out


def candidate_modes():
    return ["upper", "lower", "wide_upper", "wide_lower", "straight"]


def bridge_candidates(task, rng, jitter=0.018):
    candidates = []
    for mode in candidate_modes():
        candidates.append(make_curve(task.start, task.goal, mode, rng, noise=jitter))
        candidates.append(project_safety(make_curve(task.start, task.goal, mode, rng, noise=jitter * 0.6), task))
    return candidates


def score_path(path, task, support_weight=0.0, collision_weight=90.0, velocity_weight=80.0):
    stats = trajectory_stats(path, task)
    support_dist = support_distance(path, task.demos)
    penalty = 0.0
    if stats["collision"]:
        penalty += 140.0
    if stats["dynamic_violation"]:
        penalty += 60.0
    penalty += support_weight * max(0.0, support_dist - 0.18) ** 2
    penalty += 15.0 * risk_proxy_from_stats(stats, support_dist)
    return task_energy(path, task, collision_weight=collision_weight, velocity_weight=velocity_weight) + penalty


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
    return_candidates=False,
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

    if return_candidates:
        return candidates
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


def cem_optimizer(task, rng, population=16, iterations=3, elite_frac=0.24):
    seeds = []
    for mode in ["upper", "lower", "wide_upper", "wide_lower", "straight"]:
        seeds.append(make_curve(task.start, task.goal, mode, rng, noise=0.03))
    for _ in range(5):
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


def chomp_like_optimizer(task, rng, starts=None, steps=14):
    candidates = list(starts or [])
    candidates += bridge_candidates(task, rng, jitter=0.012)
    best = None
    best_score = float("inf")
    for seed_path in candidates[:8]:
        path = seed_path.copy()
        for k in range(steps):
            grad = energy_gradient(path, task, collision_weight=70.0, velocity_weight=70.0)
            norm = float(np.linalg.norm(grad[1:-1]))
            if norm > 25.0:
                grad[1:-1] *= 25.0 / norm
            path[1:-1] -= (0.008 * (0.96**k)) * grad[1:-1]
            path = smooth_path(path, passes=1)
            path = project_safety(path, task)
            path[:, 0] = np.clip(path[:, 0], -1.35, 1.35)
            path[:, 1] = np.clip(path[:, 1], -1.60, 1.60)
            path[0] = task.start
            path[-1] = task.goal
        score = score_path(path, task, support_weight=0.02)
        if score < best_score:
            best_score = score
            best = path.copy()
    return project_safety(best, task)


def waypoint_planner(task, rng, mode_subset=None, support_weight=0.0):
    modes = mode_subset or candidate_modes()
    candidates = []
    for mode in modes:
        for noise in [0.004, 0.012, 0.024]:
            candidates.append(project_safety(make_curve(task.start, task.goal, mode, rng, noise=noise), task))
    scores = [score_path(c, task, support_weight=support_weight) for c in candidates]
    return candidates[int(np.argmin(scores))]


def diffusion_cem_hybrid(task, rng):
    diffusion = sample_diffusion(
        task,
        rng,
        samples=8,
        steps=8,
        guidance_weight=0.70,
        collision_weight=34.0,
        projection=True,
        choose_energy=True,
    )
    local = chomp_like_optimizer(task, rng, starts=[diffusion], steps=8)
    cem = cem_optimizer(task, rng, population=12, iterations=2, elite_frac=0.25)
    candidates = [diffusion, local, cem]
    scores = [score_path(c, task, support_weight=0.01) for c in candidates]
    return candidates[int(np.argmin(scores))]


def mode_diverse_diffusion(task, rng):
    candidates = []
    candidates.extend(
        sample_diffusion(
            task,
            rng,
            samples=6,
            steps=8,
            guidance_weight=0.0,
            projection=False,
            choose_energy=True,
            return_candidates=True,
        )
    )
    candidates.extend(bridge_candidates(task, rng, jitter=0.030))
    scores = [score_path(c, task, support_weight=0.03, collision_weight=60.0) for c in candidates]
    return project_safety(candidates[int(np.argmin(scores))], task)


def support_aware_energy_bridge(task, rng):
    candidates = []
    candidates.extend(
        sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.68,
            collision_weight=max(28.0, task.guidance_collision_weight),
            projection=True,
            choose_energy=True,
            return_candidates=True,
        )
    )
    candidates.extend(bridge_candidates(task, rng, jitter=0.014))
    repaired = []
    for c in candidates[:10]:
        local = c.copy()
        for _ in range(4):
            grad = energy_gradient(local, task, collision_weight=58.0, velocity_weight=62.0)
            norm = float(np.linalg.norm(grad[1:-1]))
            if norm > 18.0:
                grad[1:-1] *= 18.0 / norm
            local[1:-1] -= 0.0065 * grad[1:-1]
            local = smooth_path(local, passes=1)
            local = project_safety(local, task)
            local[0] = task.start
            local[-1] = task.goal
        repaired.append(local)
    candidates.extend(repaired)
    scored = sorted(candidates, key=lambda c: score_path(c, task, support_weight=0.065, collision_weight=80.0, velocity_weight=80.0))
    return project_safety(scored[0], task)


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
    "support_projected_guidance",
    "mode_diverse_diffusion",
    "diffusion_cem_hybrid",
    "chomp_like_optimizer",
    "cem_trajectory_optimizer",
    "graph_search_planner",
    "support_aware_energy_bridge_v5",
    "grid_oracle",
]


def run_method(method, task, rng):
    if method == "behavior_clone_nearest":
        return behavior_clone_nearest(task)
    if method == "unguided_diffusion_prior":
        return sample_diffusion(task, rng, samples=5, steps=8, guidance_weight=0.0, choose_energy=False)
    if method == "energy_rerank_unguided":
        return sample_diffusion(task, rng, samples=6, steps=8, guidance_weight=0.0, choose_energy=True)
    if method == "energy_guided_diffusion":
        return sample_diffusion(
            task,
            rng,
            samples=6,
            steps=8,
            guidance_weight=1.0,
            collision_weight=task.guidance_collision_weight,
            projection=False,
            choose_energy=True,
        )
    if method == "strong_guided_diffusion":
        return sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.82,
            collision_weight=max(24.0, task.guidance_collision_weight),
            projection=True,
            choose_energy=True,
        )
    if method == "support_projected_guidance":
        return sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.70,
            collision_weight=36.0,
            projection=True,
            choose_energy=True,
        )
    if method == "mode_diverse_diffusion":
        return mode_diverse_diffusion(task, rng)
    if method == "diffusion_cem_hybrid":
        return diffusion_cem_hybrid(task, rng)
    if method == "chomp_like_optimizer":
        return chomp_like_optimizer(task, rng)
    if method == "cem_trajectory_optimizer":
        return cem_optimizer(task, rng)
    if method == "graph_search_planner":
        return waypoint_planner(task, rng, support_weight=0.0)
    if method == "support_aware_energy_bridge_v5":
        return support_aware_energy_bridge(task, rng)
    if method == "grid_oracle":
        return grid_oracle(task, rng)
    raise ValueError(method)


def failure_label(method, task, stats, mode, support_dist):
    if stats["success"]:
        return "success"
    if stats["collision"]:
        if (
            "guid" in method
            or "bridge" in method
            or method
            in {"full_energy_guided", "no_safety_projection", "strong_projection", "overguidance"}
        ):
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
    risk_proxy = risk_proxy_from_stats(stats, support_dist)
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
        "risk_proxy": f"{risk_proxy:.5f}",
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


def aggregate_seed_metrics(rows, methods=None):
    methods = methods or METHODS
    output = []
    for split in sorted({r["split"] for r in rows}):
        for method in methods:
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
                        "risk_proxy": f"{np.mean([float(v['risk_proxy']) for v in vals]):.5f}",
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
            for metric in [
                "success",
                "collision",
                "dynamic_violation",
                "mean_energy",
                "support_distance",
                "risk_proxy",
                "mode_escape",
            ]:
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
    for split in sorted({r["split"] for r in seed_rows}):
        for comparison in [m for m in METHODS if m != REFERENCE_METHOD]:
            success_diffs = []
            collision_reductions = []
            dynamic_reductions = []
            support_diffs = []
            risk_reductions = []
            mode_escape_diffs = []
            for seed in SEEDS:
                tv = [
                    r
                    for r in seed_rows
                    if r["split"] == split and r["method"] == REFERENCE_METHOD and int(r["seed"]) == seed
                ]
                rv = [r for r in seed_rows if r["split"] == split and r["method"] == comparison and int(r["seed"]) == seed]
                if tv and rv:
                    ref = tv[0]
                    comp = rv[0]
                    success_diffs.append(float(ref["success"]) - float(comp["success"]))
                    collision_reductions.append(float(comp["collision"]) - float(ref["collision"]))
                    dynamic_reductions.append(float(comp["dynamic_violation"]) - float(ref["dynamic_violation"]))
                    support_diffs.append(float(ref["support_distance"]) - float(comp["support_distance"]))
                    risk_reductions.append(float(comp["risk_proxy"]) - float(ref["risk_proxy"]))
                    mode_escape_diffs.append(float(ref["mode_escape"]) - float(comp["mode_escape"]))
            if success_diffs:
                pairs.append(
                    {
                        "split": split,
                        "reference": REFERENCE_METHOD,
                        "comparison": comparison,
                        "paired_success_diff": f"{np.mean(success_diffs):.5f}",
                        "ci95_success_diff": f"{ci95(success_diffs):.5f}",
                        "paired_collision_reduction": f"{np.mean(collision_reductions):.5f}",
                        "paired_dynamic_reduction": f"{np.mean(dynamic_reductions):.5f}",
                        "paired_support_distance_diff": f"{np.mean(support_diffs):.5f}",
                        "paired_risk_reduction": f"{np.mean(risk_reductions):.5f}",
                        "paired_mode_escape_diff": f"{np.mean(mode_escape_diffs):.5f}",
                        "reference_better_seeds": sum(1 for d in success_diffs if d > 0),
                        "seeds": len(success_diffs),
                    }
                )
    return pairs


def aggregate_hard_seed_metrics(seed_rows):
    output = []
    for method in METHODS:
        for seed in SEEDS:
            vals = [
                r
                for r in seed_rows
                if r["split"] in HARD_SPLITS and r["method"] == method and int(r["seed"]) == seed
            ]
            if not vals:
                continue
            output.append(
                {
                    "split": "aggregate_hard_regime",
                    "method": method,
                    "seed": seed,
                    "tasks": sum(int(v["tasks"]) for v in vals),
                    "success": f"{np.mean([float(v['success']) for v in vals]):.5f}",
                    "collision": f"{np.mean([float(v['collision']) for v in vals]):.5f}",
                    "dynamic_violation": f"{np.mean([float(v['dynamic_violation']) for v in vals]):.5f}",
                    "mean_energy": f"{np.mean([float(v['mean_energy']) for v in vals]):.5f}",
                    "support_distance": f"{np.mean([float(v['support_distance']) for v in vals]):.5f}",
                    "risk_proxy": f"{np.mean([float(v['risk_proxy']) for v in vals]):.5f}",
                    "mode_escape": f"{np.mean([float(v['mode_escape']) for v in vals]):.5f}",
                }
            )
    return output


def run_main():
    rows = []
    support_rows = []
    for split in MAIN_SPLITS:
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
    aggregate_seed_rows = aggregate_hard_seed_metrics(seed_rows)
    aggregate_metric_rows = aggregate_metrics(aggregate_seed_rows)
    aggregate_pair_rows = pairwise_stats(aggregate_seed_rows)
    write_csv(RESULTS / "rollouts.csv", rows)
    write_csv(RESULTS / "training_summary.csv", support_rows)
    write_csv(RESULTS / "raw_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "metrics.csv", metric_rows)
    write_csv(RESULTS / "pairwise_stats.csv", pair_rows)
    write_csv(RESULTS / "aggregate_seed_metrics.csv", aggregate_seed_rows)
    write_csv(RESULTS / "aggregate_metrics.csv", aggregate_metric_rows)
    write_csv(RESULTS / "aggregate_pairwise_stats.csv", aggregate_pair_rows)
    return rows, seed_rows, metric_rows, pair_rows, aggregate_seed_rows, aggregate_metric_rows, aggregate_pair_rows


ABLATIONS = [
    "support_aware_v5_full",
    "no_bridge_proposals",
    "no_support_awareness",
    "no_safety_projection",
    "no_mode_diversity",
    "no_energy_gradient",
    "no_prior_score",
    "rerank_only",
    "sample_count_only",
    "optimizer_handoff",
]


def run_ablation_variant(name, task, rng):
    if name == "support_aware_v5_full":
        return support_aware_energy_bridge(task, rng)
    if name == "no_bridge_proposals":
        return sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.68,
            collision_weight=34.0,
            projection=True,
            choose_energy=True,
        )
    if name == "no_support_awareness":
        candidates = bridge_candidates(task, rng, jitter=0.014)
        candidates.extend(
            sample_diffusion(
                task,
                rng,
                samples=7,
                steps=8,
                guidance_weight=0.68,
                collision_weight=34.0,
                projection=True,
                choose_energy=True,
                return_candidates=True,
            )
        )
        return project_safety(min(candidates, key=lambda c: score_path(c, task, support_weight=0.0)), task)
    if name == "no_safety_projection":
        return sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.88,
            collision_weight=20.0,
            projection=False,
            choose_energy=True,
        )
    if name == "no_mode_diversity":
        return sample_diffusion(
            task,
            rng,
            samples=7,
            steps=8,
            guidance_weight=0.68,
            collision_weight=34.0,
            projection=True,
            choose_energy=True,
        )
    if name == "no_energy_gradient":
        return waypoint_planner(task, rng, support_weight=0.05)
    if name == "no_prior_score":
        return chomp_like_optimizer(task, rng, starts=bridge_candidates(task, rng, jitter=0.020), steps=10)
    if name == "rerank_only":
        return sample_diffusion(task, rng, samples=8, steps=8, guidance_weight=0.0, choose_energy=True)
    if name == "sample_count_only":
        return sample_diffusion(task, rng, samples=12, steps=8, guidance_weight=0.0, projection=True, choose_energy=True)
    if name == "optimizer_handoff":
        return diffusion_cem_hybrid(task, rng)
    raise ValueError(name)


def run_ablation():
    rows = []
    for split in ["rare_mode_recovery", "off_support_corridor"]:
        for seed in SEEDS:
            for task_id in range(ABLATION_TASKS_PER_SPLIT_SEED):
                task = make_task(split, seed, task_id)
                for name in ABLATIONS:
                    rng = stable_rng("ablation", name, split, seed, task_id)
                    path = run_ablation_variant(name, task, rng)
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
    seed_summary = aggregate_seed_metrics([{**r, "method": r["ablation"]} for r in rows], methods=ABLATIONS)
    write_csv(RESULTS / "ablation_seed_metrics.csv", seed_summary)
    return rows, summary


def run_stress():
    stress_rows = []
    raw_rows = []
    methods = [
        "energy_rerank_unguided",
        "energy_guided_diffusion",
        "strong_guided_diffusion",
        "support_projected_guidance",
        "mode_diverse_diffusion",
        "diffusion_cem_hybrid",
        "chomp_like_optimizer",
        "cem_trajectory_optimizer",
        "graph_search_planner",
        "support_aware_energy_bridge_v5",
    ]
    for level in STRESS_LEVELS:
        support_frequency = max(0.0, 0.50 * (1.0 - min(1.0, level)))
        for seed in SEEDS:
            for task_id in range(STRESS_TASKS_PER_SEED):
                task = make_task("support_gap_sweep", seed, task_id, support_override=support_frequency)
                for method in methods:
                    rng = stable_rng("stress", method, seed, task_id, int(level * 100))
                    path = run_method(method, task, rng)
                    row = evaluate_row(method, task, path)
                    row["stress_level"] = f"{level:.2f}"
                    raw_rows.append(row)
            print(f"stress level={level:.2f} seed={seed} rows={len(raw_rows)}", flush=True)
    for level in STRESS_LEVELS:
        for method in methods:
            vals = [r for r in raw_rows if r["stress_level"] == f"{level:.2f}" and r["method"] == method]
            seed_means = []
            for seed in SEEDS:
                seed_vals = [r for r in vals if int(r["seed"]) == seed]
                seed_means.append(np.mean([int(r["success"]) for r in seed_vals]))
            stress_rows.append(
                {
                    "stress_level": f"{level:.2f}",
                    "support_frequency": f"{max(0.0, 0.50 * (1.0 - min(1.0, level))):.3f}",
                    "method": method,
                    "success": f"{np.mean(seed_means):.5f}",
                    "ci95": f"{ci95(seed_means):.5f}",
                    "collision": f"{np.mean([int(r['collision']) for r in vals]):.5f}",
                    "risk_proxy": f"{np.mean([float(r['risk_proxy']) for r in vals]):.5f}",
                    "mode_escape": f"{np.mean([int(r['mode_escape']) for r in vals]):.5f}",
                    "rows": len(vals),
                }
            )
    write_csv(RESULTS / "stress_sweep_raw.csv", raw_rows)
    write_csv(RESULTS / "stress_sweep.csv", stress_rows)
    write_csv(FIGURES / "stress_curve_data.csv", stress_rows)
    return raw_rows, stress_rows


def run_fixed_risk():
    methods = [
        "energy_rerank_unguided",
        "energy_guided_diffusion",
        "strong_guided_diffusion",
        "support_projected_guidance",
        "mode_diverse_diffusion",
        "diffusion_cem_hybrid",
        "cem_trajectory_optimizer",
        "support_aware_energy_bridge_v5",
    ]
    raw_rows = []
    for budget in RISK_BUDGETS:
        for seed in SEEDS:
            for task_id in range(FIXED_RISK_TASKS_PER_SEED):
                task = make_task("off_support_corridor", seed, 4000 + task_id)
                for method in methods:
                    rng = stable_rng("fixed", method, seed, task_id, int(budget * 1000))
                    path = run_method(method, task, rng)
                    row = evaluate_row(method, task, path)
                    row["risk_budget"] = f"{budget:.2f}"
                    row["fixed_risk_success"] = int(int(row["success"]) == 1 and float(row["risk_proxy"]) <= budget)
                    raw_rows.append(row)
            print(f"fixed-risk budget={budget:.2f} seed={seed} rows={len(raw_rows)}", flush=True)

    seed_rows = []
    for budget in RISK_BUDGETS:
        for method in methods:
            for seed in SEEDS:
                vals = [
                    r
                    for r in raw_rows
                    if r["risk_budget"] == f"{budget:.2f}" and r["method"] == method and int(r["seed"]) == seed
                ]
                if not vals:
                    continue
                seed_rows.append(
                    {
                        "risk_budget": f"{budget:.2f}",
                        "method": method,
                        "seed": seed,
                        "tasks": len(vals),
                        "success": f"{np.mean([int(v['success']) for v in vals]):.5f}",
                        "fixed_risk_success": f"{np.mean([int(v['fixed_risk_success']) for v in vals]):.5f}",
                        "collision": f"{np.mean([int(v['collision']) for v in vals]):.5f}",
                        "dynamic_violation": f"{np.mean([int(v['dynamic_violation']) for v in vals]):.5f}",
                        "risk_proxy": f"{np.mean([float(v['risk_proxy']) for v in vals]):.5f}",
                        "mode_escape": f"{np.mean([int(v['mode_escape']) for v in vals]):.5f}",
                    }
                )

    metric_rows = []
    for budget in RISK_BUDGETS:
        for method in methods:
            vals = [r for r in seed_rows if r["risk_budget"] == f"{budget:.2f}" and r["method"] == method]
            if not vals:
                continue
            for metric in ["success", "fixed_risk_success", "collision", "dynamic_violation", "risk_proxy", "mode_escape"]:
                numbers = [float(v[metric]) for v in vals]
                metric_rows.append(
                    {
                        "risk_budget": f"{budget:.2f}",
                        "method": method,
                        "metric": metric,
                        "mean": f"{np.mean(numbers):.5f}",
                        "ci95": f"{ci95(numbers):.5f}",
                        "seeds": len(numbers),
                        "tasks_per_seed": vals[0]["tasks"],
                    }
                )

    pair_rows = []
    for budget in RISK_BUDGETS:
        for comparison in [m for m in methods if m != REFERENCE_METHOD]:
            diffs = []
            risk_reductions = []
            for seed in SEEDS:
                tv = [
                    r
                    for r in seed_rows
                    if r["risk_budget"] == f"{budget:.2f}" and r["method"] == REFERENCE_METHOD and int(r["seed"]) == seed
                ]
                rv = [
                    r
                    for r in seed_rows
                    if r["risk_budget"] == f"{budget:.2f}" and r["method"] == comparison and int(r["seed"]) == seed
                ]
                if tv and rv:
                    diffs.append(float(tv[0]["fixed_risk_success"]) - float(rv[0]["fixed_risk_success"]))
                    risk_reductions.append(float(rv[0]["risk_proxy"]) - float(tv[0]["risk_proxy"]))
            if diffs:
                pair_rows.append(
                    {
                        "risk_budget": f"{budget:.2f}",
                        "reference": REFERENCE_METHOD,
                        "comparison": comparison,
                        "paired_fixed_risk_success_diff": f"{np.mean(diffs):.5f}",
                        "ci95_fixed_risk_success_diff": f"{ci95(diffs):.5f}",
                        "paired_risk_reduction": f"{np.mean(risk_reductions):.5f}",
                        "reference_better_seeds": sum(1 for d in diffs if d > 0),
                        "seeds": len(diffs),
                    }
                )

    write_csv(RESULTS / "fixed_risk_raw.csv", raw_rows)
    write_csv(RESULTS / "fixed_risk_seed_metrics.csv", seed_rows)
    write_csv(RESULTS / "fixed_risk_metrics.csv", metric_rows)
    write_csv(RESULTS / "fixed_risk_pairwise.csv", pair_rows)
    write_csv(FIGURES / "fixed_risk_curve_data.csv", metric_rows)
    return raw_rows, seed_rows, metric_rows, pair_rows


def decisive_value(metric_rows, split, method, metric="success"):
    vals = [r for r in metric_rows if r["split"] == split and r["method"] == method and r["metric"] == metric]
    if not vals:
        return None
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def metric_lookup(rows, split_key, split_value, method, metric, method_key="method"):
    vals = [r for r in rows if r.get(split_key) == split_value and r[method_key] == method and r.get("metric") == metric]
    if not vals:
        return None
    return float(vals[0]["mean"]), float(vals[0]["ci95"])


def best_method(rows, split, metric="success", exclude_oracle=True):
    candidates = []
    for method in METHODS:
        if exclude_oracle and method == "grid_oracle":
            continue
        val = decisive_value(rows, split, method, metric)
        if val is not None:
            candidates.append((method, val[0], val[1]))
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0]


def write_summary(
    metric_rows,
    pair_rows,
    aggregate_metric_rows,
    aggregate_pair_rows,
    ablation_summary,
    stress_summary,
    fixed_metric_rows,
    fixed_pair_rows,
    rollout_rows,
    ablation_rows,
    stress_raw,
    fixed_raw,
):
    off_v5 = decisive_value(metric_rows, "off_support_corridor", REFERENCE_METHOD)
    off_rerank = decisive_value(metric_rows, "off_support_corridor", "energy_rerank_unguided")
    off_strong = decisive_value(metric_rows, "off_support_corridor", "strong_guided_diffusion")
    off_cem = decisive_value(metric_rows, "off_support_corridor", "cem_trajectory_optimizer")
    off_best = best_method(metric_rows, "off_support_corridor")
    agg_v5 = decisive_value(aggregate_metric_rows, "aggregate_hard_regime", REFERENCE_METHOD)
    agg_best = best_method(aggregate_metric_rows, "aggregate_hard_regime")
    pair_rerank = [
        r
        for r in pair_rows
        if r["split"] == "off_support_corridor" and r["comparison"] == "energy_rerank_unguided"
    ][0]
    pair_strong = [
        r
        for r in pair_rows
        if r["split"] == "off_support_corridor" and r["comparison"] == "strong_guided_diffusion"
    ][0]
    pair_cem = [r for r in pair_rows if r["split"] == "off_support_corridor" and r["comparison"] == "cem_trajectory_optimizer"][0]

    max_level = f"{max(float(r['stress_level']) for r in stress_summary):.2f}"
    stress_v5 = [
        r for r in stress_summary if r["stress_level"] == max_level and r["method"] == REFERENCE_METHOD
    ][0]
    stress_best = sorted(
        [r for r in stress_summary if r["stress_level"] == max_level],
        key=lambda r: float(r["success"]),
        reverse=True,
    )[0]

    full_ablation = [
        r for r in ablation_summary if r["split"] == "off_support_corridor" and r["ablation"] == "support_aware_v5_full"
    ][0]
    best_removed = sorted(
        [r for r in ablation_summary if r["split"] == "off_support_corridor" and r["ablation"] != "support_aware_v5_full"],
        key=lambda r: float(r["success"]),
        reverse=True,
    )[0]

    fixed_checks = []
    for budget in RISK_BUDGETS:
        rows = [
            r
            for r in fixed_metric_rows
            if r["risk_budget"] == f"{budget:.2f}" and r["metric"] == "fixed_risk_success"
        ]
        v5_row = [r for r in rows if r["method"] == REFERENCE_METHOD][0]
        best_row = sorted(rows, key=lambda r: float(r["mean"]), reverse=True)[0]
        fixed_checks.append((budget, float(v5_row["mean"]), float(v5_row["ci95"]), best_row["method"], float(best_row["mean"])))

    failed = []
    rerank_diff = float(pair_rerank["paired_success_diff"])
    rerank_low = rerank_diff - float(pair_rerank["ci95_success_diff"])
    if off_v5[0] <= off_rerank[0] + 0.05 or rerank_low <= 0.0:
        failed.append("off_support_rerank_margin")
    if off_v5[0] < off_strong[0] - 0.02:
        failed.append("off_support_strong_guidance")
    if off_v5[0] < off_best[1] - 0.05:
        failed.append("optimizer_or_search_gap")
    if agg_v5[0] < agg_best[1] - 0.03:
        failed.append("aggregate_hard_regime")
    if float(stress_v5["success"]) < float(stress_best["success"]) - 0.05:
        failed.append("maximum_stress")
    if any(v5 < best - 0.05 for _budget, v5, _ci, _best_method, best in fixed_checks):
        failed.append("fixed_risk")
    if float(full_ablation["success"]) <= float(best_removed["success"]) + 0.02:
        failed.append("ablation_necessity")

    terminal = "STRONG_REVISE" if not failed else "KILL_ARCHIVE"
    if terminal == "STRONG_REVISE":
        reason = (
            f"{REFERENCE_METHOD} clears the frozen local support-gap gates but remains not ICLR-main-ready "
            "without real robot or external benchmark validation."
        )
    else:
        reason = (
            f"{REFERENCE_METHOD} does not honestly clear the frozen local gates: off_support={off_v5[0]:.3f} "
            f"versus rerank={off_rerank[0]:.3f}, strong_guided={off_strong[0]:.3f}, "
            f"best_non_oracle={off_best[0]}:{off_best[1]:.3f}; paired diff vs rerank="
            f"{pair_rerank['paired_success_diff']}+/-{pair_rerank['ci95_success_diff']}; "
            f"aggregate hard-regime={agg_v5[0]:.3f} versus {agg_best[0]}:{agg_best[1]:.3f}; "
            f"max-stress={float(stress_v5['success']):.3f} versus {stress_best['method']}:{float(stress_best['success']):.3f}; "
            "fixed-risk checks: "
            + "; ".join(
                f"budget {budget:.2f}: v5={v5:.3f}, best={best_method_name}:{best:.3f}"
                for budget, v5, _ci, best_method_name, best in fixed_checks
            )
            + f"; best ablation={best_removed['ablation']}:{float(best_removed['success']):.3f}; "
            + "failed gates: "
            + ", ".join(failed)
            + "."
        )

    failure_counts = {}
    for row in rollout_rows:
        if row["method"] == REFERENCE_METHOD:
            failure_counts[row["failure_label"]] = failure_counts.get(row["failure_label"], 0) + 1
    top_failures = sorted(failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:6]

    with (RESULTS / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("Paper 78 energy_guided_diffusion_policy_limits expanded v5 support-gap rebuild\n")
        f.write(f"Terminal recommendation: {terminal}\n")
        f.write(f"Reason: {reason}\n")
        f.write(f"Main rollout rows: {len(rollout_rows)}\n")
        f.write(f"Ablation rows: {len(ablation_rows)}\n")
        f.write(f"Stress rows: {len(stress_raw)}\n")
        f.write(f"Fixed-risk rows: {len(fixed_raw)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("\nDecisive off-support corridor:\n")
        for method in METHODS:
            val = decisive_value(metric_rows, "off_support_corridor", method)
            col = decisive_value(metric_rows, "off_support_corridor", method, "collision")
            risk = decisive_value(metric_rows, "off_support_corridor", method, "risk_proxy")
            dist = decisive_value(metric_rows, "off_support_corridor", method, "support_distance")
            f.write(
                f"{method} success={val[0]:.5f} ci95={val[1]:.5f} collision={col[0]:.5f} "
                f"risk={risk[0]:.5f} support_distance={dist[0]:.5f}\n"
            )
        f.write("\nPairwise off-support success for v5 reference:\n")
        for row in [pair_rerank, pair_strong, pair_cem]:
            f.write(
                f"vs {row['comparison']} diff={row['paired_success_diff']} "
                f"ci95={row['ci95_success_diff']} better_seeds={row['reference_better_seeds']}/{row['seeds']}\n"
            )
        f.write("\nAggregate hard-regime summary:\n")
        for method in METHODS:
            val = decisive_value(aggregate_metric_rows, "aggregate_hard_regime", method)
            col = decisive_value(aggregate_metric_rows, "aggregate_hard_regime", method, "collision")
            risk = decisive_value(aggregate_metric_rows, "aggregate_hard_regime", method, "risk_proxy")
            f.write(f"{method} success={val[0]:.5f} ci95={val[1]:.5f} collision={col[0]:.5f} risk={risk[0]:.5f}\n")
        f.write("\nAblation summary off-support:\n")
        for row in ablation_summary:
            if row["split"] == "off_support_corridor":
                f.write(
                    f"{row['ablation']} success={row['success']} ci95={row['ci95']} "
                    f"collision={row['collision']} mode_escape={row['mode_escape']}\n"
                )
        f.write(f"\nStress level {max_level}:\n")
        for row in stress_summary:
            if row["stress_level"] == max_level:
                f.write(
                    f"{row['method']} success={row['success']} ci95={row['ci95']} "
                    f"risk={row['risk_proxy']} mode_escape={row['mode_escape']}\n"
                )
        f.write("\nFixed-risk off-support summary:\n")
        for budget, v5, ci, best_method_name, best in fixed_checks:
            f.write(f"budget={budget:.2f} v5={v5:.5f} ci95={ci:.5f} best={best_method_name}:{best:.5f}\n")
        f.write("\nTop v5 failure labels:\n")
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
        in {
            "energy_guided_diffusion",
            "strong_guided_diffusion",
            "support_projected_guidance",
            "mode_diverse_diffusion",
            "support_aware_energy_bridge_v5",
            "energy_rerank_unguided",
            "unguided_diffusion_prior",
        }
    ]
    failures = sorted(
        failures,
        key=lambda r: (
            r["split"] != "off_support_corridor",
            r["method"] != REFERENCE_METHOD,
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


def plot_outputs(metric_rows, ablation_summary, stress_summary, fixed_metric_rows):
    split = "off_support_corridor"
    methods = METHODS
    vals = [decisive_value(metric_rows, split, m, "success")[0] for m in methods]
    errs = [decisive_value(metric_rows, split, m, "success")[1] for m in methods]
    labels = [m.replace("_", "\n") for m in methods]
    colors = ["#7a7f86"] * len(methods)
    for i, method in enumerate(methods):
        if method == REFERENCE_METHOD:
            colors[i] = "#d9480f"
        elif method in {"cem_trajectory_optimizer", "graph_search_planner", "grid_oracle", "chomp_like_optimizer"}:
            colors[i] = "#2f9e44"
        elif "diffusion" in method or "guidance" in method or "guided" in method:
            colors[i] = "#4c6ef5"
    plt.figure(figsize=(13.5, 5.0))
    plt.bar(range(len(methods)), vals, yerr=errs, color=colors, capsize=3)
    plt.xticks(range(len(methods)), labels, fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("success")
    plt.title("Off-support corridor: support-gap trajectory success")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_success.png", dpi=220)
    plt.close()

    vals = [decisive_value(metric_rows, split, m, "support_distance")[0] for m in methods]
    plt.figure(figsize=(13.5, 4.8))
    plt.bar(range(len(methods)), vals, color=colors)
    plt.xticks(range(len(methods)), labels, fontsize=8)
    plt.ylabel("nearest-demo trajectory distance")
    plt.title("Selected trajectory support distance")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_support_distance.png", dpi=220)
    plt.close()

    off = [r for r in ablation_summary if r["split"] == "off_support_corridor"]
    plt.figure(figsize=(11.5, 4.8))
    plt.bar(range(len(off)), [float(r["success"]) for r in off], yerr=[float(r["ci95"]) for r in off], color="#f08c00", capsize=3)
    plt.xticks(range(len(off)), [r["ablation"].replace("_", "\n") for r in off], fontsize=8)
    plt.ylim(0, 1.05)
    plt.ylabel("success")
    plt.title("Ablations on off-support corridor")
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_ablation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.8, 5.2))
    for method in [
        "energy_rerank_unguided",
        "energy_guided_diffusion",
        "strong_guided_diffusion",
        "diffusion_cem_hybrid",
        "cem_trajectory_optimizer",
        "graph_search_planner",
        REFERENCE_METHOD,
    ]:
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

    plt.figure(figsize=(9.8, 5.2))
    for method in [
        "energy_rerank_unguided",
        "energy_guided_diffusion",
        "strong_guided_diffusion",
        "cem_trajectory_optimizer",
        REFERENCE_METHOD,
    ]:
        rows = [
            r
            for r in fixed_metric_rows
            if r["method"] == method and r["metric"] == "fixed_risk_success"
        ]
        rows = sorted(rows, key=lambda r: float(r["risk_budget"]))
        x = [float(r["risk_budget"]) for r in rows]
        y = [float(r["mean"]) for r in rows]
        e = [float(r["ci95"]) for r in rows]
        plt.errorbar(x, y, yerr=e, marker="o", linewidth=2, capsize=3, label=method)
    plt.xlabel("allowed collision/dynamics risk budget")
    plt.ylabel("fixed-risk success")
    plt.ylim(0, 1.05)
    plt.title("Fixed-risk off-support success")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGURES / "diffusion_limit_fixed_risk.png", dpi=220)
    plt.close()


def main():
    print(
        f"Paper78 runner quick={QUICK_MODE} seeds={SEEDS} tasks={TASKS_PER_SPLIT_SEED} "
        f"methods={len(METHODS)} demos={DEMO_COUNT} horizon={T} reference={REFERENCE_METHOD}",
        flush=True,
    )
    (
        rollout_rows,
        seed_rows,
        metric_rows,
        pair_rows,
        aggregate_seed_rows,
        aggregate_metric_rows,
        aggregate_pair_rows,
    ) = run_main()
    ablation_rows, ablation_summary = run_ablation()
    stress_raw, stress_summary = run_stress()
    fixed_raw, fixed_seed_rows, fixed_metric_rows, fixed_pair_rows = run_fixed_risk()
    terminal = write_summary(
        metric_rows,
        pair_rows,
        aggregate_metric_rows,
        aggregate_pair_rows,
        ablation_summary,
        stress_summary,
        fixed_metric_rows,
        fixed_pair_rows,
        rollout_rows,
        ablation_rows,
        stress_raw,
        fixed_raw,
    )
    plot_outputs(metric_rows, ablation_summary, stress_summary, fixed_metric_rows)
    print(f"terminal={terminal}")
    print(
        f"main_rollouts={len(rollout_rows)} aggregate_seed_rows={len(aggregate_seed_rows)} "
        f"ablation_rollouts={len(ablation_rows)} stress_rollouts={len(stress_raw)} "
        f"fixed_risk_rollouts={len(fixed_raw)}"
    )
    print(f"wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
