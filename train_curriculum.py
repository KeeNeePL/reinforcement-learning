"""
Multi-phase curriculum training for GridWorld PPO.

Each phase runs after the previous one and saves to a separate run directory
so your existing checkpoints/ and ppo_gridworld_final.zip are not overwritten.

Edit CURRICULUM_PHASES below to define your training plan.
"""

import argparse
import json
import os
from datetime import datetime

from train import args_from_namespace, normalize_berry_kwargs, resolve_model_path, run_training

# ---------------------------------------------------------------------------
# Curriculum plan — edit this section
# ---------------------------------------------------------------------------
# Each phase is a dict. Required keys: name, total_timesteps
# Training flags (all optional except where noted):
#   grid_size, grid_size_min, grid_size_max, no_enemy, obstacle_density
#   initial_learning_rate, ent_coef, gamma
#   berry_count: fixed N per episode (e.g. 1 or 2)
#   berry_count_min / berry_count_max: random count each episode (e.g. 1 and 5)
#   berry_count_range: (min, max) tuple — alternative to min/max keys
#   max_episode_steps: episode step cap before truncation (default 300)
#   resume_from: "previous" (default after phase 1), path string, or None/false for fresh start
#
CURRICULUM_PHASES = [
    {
        "name": "phase_01_no_enemy_16x16_3_berries",
        "description": "16×16, no hunter, exactly 3 berries — learn multi-pickup + exit from scratch",
        "total_timesteps": 3_000_000,
        "grid_size": 16,
        "no_enemy": True,
        "berry_count": 3,
        "initial_learning_rate": 1e-4,
        "ent_coef": 0.08,
        "gamma": 0.98,
        "obstacle_density": 0.12,
        "max_episode_steps": 900,
        "resume_from": None,
    },
    {
        "name": "phase_02_no_enemy_16x16_random_berries",
        "description": "16×16, no hunter, random 2–5 berries — learn multi-pickup + exit from scratch",
        "total_timesteps": 2_000_000,
        "grid_size": 16,
        "no_enemy": True,
        "berry_count_min": 2,
        "berry_count_max": 5,
        "initial_learning_rate": 2e-4,
        "ent_coef": 0.05,
        "gamma": 0.98,
        "obstacle_density": 0.12,
        "max_episode_steps": 700,
        "resume_from": "previous",
    },
    # {
    #     "name": "phase_03_no_enemy_16x16_random_berries",
    #     "description": "16×16, no hunter, random 2–5 berries — learn multi-pickup + exit from scratch",
    #     "total_timesteps": 2_000_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count_min": 2,
    #     "berry_count_max": 5,
    #     "initial_learning_rate": 3e-4,
    #     "ent_coef": 0.03,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "max_episode_steps": 500,
    #     "resume_from": "previous",
    # },
    # --- Incremental berry-count phases (disabled; use random-berry phase above) ---
    # {
    #     "name": "phase_0_no_enemy_16x16_1_berry",
    #     ...
    # },
    # {
    #     "name": "phase_01_no_enemy_16x16_2_berries",
    #     "description": "16×16, no hunter, exactly 2 berries — learn pickup + exit",
    #     "total_timesteps": 700_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count": 2,
    #     "initial_learning_rate": 3e-4,
    #     "ent_coef": 0.06,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "resume_from": None,
    # },
    # {
    #     "name": "phase_02_no_enemy_16x16_3_berries",
    #     "total_timesteps": 700_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count": 3,
    #     "initial_learning_rate": 2e-4,
    #     "ent_coef": 0.06,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_03_no_enemy_16x16_4_berries",
    #     "total_timesteps": 700_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count": 4,
    #     "initial_learning_rate": 2e-4,
    #     "ent_coef": 0.06,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_04_no_enemy_16x16_5_berries",
    #     "total_timesteps": 700_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count": 5,
    #     "initial_learning_rate": 2e-4,
    #     "ent_coef": 0.06,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_05_no_enemy_16x16_random_berries",
    #     "total_timesteps": 700_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count_min": 1,
    #     "berry_count_max": 5,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.06,
    #     "gamma": 0.98,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_06_no_enemy_16x16_random_berries_finetune_leaving",
    #     "total_timesteps": 300_000,
    #     "grid_size": 16,
    #     "no_enemy": True,
    #     "berry_count_min": 1,
    #     "berry_count_max": 5,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.999,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_07_with_enemy_16x16_1_berry",
    #     "description": "16x16, with hunter, random berries",
    #     "total_timesteps": 1000_000,
    #     "grid_size": 16,
    #     "no_enemy": False,
    #     "berry_count": 1,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.97,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_08_with_enemy_16x16_2_berries",
    #     "description": "16x16, with hunter, random berries",
    #     "total_timesteps": 500_000,
    #     "grid_size": 16,
    #     "no_enemy": False,
    #     "berry_count": 2,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.99,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_09_with_enemy_16x16_3_berries",
    #     "description": "16x16, with hunter, random berries",
    #     "total_timesteps": 500_000,
    #     "grid_size": 16,
    #     "no_enemy": False,
    #     "berry_count": 3,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.99,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_10_with_enemy_16x16_4_berries",
    #     "description": "16x16, with hunter, random berries",
    #     "total_timesteps": 500_000,
    #     "grid_size": 16,
    #     "no_enemy": False,
    #     "berry_count": 4,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.99,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    # {
    #     "name": "phase_11_with_enemy_16x16_5_berries",
    #     "description": "16x16, with hunter, random berries",
    #     "total_timesteps": 500_000,
    #     "grid_size": 16,
    #     "no_enemy": False,
    #     "berry_count": 5,
    #     "initial_learning_rate": 1e-4,
    #     "ent_coef": 0.005,
    #     "gamma": 0.99,
    #     "obstacle_density": 0.12,
    #     "resume_from": "previous",
    # },
    {
        "name": "phase_12_with_enemy_16x16_random_berries",
        "description": "16x16, with hunter, random berries",
        "total_timesteps": 2_000_000,
        "grid_size": 16,
        "no_enemy": False,
        "berry_count_min": 2,
        "berry_count_max": 5,
        "initial_learning_rate": 1e-4,
        "ent_coef": 0.005,
        "gamma": 0.99,
        "obstacle_density": 0.12,
        "max_episode_steps": 700,
        "resume_from": "previous",
    },
    {
        "name": "phase_13_random_maps",
        "description": "Random 16×16–20×20 maps",
        "total_timesteps": 1_000_000,
        "grid_size_min": 16,
        "grid_size_max": 20,
        "no_enemy": False,
        "berry_count_min": 2,
        "berry_count_max": 5,
        "initial_learning_rate": 1e-4,
        "ent_coef": 0.005,
        "gamma": 0.99,
        "obstacle_density": 0.12,
        "resume_from": "previous",
    },
]


# Root folder for this curriculum run (per-phase subfolders created inside)
DEFAULT_RUN_NAME = "curriculum_v2"


def _console_str(text: str) -> str:
    """ASCII-safe strings for Windows cp1250 consoles and Tee-Object logs."""
    return (
        text.replace("\u2192", "->")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00d7", "x")
    )


def _safe_print(*args, **kwargs) -> None:
    print(*(_console_str(a) if isinstance(a, str) else a for a in args), **kwargs)


def _phase_slug(name: str) -> str:
    """Filesystem-safe folder name (no trailing spaces — breaks TensorBoard on Windows)."""
    slug = name.strip().replace(",", "_").replace(" ", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug


def _phase_dirs(run_root: str, phase_name: str) -> dict:
    slug = _phase_slug(phase_name)
    return {
        "checkpoint_dir": os.path.join(run_root, "checkpoints", slug),
        "log_dir": os.path.join(run_root, "tensorboard", slug),
        "model_save_path": os.path.join(run_root, "models", slug),
    }


def _print_plan(phases: list, run_root: str):
    _safe_print(f"\nCurriculum plan ({len(phases)} phases) -> {run_root}\n")
    for i, phase in enumerate(phases, start=1):
        resume = phase.get("resume_from", "previous" if i > 1 else None)
        grid = (
            f"{phase['grid_size_min']}-{phase['grid_size_max']}"
            if phase.get("grid_size_min") is not None
            else str(phase.get("grid_size", 12))
        )
        enemy = "no enemy" if phase.get("no_enemy") else "with enemy"
        if "berry_count" in phase:
            berries = str(phase["berry_count"])
        elif "berry_count_range" in phase:
            lo, hi = phase["berry_count_range"]
            berries = f"{lo}" if lo == hi else f"{lo}-{hi}"
        else:
            b_lo = phase.get("berry_count_min", 2)
            b_hi = phase.get("berry_count_max", 5)
            berries = f"{b_lo}" if b_lo == b_hi else f"{b_lo}-{b_hi}"
        _safe_print(f"  {i}. {phase['name']}")
        _safe_print(f"     {_console_str(phase.get('description', ''))}")
        _safe_print(f"     steps={phase['total_timesteps']:,}  grid={grid}  berries={berries}  {enemy}")
        _safe_print(f"     max_episode_steps={phase.get('max_episode_steps', 300)}")
        _safe_print(f"     lr={phase.get('initial_learning_rate', 3e-4)}  ent={phase.get('ent_coef', 0.01)}  gamma={phase.get('gamma', 0.99)}")
        _safe_print(f"     resume: {resume}")
        _safe_print()


def _build_train_args(phase: dict):
    kwargs = normalize_berry_kwargs(
        {k: v for k, v in phase.items() if k not in ("name", "description", "resume_from")}
    )
    kwargs.setdefault("obstacle_density", 0.12)
    kwargs.setdefault("gamma", 0.99)
    kwargs.setdefault("ent_coef", 0.01)
    kwargs.setdefault("initial_learning_rate", 3e-4)
    kwargs.setdefault("no_enemy", False)
    kwargs.setdefault("max_episode_steps", 300)
    if kwargs.get("grid_size_min") is None:
        kwargs.setdefault("grid_size", 12)
    return args_from_namespace(**kwargs)


def run_curriculum(
    phases: list,
    run_root: str,
    start_phase: int = 1,
    initial_resume_path: str | None = None,
):
    """
    Run phases sequentially. Returns path to the last saved model (without .zip).
    """
    os.makedirs(os.path.join(run_root, "models"), exist_ok=True)
    manifest_path = os.path.join(run_root, "curriculum_manifest.json")

    previous_model_path: str | None = initial_resume_path
    completed = []

    for index, phase in enumerate(phases, start=1):
        if index < start_phase:
            dirs = _phase_dirs(run_root, phase["name"])
            candidate = dirs["model_save_path"]
            if os.path.exists(candidate + ".zip") or os.path.exists(candidate):
                previous_model_path = candidate
            continue

        name = _phase_slug(phase["name"])
        _safe_print("\n" + "=" * 72)
        _safe_print(f"PHASE {index}/{len(phases)}: {name}")
        if phase.get("description"):
            _safe_print(_console_str(phase["description"]))
        _safe_print("=" * 72)

        dirs = _phase_dirs(run_root, name)
        resume_from = phase.get("resume_from", "previous" if index > 1 else None)

        resume_path = None
        if index == 1 and initial_resume_path:
            resume_path = resolve_model_path(initial_resume_path)
        elif resume_from and resume_from not in (False, "none", "None"):
            if resume_from == "previous":
                if previous_model_path is None:
                    raise RuntimeError(
                        f"Phase {name} expects resume_from='previous' but no prior model exists."
                    )
                resume_path = resolve_model_path(previous_model_path)
            else:
                resume_path = resolve_model_path(str(resume_from))

        train_args = _build_train_args(phase)
        saved = run_training(
            train_args,
            checkpoint_dir=dirs["checkpoint_dir"],
            log_dir=dirs["log_dir"],
            model_save_path=dirs["model_save_path"],
            checkpoint_prefix=f"{name}",
            resume_model=resume_path,
        )

        previous_model_path = saved
        completed.append(
            {
                "index": index,
                "name": name,
                "model_path": saved + ".zip",
                "total_timesteps": phase["total_timesteps"],
            }
        )

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_root": run_root,
                    "completed_phases": completed,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
                f,
                indent=2,
            )

    final = previous_model_path
    if final:
        _safe_print(f"\nCurriculum finished. Last model: {final}.zip")
        _safe_print(f"Manifest: {manifest_path}")
    return final


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-phase curriculum training (separate from checkpoints/)."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Subfolder under curriculum_runs/ for this training run.",
    )
    parser.add_argument(
        "--from-phase",
        type=int,
        default=1,
        help="Start from this phase number (1-based). Earlier phases are skipped; "
        "phase N still resumes from phase N-1 model if it exists on disk.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional model path for phase 1 only (instead of training from scratch).",
    )
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="Print the curriculum plan and exit.",
    )
    args = parser.parse_args()

    run_root = os.path.join("curriculum_runs", args.run_name)

    if args.list_phases:
        _print_plan(CURRICULUM_PHASES, run_root)
        return

    if args.from_phase < 1 or args.from_phase > len(CURRICULUM_PHASES):
        raise ValueError(f"--from-phase must be between 1 and {len(CURRICULUM_PHASES)}")

    _print_plan(CURRICULUM_PHASES, run_root)

    initial_resume = args.resume_from.strip() or None

    run_curriculum(
        CURRICULUM_PHASES,
        run_root=run_root,
        start_phase=args.from_phase,
        initial_resume_path=initial_resume if args.from_phase == 1 else None,
    )


if __name__ == "__main__":
    main()
