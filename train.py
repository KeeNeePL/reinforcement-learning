import os
import argparse
from types import SimpleNamespace

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

from environment import GridWorldEnv, GAME_INFO_KEYS
from callbacks import GameMetricsCallback


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def berry_count_range_for_args(args) -> tuple[int, int]:
    """Fixed count via berry_count, or random in [berry_count_min, berry_count_max]."""
    if getattr(args, "berry_count", None) is not None:
        n = int(args.berry_count)
        return (n, n)
    lo = int(getattr(args, "berry_count_min", 1))
    hi = int(getattr(args, "berry_count_max", 5))
    if lo > hi:
        raise ValueError(f"berry_count_min ({lo}) must be <= berry_count_max ({hi})")
    return (lo, hi)


def normalize_berry_kwargs(kwargs: dict) -> dict:
    """Curriculum helper: berry_count → min/max; berry_count_range tuple also accepted."""
    out = dict(kwargs)
    if "berry_count_range" in out:
        lo, hi = out.pop("berry_count_range")
        out["berry_count_min"] = int(lo)
        out["berry_count_max"] = int(hi)
    if "berry_count" in out:
        n = int(out.pop("berry_count"))
        out["berry_count_min"] = n
        out["berry_count_max"] = n
    return out


def _make_env(args):
    berry_range = berry_count_range_for_args(args)
    common = dict(
        max_episode_steps=300,
        obstacle_density=args.obstacle_density,
        no_enemy=args.no_enemy,
        berry_count_range=berry_range,
    )
    if args.grid_size_min is not None and args.grid_size_max is not None:
        if args.grid_size_min > args.grid_size_max:
            raise ValueError("--grid-size-min must be <= --grid-size-max")
        return GridWorldEnv(
            grid_size=args.grid_size_max,
            grid_size_range=(args.grid_size_min, args.grid_size_max),
            **common,
        )
    return GridWorldEnv(grid_size=args.grid_size, **common)


def resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    raise FileNotFoundError(f"Model not found: {model_path}")


def run_training(
    args,
    *,
    checkpoint_dir: str = "./checkpoints/",
    log_dir: str = "./ppo_gridworld_tensorboard/",
    model_save_path: str = "ppo_gridworld_final",
    checkpoint_prefix: str = "ppo_model",
    resume_model: str | None = None,
):
    """
    Run one PPO training session.

    `args` must provide: total_timesteps, grid_size, grid_size_min, grid_size_max,
    initial_learning_rate, ent_coef, gamma, obstacle_density, no_enemy.
  Optional `resume_model` overrides args.resume_model when set.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.grid_size_min is not None and args.grid_size_max is not None:
        print(f"Tryb treningu: losowy rozmiar mapy ({args.grid_size_min}-{args.grid_size_max})")
    else:
        print(f"Tryb treningu: stały rozmiar mapy {args.grid_size}x{args.grid_size}")
    if args.no_enemy:
        print("Tryb treningu: bez myśliwego (curriculum stage 0)")
    berry_lo, berry_hi = berry_count_range_for_args(args)
    if berry_lo == berry_hi:
        print(f"Jagód na mapie: {berry_lo} (stała)")
    else:
        print(f"Jagod na mapie: losowo {berry_lo}-{berry_hi}")

    monitor_kwargs = {"info_keywords": list(GAME_INFO_KEYS)}

    env = make_vec_env(lambda: _make_env(args), n_envs=8, monitor_kwargs=monitor_kwargs)
    eval_env = make_vec_env(lambda: _make_env(args), n_envs=1, monitor_kwargs=monitor_kwargs)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix=checkpoint_prefix,
    )
    game_metrics_callback = GameMetricsCallback(window=200, no_enemy=args.no_enemy)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
        log_path=os.path.join(log_dir, "evaluations"),
        eval_freq=2500,
        n_eval_episodes=20,
        deterministic=False,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, game_metrics_callback, eval_callback])

    print("Inicjalizacja modelu PPO...")
    device = "cpu"
    lr_schedule = linear_schedule(args.initial_learning_rate)

    resume_path = resume_model if resume_model is not None else getattr(args, "resume_model", "") or ""
    reset_num_timesteps = True

    if resume_path:
        resume_path = resolve_model_path(resume_path)
        print(f"Wznawianie treningu z modelu: {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
                "ent_coef": args.ent_coef,
                "gamma": args.gamma,
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            },
        )
        model.tensorboard_log = log_dir
        model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=lr_schedule,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            device=device,
        )

    print("Rozpoczęcie treningu...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
    )

    print("Trening zakończony! Zapisywanie końcowego modelu...")
    model.save(model_save_path)

    env.close()
    eval_env.close()
    print(f"Model zapisany: {model_save_path}.zip")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    return model_save_path


def args_from_namespace(**kwargs):
    """Build a training args object with defaults for optional fields."""
    defaults = dict(
        total_timesteps=500_000,
        resume_model="",
        grid_size=12,
        grid_size_min=None,
        grid_size_max=None,
        initial_learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        obstacle_density=0.12,
        no_enemy=False,
        berry_count=None,
        berry_count_min=1,
        berry_count_max=5,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def main():
    parser = argparse.ArgumentParser(description="Train or resume PPO on GridWorld.")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Timesteps to train in this run.")
    parser.add_argument("--resume-model", type=str, default="", help="Path to existing model/checkpoint (.zip optional).")
    parser.add_argument("--grid-size", type=int, default=12, help="Fixed grid size (used when min/max not set).")
    parser.add_argument("--grid-size-min", type=int, default=None, help="Min grid size for random map size each episode.")
    parser.add_argument("--grid-size-max", type=int, default=None, help="Max grid size for random map size each episode.")
    parser.add_argument("--initial-learning-rate", type=float, default=3e-4, help="Initial learning rate for linear decay schedule.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient to encourage exploration.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--obstacle-density", type=float, default=0.12, help="Approximate blocked cells ratio (0.0-0.5).")
    parser.add_argument("--no-enemy", action="store_true", help="Disable hunter movement, death, and danger zone (curriculum stage 0).")
    parser.add_argument("--berry-count", type=int, default=None, help="Fixed berries per episode (overrides min/max).")
    parser.add_argument("--berry-count-min", type=int, default=1, help="Min berries per episode when count is random.")
    parser.add_argument("--berry-count-max", type=int, default=5, help="Max berries per episode when count is random.")
    args = parser.parse_args()

    print("Inicjalizacja środowiska treningowego...")
    run_training(args)


if __name__ == "__main__":
    main()
