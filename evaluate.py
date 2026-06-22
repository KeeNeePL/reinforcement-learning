import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from environment import GridWorldEnv
from train import berry_count_range_for_args


def resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    raise FileNotFoundError(f"Model not found: {model_path}")


def evaluate_group(
    model,
    episodes: int,
    max_steps: int,
    grid_size: int,
    deterministic: bool,
    no_enemy: bool = False,
    berry_count_range=(2, 5),
):
    env = GridWorldEnv(
        grid_size=grid_size,
        render_mode=None,
        max_episode_steps=max_steps,
        obstacle_density=0.12,
        no_enemy=no_enemy,
        berry_count_range=berry_count_range,
    )

    total_rewards = []
    total_steps = []
    total_collected = 0
    total_spawned = 0

    extraction_successes = 0
    full_extraction_successes = 0
    any_reward_collected = 0
    all_rewards_collected = 0
    enemy_deaths = 0
    timeouts = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while not terminated and not truncated and steps < max_steps:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward
            steps += 1

        spawned = env.spawned_berry_count()
        collected_count = env.collected_berry_count()
        total_collected += collected_count
        total_spawned += spawned
        total_rewards.append(episode_reward)
        total_steps.append(steps)

        if collected_count > 0:
            any_reward_collected += 1
        if env._all_spawned_berries_collected():
            all_rewards_collected += 1

        if env._episode_extracted:
            extraction_successes += 1
        if env._episode_full_extraction:
            full_extraction_successes += 1
        if terminated and not no_enemy and not env._episode_extracted:
            enemy_deaths += 1
        elif truncated or steps >= max_steps:
            timeouts += 1

    env.close()

    rewards_arr = np.array(total_rewards, dtype=np.float32)
    steps_arr = np.array(total_steps, dtype=np.float32)

    return {
        "episodes": episodes,
        "grid_size": grid_size,
        "avg_episode_reward": float(np.mean(rewards_arr)),
        "std_episode_reward": float(np.std(rewards_arr)),
        "avg_steps": float(np.mean(steps_arr)),
        "extraction_success_rate": extraction_successes / episodes,
        "full_extraction_success_rate": full_extraction_successes / episodes,
        "any_reward_collected_rate": any_reward_collected / episodes,
        "all_rewards_collected_rate": all_rewards_collected / episodes,
        "mean_rewards_collected_per_episode": total_collected / episodes,
        "mean_berries_spawned_per_episode": total_spawned / episodes,
        "enemy_death_rate": enemy_deaths / episodes,
        "timeout_rate": timeouts / episodes,
    }


def print_metrics(name: str, metrics: dict, no_enemy: bool = False):
    print(f"\n=== {name} ===")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Grid size: {metrics['grid_size']}")
    if no_enemy:
        print("Enemy: disabled")
    print(f"Avg episode reward: {metrics['avg_episode_reward']:.3f}")
    print(f"Std episode reward: {metrics['std_episode_reward']:.3f}")
    print(f"Avg steps: {metrics['avg_steps']:.2f}")
    print(f"Extraction success rate (any berries): {metrics['extraction_success_rate']:.2%}")
    print(f"Full extraction rate (all berries + exit): {metrics['full_extraction_success_rate']:.2%}")
    print(f"Any reward collected rate: {metrics['any_reward_collected_rate']:.2%}")
    print(f"All rewards collected rate: {metrics['all_rewards_collected_rate']:.2%}")
    print(f"Mean rewards collected / episode: {metrics['mean_rewards_collected_per_episode']:.2f}")
    print(f"Mean berries spawned / episode: {metrics['mean_berries_spawned_per_episode']:.2f}")
    if not no_enemy:
        print(f"Enemy death rate: {metrics['enemy_death_rate']:.2%}")
    print(f"Timeout rate: {metrics['timeout_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on in-distribution and generalization map sets.")
    parser.add_argument("--model-path", type=str, default="ppo_gridworld_final.zip", help="Path to PPO model (.zip or base path).")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per evaluation group.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument("--no-enemy", action="store_true", help="Evaluate without hunter (matches curriculum stage 0).")
    parser.add_argument("--berry-count", type=int, default=None, help="Fixed berries per episode (overrides min/max).")
    parser.add_argument("--berry-count-min", type=int, default=2, help="Min berries per episode when count is random.")
    parser.add_argument("--berry-count-max", type=int, default=5, help="Max berries per episode when count is random.")
    args = parser.parse_args()

    berry_range = berry_count_range_for_args(args)
    if berry_range[0] == berry_range[1]:
        print(f"Berries per episode: {berry_range[0]} (fixed)")
    else:
        print(f"Berries per episode: random {berry_range[0]}–{berry_range[1]}")

    model_path = resolve_model_path(args.model_path)
    print(f"Loading model: {model_path}")
    if args.no_enemy:
        print("Evaluation mode: no enemy")
    # For MLP PPO in SB3, CPU is typically faster/more efficient than CUDA.
    model = PPO.load(model_path, device="cpu")

    in_distribution = evaluate_group(
        model=model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        grid_size=12,
        deterministic=args.deterministic,
        no_enemy=args.no_enemy,
        berry_count_range=berry_range,
    )

    generalization = evaluate_group(
        model=model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        grid_size=16,
        deterministic=args.deterministic,
        no_enemy=args.no_enemy,
        berry_count_range=berry_range,
    )

    print_metrics("In-distribution (train-like maps)", in_distribution, no_enemy=args.no_enemy)
    print_metrics("Generalization (harder maps)", generalization, no_enemy=args.no_enemy)


if __name__ == "__main__":
    main()
