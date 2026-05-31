import argparse
import os
import numpy as np
from stable_baselines3 import PPO

from environment import GridWorldEnv


def resolve_model_path(model_path: str) -> str:
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    raise FileNotFoundError(f"Model not found: {model_path}")


def evaluate_group(model, episodes: int, max_steps: int, grid_size: int, deterministic: bool):
    env = GridWorldEnv(grid_size=grid_size, render_mode=None, max_episode_steps=max_steps, obstacle_density=0.12)

    total_rewards = []
    total_steps = []
    total_collected = 0

    extraction_successes = 0
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

        collected_count = int(sum(env.rewards_collected))
        total_collected += collected_count
        total_rewards.append(episode_reward)
        total_steps.append(steps)

        if collected_count > 0:
            any_reward_collected += 1
        if collected_count == len(env.rewards_collected):
            all_rewards_collected += 1

        extracted = np.array_equal(env.agent_pos, env.goal_pos) and all(env.rewards_collected)
        if extracted:
            extraction_successes += 1
        elif terminated:
            # In current environment logic, non-extraction termination means enemy collision.
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
        "any_reward_collected_rate": any_reward_collected / episodes,
        "all_rewards_collected_rate": all_rewards_collected / episodes,
        "mean_rewards_collected_per_episode": total_collected / episodes,
        "enemy_death_rate": enemy_deaths / episodes,
        "timeout_rate": timeouts / episodes,
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    print(f"Episodes: {metrics['episodes']}")
    print(f"Grid size: {metrics['grid_size']}")
    print(f"Avg episode reward: {metrics['avg_episode_reward']:.3f}")
    print(f"Std episode reward: {metrics['std_episode_reward']:.3f}")
    print(f"Avg steps: {metrics['avg_steps']:.2f}")
    print(f"Extraction success rate: {metrics['extraction_success_rate']:.2%}")
    print(f"Any reward collected rate: {metrics['any_reward_collected_rate']:.2%}")
    print(f"All rewards collected rate: {metrics['all_rewards_collected_rate']:.2%}")
    print(f"Mean rewards collected / episode: {metrics['mean_rewards_collected_per_episode']:.2f}")
    print(f"Enemy death rate: {metrics['enemy_death_rate']:.2%}")
    print(f"Timeout rate: {metrics['timeout_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy on in-distribution and generalization map sets.")
    parser.add_argument("--model-path", type=str, default="ppo_gridworld_final.zip", help="Path to PPO model (.zip or base path).")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per evaluation group.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    print(f"Loading model: {model_path}")
    # For MLP PPO in SB3, CPU is typically faster/more efficient than CUDA.
    model = PPO.load(model_path, device="cpu")

    in_distribution = evaluate_group(
        model=model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        grid_size=12,
        deterministic=args.deterministic,
    )

    generalization = evaluate_group(
        model=model,
        episodes=args.episodes,
        max_steps=args.max_steps,
        grid_size=16,
        deterministic=args.deterministic,
    )

    print_metrics("In-distribution (train-like maps)", in_distribution)
    print_metrics("Generalization (harder maps)", generalization)


if __name__ == "__main__":
    main()
