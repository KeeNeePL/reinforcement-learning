import argparse
import os
import time

from stable_baselines3 import PPO

from environment import GridWorldEnv
from train import berry_count_range_for_args


def resolve_model_path(model_path: str) -> str | None:
    if not model_path:
        return None
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    return None


def main():
    parser = argparse.ArgumentParser(description="Visual demo of a trained PPO agent in GridWorld.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_gridworld_final.zip",
        help="Path to PPO model (.zip or base path).",
    )
    parser.add_argument("--grid-size", type=int, default=12, help="Grid size for the demo map.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--obstacle-density", type=float, default=0.12, help="Blocked cells ratio.")
    parser.add_argument("--render-fps", type=int, default=4, help="Render speed (lower = slower).")
    parser.add_argument("--no-enemy", action="store_true", help="Disable hunter (matches stage 0 training).")
    parser.add_argument("--berry-count", type=int, default=None, help="Fixed berries per episode (overrides min/max).")
    parser.add_argument("--berry-count-min", type=int, default=1, help="Min berries per episode when count is random.")
    parser.add_argument("--berry-count-max", type=int, default=5, help="Max berries per episode when count is random.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    args = parser.parse_args()

    berry_range = berry_count_range_for_args(args)

    print("Inicjalizacja środowiska...")
    env = GridWorldEnv(
        grid_size=args.grid_size,
        render_mode="human",
        max_episode_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        render_fps=args.render_fps,
        no_enemy=args.no_enemy,
        berry_count_range=berry_range,
    )
    if args.no_enemy:
        print("Tryb demo: bez myśliwego")
    if berry_range[0] == berry_range[1]:
        print(f"Jagód na mapie: {berry_range[0]} (stała)")
    else:
        print(f"Jagód na mapie: losowo {berry_range[0]}–{berry_range[1]}")

    model_path = resolve_model_path(args.model_path)
    if model_path is not None:
        print(f"Ładowanie zapisanego modelu z: {model_path}")
        model = PPO.load(model_path, device="cpu")
    else:
        print(f"Brak modelu: {args.model_path}")
        print("Uruchom najpierw 'train.py' lub podaj --model-path. Test na losowych akcjach...")
        model = None

    for episode in range(args.episodes):
        obs, _ = env.reset()
        print(f"\n--- Epizod {episode + 1} ---")
        env.render()

        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        while not terminated and not truncated:
            if model is not None:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            step += 1
            env.render()

            if terminated or truncated:
                if env._episode_full_extraction:
                    reason = "pełna ekstrakcja (wszystkie jagody)"
                elif env._episode_extracted:
                    reason = f"wczesna ekstrakcja ({env.collected_berry_count()}/{env.spawned_berry_count()} jagód)"
                elif terminated:
                    reason = "śmierć u wroga"
                elif truncated:
                    reason = "timeout"
                else:
                    reason = "koniec epizodu"
                berries = env.collected_berry_count()
                spawned = env.spawned_berry_count()
                print(
                    f"Epizod zakończony ({reason}) po {step} krokach. "
                    f"Nagroda: {total_reward:.2f}. Jagody: {berries}/{spawned}."
                )
                time.sleep(1)

    env.close()
    print("Test środowiska zakończony pomyślnie.")


if __name__ == "__main__":
    main()
