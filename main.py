import argparse
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk

from stable_baselines3 import PPO

from environment import MAX_BERRIES, GridWorldEnv


def resolve_model_path(model_path: str) -> str | None:
    if not model_path:
        return None
    if os.path.exists(model_path):
        return model_path
    if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
        return model_path + ".zip"
    return None


def show_settings_dialog(
    *,
    grid_size: int,
    berry_count: int,
    no_enemy: bool,
) -> tuple[int, int, bool] | None:
    """Show a small window to edit demo settings. Returns None if cancelled."""
    result: dict = {}
    cancelled = False

    root = tk.Tk()
    root.title("Forest Escape — Demo settings")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=12)
    frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frame, text="Map size (grid cells per side):").grid(
        row=0, column=0, sticky="w", pady=(0, 4)
    )
    grid_var = tk.StringVar(value=str(grid_size))
    grid_spin = ttk.Spinbox(
        frame,
        from_=8,
        to=24,
        width=8,
        textvariable=grid_var,
    )
    grid_spin.grid(row=1, column=0, sticky="w", pady=(0, 12))

    ttk.Label(frame, text=f"Berries per episode (1–{MAX_BERRIES}):").grid(
        row=2, column=0, sticky="w", pady=(0, 4)
    )
    berry_var = tk.StringVar(value=str(berry_count))
    berry_spin = ttk.Spinbox(
        frame,
        from_=1,
        to=MAX_BERRIES,
        width=8,
        textvariable=berry_var,
    )
    berry_spin.grid(row=3, column=0, sticky="w", pady=(0, 12))

    hunter_var = tk.BooleanVar(value=not no_enemy)
    ttk.Checkbutton(frame, text="Enable hunter (enemy)", variable=hunter_var).grid(
        row=4, column=0, sticky="w", pady=(0, 12)
    )

    ttk.Label(
        frame,
        text="Press Space during play to stop the demo.",
        foreground="gray",
    ).grid(row=5, column=0, sticky="w", pady=(0, 12))

    btn_row = ttk.Frame(frame)
    btn_row.grid(row=6, column=0, sticky="e")

    def on_start() -> None:
        try:
            g = int(grid_var.get())
            b = int(berry_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Map size and berries must be whole numbers.")
            return
        if not 8 <= g <= 24:
            messagebox.showerror("Invalid input", "Map size must be between 8 and 24.")
            return
        if not 1 <= b <= MAX_BERRIES:
            messagebox.showerror(
                "Invalid input",
                f"Berry count must be between 1 and {MAX_BERRIES}.",
            )
            return
        result["grid_size"] = g
        result["berry_count"] = b
        result["no_enemy"] = not hunter_var.get()
        root.destroy()

    def on_cancel() -> None:
        nonlocal cancelled
        cancelled = True
        root.destroy()

    ttk.Button(btn_row, text="Cancel", command=on_cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_row, text="Start", command=on_start).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.bind("<Return>", lambda _e: on_start())
    root.eval("tk::PlaceWindow . center")
    root.mainloop()

    if cancelled or not result:
        return None
    return result["grid_size"], result["berry_count"], result["no_enemy"]


def _renderer_stop_requested(env: GridWorldEnv) -> bool:
    renderer = getattr(env, "renderer", None)
    if renderer is None:
        return False
    return renderer.consume_stop_request()


def main():
    parser = argparse.ArgumentParser(description="Visual demo of a trained PPO agent in GridWorld.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="ppo_gridworld_final.zip",
        help="Path to PPO model (.zip or base path).",
    )
    parser.add_argument("--grid-size", type=int, default=16, help="Grid size for the demo map.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to play.")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode.")
    parser.add_argument("--obstacle-density", type=float, default=0.12, help="Blocked cells ratio.")
    parser.add_argument("--render-fps", type=int, default=4, help="Render speed (lower = slower).")
    parser.add_argument("--no-enemy", action="store_true", help="Disable hunter (matches stage 0 training).")
    parser.add_argument("--berry-count", type=int, default=None, help="Fixed berries per episode (overrides min/max).")
    parser.add_argument("--berry-count-min", type=int, default=2, help="Min berries per episode when count is random.")
    parser.add_argument("--berry-count-max", type=int, default=5, help="Max berries per episode when count is random.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions.")
    parser.add_argument(
        "--no-settings",
        action="store_true",
        help="Skip the settings window; use CLI grid size and berry flags only.",
    )
    args = parser.parse_args()

    grid_size = args.grid_size
    berry_count = args.berry_count if args.berry_count is not None else args.berry_count_min
    no_enemy = args.no_enemy

    if not args.no_settings:
        picked = show_settings_dialog(
            grid_size=grid_size,
            berry_count=berry_count,
            no_enemy=no_enemy,
        )
        if picked is None:
            print("Demo cancelled.")
            return
        grid_size, berry_count, no_enemy = picked

    berry_range = (berry_count, berry_count)

    print("Inicjalizacja środowiska...")
    env = GridWorldEnv(
        grid_size=grid_size,
        render_mode="human",
        max_episode_steps=args.max_steps,
        obstacle_density=args.obstacle_density,
        render_fps=args.render_fps,
        no_enemy=no_enemy,
        berry_count_range=berry_range,
    )
    if no_enemy:
        print("Tryb demo: bez myśliwego")
    else:
        print("Tryb demo: z myśliwym")
    print(f"Mapa: {grid_size}×{grid_size}, jagód: {berry_count}")

    model_path = resolve_model_path(args.model_path)
    if model_path is not None:
        print(f"Ładowanie zapisanego modelu z: {model_path}")
        model = PPO.load(model_path, device="cpu")
    else:
        print(f"Brak modelu: {args.model_path}")
        print("Uruchom najpierw 'train.py' lub podaj --model-path. Test na losowych akcjach...")
        model = None

    stop_demo = False
    for episode in range(args.episodes):
        if stop_demo:
            break

        obs, _ = env.reset()
        print(f"\n--- Epizod {episode + 1} ---")
        env.render()
        if _renderer_stop_requested(env):
            print("Demo zatrzymane (Space).")
            break

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

            if _renderer_stop_requested(env):
                print("Demo zatrzymane (Space).")
                stop_demo = True
                break

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
