import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import GridWorldEnv

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    parser = argparse.ArgumentParser(description="Train or resume PPO on GridWorld.")
    parser.add_argument("--total-timesteps", type=int, default=500000, help="Timesteps to train in this run.")
    parser.add_argument("--resume-model", type=str, default="", help="Path to existing model/checkpoint (.zip optional).")
    parser.add_argument("--grid-size", type=int, default=12, help="Grid size used for training.")
    parser.add_argument("--initial-learning-rate", type=float, default=3e-4, help="Initial learning rate for linear decay schedule.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient to encourage exploration.")
    parser.add_argument("--obstacle-density", type=float, default=0.12, help="Approximate blocked cells ratio (0.0-0.5).")
    args = parser.parse_args()

    print("Inicjalizacja środowiska treningowego...")
    
    # Katalog na logi dla TensorBoard
    log_dir = "./ppo_gridworld_tensorboard/"
    # Katalog na checkpointy
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Utworzenie środowiska. Używamy wektoryzacji (DummyVecEnv), 
    # co jest wymagane i zalecane w Stable Baselines3.
    # Używamy render_mode=None (domyślnie), aby trening przebiegał szybko w tle, bez UI.
    # Więcej równoległych środowisk poprawia eksplorację i stabilność PPO.
    env = make_vec_env(
        lambda: GridWorldEnv(
            grid_size=args.grid_size,
            max_episode_steps=300,
            obstacle_density=args.obstacle_density
        ),
        n_envs=8
    )
    
    # Callback zapisu modelu co określoną liczbę kroków
    # Zapisze model np. w checkpoints/ppo_model_10000_steps.zip
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=checkpoint_dir,
        name_prefix="ppo_model"
    )

    # Inicjalizacja modelu PPO
    print("Inicjalizacja modelu PPO...")
    
    device = "cpu" 

    lr_schedule = linear_schedule(args.initial_learning_rate)

    reset_num_timesteps = True
    if args.resume_model:
        resume_path = args.resume_model
        if not os.path.exists(resume_path) and os.path.exists(resume_path + ".zip"):
            resume_path = resume_path + ".zip"
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Nie znaleziono modelu do wznowienia: {args.resume_model}")

        print(f"Wznawianie treningu z modelu: {resume_path}")
        # Override saved spaces so checkpoints trained on one grid_size can resume on another.
        model = PPO.load(
            resume_path,
            env=env,
            device=device,
            custom_objects={
                "learning_rate": lr_schedule,
                "lr_schedule": lr_schedule,
                "ent_coef": args.ent_coef,
                "observation_space": env.observation_space,
                "action_space": env.action_space,
            }
        )
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
            gamma=0.99,
            ent_coef=args.ent_coef,
            device=device
        )
    
    print("Rozpoczęcie treningu...")
    # Trening przez zadaną liczbę kroków; przy wznowieniu licznik kroków nie jest resetowany.
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps
    )
    
    print("Trening zakończony! Zapisywanie końcowego modelu...")
    model.save("ppo_gridworld_final")
    
    # Zamknięcie środowiska
    env.close()
    print("Model 'ppo_gridworld_final.zip' zapisany pomyślnie.")

if __name__ == "__main__":
    main()
0