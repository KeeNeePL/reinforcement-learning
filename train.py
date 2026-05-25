import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import GridWorldEnv

def main():
    print("Inicjalizacja środowiska treningowego...")
    
    # Katalog na logi dla TensorBoard
    log_dir = "./ppo_gridworld_tensorboard/"
    # Katalog na checkpointy
    checkpoint_dir = "./checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Utworzenie środowiska. Używamy wektoryzacji (DummyVecEnv), 
    # co jest wymagane i zalecane w Stable Baselines3.
    # Używamy render_mode=None (domyślnie), aby trening przebiegał szybko w tle, bez UI.
    env = make_vec_env(lambda: GridWorldEnv(grid_size=20), n_envs=1)
    
    # Callback zapisu modelu co określoną liczbę kroków
    # Zapisze model np. w checkpoints/ppo_model_10000_steps.zip
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=checkpoint_dir,
        name_prefix="ppo_model"
    )

    # Inicjalizacja modelu PPO
    print("Inicjalizacja modelu PPO...")
    
    # Sprawdzenie dostępności CUDA (GPU Nvidia)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Używane urządzenie do treningu: {device.upper()}")
    if device == "cpu":
        print("UWAGA: Nie wykryto karty graficznej! Upewnij się, że masz zainstalowaną wersję PyTorch ze wsparciem dla CUDA.")

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device=device
    )
    
    print("Rozpoczęcie treningu...")
    # Trening przez 500 000 kroków (więcej czasu na eksplorację i zrozumienie mapy)
    model.learn(total_timesteps=5000000, callback=checkpoint_callback)
    
    print("Trening zakończony! Zapisywanie końcowego modelu...")
    model.save("ppo_gridworld_final")
    
    # Zamknięcie środowiska
    env.close()
    print("Model 'ppo_gridworld_final.zip' zapisany pomyślnie.")

if __name__ == "__main__":
    main()
0