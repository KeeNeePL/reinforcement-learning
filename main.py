import time
import os
from stable_baselines3 import PPO
from environment import GridWorldEnv

def main():
    print("Inicjalizacja środowiska...")
    # Utworzenie środowiska z włączonym renderowaniem na żywo ("human")
    env = GridWorldEnv(
        grid_size=16,
        render_mode="human",
        max_episode_steps=300,
        obstacle_density=0.12,
        render_fps=4,  # lower = slower (steps per second while viewing)
    )
    
    model_path = r"C:\Python Projects\reinforcement-learning\checkpoints\ppo_model_4480000_steps.zip"
    
    if os.path.exists(model_path):
        print(f"Ładowanie zapisanego modelu z: {model_path}")
        model = PPO.load(model_path)
    else:
        print("Brak zapisanego modelu PPO. Uruchom najpierw 'train.py', by wygenerować model.")
        print("Wykonuję test na losowych akcjach...")
        model = None
    
    # Wykonanie kilku testowych epizodów
    for episode in range(3):
        obs, info = env.reset()
        print(f"\n--- Epizod {episode + 1} ---")
        env.render()
        
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        
        while not terminated and not truncated:
            if model is not None:
                # Agent z modelem PPO: przewidywanie najlepszej akcji dla stanu
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Losowa akcja w przypadku braku modelu
                action = env.action_space.sample()
            
            # W środowiskach PPO używamy skalara int, jeśli action space jest Discrete
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            step += 1
            
            env.render()
            
            if terminated or truncated:
                reason = "terminacja" if terminated else "timeout (truncated)"
                print(f"Epizod zakończony ({reason}) po {step} krokach. Całkowita nagroda: {total_reward:.2f}")
                time.sleep(1) # Krótka pauza przed kolejnym epizodem
                
    env.close()
    print("Test środowiska zakończony pomyślnie.")

if __name__ == "__main__":
    main()
