import time
import os
from stable_baselines3 import PPO
from environment import GridWorldEnv

def main():
    print("Inicjalizacja środowiska...")
    # Utworzenie środowiska z włączonym renderowaniem na żywo ("human")
    env = GridWorldEnv(grid_size=20, render_mode="human")
    
    model_path = "ppo_gridworld_final"
    
    if os.path.exists(model_path + ".zip"):
        print(f"Ładowanie zapisanego modelu z: {model_path}.zip")
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
        total_reward = 0
        step = 0
        
        while not terminated and step < 200:
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
            
            if terminated:
                print(f"Epizod zakończony po {step} krokach. Całkowita nagroda: {total_reward:.2f}")
                time.sleep(1) # Krótka pauza przed kolejnym epizodem
                
    env.close()
    print("Test środowiska zakończony pomyślnie.")

if __name__ == "__main__":
    main()
