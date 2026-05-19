import time
from environment import GridWorldEnv

def main():
    print("Inicjalizacja środowiska...")
    # Utworzenie środowiska z włączonym renderowaniem na żywo ("human")
    env = GridWorldEnv(grid_size=20, render_mode="human")
    
    # Wykonanie kilku testowych epizodów z losowymi akcjami
    for episode in range(3):
        obs, info = env.reset()
        print(f"\n--- Epizod {episode + 1} ---")
        env.render()
        
        terminated = False
        total_reward = 0
        step = 0
        
        while not terminated and step < 200:
            # Losowa akcja (0: Góra, 1: Dół, 2: Lewo, 3: Prawo)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
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
