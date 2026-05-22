import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=10, render_mode=None):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        self.agent_pos = None
        self.goal_pos = None
        self.enemy_pos = None
        self.reward_pos = None
        self.reward_collected = False
        
        # Stałe przeszkody - ułożenie w sensowne kształty
        self.obstacles = self._generate_static_obstacles()
        self.num_obstacles = len(self.obstacles)
        
        # State: Agent, Goal, Enemy, Reward (x, y) + obstacles (x, y)
        obs_shape = 11
        self.observation_space = spaces.Box(
            low=-self.grid_size, 
            high=self.grid_size, 
            shape=(obs_shape,), 
            dtype=np.float32
        )

        self.renderer = None

    def _generate_static_obstacles(self):
        obstacles = []
        
        # 1. Pionowa ściana (na środku, otwory na brzegach)
        mid_x = self.grid_size // 2
        for y in range(2, self.grid_size - 2):
            obstacles.append([mid_x, y])
            
        # 2. Pozioma ściana po prawej stronie (pokój z wyjściem do celu)
        mid_y = self.grid_size // 2
        for x in range(mid_x + 2, self.grid_size - 1):
            obstacles.append([x, mid_y])
            
        # 3. Zamknięty kwadrat wyłączony z użytku (np. 3x3) po lewej
        for x in range(2, 5):
            for y in range(2, 5):
                # Sprawdzenie by nie zablokować mapy dla mniejszych rozmiarów siatki
                if x < mid_x and x < self.grid_size and y < self.grid_size:
                    obstacles.append([x, y])
                    
        return np.array(obstacles, dtype=int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize positions
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        
        # Random enemy position
        while True:
            self.enemy_pos = np.random.randint(0, self.grid_size, size=2)
            if np.max(np.abs(self.agent_pos - self.enemy_pos)) > 2:
                overlap = np.array_equal(self.enemy_pos, self.goal_pos)
                for obs in self.obstacles:
                    if np.array_equal(self.enemy_pos, obs):
                        overlap = True
                        break
                if not overlap:
                    break
                
        self.reward_pos = np.array([self.grid_size // 3, (self.grid_size // 3) * 2])
        # Ensure reward is not on obstacle
        for obs in self.obstacles:
            if np.array_equal(self.reward_pos, obs):
                while True:
                    self.reward_pos = np.random.randint(0, self.grid_size, size=2)
                    if not any(np.array_equal(self.reward_pos, o) for o in self.obstacles) and \
                       not np.array_equal(self.reward_pos, self.goal_pos) and \
                       not np.array_equal(self.reward_pos, self.agent_pos):
                        break
                break
                
        self.reward_collected = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. Deltas (wektory od agenta do celu)
        delta_goal = self.goal_pos - self.agent_pos
        delta_enemy = self.enemy_pos - self.agent_pos
        
        # Jeśli nagroda zebrana, wektor kierunkowy zerujemy
        if self.reward_collected:
            delta_reward = np.array([0, 0])
            has_reward = 1.0
        else:
            delta_reward = self.reward_pos - self.agent_pos
            has_reward = 0.0

        # 2. Czujniki odległości (raycasting)
        sensors = self._get_sensors()

        # 3. Złączenie w jeden płaski wektor o stałej długości (11 elementów)
        obs = np.concatenate([
            delta_goal, 
            delta_enemy, 
            delta_reward, 
            [has_reward], 
            sensors
        ])
        
        return obs.astype(np.float32)
    
    def _get_sensors(self):
        # Kolejność: Góra(0), Dół(1), Lewo(2), Prawo(3)
        sensors = np.zeros(4, dtype=np.float32)
        
        # Wektory kierunkowe odpowiadające Twoim akcjom w step():
        # [dx, dy]
        directions = [
            np.array([0, -1]), # Góra (zmniejsza Y)
            np.array([0, 1]),  # Dół (zwiększa Y)
            np.array([-1, 0]), # Lewo (zmniejsza X)
            np.array([1, 0])   # Prawo (zwiększa X)
        ]
        
        for i, direction in enumerate(directions):
            current_pos = self.agent_pos.copy()
            distance = 0
            
            while True:
                current_pos += direction
                distance += 1
                
                # Uderzenie w granicę mapy
                if current_pos[0] < 0 or current_pos[0] >= self.grid_size or \
                   current_pos[1] < 0 or current_pos[1] >= self.grid_size:
                    break
                    
                # Uderzenie w przeszkodę
                if any(np.array_equal(current_pos, obs) for obs in self.obstacles):
                    break
                    
            sensors[i] = distance
            
        return sensors
        
    def step(self, action):
        # Obliczamy odległość przed ruchem
        old_dist = np.linalg.norm(self.goal_pos - self.agent_pos)

        # Move Agent
        new_agent_pos = self.agent_pos.copy()
        if action == 0: new_agent_pos[1] = max(0, self.agent_pos[1] - 1)              # Up (decreases Y)
        elif action == 1: new_agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)  # Down (increases Y)
        elif action == 2: new_agent_pos[0] = max(0, self.agent_pos[0] - 1)              # Left (decreases X)
        elif action == 3: new_agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)  # Right (increases X)

        if not any(np.array_equal(new_agent_pos, obs) for obs in self.obstacles):
            self.agent_pos = new_agent_pos
            
        # Obliczamy odległość po ruchu
        new_dist = np.linalg.norm(self.goal_pos - self.agent_pos)

        # Move Enemy (Random)
        enemy_action = np.random.choice([0, 1, 2, 3])
        new_enemy_pos = self.enemy_pos.copy()
        if enemy_action == 0: new_enemy_pos[1] = max(0, self.enemy_pos[1] - 1)
        elif enemy_action == 1: new_enemy_pos[1] = min(self.grid_size - 1, self.enemy_pos[1] + 1)
        elif enemy_action == 2: new_enemy_pos[0] = max(0, self.enemy_pos[0] - 1)
        elif enemy_action == 3: new_enemy_pos[0] = min(self.grid_size - 1, self.enemy_pos[0] + 1)
        
        if not any(np.array_equal(new_enemy_pos, obs) for obs in self.obstacles):
            self.enemy_pos = new_enemy_pos

        # Podstawowa kara za krok (zachęca do szybkości) + nagroda za skrócenie dystansu (Reward Shaping)
        reward = -0.05 + (old_dist - new_dist) * 0.5
        terminated = False

        # Enemy collision (Distance <= 1 zmniejszono z 2)
        if np.max(np.abs(self.agent_pos - self.enemy_pos)) <= 1:
            reward = -20.0
            terminated = True

        # Reward collection
        if not self.reward_collected and np.array_equal(self.agent_pos, self.reward_pos):
            reward += 5.0
            self.reward_collected = True
            self.reward_pos = np.array([-1, -1])  # Move off-grid

        # Goal reached
        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 20.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            return

        # Initialize renderer lazily to avoid pygame overhead if unused
        if self.renderer is None:
            from renderer import PygameRenderer
            self.renderer = PygameRenderer(self.grid_size, self.render_mode)
        
        return self.renderer.render(
            self.agent_pos, 
            self.goal_pos, 
            self.enemy_pos, 
            self.reward_pos, 
            self.reward_collected,
            self.obstacles
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
