import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

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
        self.rewards_pos = None
        self.rewards_collected = None
        self.obstacles = np.array([])
        
        # State:
        # Delta Goal (2), Delta Enemy (2), Delta Reward1 (2), HasReward1 (1), 
        # Delta Reward2 (2), HasReward2 (1), Sensors 8 dirs (8) = 18
        obs_shape = 18
        self.observation_space = spaces.Box(
            low=-self.grid_size, 
            high=self.grid_size, 
            shape=(obs_shape,), 
            dtype=np.float32
        )

        self.renderer = None

    def _generate_random_obstacles(self):
        obstacles = []
        # Próba wygenerowania zablokowania około 15% mapy
        num_blocks = int((self.grid_size * self.grid_size) * 0.15)
        
        # Generujemy bloki wielkości np. 1x3, 3x1, 2x2
        while len(obstacles) < num_blocks:
            w = np.random.choice([1, 2, 3])
            h = np.random.choice([1, 2, 3])
            
            x = np.random.randint(0, self.grid_size - w + 1)
            y = np.random.randint(0, self.grid_size - h + 1)
            
            for ix in range(x, x + w):
                for iy in range(y, y + h):
                    obstacles.append([ix, iy])
                    
        # Usuń duplikaty
        unique_obs = []
        for o in obstacles:
            if o not in unique_obs:
                unique_obs.append(o)
                
        return np.array(unique_obs, dtype=int) if len(unique_obs) > 0 else np.array([[-1, -1]])

    def _is_reachable(self, start, end):
        if np.array_equal(start, end):
            return True
        
        queue = deque([tuple(start)])
        visited = set([tuple(start)])
        
        # Convert obstacles to set of tuples for O(1) lookup
        obs_set = set(tuple(o) for o in self.obstacles)
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            current = queue.popleft()
            if current[0] == end[0] and current[1] == end[1]:
                return True
                
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    n_pos = (nx, ny)
                    if n_pos not in visited and n_pos not in obs_set:
                        visited.add(n_pos)
                        queue.append(n_pos)
                        
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        while True:
            # 1. Losuj przeszkody
            self.obstacles = self._generate_random_obstacles()
            obs_set = set(tuple(o) for o in self.obstacles)
            
            # 2. Losuj start
            while True:
                self.agent_pos = np.random.randint(0, self.grid_size, size=2)
                if tuple(self.agent_pos) not in obs_set:
                    break
                    
            # 3. Losuj cel (minimum odległość: grid_size / 2)
            while True:
                self.goal_pos = np.random.randint(0, self.grid_size, size=2)
                if tuple(self.goal_pos) not in obs_set and tuple(self.goal_pos) != tuple(self.agent_pos):
                    dist = np.linalg.norm(self.goal_pos - self.agent_pos)
                    if dist >= self.grid_size / 2.0:
                        break
                        
            # 4. Sprawdź BFS
            if self._is_reachable(self.agent_pos, self.goal_pos):
                break # Jeśli cel jest osiągalny, to mapa jest ok (nie zablokowana)

        # 5. Losuj dwie nagrody (muszą być dostępne ze startu)
        self.rewards_pos = []
        self.rewards_collected = [False, False]
        for _ in range(2):
            while True:
                r_pos = np.random.randint(0, self.grid_size, size=2)
                if tuple(r_pos) not in obs_set and \
                   tuple(r_pos) != tuple(self.agent_pos) and \
                   tuple(r_pos) != tuple(self.goal_pos) and \
                   not any(np.array_equal(r_pos, existing_r) for existing_r in self.rewards_pos):
                    if self._is_reachable(self.agent_pos, r_pos):
                        self.rewards_pos.append(r_pos)
                        break

        # 6. Random enemy position
        while True:
            self.enemy_pos = np.random.randint(0, self.grid_size, size=2)
            if tuple(self.enemy_pos) not in obs_set and tuple(self.enemy_pos) != tuple(self.goal_pos) and tuple(self.enemy_pos) != tuple(self.agent_pos):
                if np.linalg.norm(self.enemy_pos - self.agent_pos) > 3.0: # Wróg dalej od startu
                    break
        
        return self._get_obs(), {}

    def _get_obs(self):
        delta_goal = self.goal_pos - self.agent_pos
        delta_enemy = self.enemy_pos - self.agent_pos
        
        obs_elements = [delta_goal, delta_enemy]
        
        for i in range(2):
            if self.rewards_collected[i]:
                obs_elements.append(np.array([0, 0]))
                obs_elements.append([1.0])
            else:
                obs_elements.append(self.rewards_pos[i] - self.agent_pos)
                obs_elements.append([0.0])

        sensors = self._get_sensors()
        obs_elements.append(sensors)

        obs = np.concatenate(obs_elements)
        return obs.astype(np.float32)
    
    def _get_sensors(self):
        # 8 directions
        sensors = np.zeros(8, dtype=np.float32)
        directions = [
            np.array([0, -1]),  # Up
            np.array([0, 1]),   # Down
            np.array([-1, 0]),  # Left
            np.array([1, 0]),   # Right
            np.array([-1, -1]), # Up-Left
            np.array([1, -1]),  # Up-Right
            np.array([-1, 1]),  # Down-Left
            np.array([1, 1])    # Down-Right
        ]
        
        obs_set = set(tuple(o) for o in self.obstacles)
        
        for i, direction in enumerate(directions):
            current_pos = self.agent_pos.copy()
            distance = 0
            
            while True:
                current_pos += direction
                distance += 1
                
                if current_pos[0] < 0 or current_pos[0] >= self.grid_size or \
                   current_pos[1] < 0 or current_pos[1] >= self.grid_size:
                    break
                    
                if tuple(current_pos) in obs_set:
                    break
                    
            sensors[i] = distance
            
        return sensors
        
    def step(self, action):
        old_dist = np.linalg.norm(self.goal_pos - self.agent_pos)

        new_agent_pos = self.agent_pos.copy()
        if action == 0: new_agent_pos[1] = max(0, self.agent_pos[1] - 1)              
        elif action == 1: new_agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)  
        elif action == 2: new_agent_pos[0] = max(0, self.agent_pos[0] - 1)              
        elif action == 3: new_agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)  

        obs_set = set(tuple(o) for o in self.obstacles)
        reward = 0.0

        if tuple(new_agent_pos) in obs_set:
            # Uderzenie w mur!
            reward -= 0.5
        else:
            self.agent_pos = new_agent_pos
            
        new_dist = np.linalg.norm(self.goal_pos - self.agent_pos)

        # Move Enemy (Random)
        enemy_action = np.random.choice([0, 1, 2, 3])
        new_enemy_pos = self.enemy_pos.copy()
        if enemy_action == 0: new_enemy_pos[1] = max(0, self.enemy_pos[1] - 1)
        elif enemy_action == 1: new_enemy_pos[1] = min(self.grid_size - 1, self.enemy_pos[1] + 1)
        elif enemy_action == 2: new_enemy_pos[0] = max(0, self.enemy_pos[0] - 1)
        elif enemy_action == 3: new_enemy_pos[0] = min(self.grid_size - 1, self.enemy_pos[0] + 1)
        
        if tuple(new_enemy_pos) not in obs_set:
            self.enemy_pos = new_enemy_pos

        reward += -0.05 + (old_dist - new_dist) * 0.5
        terminated = False

        if np.max(np.abs(self.agent_pos - self.enemy_pos)) <= 1:
            reward = -20.0
            terminated = True

        for i in range(2):
            if not self.rewards_collected[i] and np.array_equal(self.agent_pos, self.rewards_pos[i]):
                reward += 5.0
                self.rewards_collected[i] = True

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward += 20.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.renderer is None:
            from renderer import PygameRenderer
            self.renderer = PygameRenderer(self.grid_size, self.render_mode)
        
        return self.renderer.render(
            self.agent_pos, 
            self.goal_pos, 
            self.enemy_pos, 
            self.rewards_pos, 
            self.rewards_collected,
            self.obstacles
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

