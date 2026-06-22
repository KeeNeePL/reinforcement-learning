import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

MAX_BERRIES = 5
# Fixed totals when all berries collected (scaled per episode by 1 / n_berries).
TOTAL_BERRY_PICKUP_REWARD = 15 * MAX_BERRIES   # 75
TOTAL_BERRY_LANDING_REWARD = 5 * MAX_BERRIES   # 25
LAST_BERRY_COLLECTION_BONUS = 10  # one-time when final spawned berry is picked
STEP_PENALTY = 0.08 # per-step cost (was 0.1); discourages wandering / timeouts
WALL_COLLISION_PENALTY = 5.0  # moving into an obstacle
EMPTY_EXIT_PENALTY = 5.0  # reaching shelter with zero berries collected
TOTAL_EXTRACTION_REWARD = 50.0  # scaled by berries_collected / berries_spawned on exit
FULL_EXTRACTION_BONUS = 40.0  # extra when exiting with every spawned berry collected
GOAL_SHAPING_MAX = 1.2  # max goal pull at 100% berries; scaled by (collected/spawned)^2
BERRY_SHAPING_MIN = 0.40  # per-berry smell coefficient when far (BFS distance at cap)
BERRY_SHAPING_MAX = 1  # per-berry smell coefficient when adjacent or on berry (BFS = 0)
BERRY_RETREAT_PENALTY = 0.5  # stepping away from an adjacent uncollected berry
ENEMY_DEATH_PENALTY = 25.0  # additive on death step (does not replace other step rewards)
ENEMY_DANGER_RADIUS = 4  # Chebyshev distance for danger-zone penalty
# Per-step danger cost when 1 < dist <= ENEMY_DANGER_RADIUS (stronger when closer).
ENEMY_DANGER_COEF = 0.5
ENEMY_DANGER_EXP_BASE = 1.65  # exponential bump: dist 2 hurts much more than dist 4
# Obs: delta goal (2) + delta enemy (2) + per-berry slot (5×5) + sensors (8) = 37
# Each berry slot: active, collected, delta_x, delta_y, smell_distance (BFS)
OBS_DIM = 2 + 2 + MAX_BERRIES * 5 + 8

# Logged via SB3 Monitor info_keywords (VecMonitor copies these into info["episode"]).
GAME_INFO_KEYS = (
    "extracted",
    "full_extraction",
    "all_berries_collected",
    "any_berry_collected",
    "berries_collected",
    "berries_spawned",
    "enemy_death",
    "timeout",
    "is_success",
)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        grid_size=10,
        render_mode=None,
        max_episode_steps=300,
        obstacle_density=0.12,
        render_fps=6,
        grid_size_range=None,
        berry_count_range=(2, 5),
        no_enemy=False,
    ):
        super(GridWorldEnv, self).__init__()
        if grid_size_range is not None:
            lo, hi = grid_size_range
            if lo > hi:
                raise ValueError(f"grid_size_range min ({lo}) must be <= max ({hi})")
            self.grid_size_range = (int(lo), int(hi))
            self.grid_size = int(hi)
        else:
            self.grid_size_range = None
            self.grid_size = int(grid_size)

        b_lo, b_hi = berry_count_range
        if b_lo < 1 or b_hi > MAX_BERRIES or b_lo > b_hi:
            raise ValueError(f"berry_count_range must be within 1..{MAX_BERRIES} and min <= max")
        self.berry_count_range = (int(b_lo), int(b_hi))
        self.no_enemy = bool(no_enemy)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.obstacle_density = obstacle_density
        self.render_fps = render_fps
        
        self._episode_return = 0.0
        self._episode_extracted = False
        self._episode_full_extraction = False
        
        # Actions: 0: Up, 1: Down, 2: Left, 3: Right
        self.action_space = spaces.Discrete(4)
        
        self.agent_pos = None
        self.goal_pos = None
        self.enemy_pos = None
        self.n_berries = 0
        self.rewards_pos = np.zeros((MAX_BERRIES, 2), dtype=int)
        self.rewards_active = [False] * MAX_BERRIES
        self.rewards_collected = [False] * MAX_BERRIES
        self.obstacles = np.array([])
        
        self.goal_dist_map = None
        self.rewards_dist_maps = [None] * MAX_BERRIES
        self.current_step = 0
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32
        )

        self.renderer = None

    def _generate_random_obstacles(self):
        obstacles = []
        # Docelowo blokujemy część mapy (domyślnie 12% dla łatwiejszej eksploracji na 12x12).
        num_blocks = int((self.grid_size * self.grid_size) * self.obstacle_density)
        
        # Na mniejszych mapach używamy mniejszych bloków, żeby nie tworzyć wąskich korytarzy.
        while len(obstacles) < num_blocks:
            w = np.random.choice([1, 2])
            h = np.random.choice([1, 2])
            
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

    def _compute_distance_map(self, target_pos):
        dist_map = np.full((self.grid_size, self.grid_size), fill_value=-1, dtype=np.float32)
        
        if target_pos[0] < 0 or target_pos[0] >= self.grid_size or target_pos[1] < 0 or target_pos[1] >= self.grid_size:
            return dist_map

        queue = deque([(tuple(target_pos), 0)])
        visited = set([tuple(target_pos)])
        
        obs_set = set(tuple(o) for o in self.obstacles)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            current, dist = queue.popleft()
            dist_map[current[0], current[1]] = dist
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    n_pos = (nx, ny)
                    if n_pos not in visited and n_pos not in obs_set:
                        visited.add(n_pos)
                        queue.append((n_pos, dist + 1))
                        
        dist_map[dist_map == -1] = self.grid_size * 2
        return dist_map

    def _all_spawned_berries_collected(self):
        return all(
            self.rewards_collected[i]
            for i in range(MAX_BERRIES)
            if self.rewards_active[i]
        )

    def spawned_berry_count(self):
        return int(sum(self.rewards_active))

    def collected_berry_count(self):
        return int(
            sum(
                self.rewards_collected[i]
                for i in range(MAX_BERRIES)
                if self.rewards_active[i]
            )
        )

    def _spawn_berries(self, obs_set):
        b_lo, b_hi = self.berry_count_range
        self.n_berries = int(self.np_random.integers(b_lo, b_hi + 1))
        self.rewards_active = [False] * MAX_BERRIES
        self.rewards_collected = [False] * MAX_BERRIES
        self.rewards_pos = np.zeros((MAX_BERRIES, 2), dtype=int)
        self.rewards_dist_maps = [None] * MAX_BERRIES

        for i in range(self.n_berries):
            while True:
                r_pos = self.np_random.integers(0, self.grid_size, size=2)
                if tuple(r_pos) not in obs_set and \
                   not np.array_equal(r_pos, self.agent_pos) and \
                   not np.array_equal(r_pos, self.goal_pos) and \
                   not any(
                       self.rewards_active[j] and np.array_equal(r_pos, self.rewards_pos[j])
                       for j in range(MAX_BERRIES)
                   ):
                    if self._is_reachable(self.agent_pos, r_pos):
                        self.rewards_pos[i] = r_pos
                        self.rewards_active[i] = True
                        self.rewards_dist_maps[i] = self._compute_distance_map(r_pos)
                        break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._episode_return = 0.0
        self._episode_extracted = False
        self._episode_full_extraction = False

        if self.grid_size_range is not None:
            lo, hi = self.grid_size_range
            new_size = int(self.np_random.integers(lo, hi + 1))
            if new_size != self.grid_size and self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            self.grid_size = new_size
        
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
                self.goal_dist_map = self._compute_distance_map(self.goal_pos)
                break # Jeśli cel jest osiągalny, to mapa jest ok (nie zablokowana)

        # 5. Losuj 1–5 jagód (muszą być dostępne ze startu)
        self._spawn_berries(obs_set)

        # 6. Hunter position (optional — curriculum stage 0)
        if self.no_enemy:
            self.enemy_pos = self.agent_pos.copy()
        else:
            while True:
                self.enemy_pos = np.random.randint(0, self.grid_size, size=2)
                if tuple(self.enemy_pos) not in obs_set and tuple(self.enemy_pos) != tuple(self.goal_pos) and tuple(self.enemy_pos) != tuple(self.agent_pos):
                    if np.linalg.norm(self.enemy_pos - self.agent_pos) > 3.0:
                        break
        
        return self._get_obs(), {}

    def _get_obs(self):
        berry_feats = []
        for i in range(MAX_BERRIES):
            if not self.rewards_active[i]:
                berry_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            elif self.rewards_collected[i]:
                berry_feats.extend([1.0, 1.0, 0.0, 0.0, 0.0])
            else:
                smell = float(
                    self.rewards_dist_maps[i][self.agent_pos[0], self.agent_pos[1]]
                )
                delta = self.rewards_pos[i] - self.agent_pos
                berry_feats.extend([1.0, 0.0, float(delta[0]), float(delta[1]), smell])

        if self.no_enemy:
            enemy_dx, enemy_dy = 0.0, 0.0
        else:
            enemy_dx = float(self.enemy_pos[0] - self.agent_pos[0])
            enemy_dy = float(self.enemy_pos[1] - self.agent_pos[1])

        obs = np.array(
            [
                self.goal_pos[0] - self.agent_pos[0],
                self.goal_pos[1] - self.agent_pos[1],
                enemy_dx,
                enemy_dy,
                *berry_feats,
                *self._get_sensors(),
            ],
            dtype=np.float32,
        )
        obs /= float(self.grid_size)
        return obs
    
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
        
    def _enemy_chebyshev_dist(self, agent_pos, enemy_pos):
        return int(np.max(np.abs(agent_pos - enemy_pos)))

    def _enemy_danger_penalty(self, dist: int) -> float:
        if dist <= 1 or dist > ENEMY_DANGER_RADIUS:
            return 0.0
        linear = (ENEMY_DANGER_RADIUS + 1 - dist) * ENEMY_DANGER_COEF
        return linear * (ENEMY_DANGER_EXP_BASE ** (ENEMY_DANGER_RADIUS + 1 - dist))

    def step(self, action):
        self.current_step += 1
        old_dist = self.goal_dist_map[self.agent_pos[0], self.agent_pos[1]]
        old_dist_rewards = [
            self.rewards_dist_maps[i][self.agent_pos[0], self.agent_pos[1]]
            if self.rewards_active[i] and not self.rewards_collected[i]
            else 0.0
            for i in range(MAX_BERRIES)
        ]

        new_agent_pos = self.agent_pos.copy()
        if action == 0: new_agent_pos[1] = max(0, self.agent_pos[1] - 1)              
        elif action == 1: new_agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)  
        elif action == 2: new_agent_pos[0] = max(0, self.agent_pos[0] - 1)              
        elif action == 3: new_agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)  

        obs_set = set(tuple(o) for o in self.obstacles)
        reward = 0.0

        if tuple(new_agent_pos) in obs_set:
            # Uderzenie w mur!
            reward -= WALL_COLLISION_PENALTY
        else:
            self.agent_pos = new_agent_pos
            
        new_dist = self.goal_dist_map[self.agent_pos[0], self.agent_pos[1]]
        new_dist_rewards = [
            self.rewards_dist_maps[i][self.agent_pos[0], self.agent_pos[1]]
            if self.rewards_active[i] and not self.rewards_collected[i]
            else 0.0
            for i in range(MAX_BERRIES)
        ]

        # Move hunter (skipped in curriculum stage without enemy)
        if not self.no_enemy:
            enemy_action = np.random.choice([0, 1, 2, 3])
            new_enemy_pos = self.enemy_pos.copy()
            if enemy_action == 0: new_enemy_pos[1] = max(0, self.enemy_pos[1] - 1)
            elif enemy_action == 1: new_enemy_pos[1] = min(self.grid_size - 1, self.enemy_pos[1] + 1)
            elif enemy_action == 2: new_enemy_pos[0] = max(0, self.enemy_pos[0] - 1)
            elif enemy_action == 3: new_enemy_pos[0] = min(self.grid_size - 1, self.enemy_pos[0] + 1)

            if tuple(new_enemy_pos) not in obs_set:
                self.enemy_pos = new_enemy_pos

        # Step cost — pressure to finish before the episode cap
        reward -= STEP_PENALTY

        # Exit pull ramps slowly until most berries are collected (quadratic in collection fraction).
        spawned = self.spawned_berry_count()
        collected = self.collected_berry_count()
        frac = collected / spawned if spawned > 0 else 0.0
        goal_weight = GOAL_SHAPING_MAX * (frac ** 2)
        reward += (old_dist - new_dist) * goal_weight

        # Berry smell — divide by uncollected count so the last berry pulls as strongly as the first.
        uncollected = spawned - collected
        landing_bonus = TOTAL_BERRY_LANDING_REWARD / self.n_berries
        for i in range(MAX_BERRIES):
            if self.rewards_active[i] and not self.rewards_collected[i]:
                old_r = old_dist_rewards[i]
                new_r = new_dist_rewards[i]
                cap = self.grid_size * 2
                proximity = 1.0 - min(new_r, cap) / cap
                smell_weight = BERRY_SHAPING_MIN + (BERRY_SHAPING_MAX - BERRY_SHAPING_MIN) * proximity
                if uncollected > 0:
                    reward += (old_r - new_r) * smell_weight / uncollected
                if old_r > 0 and new_r == 0:
                    reward += landing_bonus
                if old_r == 1 and new_r > old_r:
                    reward -= BERRY_RETREAT_PENALTY

        terminated = False
        truncated = False

        # Hunter danger zone and death (disabled when no_enemy)
        if not self.no_enemy:
            new_enemy_dist = self._enemy_chebyshev_dist(self.agent_pos, self.enemy_pos)

            if new_enemy_dist <= 1:
                reward -= ENEMY_DEATH_PENALTY
                terminated = True
            elif new_enemy_dist <= ENEMY_DANGER_RADIUS:
                reward -= self._enemy_danger_penalty(new_enemy_dist)

        pickup_bonus = TOTAL_BERRY_PICKUP_REWARD / self.n_berries
        for i in range(MAX_BERRIES):
            if self.rewards_active[i] and not self.rewards_collected[i] and \
               np.array_equal(self.agent_pos, self.rewards_pos[i]):
                reward += pickup_bonus
                self.rewards_collected[i] = True
                if self._all_spawned_berries_collected():
                    reward += LAST_BERRY_COLLECTION_BONUS

        if np.array_equal(self.agent_pos, self.goal_pos):
            collected = self.collected_berry_count()
            spawned = self.spawned_berry_count()
            if collected == 0:
                reward -= EMPTY_EXIT_PENALTY
                terminated = True
            else:
                reward += TOTAL_EXTRACTION_REWARD * (collected / spawned)
                if self._all_spawned_berries_collected():
                    reward += FULL_EXTRACTION_BONUS
                    self._episode_full_extraction = True
                self._episode_extracted = True
                terminated = True

        if not terminated and self.current_step >= self.max_episode_steps:
            truncated = True

        self._episode_return += reward
        info = {}
        if terminated or truncated:
            collected = self.collected_berry_count()
            spawned = self.spawned_berry_count()
            info["extracted"] = float(self._episode_extracted)
            info["full_extraction"] = float(self._episode_full_extraction)
            info["all_berries_collected"] = float(self._all_spawned_berries_collected())
            info["any_berry_collected"] = float(collected > 0)
            info["berries_collected"] = float(collected)
            info["berries_spawned"] = float(spawned)
            info["enemy_death"] = float(
                terminated and not self._episode_extracted and not self.no_enemy
            )
            info["timeout"] = float(truncated)
            info["is_success"] = float(self._episode_extracted)

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.renderer is None:
            from renderer import PygameRenderer
            self.renderer = PygameRenderer(
                self.grid_size,
                self.render_mode,
                cell_size=48,
                fps=self.render_fps,
            )
            self._renderer_grid_size = self.grid_size
        elif getattr(self, "_renderer_grid_size", None) != self.grid_size:
            self.renderer.close()
            self.renderer = PygameRenderer(
                self.grid_size,
                self.render_mode,
                cell_size=48,
                fps=self.render_fps,
            )
            self._renderer_grid_size = self.grid_size
        
        active_positions = [self.rewards_pos[i] for i in range(MAX_BERRIES) if self.rewards_active[i]]
        active_collected = [self.rewards_collected[i] for i in range(MAX_BERRIES) if self.rewards_active[i]]

        enemy_for_render = None if self.no_enemy else self.enemy_pos

        return self.renderer.render(
            self.agent_pos, 
            self.goal_pos, 
            enemy_for_render, 
            active_positions, 
            active_collected,
            self.obstacles,
            getattr(self, 'goal_dist_map', None)
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

