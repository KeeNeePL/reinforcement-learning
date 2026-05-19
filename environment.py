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
        
        # State: Agent, Goal, Enemy, Reward (x, y coordinates) -> 8 values
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(8,), dtype=np.float32
        )
        
        self.agent_pos = None
        self.goal_pos = None
        self.enemy_pos = None
        self.reward_pos = None
        self.reward_collected = False

        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize positions
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.grid_size - 1, self.grid_size - 1])
        
        # Random enemy position (must be at least 3 cells away from agent to avoid instant death)
        while True:
            self.enemy_pos = np.random.randint(0, self.grid_size, size=2)
            if np.max(np.abs(self.agent_pos - self.enemy_pos)) > 2:
                break
                
        self.reward_pos = np.array([self.grid_size // 3, (self.grid_size // 3) * 2])
        self.reward_collected = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.agent_pos, self.goal_pos, self.enemy_pos, self.reward_pos
        ]).astype(np.float32)

    def step(self, action):
        # Move Agent
        if action == 0: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)              # Up (decreases Y)
        elif action == 1: self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)  # Down (increases Y)
        elif action == 2: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)              # Left (decreases X)
        elif action == 3: self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)  # Right (increases X)

        # Move Enemy (Random)
        enemy_action = np.random.choice([0, 1, 2, 3])
        if enemy_action == 0: self.enemy_pos[1] = max(0, self.enemy_pos[1] - 1)
        elif enemy_action == 1: self.enemy_pos[1] = min(self.grid_size - 1, self.enemy_pos[1] + 1)
        elif enemy_action == 2: self.enemy_pos[0] = max(0, self.enemy_pos[0] - 1)
        elif enemy_action == 3: self.enemy_pos[0] = min(self.grid_size - 1, self.enemy_pos[0] + 1)

        reward = -0.05 # Step penalty to encourage speed
        terminated = False

        # Enemy collision (Distance <= 2)
        if np.max(np.abs(self.agent_pos - self.enemy_pos)) <= 2:
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
            self.reward_collected
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
