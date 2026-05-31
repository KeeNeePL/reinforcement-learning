import pygame
import numpy as np

from visual_assets import VisualAssets


def _direction_from_delta(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    if dy != 0:
        return "down" if dy > 0 else "up"
    return "down"


class PygameRenderer:
    def __init__(
        self,
        grid_size,
        render_mode="human",
        cell_size=48,
        show_distance_map=False,
        fps=6,
    ):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.show_distance_map = show_distance_map
        self.fps = fps
        self.window_size = self.grid_size * self.cell_size

        self.window = None
        self.clock = None
        self.assets = None

        self.anim_frame = 0
        self.prev_agent_pos = None
        self.prev_enemy_pos = None
        self.agent_facing = "down"
        self.enemy_facing = "down"
        self._grass_background = None
        self._grass_background_grid_size = None

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Forest Escape — Deer vs Hunter")
            self.clock = pygame.time.Clock()
            self.assets = VisualAssets(self.cell_size)
            self._grass_background = self.assets.build_grass_floor(self.grid_size)
            self._grass_background_grid_size = self.grid_size

    def _cell_center(self, pos):
        return (
            int((pos[0] + 0.5) * self.cell_size),
            int((pos[1] + 0.5) * self.cell_size),
        )

    def _blit_centered(self, canvas, sprite, pos):
        cx, cy = self._cell_center(pos)
        rect = sprite.get_rect(center=(cx, cy))
        canvas.blit(sprite, rect)

    def _ensure_grass_background(self):
        if (
            self._grass_background is not None
            and self._grass_background_grid_size == self.grid_size
        ):
            return
        if self.assets is None:
            self.assets = VisualAssets(self.cell_size)
        self._grass_background = self.assets.build_grass_floor(self.grid_size)
        self._grass_background_grid_size = self.grid_size

    def _update_facing(self):
        if self.prev_agent_pos is not None:
            dx = int(self.agent_pos[0] - self.prev_agent_pos[0])
            dy = int(self.agent_pos[1] - self.prev_agent_pos[1])
            if dx != 0 or dy != 0:
                self.agent_facing = _direction_from_delta(dx, dy)
        if self.prev_enemy_pos is not None:
            dx = int(self.enemy_pos[0] - self.prev_enemy_pos[0])
            dy = int(self.enemy_pos[1] - self.prev_enemy_pos[1])
            if dx != 0 or dy != 0:
                self.enemy_facing = _direction_from_delta(dx, dy)

    def render(
        self,
        agent_pos,
        goal_pos,
        enemy_pos,
        reward_pos,
        reward_collected,
        obstacles=None,
        goal_dist_map=None,
    ):
        if self.render_mode == "human" and self.window is None:
            self.__init__(self.grid_size, self.render_mode, self.cell_size, self.show_distance_map)

        self.agent_pos = np.asarray(agent_pos)
        self.goal_pos = np.asarray(goal_pos)
        self.enemy_pos = np.asarray(enemy_pos)
        self._update_facing()

        if self.assets is None:
            self.assets = VisualAssets(self.cell_size)

        canvas = pygame.Surface((self.window_size, self.window_size))

        # --- Grass floor (static cached layer — no per-frame tile swapping) ---
        self._ensure_grass_background()
        canvas.blit(self._grass_background, (0, 0))

        # --- Rocks (obstacles) ---
        if obstacles is not None:
            for obs in obstacles:
                ox, oy = int(obs[0]), int(obs[1])
                if ox < 0 or oy < 0 or ox >= self.grid_size or oy >= self.grid_size:
                    continue
                rock = self.assets.rock_tile(ox, oy)
                canvas.blit(rock, (ox * self.cell_size, oy * self.cell_size))

        # Optional debug distance map
        if self.show_distance_map and goal_dist_map is not None:
            font = pygame.font.SysFont(None, int(self.cell_size * 0.45))
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    val = goal_dist_map[x, y]
                    if val < self.grid_size * 2:
                        text = font.render(str(int(val)), True, (40, 50, 40))
                        canvas.blit(
                            text,
                            text.get_rect(
                                center=(
                                    x * self.cell_size + self.cell_size // 2,
                                    y * self.cell_size + self.cell_size // 2,
                                )
                            ),
                        )

        # --- Exit shelter ---
        self._blit_centered(canvas, self.assets.exit_tile(), self.goal_pos)

        # --- Pickups (berries) ---
        rewards = reward_pos
        collected = reward_collected
        if isinstance(reward_pos, np.ndarray) and reward_pos.ndim == 1:
            rewards = [reward_pos]
            collected = [reward_collected]
        for r_pos, r_collected in zip(rewards, collected):
            if not r_collected:
                self._blit_centered(canvas, self.assets.reward_sprite(self.anim_frame), r_pos)

        # --- Hunter danger zone (pulsing) ---
        pulse = 0.85 + 0.15 * ((self.anim_frame % 8) / 8.0)
        danger = self.assets.danger_overlay()
        dw, dh = danger.get_size()
        scaled = pygame.transform.smoothscale(
            danger, (int(dw * pulse), int(dh * pulse))
        )
        ecx, ecy = self._cell_center(self.enemy_pos)
        canvas.blit(scaled, scaled.get_rect(center=(ecx, ecy)))

        # --- Hunter ---
        self._blit_centered(
            canvas,
            self.assets.hunter_sprite(self.enemy_facing, self.anim_frame),
            self.enemy_pos,
        )

        # --- Deer (agent) on top ---
        self._blit_centered(
            canvas,
            self.assets.deer_sprite(self.agent_facing, self.anim_frame),
            self.agent_pos,
        )

        self.prev_agent_pos = self.agent_pos.copy()
        self.prev_enemy_pos = self.enemy_pos.copy()
        self.anim_frame += 1

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.assets = None
            self._grass_background = None
            self._grass_background_grid_size = None
