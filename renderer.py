import pygame
import numpy as np

class PygameRenderer:
    def __init__(self, grid_size, render_mode="human", cell_size=40):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.window_size = self.grid_size * self.cell_size
        
        self.window = None
        self.clock = None
        
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("GridWorld RL Environment")
            self.clock = pygame.time.Clock()
            
    def render(self, agent_pos, goal_pos, enemy_pos, reward_pos, reward_collected, obstacles=None):
        # Odtwórz instancję Pygame, jeżeli zamknięto okno a program chce renderować dalej
        if self.render_mode == "human" and self.window is None:
            self.__init__(self.grid_size, self.render_mode, self.cell_size)
            
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # Białe tło
        
        # Rysowanie siatki
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                pygame.draw.rect(canvas, (200, 200, 200), rect, 1)

        # Rysowanie przeszkód (Szare)
        if obstacles is not None:
            for obs in obstacles:
                pygame.draw.rect(
                    canvas,
                    (128, 128, 128),
                    pygame.Rect(
                        obs[0] * self.cell_size,
                        obs[1] * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )

        # Rysowanie wyjścia (Zielony)
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                goal_pos[0] * self.cell_size,
                goal_pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        # Rysowanie nagrody (Złote kółko)
        if not reward_collected:
            pygame.draw.circle(
                canvas,
                (255, 215, 0),
                (
                    int((reward_pos[0] + 0.5) * self.cell_size),
                    int((reward_pos[1] + 0.5) * self.cell_size),
                ),
                self.cell_size // 3,
            )

        # Strefa niebezpieczeństwa (Jasnoczerwony, przezroczysty)
        danger_surface = pygame.Surface((self.cell_size * 5, self.cell_size * 5), pygame.SRCALPHA)
        danger_surface.fill((255, 0, 0, 50))  # 50 to poziom przezroczystości
        
        danger_x = (enemy_pos[0] - 2) * self.cell_size
        danger_y = (enemy_pos[1] - 2) * self.cell_size
        canvas.blit(danger_surface, (danger_x, danger_y))

        # Rysowanie przeciwnika (Czerwony)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                enemy_pos[0] * self.cell_size,
                enemy_pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        # Rysowanie agenta (Niebieski)
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                agent_pos[0] * self.cell_size,
                agent_pos[1] * self.cell_size,
                self.cell_size,
                self.cell_size,
            ),
        )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(4)  # 4 FPS dla płynności podglądu
            return None
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
