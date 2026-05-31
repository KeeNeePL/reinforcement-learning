"""Procedural sprites and tiles for GridWorld visualization (no external image files)."""

import random
import pygame


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class VisualAssets:
    def __init__(self, cell_size: int = 48):
        self.cell_size = cell_size
        self._grass_tiles = []
        self._rock_tiles = []
        self._reward_frames = []
        self._exit_tile = None
        self._deer_frames = {"down": [], "up": [], "left": [], "right": []}
        self._hunter_frames = {"down": [], "up": [], "left": [], "right": []}
        self._danger_overlay = None
        self._build_all()

    def _build_all(self):
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()
        for i in range(4):
            self._grass_tiles.append(self._make_grass_tile(i))
        for i in range(5):
            self._rock_tiles.append(self._make_rock_tile(i))
        for i in range(4):
            self._reward_frames.append(self._make_reward_sprite(i))
        self._exit_tile = self._make_exit_tile()
        for direction in self._deer_frames:
            for frame in range(4):
                self._deer_frames[direction].append(self._make_deer_sprite(direction, frame))
                self._hunter_frames[direction].append(self._make_hunter_sprite(direction, frame))
        self._danger_overlay = self._make_danger_overlay()

    def _make_grass_tile(self, variant: int) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s))
        rng = random.Random(variant * 17 + s)
        # Similar greens so adjacent cells don't look like a flashing checkerboard.
        bases = [(58, 125, 68), (62, 130, 72), (55, 122, 65), (60, 128, 70)]
        base = bases[variant % len(bases)]
        surf.fill(base)

        for _ in range(18):
            x1 = rng.randint(0, s)
            y1 = rng.randint(0, s)
            x2 = x1 + rng.randint(-4, 4)
            y2 = y1 + rng.randint(-6, 2)
            shade = (
                _clamp(base[0] + rng.randint(-18, 12), 0, 255),
                _clamp(base[1] + rng.randint(-18, 12), 0, 255),
                _clamp(base[2] + rng.randint(-18, 12), 0, 255),
            )
            pygame.draw.line(surf, shade, (x1, y1), (x2, y2), 1)

        for _ in range(6):
            fx = rng.randint(4, s - 4)
            fy = rng.randint(4, s - 4)
            pygame.draw.circle(surf, (90, 165, 85), (fx, fy), 2)

        return surf

    def _make_rock_tile(self, variant: int) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        rng = random.Random(variant * 31 + 7)
        cx, cy = s // 2 + rng.randint(-3, 3), s // 2 + rng.randint(-2, 4)
        rx = rng.randint(s // 4, s // 3)
        ry = rng.randint(s // 5, s // 3)
        gray = (95 + variant * 8, 98 + variant * 6, 105 + variant * 5)
        pygame.draw.ellipse(surf, gray, (cx - rx, cy - ry, rx * 2, ry * 2))
        pygame.draw.ellipse(
            surf,
            (_clamp(gray[0] + 35, 0, 255), _clamp(gray[1] + 35, 0, 255), _clamp(gray[2] + 35, 0, 255)),
            (cx - rx + 4, cy - ry + 2, rx, ry // 2),
        )
        pygame.draw.ellipse(surf, (60, 62, 68), (cx - rx, cy + ry - 4, rx * 2, 6))
        return surf

    def _make_reward_sprite(self, frame: int) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        cx, cy = s // 2, s // 2 + (frame % 2)
        pulse = 1.0 + 0.08 * (frame % 2)
        r = int(s * 0.22 * pulse)
        pygame.draw.circle(surf, (180, 40, 50), (cx, cy), r)
        pygame.draw.circle(surf, (220, 60, 70), (cx - 3, cy - 3), r - 4)
        pygame.draw.line(surf, (45, 110, 40), (cx, cy + r), (cx, cy + r + 6), 3)
        sparkle = (255, 255, 200, 180 - frame * 30)
        for ox, oy in [(-r, -2), (r - 2, -r), (0, r)]:
            pygame.draw.circle(surf, sparkle, (cx + ox, cy + oy), 2)
        return surf

    def _make_exit_tile(self) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        # Wooden shelter / den entrance
        pygame.draw.rect(surf, (110, 75, 45), (8, s // 2 - 4, s - 16, s // 2))
        pygame.draw.rect(surf, (85, 55, 30), (12, s // 2 + 2, s - 24, s // 2 - 6))
        pygame.draw.ellipse(surf, (25, 35, 25), (14, s // 2 - 8, s - 28, s // 2 + 4))
        font = pygame.font.SysFont("arial", max(10, s // 5), bold=True)
        label = font.render("EXIT", True, (240, 230, 180))
        surf.blit(label, label.get_rect(center=(s // 2, s - 12)))
        return surf

    def _make_deer_sprite(self, direction: str, frame: int) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        bob = -3 if frame % 2 == 1 else 0
        cx, cy = s // 2, s // 2 + bob

        body_color = (160, 105, 65)
        pygame.draw.ellipse(surf, body_color, (cx - 14, cy - 6, 28, 16))

        head_offset = {"down": (0, 8), "up": (0, -12), "left": (-12, 0), "right": (12, 0)}[direction]
        hx, hy = cx + head_offset[0], cy + head_offset[1]
        pygame.draw.ellipse(surf, (175, 115, 72), (hx - 8, hy - 8, 16, 14))

        # Antlers
        if direction != "down":
            pygame.draw.line(surf, (90, 60, 40), (hx - 4, hy - 8), (hx - 10, hy - 16), 2)
            pygame.draw.line(surf, (90, 60, 40), (hx + 4, hy - 8), (hx + 10, hy - 16), 2)

        # Legs animation
        leg_y = cy + 8
        spread = 5 if frame % 2 == 0 else 2
        for lx in (-spread, spread):
            pygame.draw.line(surf, (120, 80, 50), (cx + lx, cy + 4), (cx + lx, leg_y + 6), 3)

        # White tail
        tail = {"down": (0, -10), "up": (0, 10), "left": (10, 0), "right": (-10, 0)}[direction]
        pygame.draw.circle(surf, (240, 240, 235), (cx + tail[0], cy + tail[1]), 4)

        return surf

    def _make_hunter_sprite(self, direction: str, frame: int) -> pygame.Surface:
        s = self.cell_size
        surf = pygame.Surface((s, s), pygame.SRCALPHA)
        bob = -2 if frame % 2 == 1 else 0
        cx, cy = s // 2, s // 2 + bob

        # Body
        pygame.draw.rect(surf, (45, 55, 70), (cx - 10, cy - 4, 20, 22), border_radius=4)
        # Head + hat
        pygame.draw.circle(surf, (210, 180, 150), (cx, cy - 12), 8)
        pygame.draw.rect(surf, (35, 30, 25), (cx - 11, cy - 20, 22, 8))

        # Rifle / direction hint
        offsets = {"down": (0, 10, 0, 18), "up": (0, -10, 0, -18), "left": (-14, 0, -22, 0), "right": (14, 0, 22, 0)}
        ox1, oy1, ox2, oy2 = offsets[direction]
        pygame.draw.line(surf, (55, 45, 35), (cx + ox1, cy + oy1), (cx + ox2, cy + oy2), 3)

        # Legs
        step = 4 if frame % 2 == 0 else -4
        pygame.draw.line(surf, (30, 35, 45), (cx - 5, cy + 16), (cx - 5 + step, cy + 24), 3)
        pygame.draw.line(surf, (30, 35, 45), (cx + 5, cy + 16), (cx + 5 - step, cy + 24), 3)

        return surf

    def _make_danger_overlay(self) -> pygame.Surface:
        size = self.cell_size * 5
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        for r in range(center, 0, -4):
            alpha = int(35 * (r / center))
            pygame.draw.circle(surf, (220, 40, 40, alpha), (center, center), r)
        return surf

    def grass_tile(self, x: int, y: int) -> pygame.Surface:
        # Fixed variant per cell — never tied to anim_frame (avoids blinking).
        idx = (x * 3 + y * 7) % len(self._grass_tiles)
        return self._grass_tiles[idx]

    def build_grass_floor(self, grid_size: int) -> pygame.Surface:
        """Pre-compose the full grass layer once per map size."""
        s = self.cell_size
        floor = pygame.Surface((grid_size * s, grid_size * s))
        for x in range(grid_size):
            for y in range(grid_size):
                floor.blit(self.grass_tile(x, y), (x * s, y * s))
        return floor

    def rock_tile(self, x: int, y: int) -> pygame.Surface:
        return self._rock_tiles[(x * 7 + y * 13) % len(self._rock_tiles)]

    def reward_sprite(self, anim_frame: int) -> pygame.Surface:
        return self._reward_frames[anim_frame % len(self._reward_frames)]

    def exit_tile(self) -> pygame.Surface:
        return self._exit_tile

    def deer_sprite(self, direction: str, anim_frame: int) -> pygame.Surface:
        frames = self._deer_frames.get(direction, self._deer_frames["down"])
        return frames[anim_frame % len(frames)]

    def hunter_sprite(self, direction: str, anim_frame: int) -> pygame.Surface:
        frames = self._hunter_frames.get(direction, self._hunter_frames["down"])
        return frames[anim_frame % len(frames)]

    def danger_overlay(self) -> pygame.Surface:
        return self._danger_overlay
