from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class GameMetricsCallback(BaseCallback):
    """Log GridWorld episode stats to TensorBoard from training rollouts."""

    def __init__(self, window: int = 200, no_enemy: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.no_enemy = no_enemy
        self._episodes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if isinstance(ep, dict) and "extracted" in ep:
                self._episodes.append(ep)
        return True

    def _mean(self, episodes, key: str) -> float:
        return float(np.mean([e.get(key, 0.0) for e in episodes]))

    def _on_rollout_end(self) -> None:
        if not self._episodes:
            return

        episodes = list(self._episodes)
        self.logger.record("game/extraction_rate", self._mean(episodes, "extracted"))
        self.logger.record("game/full_extraction_rate", self._mean(episodes, "full_extraction"))
        self.logger.record(
            "game/all_berries_collected_rate",
            self._mean(episodes, "all_berries_collected"),
        )
        self.logger.record(
            "game/any_berry_collected_rate",
            self._mean(episodes, "any_berry_collected"),
        )
        self.logger.record(
            "game/mean_berries_collected",
            self._mean(episodes, "berries_collected"),
        )
        self.logger.record(
            "game/mean_berries_spawned",
            self._mean(episodes, "berries_spawned"),
        )
        self.logger.record("game/timeout_rate", self._mean(episodes, "timeout"))
        if not self.no_enemy:
            self.logger.record("game/enemy_death_rate", self._mean(episodes, "enemy_death"))
