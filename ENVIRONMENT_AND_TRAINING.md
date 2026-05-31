# GridWorld RL — Environment & Training Summary

This document describes the current learning setup: map generation, observations, rewards/penalties, episode rules, and how PPO training/evaluation is run.

---

## 1. Task objective

The agent must complete a **two-phase mission** on each random map:

1. **Collect** both pickup rewards on the map.
2. **Extract** at the exit (goal) — but extraction only counts as success if **both** rewards were collected first.

The agent must also deal with:

- **Obstacles** (walls) — movement blocked, small penalty.
- **Moving enemy** — random movement each step; proximity penalties and death on contact.

---

## 2. Environment (`GridWorldEnv`)

**Implementation:** `environment.py`  
**Framework:** Gymnasium

### 2.1 Configurable parameters

| Parameter | Default (training) | Description |
|-----------|-------------------|-------------|
| `grid_size` | `12` (CLI: `--grid-size`) | Width/height of the square grid |
| `max_episode_steps` | `300` | Hard step cap; episode ends with `truncated=True` |
| `obstacle_density` | `0.12` | ~12% of cells targeted as obstacles |
| `render_mode` | `None` (train) / `"human"` (visual demo) | Pygame rendering optional |

### 2.2 Map generation (`reset`)

Each episode builds a **new random map**:

1. **Obstacles** — rectangular blocks (1×1, 1×2, 2×1, 2×2) until ~`grid_size² × obstacle_density` cells are blocked (duplicates removed).
2. **Agent start** — random free cell.
3. **Exit (goal)** — random free cell, at least `grid_size / 2` Euclidean distance from start; map must be **reachable** (BFS).
4. **Two pickup rewards** — random free cells, reachable from start, not on goal/agent/other reward.
5. **Enemy** — random free cell, not on goal/agent, at least **> 3** units from agent (Euclidean).

**Distance maps** (BFS over walkable cells, obstacles blocked) are precomputed for:

- Goal (`goal_dist_map`)
- Each uncollected reward (`rewards_dist_maps`)

These maps power **dense shaping rewards** (distance decreases give positive reward).

### 2.3 Actions

Discrete **4** actions:

| ID | Action |
|----|--------|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

Movement is axis-aligned, one cell per step, clamped to grid bounds.

### 2.4 Enemy behavior

After the agent moves, the enemy takes **one random action** (same 4 directions).  
If the new enemy cell is an obstacle, the enemy **does not move** that step.

---

## 3. Observation space (18 values)

**Declared space:** `Box(-1, 1, shape=(18,), float32)` — grid-independent bounds because values are normalized.

**Vector layout** (before normalization, then divided by `grid_size`):

| Index | Size | Content |
|-------|------|---------|
| 0–1 | 2 | Delta to **exit** (goal_pos − agent_pos) |
| 2–3 | 2 | Delta to **enemy** |
| 4–5 | 2 | Delta to **reward 1** (or `[0,0]` if collected) |
| 6 | 1 | **Has reward 1** flag (`1.0` if collected, else `0.0`) |
| 7–8 | 2 | Delta to **reward 2** |
| 9 | 1 | **Has reward 2** flag |
| 10–17 | 8 | **Ray sensors** — distance to wall/obstacle/border in 8 directions (cardinal + diagonal) |

Sensors cast rays until grid edge or obstacle; stored distance is in cells, then normalized.

---

## 4. Reward & penalty structure (`step`)

Rewards are computed **every step** in a fixed order. Understanding this order matters for debugging policy behavior.

### 4.1 Movement & obstacles

| Event | Effect | Notes |
|-------|--------|-------|
| Move into obstacle | **−0.5** | Agent position **unchanged** |
| Valid move | 0 | Position updates |

### 4.2 Per-step costs & shaping (dense)

| Component | Value | When applied |
|-----------|-------|----------------|
| **Step penalty** | **−0.03** | Every step |
| **Goal distance shaping** | `(old_goal_dist − new_goal_dist) × 0.6` | Only if **both** rewards already collected (`goal_weight = 0.6`, else `0`) |
| **Reward pickup shaping** | `(old_reward_dist − new_reward_dist) × 0.3` | Per **uncollected** reward item |

Distance uses precomputed BFS maps (path length through free cells).

**Design intent:** Before both pickups, the agent is **not** shaped toward the exit (goal weight = 0). After both are collected, shaping guides extraction.

### 4.3 Enemy interaction

Distance metric: **Chebyshev** (L∞): `max(|dx|, |dy|)`.

| Enemy distance | Effect | Episode |
|----------------|--------|---------|
| **≤ 1** (adjacent or same cell) | `reward = **−8.0**` (replaces step reward sum for that step) | **`terminated = True`** (death) |
| **2 or 3** | `−(4 − enemy_dist) × 0.2` | Continues (soft danger zone) |
| **≥ 4** | No enemy penalty | Continues |

Examples for soft zone:

- Distance 2 → `−0.4`
- Distance 3 → `−0.2`

### 4.4 Sparse events (pickups & exit)

| Event | Reward | Episode |
|-------|--------|---------|
| Step on **uncollected** pickup | **+15.0** | Marks that reward collected |
| Step on **exit** with **both** rewards collected | **+50.0** | **`terminated = True`** (successful extraction) |
| Step on **exit** without both rewards | **−10.0** | Continues (penalty for premature extract attempt) |

### 4.5 Time limit

| Event | Effect |
|-------|--------|
| `current_step >= max_episode_steps` (300) | **`truncated = True`**, no extra penalty beyond step costs already incurred |

### 4.6 Typical successful episode reward (rough)

If agent collects both rewards and extracts in ~60 steps with few wall hits and moderate enemy pressure, episode return can be **strongly positive** (e.g. +15 + +15 + +50 plus shaping minus step costs).

Failed episodes often end by **enemy death (−8)** or **timeout** after many **−0.03** steps.

---

## 5. Episode termination summary

| End reason | Flag | Typical cause |
|------------|------|----------------|
| Successful extraction | `terminated` | On goal with both rewards collected |
| Enemy death | `terminated` | Chebyshev distance ≤ 1 to enemy |
| Time limit | `truncated` | 300 steps reached |
| Premature exit visit | Neither | Only −10 penalty; episode continues |

---

## 6. Training process (PPO)

**Script:** `train.py`  
**Algorithm:** Stable-Baselines3 **PPO** with **MLP** policy

### 6.1 Environment vectorization

- **8 parallel envs** (`make_vec_env`, `n_envs=8`)
- Each env: new random map every `reset()`
- Default training map: **`grid_size=12`**, `max_episode_steps=300`, `obstacle_density=0.12`

### 6.2 PPO hyperparameters

| Parameter | Value |
|-----------|-------|
| Policy | `MlpPolicy` |
| `n_steps` | 2048 (per env, per rollout) |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `ent_coef` | `0.01` (default, CLI: `--ent-coef`) |
| Learning rate | **Linear decay** from `--initial-learning-rate` (default `3e-4`) to `0` over the run |
| Device | **CPU** (configured in `train.py`) |

### 6.3 Checkpoints & logging

- Checkpoints every **10,000** timesteps → `./checkpoints/ppo_model_*_steps.zip`
- TensorBoard logs → `./ppo_gridworld_tensorboard/`
- Final model → `ppo_gridworld_final.zip`

### 6.4 Resume / transfer

```bash
python train.py --grid-size 16 --total-timesteps 300000 \
  --resume-model "./checkpoints/ppo_model_XXXXX_steps.zip" \
  --initial-learning-rate 1e-4 --ent-coef 0.005
```

- Loads weights from checkpoint; **`reset_num_timesteps=False`** keeps global step counter.
- Observation/action spaces from checkpoint are overridden so training can resume on a **different `grid_size`** (observations are normalized, so transfer is supported).

### 6.5 What one “timestep” means

- **1 timestep = 1 action in 1 parallel env**
- With `n_envs=8`, one “cycle” across all envs counts as **8 timesteps**
- **Not** one map and **not** one episode

---

## 7. Evaluation process

**Script:** `evaluate.py`

| Group | Grid | Purpose |
|-------|------|---------|
| In-distribution | **12×12** | Train-like conditions |
| Generalization | **16×16** | Harder / OOD size |

Default: **500 episodes** per group, `max_steps=300`, stochastic policy unless `--deterministic`.

**Metrics tracked:**

- Extraction success rate (goal + both rewards)
- Any / all rewards collected rate
- Mean rewards per episode
- Enemy death rate
- Timeout rate
- Average episode reward & steps

---

## 8. Visualization demo

**Script:** `main.py`

- Renders with Pygame (`render_mode="human"`)
- Uses `deterministic=True` by default in the simple version — can cause **repeated wall bumps** (same obs → same action) when movement is blocked
- For behavior closer to evaluation, use **stochastic** `predict` or match `grid_size` / checkpoint from training

---

## 9. Reward design philosophy (current)

```text
Phase A — Collection
  • Shape toward uncollected rewards (0.3 × distance improvement)
  • No shaping toward exit (goal_weight = 0)
  • Pickup bonus +15 each

Phase B — Extraction (after both collected)
  • Shape toward exit (0.6 × distance improvement)
  • Exit bonus +50 on success

Constraints / risks
  • Step cost −0.03 encourages efficiency
  • Enemy soft zone + death −8 discourage rushing through danger
  • Premature exit −10 discourages skipping pickups
  • Wall −0.5 discourages bumping obstacles
```

---

## 10. File map

| File | Role |
|------|------|
| `environment.py` | Gym env: map gen, step logic, rewards |
| `train.py` | PPO training & resume |
| `evaluate.py` | Batch metrics on 12×12 and 16×16 |
| `main.py` | Visual playtest |
| `plot_training.py` | Charts from TensorBoard logs |
| `renderer.py` | Pygame rendering |

---

*Last aligned with `environment.py` and `train.py` as of project state at documentation time.*
