# Reward, Penalty & Observation Reference

This document describes every reward and penalty used in `GridWorldEnv` (`environment.py`), plus the **37-dimensional observation vector** the policy sees. All reward values are defined as module-level constants unless noted as inline literals.

**Task summary:** Collect berries on a grid, optionally avoid a hunter, then reach the shelter (goal). Partial extraction is allowed; full extraction is incentivized but not required.

**Distance metric:** Berry smell and goal shaping use **BFS path distance** through walkable cells (obstacles block movement). Unreachable cells are treated as distance `grid_size * 2`. Hunter logic uses **Chebyshev distance** (max of |Δx|, |Δy|).

---

## Constants at a glance

| Constant | Value | Role |
|----------|-------|------|
| `MAX_BERRIES` | 5 | Max berries per episode; caps observation slots |
| `TOTAL_BERRY_PICKUP_REWARD` | 75 (`15 × MAX_BERRIES`) | Total pickup bonus if all berries collected |
| `TOTAL_BERRY_LANDING_REWARD` | 25 (`5 × MAX_BERRIES`) | Total “arrive at berry cell” bonus if all collected |
| `LAST_BERRY_COLLECTION_BONUS` | 10 | One-time bonus when the final spawned berry is picked |
| `STEP_PENALTY` | 0.08 | Per-step living cost |
| `TOTAL_EXTRACTION_REWARD` | 50 | Max shelter reward (scaled by collection fraction) |
| `FULL_EXTRACTION_BONUS` | 20 | Extra shelter reward when every spawned berry was collected |
| `GOAL_SHAPING_MAX` | 1.2 | Max per-step goal pull coefficient (at 100% berries) |
| `BERRY_SHAPING_MIN` | 0.15 | Berry smell coefficient when far |
| `BERRY_SHAPING_MAX` | 0.40 | Berry smell coefficient when on/adjacent to berry |
| `BERRY_RETREAT_PENALTY` | 0.5 | Stepping away from an adjacent uncollected berry |
| `WALL_COLLISION_PENALTY` | 5.0 | Moving into an obstacle |
| `EMPTY_EXIT_PENALTY` | 5.0 | Reaching shelter with zero berries |
| `ENEMY_DEATH_PENALTY` | 25 | Penalty on hunter kill step |
| `ENEMY_DANGER_RADIUS` | 4 | Chebyshev cells for danger-zone penalty |
| `ENEMY_DANGER_COEF` | 0.5 | Base scale for danger-zone penalty |
| `ENEMY_DANGER_EXP_BASE` | 1.65 | Exponential multiplier when hunter is closer |

---

## Per-step rewards (every action)

Applied in `step()` after movement, in roughly this order.

### 1. Wall collision penalty

- **Formula:** `reward -= WALL_COLLISION_PENALTY` → **−5.0** if the agent tries to move into an obstacle.
- **Why:** Discourages banging into walls. The agent stays in place; no distance-based shaping is lost beyond that step’s opportunity cost.

### 2. Step penalty (time pressure)

- **Formula:** `reward -= STEP_PENALTY` → **−0.08** every step.
- **Why:** Creates urgency to finish before the episode cap (`max_episode_steps`, default 300). Without this, wandering and timeouts are less costly. Tuned down from 0.10 to reduce harshness on longer berry routes.

### 3. Goal shaping (shelter pull)

- **Formula:**
  ```
  goal_weight = GOAL_SHAPING_MAX * (collected / spawned)   # 0 if no berries collected
  reward += (old_goal_dist - new_goal_dist) * goal_weight
  ```
- **Values:** `GOAL_SHAPING_MAX = 1.2`. Weight scales linearly with the fraction of spawned berries already collected.
- **Examples (moving 1 cell closer to shelter):**
  - 0 berries collected → weight 0 → **+0**
  - 2 of 5 berries → weight 0.48 → **+0.48**
  - All berries → weight 1.2 → **+1.2**
- **Why:** The shelter is always *reachable*, but the agent is only *pulled* toward it after collecting berries. This teaches “berries first, then exit,” while still allowing early partial extraction (see extraction reward). Partial pull during partial collection avoids a hard gate that blocks learning exit behavior.

### 4. Berry smell shaping (all uncollected berries)

Each active, uncollected berry contributes independently every step.

- **Distance shaping:**
  ```
  proximity = 1 - min(new_r, cap) / cap
  cap = grid_size * 2
  smell_weight = BERRY_SHAPING_MIN + (BERRY_SHAPING_MAX - BERRY_SHAPING_MIN) * proximity
  reward += (old_r - new_r) * smell_weight
  ```
  where `old_r` / `new_r` are BFS distances to that berry before/after the move.

- **Smell weight range:** **0.15** (`BERRY_SHAPING_MIN`, far) → **0.40** (`BERRY_SHAPING_MAX`, adjacent/on berry). Stronger pull when already near a berry.

- **Landing bonus (per berry):**
  ```
  landing_bonus = TOTAL_BERRY_LANDING_REWARD / n_berries   # 25 / n_berries
  if old_r > 0 and new_r == 0: reward += landing_bonus
  ```
  Fires when the agent steps onto the berry’s cell (distance becomes 0).

- **Retreat penalty (per berry):**
  ```
  if old_r == 1 and new_r > old_r: reward -= BERRY_RETREAT_PENALTY
  ```
  Small penalty for stepping *away* from a berry you were adjacent to.

- **Why “all berries at once”:** Every uncollected berry adds its own gradient. On multi-berry maps the agent feels pulls toward multiple targets simultaneously (summed), which encourages routing without picking a single “nearest only” target artificially.

- **Scaling with berry count:** Pickup and landing totals are fixed across episodes (`75` and `25` if all berries collected), divided by `n_berries` per berry so a 1-berry episode and a 5-berry episode have comparable *maximum* berry-related reward.

### 5. Hunter danger-zone penalty *(only when `no_enemy=False`)*

- **Active when:** `1 < new_enemy_dist <= ENEMY_DANGER_RADIUS` (adjacent dist 1 is death, not this penalty).
- **Formula (`_enemy_danger_penalty`):**
  ```
  linear = (ENEMY_DANGER_RADIUS + 1 - dist) * ENEMY_DANGER_COEF
  penalty = linear * (ENEMY_DANGER_EXP_BASE ** (ENEMY_DANGER_RADIUS + 1 - dist))
  reward -= penalty
  ```
- **Example values (per step, before other rewards):**

  | Chebyshev dist | Approx. penalty |
  |----------------|-----------------|
  | 2 | ~1.36 |
  | 3 | ~0.66 |
  | 4 | ~0.25 |

- **Why:** Continuous “heat” near the hunter, exponentially worse when closer. Teaches staying out of the threat bubble, not only avoiding the kill tile.

---

## Event-based rewards (triggered on specific conditions)

### 7. Berry pickup bonus

- **When:** Agent occupies an uncollected berry cell after movement.
- **Formula:** `reward += TOTAL_BERRY_PICKUP_REWARD / n_berries` → **75 / n_berries** per berry.
- **Examples:** 1 berry → +75; 5 berries → +15 each (75 total if all picked).
- **Why:** Sparse milestone reward confirming collection. Scaled so total pickup value is consistent regardless of spawn count.

### 8. Last berry collection bonus

- **When:** Picking up the final remaining spawned berry in the episode.
- **Formula:** `reward += LAST_BERRY_COLLECTION_BONUS` → **+10** (once per episode).
- **Why:** Extra incentive to finish the collection phase before heading to shelter, on top of max goal shaping kicking in.

### 9. Shelter / extraction rewards

- **When:** Agent reaches `goal_pos` after movement.

| Condition | Reward | Episode end |
|-----------|--------|-------------|
| 0 berries collected | **−EMPTY_EXIT_PENALTY (−5)** | `terminated` (failed extract) |
| ≥1 berry, partial | `50 * (collected / spawned)` | `terminated`, `extracted=True` |
| All spawned berries | above + **+20** `FULL_EXTRACTION_BONUS` | `terminated`, `full_extraction=True` |

- **Examples (5 berries spawned):**

  | Collected | Extraction reward | Full bonus | Total on exit |
  |-----------|-------------------|------------|---------------|
  | 0 | — | — | −5 |
  | 1 | 10 | 0 | 10 |
  | 3 | 30 | 0 | 30 |
  | 5 | 50 | 20 | 70 |

- **Why:** Partial extraction supports curriculum learning and hunter scenarios where full clears are hard. The +20 bonus and stronger goal shaping still prefer complete runs without blocking partial success.

### 10. Hunter death penalty *(only when `no_enemy=False`)*

- **When:** Chebyshev distance to hunter ≤ 1 after the hunter moves.
- **Formula:** `reward -= ENEMY_DEATH_PENALTY` → **−25**, plus any danger/step rewards on that same step. Episode `terminated`.
- **Why:** Clear failure signal. Additive (does not zero out other step rewards) so the value function can still attribute pre-death behavior. Sized to hurt but not dwarf a successful full extraction (~70+).

---

## What is *not* penalized directly

| Outcome | Reward on last step | Notes |
|---------|---------------------|-------|
| **Timeout** (`truncated`) | Only the usual −0.08 step penalty | No extra timeout penalty constant; failure is implicit via lost extraction reward + accumulated step cost |
| **Standing still** (into wall) | −WALL_COLLISION_PENALTY + −STEP_PENALTY | No separate idle penalty |
| **Hunter enabled but far away** | No danger terms | Hunter only penalizes inside danger radius |

---

## Episode success metrics (`info` at episode end)

Logged for TensorBoard / evaluation via `GAME_INFO_KEYS`:

| Key | Meaning |
|-----|---------|
| `is_success` / `extracted` | Reached shelter with ≥1 berry |
| `full_extraction` | Reached shelter with every spawned berry |
| `all_berries_collected` | All berries picked (may still be on map if not yet at shelter) |
| `any_berry_collected` | At least one berry picked |
| `enemy_death` | Terminated by hunter without extracting |
| `timeout` | Hit `max_episode_steps` without terminal extract/death |

---

## Typical reward flow (conceptual)

```
Every step:
  −STEP_PENALTY
  + goal shaping     (0 until berries collected; ramps up)
  + berry smell      (sum over all uncollected berries)
  − danger penalty   (if hunter nearby, not adjacent)
  −WALL_COLLISION_PENALTY (if wall)

On events:
  + pickup / landing / last-berry bonuses
  + extraction at shelter (or −EMPTY_EXIT_PENALTY)
  −ENEMY_DEATH_PENALTY (hunter adjacent)
```

**Design intent:** Berry smell and zero initial goal weight keep early behavior focused on collection. Goal pull and extraction scale with progress so exit becomes attractive after partial success. Hunter danger terms add local avoidance without overriding the main task. Fixed totals divided by `n_berries` keep reward magnitude stable across curriculum phases that change berry count.

---

## Observation vector (`OBS_DIM = 37`)

Built in `GridWorldEnv._get_obs()`. The policy receives a single `float32` vector of length **37**, declared as `spaces.Box(low=-1.0, high=1.0, shape=(37,))`. **Every component is divided by `grid_size`** before being returned, so values are roughly map-scale-invariant when grid size changes between episodes.

### Layout (indices 0–36)

| Index range | Size | Name | Description |
|-------------|------|------|-------------|
| 0–1 | 2 | Goal delta | `(goal_x − agent_x, goal_y − agent_y)` — shelter position relative to agent |
| 2–3 | 2 | Enemy delta | `(enemy_x − agent_x, enemy_y − agent_y)`; **(0, 0)** when `no_enemy=True` |
| 4–28 | 25 | Berry slots | 5 fixed slots × 5 features each (see below) |
| 29–36 | 8 | Ray sensors | Clear-cell distance in 8 directions until wall, obstacle, or map edge |

Formula from code: `2 + 2 + MAX_BERRIES×5 + 8 = 37`.

### Berry slot features (5 per slot, slots 0–4)

Slots are **fixed index positions** — slot 0 is not necessarily “the first spawned berry.” Each episode spawns `n_berries` (1–5) into the first `n_berries` slots; unused slots are zeroed.

| Feature | Index within slot | Meaning |
|---------|-------------------|---------|
| `active` | +0 | `1` if this slot has a berry this episode, else `0` |
| `collected` | +1 | `1` if picked up, `0` if still on map; `0` when inactive |
| `delta_x` | +2 | Berry grid x minus agent x (uncollected only) |
| `delta_y` | +3 | Berry grid y minus agent y (uncollected only) |
| `smell` | +4 | BFS path distance from agent to berry (uncollected only) |

**Three slot states:**

| State | Vector `[active, collected, Δx, Δy, smell]` |
|-------|-----------------------------------------------|
| Not spawned this episode | `[0, 0, 0, 0, 0]` |
| Spawned, already collected | `[1, 1, 0, 0, 0]` |
| Spawned, uncollected | `[1, 0, Δx, Δy, smell]` |

- **Smell distance** uses the same BFS maps as berry reward shaping. Unreachable cells map to `grid_size × 2` before normalization.
- **Why smell + delta:** Delta gives direction; smell gives path length around obstacles (the policy cannot infer maze distance from deltas alone).

### Ray sensors (indices 29–36)

Eight rays cast from the agent in fixed directions until the ray hits a **map boundary** or **obstacle**:

| Index | Direction |
|-------|-----------|
| 29 | Up |
| 30 | Down |
| 31 | Left |
| 32 | Right |
| 33 | Up-left (diagonal) |
| 34 | Up-right (diagonal) |
| 35 | Down-left (diagonal) |
| 36 | Down-right (diagonal) |

Each value is the **count of clear cells** along that ray (not including the agent’s cell). Berries, the shelter, and the hunter do **not** block rays — only walls/obstacles and edges do.

### Normalization

```python
obs = [goal_Δ, enemy_Δ, berry_slots…, sensors…]  # raw grid units
obs /= grid_size
```

Examples on a 16×16 map:

- Goal 8 cells east → index 0 ≈ `0.5`
- Berry smell distance 4 → slot smell feature ≈ `0.25`
- Sensor sees 3 clear cells north → index 29 ≈ `0.1875`

### What the observation does *not* include

- Absolute agent position (only relative deltas)
- One-hot map layout (obstacles are implicit via ray sensors only)
- Hunter distance as a scalar (only Δx, Δy — Chebyshev danger zone is not encoded explicitly)
- Number of berries spawned (must be inferred from how many slots have `active=1`)
- Fraction of berries collected (must be inferred from `collected` flags)

### Relation to rewards

| Obs feature | Related reward mechanism |
|-------------|-------------------------|
| Goal delta | Goal shaping toward shelter |
| Berry smell (per slot) | Same BFS maps as berry smell shaping |
| Berry deltas | Pickup when agent reaches berry cell |
| Enemy delta | Danger-zone penalty uses Chebyshev dist (not identical to Euclidean delta in obs) |
| Ray sensors | No direct reward; helps navigation around walls |

---

## Source of truth

Reward logic: `environment.py` → `GridWorldEnv.step()`.  
Observation logic: `environment.py` → `GridWorldEnv._get_obs()` and `_get_sensors()`.  
If this document and the code disagree, **trust the code** and update this file.
