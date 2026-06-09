# GridWorld Reinforcement Learning

A custom GridWorld reinforcement learning environment built with Gymnasium and trained using Stable-Baselines3 (PPO). The project challenges an AI agent to complete a two-phase mission on randomly generated maps while avoiding obstacles and a moving enemy.

## Task Objective

The agent must navigate a dynamically generated grid to complete a mission:
1. **Collect** two pickup rewards scattered on the map.
2. **Extract** at the exit (goal). The extraction is only successful if *both* rewards were collected first.

The agent must also navigate around:
- **Obstacles (walls)**: Block movement and yield a small penalty.
- **Moving enemy**: Moves randomly each step; getting too close results in penalties, and contact results in episode termination (death).

## Features

- Custom Gymnasium environment with procedurally generated maps.
- Dense shaping rewards based on precomputed BFS distance maps.
- Complex observation space including ray-cast sensing for walls/obstacles.
- PPO algorithm implementation via Stable-Baselines3.
- Pygame-based renderer for visual evaluation.
- TensorBoard integration for training metrics.

## Installation

1. Clone this repository (or navigate to the project folder).
2. Create and activate a Python virtual environment (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Agent

To train a new agent from scratch, run `train.py`. The script uses Stable-Baselines3's PPO algorithm with 8 parallel environments.

```bash
python train.py
```

You can customize the training process with various arguments:
```bash
python train.py --grid-size 12 --total-timesteps 500000 --initial-learning-rate 3e-4
```

To resume training from a checkpoint:
```bash
python train.py --resume-model "./checkpoints/ppo_model_XXXXX_steps.zip"
```

### Visualizing the Agent

Watch the trained agent perform in the environment using Pygame by running `main.py`:

```bash
python main.py
```

### Evaluating Performance

To run batch metrics on the agent (evaluating it on both in-distribution 12x12 grids and out-of-distribution 16x16 grids), run:

```bash
python evaluate.py
```

### Plotting Training Logs

You can visualize training logs extracted from TensorBoard:

```bash
python plot_training.py
```

## Project Structure

- `environment.py`: The Gymnasium environment defining the map generation, state, actions, and rewards.
- `train.py`: The main script for training the PPO agent.
- `evaluate.py`: Evaluates the trained agent's performance and generalization.
- `main.py`: Visualizes the trained policy using Pygame.
- `renderer.py`: Handles the Pygame visualization logic.
- `visual_assets.py`: Utilities for rendering assets in Pygame.
- `plot_training.py`: Generates graphs from TensorBoard logs.
- `ENVIRONMENT_AND_TRAINING.md`: Deep dive into the environment rules, observation/action spaces, and reward shaping.

## Environment Design & Reward Shaping

For an in-depth look at how the environment is structured, including the observation space (ray sensors, delta vectors) and the dense reward shaping philosophy, please see [ENVIRONMENT_AND_TRAINING.md](ENVIRONMENT_AND_TRAINING.md).

## Dependencies
- `gymnasium`
- `numpy`
- `pygame`
- `stable-baselines3`
- `tensorboard`
