import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def find_event_files(log_dir: str) -> List[str]:
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    return sorted(glob.glob(pattern, recursive=True))


def load_scalars(event_file: str) -> Dict[str, List[Tuple[int, float]]]:
    acc = event_accumulator.EventAccumulator(event_file, size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])

    scalars: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        events = acc.Scalars(tag)
        scalars[tag] = [(e.step, float(e.value)) for e in events]
    return scalars


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def pick_first_existing_tag(candidates: List[str], available: Dict[str, List[Tuple[int, float]]]) -> Optional[str]:
    for tag in candidates:
        if tag in available and len(available[tag]) > 0:
            return tag
    return None


def plot_scalar(
    scalars: Dict[str, List[Tuple[int, float]]],
    tag_candidates: List[str],
    title: str,
    y_label: str,
    out_path: str,
    smooth_window: int,
):
    tag = pick_first_existing_tag(tag_candidates, scalars)
    if tag is None:
        return False

    data = scalars[tag]
    steps = np.array([p[0] for p in data], dtype=np.float32)
    values = np.array([p[1] for p in data], dtype=np.float32)
    smooth_values = smooth(values, smooth_window)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, alpha=0.35, label="raw")
    plt.plot(steps, smooth_values, linewidth=2, label=f"smoothed (w={smooth_window})")
    plt.title(f"{title}\n(tag: {tag})")
    plt.xlabel("Training steps")
    plt.ylabel(y_label)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Create training charts from TensorBoard event logs.")
    parser.add_argument("--log-dir", type=str, default="./ppo_gridworld_tensorboard", help="TensorBoard log directory.")
    parser.add_argument("--output-dir", type=str, default="./training_charts", help="Directory for generated PNG charts.")
    parser.add_argument("--event-file", type=str, default="", help="Optional specific event file path.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Moving-average window for smoothing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.event_file:
        if not os.path.exists(args.event_file):
            raise FileNotFoundError(f"Event file not found: {args.event_file}")
        event_file = args.event_file
    else:
        event_files = find_event_files(args.log_dir)
        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files found under: {args.log_dir}")
        event_file = event_files[-1]

    print(f"Using event file: {event_file}")
    scalars = load_scalars(event_file)
    print(f"Found {len(scalars)} scalar tags.")

    chart_specs = [
        {
            "name": "episodic_reward",
            "title": "Episodic Reward",
            "y_label": "Reward",
            "tags": ["rollout/ep_rew_mean", "eval/mean_reward"],
        },
        {
            "name": "episodic_length",
            "title": "Episodic Length",
            "y_label": "Steps",
            "tags": ["rollout/ep_len_mean"],
        },
        {
            "name": "policy_loss",
            "title": "Policy Gradient Loss",
            "y_label": "Loss",
            "tags": ["train/policy_gradient_loss"],
        },
        {
            "name": "value_loss",
            "title": "Value Loss",
            "y_label": "Loss",
            "tags": ["train/value_loss"],
        },
        {
            "name": "entropy_loss",
            "title": "Entropy Loss",
            "y_label": "Loss",
            "tags": ["train/entropy_loss"],
        },
        {
            "name": "approx_kl",
            "title": "Approximate KL",
            "y_label": "KL",
            "tags": ["train/approx_kl"],
        },
        {
            "name": "clip_fraction",
            "title": "Clip Fraction",
            "y_label": "Fraction",
            "tags": ["train/clip_fraction"],
        },
        {
            "name": "explained_variance",
            "title": "Explained Variance",
            "y_label": "Variance",
            "tags": ["train/explained_variance"],
        },
    ]

    generated = 0
    for spec in chart_specs:
        out_path = os.path.join(args.output_dir, f"{spec['name']}.png")
        ok = plot_scalar(
            scalars=scalars,
            tag_candidates=spec["tags"],
            title=spec["title"],
            y_label=spec["y_label"],
            out_path=out_path,
            smooth_window=args.smooth_window,
        )
        if ok:
            generated += 1
            print(f"Generated: {out_path}")
        else:
            print(f"Skipped (tag not found): {spec['name']}")

    tags_file = os.path.join(args.output_dir, "available_tags.txt")
    with open(tags_file, "w", encoding="utf-8") as f:
        for tag in sorted(scalars.keys()):
            f.write(tag + "\n")
    print(f"Saved available tags list: {tags_file}")
    print(f"Done. Generated {generated} charts.")


if __name__ == "__main__":
    main()
