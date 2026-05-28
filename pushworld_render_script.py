import argparse
import importlib
import json
import os
import random
import re
from typing import List
from typing import Sequence
from typing import Tuple

from envs.pushworld.puzzle import Actions
from envs.pushworld.puzzle import PushWorldPuzzle


imageio = importlib.import_module("imageio.v2")


def parse_indices(indices: str | None) -> List[int] | None:
    if indices is None:
        return None
    parsed = [int(part.strip()) for part in indices.split(",") if part.strip()]
    return parsed or None


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "trajectory"


def rollout_from_state(
    puzzle: PushWorldPuzzle,
    start_state: Tuple[Tuple[int, int], ...],
    plan: Sequence[int],
):
    states = [start_state]
    state = start_state
    for action in plan:
        state = puzzle.get_next_state(state, action)
        states.append(state)

    if not puzzle.is_goal_state(states[-1]):
        raise ValueError(
            "Manifest plan does not solve the puzzle from the stored start state"
        )

    return states


def select_entries(manifest, count: int, seed: int, indices: List[int] | None):
    if indices is not None:
        selected = []
        for idx in indices:
            if idx < 0 or idx >= len(manifest):
                raise IndexError(
                    f"Trajectory index {idx} is out of range for manifest of size {len(manifest)}"
                )
            selected.append((idx, manifest[idx]))
        return selected

    if count <= 0:
        raise ValueError("num_trajectories must be positive")

    count = min(count, len(manifest))
    all_indices = list(range(len(manifest)))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    chosen = sorted(all_indices[:count])
    return [(idx, manifest[idx]) for idx in chosen]


def resolve_puzzle_path(dataset_path: str, metadata: dict, entry: dict) -> str:
    puzzle_file_path = entry.get("puzzle_file_path")
    if puzzle_file_path:
        return puzzle_file_path

    puzzles_path = metadata.get("puzzles_path")
    if puzzles_path is None:
        raise ValueError("Dataset metadata does not contain puzzles_path")

    return os.path.join(puzzles_path, f"{entry['puzzle_name']}.pwp")


def render_trajectory_gif(
    output_file_path: str,
    puzzle: PushWorldPuzzle,
    states,
    fps: int,
    border_width: int,
    pixels_per_cell: int,
):
    frames = [
        puzzle.render(
            state,
            border_width=border_width,
            pixels_per_cell=pixels_per_cell,
        )
        for state in states
    ]
    imageio.mimsave(output_file_path, frames, duration=1.0 / fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--num_trajectories", type=int, default=8)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--border_width", type=int, default=2)
    parser.add_argument("--pixels_per_cell", type=int, default=20)
    args = parser.parse_args()

    metadata_path = os.path.join(args.dataset_path, "metadata.json")
    manifest_path = os.path.join(
        args.dataset_path, args.split, f"{args.split}_manifest.json"
    )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find dataset metadata at {metadata_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Could not find manifest at {manifest_path}. This script expects planner-generated PushWorld datasets."
        )

    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    with open(manifest_path, "r") as manifest_file:
        manifest = json.load(manifest_file)

    selected = select_entries(
        manifest=manifest,
        count=args.num_trajectories,
        seed=args.seed,
        indices=parse_indices(args.indices),
    )

    os.makedirs(args.output_path, exist_ok=True)
    review_manifest = []
    for entry_index, entry in selected:
        puzzle_file_path = resolve_puzzle_path(args.dataset_path, metadata, entry)
        puzzle = PushWorldPuzzle(puzzle_file_path)
        start_state = tuple(tuple(position) for position in entry["start_state"])
        plan = [Actions.FROM_CHAR[action_char] for action_char in entry["plan_string"]]
        states = rollout_from_state(puzzle, start_state, plan)

        output_name = (
            f"{entry_index:05d}_"
            f"{sanitize_name(entry['puzzle_name'])}_"
            f"len{len(states) - 1}.gif"
        )
        output_file_path = os.path.join(args.output_path, output_name)
        render_trajectory_gif(
            output_file_path=output_file_path,
            puzzle=puzzle,
            states=states,
            fps=args.fps,
            border_width=args.border_width,
            pixels_per_cell=args.pixels_per_cell,
        )

        review_manifest.append(
            {
                "manifest_index": entry_index,
                "output_file": output_name,
                "puzzle_name": entry["puzzle_name"],
                "trajectory_length": len(states),
                "plan_length": len(plan),
                "start_source": entry.get("start_source"),
                "solution_source": entry.get("solution_source"),
            }
        )

    with open(
        os.path.join(args.output_path, "review_manifest.json"), "w"
    ) as review_manifest_file:
        json.dump(review_manifest, review_manifest_file, indent=2)


if __name__ == "__main__":
    main()
