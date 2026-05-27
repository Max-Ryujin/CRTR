import json

import gin
import numpy as np

from envs.pushworld.data import build_encoder_from_metadata
from envs.pushworld.data import build_observation_encoder
from envs.pushworld.data import load_solution_records
from envs.pushworld.data import rollout_plan
from envs.pushworld.puzzle import PushWorldPuzzle


@gin.configurable()
def generate_problems_pushworld(
    n_problems,
    *unused_args,
    planning_results_path,
    puzzles_path,
    metadata_path=None,
    puzzle_names=None,
    grid_size=None,
    center_pad=True,
    encoder_name="categorical_grid",
    seed=0,
):
    records = load_solution_records(
        planning_results_path=planning_results_path,
        puzzles_path=puzzles_path,
        puzzle_names=puzzle_names,
    )

    if metadata_path is not None:
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
        encoder = build_encoder_from_metadata(metadata)
    else:
        encoder = build_observation_encoder(
            records=records,
            encoder_name=encoder_name,
            grid_size=grid_size,
            center_pad=center_pad,
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records)) if records else np.array([], dtype=int)

    problems = []
    for problem_idx in range(n_problems):
        record = records[int(order[problem_idx % len(order)])]
        puzzle = PushWorldPuzzle(record["puzzle_file_path"])
        goal_state = rollout_plan(puzzle, record["plan"])[-1]
        problems.append(
            (
                encoder.flatten(puzzle, puzzle.initial_state).astype(np.float32),
                encoder.flatten(puzzle, goal_state).astype(np.float32),
                {
                    "puzzle_name": record["puzzle_name"],
                    "puzzle_file_path": record["puzzle_file_path"],
                    "initial_raw_state": puzzle.initial_state,
                },
            )
        )

    return problems
