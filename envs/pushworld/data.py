import json
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import gin
import joblib
import numpy as np
import yaml

from pushworld.utils.filesystem import get_puzzle_file_paths
from pushworld.utils.filesystem import iter_files_with_extension

from envs.pushworld.puzzle import Actions
from envs.pushworld.puzzle import PushWorldPuzzle

NUM_PUSHWORLD_CELL_TYPES = 10


class PushWorldFields(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT_WALL = 2
    GOAL = 3
    MOVABLE = 4
    MOVABLE_ON_GOAL = 5
    GOAL_MOVABLE = 6
    GOAL_MOVABLE_ON_GOAL = 7
    AGENT = 8
    AGENT_ON_GOAL = 9


class PushWorldObservationEncoder:
    encoder_name = None
    output_kind = "grid_tokens"

    def encode(self, puzzle, state):
        raise NotImplementedError

    def flatten(self, puzzle, state):
        return self.encode(puzzle, state).reshape(-1)

    def to_metadata(self):
        raise NotImplementedError


def _get_goal_objects(puzzle):
    if hasattr(puzzle, "goal_objects"):
        return puzzle.goal_objects
    return puzzle._goals


class PushWorldCategoricalGridEncoder(PushWorldObservationEncoder):
    encoder_name = "categorical_grid"

    def __init__(self, grid_size: int, center_pad: bool = True):
        self.grid_size = grid_size
        self.center_pad = center_pad
        self.num_cell_types = NUM_PUSHWORLD_CELL_TYPES

    def _offsets(self, puzzle: PushWorldPuzzle) -> Tuple[int, int]:
        width, height = puzzle.dimensions
        if width > self.grid_size or height > self.grid_size:
            raise ValueError(
                f"Puzzle dimensions {(width, height)} exceed encoder grid size {self.grid_size}"
            )

        if not self.center_pad:
            return 0, 0

        offset_x = (self.grid_size - width) // 2
        offset_y = (self.grid_size - height) // 2
        return offset_x, offset_y

    def encode(self, puzzle, state):
        board = np.full(
            (self.grid_size, self.grid_size), PushWorldFields.EMPTY, dtype=np.uint8
        )
        offset_x, offset_y = self._offsets(puzzle)
        goal_cells = set()

        for x, y in puzzle.wall_positions:
            board[y + offset_y, x + offset_x] = PushWorldFields.WALL

        for x, y in puzzle.agent_wall_positions:
            board[y + offset_y, x + offset_x] = PushWorldFields.AGENT_WALL

        for goal_object in _get_goal_objects(puzzle):
            goal_position = np.array(goal_object.position)
            for cell in goal_object.cells:
                x, y = goal_position + np.array(cell)
                board[y + offset_y, x + offset_x] = PushWorldFields.GOAL
                goal_cells.add((int(x), int(y)))

        for movable_idx, (movable, position) in enumerate(
            zip(puzzle.movable_objects, state)
        ):
            position = np.array(position)
            if movable_idx == 0:
                base_value = PushWorldFields.AGENT
                overlay_value = PushWorldFields.AGENT_ON_GOAL
            elif movable_idx <= len(puzzle.goal_state):
                base_value = PushWorldFields.GOAL_MOVABLE
                overlay_value = PushWorldFields.GOAL_MOVABLE_ON_GOAL
            else:
                base_value = PushWorldFields.MOVABLE
                overlay_value = PushWorldFields.MOVABLE_ON_GOAL

            for cell in movable.cells:
                x, y = position + np.array(cell)
                board[y + offset_y, x + offset_x] = (
                    overlay_value if (int(x), int(y)) in goal_cells else base_value
                )

        return board

    def to_metadata(self):
        return {
            "encoder_name": self.encoder_name,
            "output_kind": self.output_kind,
            "grid_size": self.grid_size,
            "center_pad": self.center_pad,
            "num_cell_types": self.num_cell_types,
        }


class PushWorldObjectIdentityGridEncoder(PushWorldObservationEncoder):
    encoder_name = "object_identity_grid"

    def __init__(self, grid_size: int, max_num_objects: int, center_pad: bool = True):
        self.grid_size = grid_size
        self.max_num_objects = max_num_objects
        self.center_pad = center_pad
        self.num_static_tokens = 4
        self.num_cell_types = self.num_static_tokens + 2 * max_num_objects

    def _offsets(self, puzzle: PushWorldPuzzle) -> Tuple[int, int]:
        width, height = puzzle.dimensions
        if width > self.grid_size or height > self.grid_size:
            raise ValueError(
                f"Puzzle dimensions {(width, height)} exceed encoder grid size {self.grid_size}"
            )

        if not self.center_pad:
            return 0, 0

        offset_x = (self.grid_size - width) // 2
        offset_y = (self.grid_size - height) // 2
        return offset_x, offset_y

    def encode(self, puzzle, state):
        board = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        offset_x, offset_y = self._offsets(puzzle)
        goal_cells = set()

        for x, y in puzzle.wall_positions:
            board[y + offset_y, x + offset_x] = 1

        for x, y in puzzle.agent_wall_positions:
            board[y + offset_y, x + offset_x] = 2

        for goal_object in _get_goal_objects(puzzle):
            goal_position = np.array(goal_object.position)
            for cell in goal_object.cells:
                x, y = goal_position + np.array(cell)
                board[y + offset_y, x + offset_x] = 3
                goal_cells.add((int(x), int(y)))

        for movable_idx, (movable, position) in enumerate(
            zip(puzzle.movable_objects, state)
        ):
            if movable_idx >= self.max_num_objects:
                raise ValueError(
                    f"Puzzle has {len(state)} objects but encoder only supports {self.max_num_objects}"
                )

            position = np.array(position)
            object_token = self.num_static_tokens + 2 * movable_idx
            object_token_on_goal = object_token + 1

            for cell in movable.cells:
                x, y = position + np.array(cell)
                board[y + offset_y, x + offset_x] = (
                    object_token_on_goal
                    if (int(x), int(y)) in goal_cells
                    else object_token
                )

        return board

    def to_metadata(self):
        return {
            "encoder_name": self.encoder_name,
            "output_kind": self.output_kind,
            "grid_size": self.grid_size,
            "center_pad": self.center_pad,
            "num_cell_types": self.num_cell_types,
            "max_num_objects": self.max_num_objects,
        }


@dataclass(frozen=True)
class RolloutStrategySpec:
    name: str
    count: int = 1
    params: Optional[Dict[str, object]] = None


def parse_puzzle_names(puzzle_names: Optional[str]) -> Optional[Sequence[str]]:
    if puzzle_names is None:
        return None
    parsed_names = [name.strip() for name in puzzle_names.split(",") if name.strip()]
    return parsed_names if parsed_names else None


def load_solution_records(
    planning_results_path: str,
    puzzles_path: str,
    puzzle_names: Optional[str] = None,
) -> List[dict]:
    puzzle_paths = get_puzzle_file_paths(puzzles_path)
    selected_puzzle_names = parse_puzzle_names(puzzle_names)
    selected_set = None if selected_puzzle_names is None else set(selected_puzzle_names)

    records = []
    for solution_file_path in sorted(
        iter_files_with_extension(planning_results_path, ".yaml")
    ):
        with open(solution_file_path, "r") as solution_file:
            solution = yaml.safe_load(solution_file) or {}

        puzzle_name = solution.get("puzzle")
        plan_string = (solution.get("plan") or "").upper()
        if puzzle_name is None or not plan_string:
            continue
        if selected_set is not None and puzzle_name not in selected_set:
            continue
        if puzzle_name not in puzzle_paths:
            raise ValueError(f'Could not find puzzle file for solution "{puzzle_name}"')
        if not set(plan_string).issubset(set(Actions.FROM_CHAR.keys())):
            raise ValueError(
                f'Planning result contains unsupported actions for "{puzzle_name}": {plan_string}'
            )

        records.append(
            {
                "puzzle_name": puzzle_name,
                "puzzle_file_path": puzzle_paths[puzzle_name],
                "solution_file_path": solution_file_path,
                "plan": [Actions.FROM_CHAR[action] for action in plan_string],
                "plan_string": plan_string,
            }
        )

    if not records:
        raise ValueError(f"No solved PushWorld plans found in {planning_results_path}")

    return records


def compute_dataset_stats(records: Sequence[dict]):
    max_width = 0
    max_height = 0
    max_num_objects = 0
    for record in records:
        puzzle = PushWorldPuzzle(record["puzzle_file_path"])
        width, height = puzzle.dimensions
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        max_num_objects = max(max_num_objects, len(puzzle.initial_state))
    return {
        "grid_size": max(max_width, max_height),
        "max_width": max_width,
        "max_height": max_height,
        "max_num_objects": max_num_objects,
    }


def infer_grid_size(records: Sequence[dict]) -> int:
    return compute_dataset_stats(records)["grid_size"]


def build_observation_encoder(
    records: Sequence[dict],
    encoder_name: str = "categorical_grid",
    grid_size: Optional[int] = None,
    center_pad: bool = True,
):
    stats = compute_dataset_stats(records)
    if grid_size is None:
        grid_size = stats["grid_size"]

    if encoder_name == "categorical_grid":
        return PushWorldCategoricalGridEncoder(
            grid_size=grid_size,
            center_pad=center_pad,
        )
    if encoder_name == "object_identity_grid":
        return PushWorldObjectIdentityGridEncoder(
            grid_size=grid_size,
            max_num_objects=stats["max_num_objects"],
            center_pad=center_pad,
        )

    raise ValueError(f"Unsupported PushWorld encoder: {encoder_name}")


def build_encoder_from_metadata(metadata: Dict[str, object]):
    encoder_name = metadata.get("encoder_name", "categorical_grid")
    grid_size = metadata["grid_size"]
    center_pad = metadata.get("center_pad", True)

    if encoder_name == "categorical_grid":
        return PushWorldCategoricalGridEncoder(
            grid_size=grid_size,
            center_pad=center_pad,
        )
    if encoder_name == "object_identity_grid":
        return PushWorldObjectIdentityGridEncoder(
            grid_size=grid_size,
            max_num_objects=metadata["max_num_objects"],
            center_pad=center_pad,
        )

    raise ValueError(f"Unsupported PushWorld encoder in metadata: {encoder_name}")


def rollout_plan(puzzle: PushWorldPuzzle, plan: Sequence[int]):
    states = [puzzle.initial_state]
    state = puzzle.initial_state
    for action in plan:
        state = puzzle.get_next_state(state, action)
        states.append(state)

    if not puzzle.is_goal_state(states[-1]):
        raise ValueError("Provided plan does not solve the puzzle")

    return states


def split_records(records: Sequence[dict], test_fraction: float, seed: int):
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0.0, 1.0)")

    unique_puzzles = sorted({record["puzzle_name"] for record in records})
    shuffled = list(unique_puzzles)
    np.random.default_rng(seed).shuffle(shuffled)

    num_test = int(round(len(shuffled) * test_fraction))
    if test_fraction > 0.0 and len(shuffled) > 1:
        num_test = max(1, min(num_test, len(shuffled) - 1))

    test_puzzles = set(shuffled[:num_test])
    train_records, test_records = [], []
    for record in records:
        if record["puzzle_name"] in test_puzzles:
            test_records.append(record)
        else:
            train_records.append(record)

    return train_records, test_records


def _coerce_rollout_strategy_specs(
    rollout_strategy_specs=None,
    rollout_strategy_specs_json: Optional[str] = None,
):
    if rollout_strategy_specs is not None and rollout_strategy_specs_json is not None:
        raise ValueError(
            "Provide rollout_strategy_specs or rollout_strategy_specs_json, not both"
        )

    if rollout_strategy_specs_json is not None:
        rollout_strategy_specs = json.loads(rollout_strategy_specs_json)

    if rollout_strategy_specs is None:
        rollout_strategy_specs = [{"name": "expert", "count": 1}]

    if isinstance(rollout_strategy_specs, dict):
        rollout_strategy_specs = [rollout_strategy_specs]

    coerced_specs = []
    for spec in rollout_strategy_specs:
        if isinstance(spec, RolloutStrategySpec):
            coerced_specs.append(spec)
            continue

        name = spec["name"]
        count = int(spec.get("count", 1))
        params = {
            key: value for key, value in spec.items() if key not in {"name", "count"}
        }
        coerced_specs.append(RolloutStrategySpec(name=name, count=count, params=params))

    return coerced_specs


def _expert_trajectory(puzzle, plan, rng, expert_states=None, **unused_params):
    del puzzle, plan, rng
    return list(expert_states)


def _solution_suffix_trajectory(
    puzzle,
    plan,
    rng,
    expert_states=None,
    min_length=2,
    **unused_params,
):
    del puzzle, plan
    max_start = max(0, len(expert_states) - int(min_length))
    start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    return list(expert_states[start_index:])


def _epsilon_plan_trajectory(
    puzzle,
    plan,
    rng,
    epsilon=0.1,
    extra_steps=0,
    max_steps=None,
    **unused_params,
):
    state = puzzle.initial_state
    trajectory = [state]
    total_steps = len(plan) if max_steps is None else min(int(max_steps), len(plan))
    for step_idx in range(total_steps):
        action = plan[step_idx]
        if rng.random() < float(epsilon):
            action = int(rng.integers(0, len(Actions.DISPLACEMENTS)))
        state = puzzle.get_next_state(state, action)
        trajectory.append(state)

    for _ in range(int(extra_steps)):
        action = int(rng.integers(0, len(Actions.DISPLACEMENTS)))
        state = puzzle.get_next_state(state, action)
        trajectory.append(state)

    return trajectory


def _random_walk_trajectory(
    puzzle,
    plan,
    rng,
    expert_states=None,
    walk_length=8,
    start_source="expert_nonterminal",
    **unused_params,
):
    del plan
    if start_source == "initial":
        state = puzzle.initial_state
    elif start_source == "expert_any":
        start_index = int(rng.integers(0, len(expert_states)))
        state = expert_states[start_index]
    else:
        start_index = int(rng.integers(0, max(1, len(expert_states) - 1)))
        state = expert_states[start_index]

    trajectory = [state]
    for _ in range(max(1, int(walk_length))):
        action = int(rng.integers(0, len(Actions.DISPLACEMENTS)))
        state = puzzle.get_next_state(state, action)
        trajectory.append(state)
    return trajectory


ROLLOUT_STRATEGIES = {
    "expert": _expert_trajectory,
    "solution_suffix": _solution_suffix_trajectory,
    "epsilon_plan": _epsilon_plan_trajectory,
    "random_walk": _random_walk_trajectory,
}


def _generate_record_trajectories(record, strategy_specs, rng):
    puzzle = PushWorldPuzzle(record["puzzle_file_path"])
    expert_states = rollout_plan(puzzle, record["plan"])
    trajectories = []

    for spec in strategy_specs:
        if spec.name not in ROLLOUT_STRATEGIES:
            raise ValueError(f"Unsupported PushWorld rollout strategy: {spec.name}")

        strategy = ROLLOUT_STRATEGIES[spec.name]
        for _ in range(spec.count):
            trajectory = strategy(
                puzzle,
                record["plan"],
                rng,
                expert_states=expert_states,
                **(spec.params or {}),
            )
            if len(trajectory) >= 2:
                trajectories.append(trajectory)

    return puzzle, trajectories


def _build_split_tensor(records, encoder, strategy_specs, rng):
    if not records:
        empty_trajs = np.zeros(
            (0, 0, encoder.grid_size, encoder.grid_size), dtype=np.uint8
        )
        empty_lens = np.zeros((0,), dtype=np.int64)
        return empty_trajs, empty_lens

    encoded_trajectories = []
    lengths = []
    max_length = 0
    for record in records:
        puzzle, raw_trajectories = _generate_record_trajectories(
            record, strategy_specs, rng
        )
        for raw_trajectory in raw_trajectories:
            encoded_trajectory = np.stack(
                [encoder.encode(puzzle, state) for state in raw_trajectory]
            )
            encoded_trajectories.append(encoded_trajectory)
            lengths.append(len(encoded_trajectory))
            max_length = max(max_length, len(encoded_trajectory))

    if not encoded_trajectories:
        empty_trajs = np.zeros(
            (0, 0, encoder.grid_size, encoder.grid_size), dtype=np.uint8
        )
        empty_lens = np.zeros((0,), dtype=np.int64)
        return empty_trajs, empty_lens

    padded = np.zeros(
        (len(encoded_trajectories), max_length, encoder.grid_size, encoder.grid_size),
        dtype=np.uint8,
    )
    for trajectory_idx, trajectory in enumerate(encoded_trajectories):
        padded[trajectory_idx, : len(trajectory)] = trajectory

    return padded, np.asarray(lengths, dtype=np.int64)


def _write_split(split_path: str, split_name: str, trajectories, lengths):
    os.makedirs(split_path, exist_ok=True)
    with open(
        os.path.join(split_path, f"{split_name}_trajectories.pkl"), "wb"
    ) as trajectory_file:
        joblib.dump(trajectories, trajectory_file)
    with open(os.path.join(split_path, f"{split_name}_lens.pkl"), "wb") as lengths_file:
        joblib.dump(lengths, lengths_file)


@gin.configurable()
def generate_pushworld_dataset(
    output_path,
    planning_results_path,
    puzzles_path,
    test_fraction=0.2,
    seed=0,
    puzzle_names=None,
    grid_size=None,
    center_pad=True,
    encoder_name="categorical_grid",
    rollout_strategy_specs=None,
    rollout_strategy_specs_json=None,
):
    records = load_solution_records(
        planning_results_path=planning_results_path,
        puzzles_path=puzzles_path,
        puzzle_names=puzzle_names,
    )
    encoder = build_observation_encoder(
        records=records,
        encoder_name=encoder_name,
        grid_size=grid_size,
        center_pad=center_pad,
    )
    strategy_specs = _coerce_rollout_strategy_specs(
        rollout_strategy_specs=rollout_strategy_specs,
        rollout_strategy_specs_json=rollout_strategy_specs_json,
    )

    train_records, test_records = split_records(
        records, test_fraction=test_fraction, seed=seed
    )
    rng = np.random.default_rng(seed)
    train_trajectories, train_lengths = _build_split_tensor(
        train_records,
        encoder,
        strategy_specs,
        rng,
    )
    test_trajectories, test_lengths = _build_split_tensor(
        test_records,
        encoder,
        strategy_specs,
        rng,
    )

    os.makedirs(output_path, exist_ok=True)
    _write_split(
        os.path.join(output_path, "train"),
        "train",
        train_trajectories,
        train_lengths,
    )
    _write_split(
        os.path.join(output_path, "test"),
        "test",
        test_trajectories,
        test_lengths,
    )

    metadata = {
        **compute_dataset_stats(records),
        **encoder.to_metadata(),
        "planning_results_path": planning_results_path,
        "puzzles_path": puzzles_path,
        "num_train_trajectories": int(len(train_lengths)),
        "num_test_trajectories": int(len(test_lengths)),
        "rollout_strategy_specs": [
            {
                "name": spec.name,
                "count": spec.count,
                **(spec.params or {}),
            }
            for spec in strategy_specs
        ],
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata
