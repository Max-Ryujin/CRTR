import argparse
import json
import os
import pickle
import resource
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from envs.pushworld import ensure_pushworld_on_path
from envs.pushworld.data import build_observation_encoder
from envs.pushworld.data import compute_dataset_stats
from envs.pushworld.data import generate_pushworld_dataset
from envs.pushworld.data import load_solution_records
from envs.pushworld.data import rollout_plan
from envs.pushworld.puzzle import Actions
from envs.pushworld.puzzle import PushWorldPuzzle

KILOBYTE = 1000
MEGABYTE = 1000 * KILOBYTE
GIGABYTE = 1000 * MEGABYTE


def resolve_default_planner_path() -> str:
    candidates = []
    pushworld_src_path = ensure_pushworld_on_path()
    if pushworld_src_path is not None:
        candidates.append(
            Path(pushworld_src_path).resolve().parents[2]
            / "cpp"
            / "build"
            / "bin"
            / "run_planner"
        )

    env_path = os.getenv("PUSHWORLD_SRC_PATH")
    if env_path:
        candidates.append(
            Path(env_path).expanduser().absolute().parents[2]
            / "cpp"
            / "build"
            / "bin"
            / "run_planner"
        )

    script_dir = Path(__file__).expanduser().absolute().parent
    cwd = Path.cwd().expanduser().absolute()
    candidates.extend(
        [
            script_dir.parent / "pushworld" / "cpp" / "build" / "bin" / "run_planner",
            cwd.parent / "pushworld" / "cpp" / "build" / "bin" / "run_planner",
            cwd / "../pushworld/cpp/build/bin/run_planner",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def get_child_process_cpu_time() -> float:
    child_times = resource.getrusage(resource.RUSAGE_CHILDREN)
    return child_times.ru_utime + child_times.ru_stime


def run_process(
    command: List[str],
    time_limit: Optional[int] = None,
    memory_limit: Optional[int] = None,
):
    if not isinstance(time_limit, (int, type(None))):
        raise TypeError("time_limit must be an integer or None")
    if time_limit is not None and time_limit <= 0:
        raise ValueError("time_limit must be a positive integer")

    if not isinstance(memory_limit, (int, type(None))):
        raise TypeError("memory_limit must be an integer or None")
    if memory_limit is not None and memory_limit <= 0:
        raise ValueError("memory_limit must be a positive integer")

    def preexec_fn():
        if time_limit is not None:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit + 1))
            except ValueError:
                resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit))

        if memory_limit is not None:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

    start_cpu_time = get_child_process_cpu_time()
    process = subprocess.Popen(
        command,
        preexec_fn=preexec_fn,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = process.communicate()[0]
    cpu_run_time = get_child_process_cpu_time() - start_cpu_time

    return output.strip().decode("utf-8"), process.returncode, cpu_run_time


RGD_PLANNER_PATH = resolve_default_planner_path()

ACTION_TO_CHAR = {value: key for key, value in Actions.FROM_CHAR.items()}


@dataclass
class PlannerSolveResult:
    plan: Optional[List[int]]
    plan_string: Optional[str]
    cpu_time: float
    return_code: int
    stdout: str
    mode: str


@dataclass
class StartStatePlan:
    actions: List[int]
    plan_string: str
    prefix_length: int
    planner_cpu_time: float
    planner_mode: str
    source: str


@dataclass
class StartStateBundle:
    base_puzzle_name: str
    puzzle_file_path: str
    state: Tuple[Tuple[int, int], ...]
    source: str
    anchor_step: Optional[int]
    plans: List[StartStatePlan]


class PushWorldPuzzleSerializer:
    def __init__(self, puzzle: PushWorldPuzzle):
        width, height = puzzle.dimensions
        self.width = width - 2
        self.height = height - 2
        self._static_tokens = [
            [[] for _ in range(self.width)] for _ in range(self.height)
        ]
        self._movable_cells = [
            tuple(sorted(obj.cells)) for obj in puzzle.movable_objects
        ]

        for x, y in puzzle.wall_positions:
            if 1 <= x <= self.width and 1 <= y <= self.height:
                self._static_tokens[y - 1][x - 1].append("W")

        for x, y in puzzle.agent_wall_positions:
            if 1 <= x <= self.width and 1 <= y <= self.height:
                self._static_tokens[y - 1][x - 1].append("AW")

        goals = getattr(puzzle, "_goals")
        for goal_idx, goal in enumerate(goals):
            goal_x, goal_y = goal.position
            for delta_x, delta_y in goal.cells:
                x = goal_x + delta_x
                y = goal_y + delta_y
                if 1 <= x <= self.width and 1 <= y <= self.height:
                    self._static_tokens[y - 1][x - 1].append(f"G{goal_idx}")

    def serialize(self, state: Tuple[Tuple[int, int], ...]) -> str:
        cells = [[list(tokens) for tokens in row] for row in self._static_tokens]
        for movable_idx, (origin_x, origin_y) in enumerate(state):
            label = "A" if movable_idx == 0 else f"M{movable_idx - 1}"
            for delta_x, delta_y in self._movable_cells[movable_idx]:
                x = origin_x + delta_x
                y = origin_y + delta_y
                if 1 <= x <= self.width and 1 <= y <= self.height:
                    cells[y - 1][x - 1].append(label)

        lines = []
        for row in cells:
            lines.append(" ".join("+".join(cell) if cell else "." for cell in row))
        return "\n".join(lines) + "\n"


def parse_comma_separated_names(puzzle_names: Optional[str]) -> Optional[List[str]]:
    if puzzle_names is None:
        return None
    names = [name.strip() for name in puzzle_names.split(",") if name.strip()]
    return names or None


def select_records(
    records: Sequence[dict], num_puzzles: Optional[int], seed: int
) -> List[dict]:
    if num_puzzles is None:
        return list(records)

    unique_names = sorted(record["puzzle_name"] for record in records)
    if num_puzzles <= 0:
        raise ValueError("num_puzzles must be positive when provided")
    if num_puzzles > len(unique_names):
        raise ValueError(
            f"Requested {num_puzzles} puzzles, but only {len(unique_names)} are available"
        )

    rng = np.random.default_rng(seed)
    chosen_names = set(
        rng.choice(unique_names, size=num_puzzles, replace=False).tolist()
    )
    return [record for record in records if record["puzzle_name"] in chosen_names]


def split_records_by_base_puzzle(
    records: Sequence[dict],
    test_fraction: float,
    seed: int,
    num_test_puzzles: Optional[int] = None,
):
    unique_puzzles = sorted(record["puzzle_name"] for record in records)
    if not unique_puzzles:
        raise ValueError("No PushWorld puzzles were selected")

    shuffled = list(unique_puzzles)
    np.random.default_rng(seed).shuffle(shuffled)

    if num_test_puzzles is not None:
        if num_test_puzzles < 0 or num_test_puzzles >= len(shuffled):
            raise ValueError(
                "num_test_puzzles must be in [0, number_of_selected_puzzles)"
            )
        num_test = num_test_puzzles
    else:
        if not 0.0 <= test_fraction < 1.0:
            raise ValueError("test_fraction must be in [0.0, 1.0)")
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


def plan_to_string(plan: Sequence[int]) -> str:
    return "".join(ACTION_TO_CHAR[action] for action in plan)


def rollout_from_state(puzzle: PushWorldPuzzle, start_state, plan: Sequence[int]):
    states = [start_state]
    state = start_state
    for action in plan:
        state = puzzle.get_next_state(state, action)
        states.append(state)
    if not puzzle.is_goal_state(states[-1]):
        raise ValueError(
            "Provided plan does not solve the puzzle from the selected state"
        )
    return states


def evenly_spaced_indices(total_count: int, desired_count: int) -> List[int]:
    if total_count <= 0 or desired_count <= 0:
        return []
    desired_count = min(total_count, desired_count)
    if desired_count == total_count:
        return list(range(total_count))
    values = np.linspace(0, total_count - 1, num=desired_count)
    indices = []
    seen = set()
    for value in values:
        idx = int(round(float(value)))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def sample_random_walk_state(
    puzzle: PushWorldPuzzle,
    anchor_state,
    rng: np.random.Generator,
    min_steps: int,
    max_steps: int,
):
    if min_steps < 0 or max_steps < min_steps:
        raise ValueError("Random-walk limits must satisfy 0 <= min_steps <= max_steps")

    num_steps = int(rng.integers(min_steps, max_steps + 1)) if max_steps > 0 else 0
    state = anchor_state
    actions = []
    for _ in range(num_steps):
        action = int(rng.integers(0, len(Actions.DISPLACEMENTS)))
        actions.append(action)
        state = puzzle.get_next_state(state, action)
    return state, actions


def solve_state_with_planner(
    serializer: PushWorldPuzzleSerializer,
    state,
    planner_path: str,
    planner_mode: str,
    time_limit: Optional[int],
    memory_limit_gb: Optional[float],
) -> PlannerSolveResult:
    with tempfile.NamedTemporaryFile("w", suffix=".pwp", delete=False) as puzzle_file:
        puzzle_file.write(serializer.serialize(state))
        temp_puzzle_path = puzzle_file.name

    try:
        stdout, return_code, cpu_time = run_process(
            [planner_path, planner_mode, temp_puzzle_path],
            time_limit=time_limit,
            memory_limit=(
                None if memory_limit_gb is None else int(memory_limit_gb * GIGABYTE)
            ),
        )
    finally:
        os.unlink(temp_puzzle_path)

    if (
        return_code == 0
        and stdout
        and set(stdout).issubset(set(Actions.FROM_CHAR.keys()))
    ):
        plan = [Actions.FROM_CHAR[char] for char in stdout]
        return PlannerSolveResult(
            plan=plan,
            plan_string=stdout,
            cpu_time=cpu_time,
            return_code=return_code,
            stdout=stdout,
            mode=planner_mode,
        )

    return PlannerSolveResult(
        plan=None,
        plan_string=None,
        cpu_time=cpu_time,
        return_code=return_code,
        stdout=stdout,
        mode=planner_mode,
    )


def solve_states_in_parallel(
    requests,
    planner_path: str,
    planner_mode: str,
    time_limit: Optional[int],
    memory_limit_gb: Optional[float],
    max_workers: int,
):
    if not requests:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                solve_state_with_planner,
                request["serializer"],
                request["state"],
                planner_path,
                planner_mode,
                time_limit,
                memory_limit_gb,
            )
            for request in requests
        ]
        return [future.result() for future in futures]


def add_plan_if_unique(bundle: StartStateBundle, plan: StartStatePlan):
    existing = {existing_plan.plan_string for existing_plan in bundle.plans}
    if plan.plan_string not in existing:
        bundle.plans.append(plan)


def generate_start_state_bundles_for_record(
    record: dict,
    planner_path: str,
    planner_mode: str,
    planner_workers: int,
    planner_time_limit: Optional[int],
    planner_memory_limit_gb: Optional[float],
    states_per_puzzle: int,
    expert_starts_per_puzzle: int,
    start_walk_min_steps: int,
    start_walk_max_steps: int,
    solutions_per_state: int,
    detour_min_steps: int,
    detour_max_steps: int,
    max_state_attempts_multiplier: int,
    max_solution_attempts_multiplier: int,
    rng: np.random.Generator,
):
    puzzle = PushWorldPuzzle(record["puzzle_file_path"])
    serializer = PushWorldPuzzleSerializer(puzzle)
    expert_states = rollout_plan(puzzle, record["plan"])
    bundles_by_state: Dict[Tuple[Tuple[int, int], ...], StartStateBundle] = {}

    for start_idx in evenly_spaced_indices(
        len(record["plan"]), expert_starts_per_puzzle
    ):
        start_state = expert_states[start_idx]
        suffix_plan = list(record["plan"][start_idx:])
        if not suffix_plan:
            continue
        bundle = StartStateBundle(
            base_puzzle_name=record["puzzle_name"],
            puzzle_file_path=record["puzzle_file_path"],
            state=start_state,
            source="expert_suffix",
            anchor_step=start_idx,
            plans=[],
        )
        add_plan_if_unique(
            bundle,
            StartStatePlan(
                actions=suffix_plan,
                plan_string=plan_to_string(suffix_plan),
                prefix_length=0,
                planner_cpu_time=0.0,
                planner_mode="expert_suffix",
                source="expert_suffix",
            ),
        )
        bundles_by_state[start_state] = bundle

    max_state_attempts = max(1, states_per_puzzle * max_state_attempts_multiplier)
    state_attempts = 0
    while (
        len(bundles_by_state) < states_per_puzzle
        and state_attempts < max_state_attempts
    ):
        batch_size = max(planner_workers * 2, 1)
        requests = []
        request_metadata = []
        while len(requests) < batch_size and state_attempts < max_state_attempts:
            anchor_idx = int(rng.integers(0, len(expert_states) - 1))
            candidate_state, _ = sample_random_walk_state(
                puzzle,
                expert_states[anchor_idx],
                rng,
                start_walk_min_steps,
                start_walk_max_steps,
            )
            state_attempts += 1
            if puzzle.is_goal_state(candidate_state):
                continue
            if candidate_state in bundles_by_state:
                continue
            requests.append({"serializer": serializer, "state": candidate_state})
            request_metadata.append((candidate_state, anchor_idx))

        for (candidate_state, anchor_idx), solve_result in zip(
            request_metadata,
            solve_states_in_parallel(
                requests,
                planner_path=planner_path,
                planner_mode=planner_mode,
                time_limit=planner_time_limit,
                memory_limit_gb=planner_memory_limit_gb,
                max_workers=planner_workers,
            ),
        ):
            if solve_result.plan is None:
                continue
            bundle = StartStateBundle(
                base_puzzle_name=record["puzzle_name"],
                puzzle_file_path=record["puzzle_file_path"],
                state=candidate_state,
                source="random_walk_start",
                anchor_step=anchor_idx,
                plans=[],
            )
            add_plan_if_unique(
                bundle,
                StartStatePlan(
                    actions=solve_result.plan,
                    plan_string=solve_result.plan_string,
                    prefix_length=0,
                    planner_cpu_time=solve_result.cpu_time,
                    planner_mode=solve_result.mode,
                    source="planner_start",
                ),
            )
            bundles_by_state[candidate_state] = bundle
            if len(bundles_by_state) >= states_per_puzzle:
                break

    bundles = list(bundles_by_state.values())
    for bundle in bundles:
        max_solution_attempts = max(
            1, solutions_per_state * max_solution_attempts_multiplier
        )
        solution_attempts = 0
        while (
            len(bundle.plans) < solutions_per_state
            and solution_attempts < max_solution_attempts
        ):
            requests = []
            request_metadata = []
            batch_size = max(
                1, min(planner_workers, solutions_per_state - len(bundle.plans))
            )
            while (
                len(requests) < batch_size and solution_attempts < max_solution_attempts
            ):
                detour_state, prefix_actions = sample_random_walk_state(
                    puzzle,
                    bundle.state,
                    rng,
                    detour_min_steps,
                    detour_max_steps,
                )
                solution_attempts += 1
                if not prefix_actions:
                    continue
                if puzzle.is_goal_state(detour_state):
                    continue
                requests.append({"serializer": serializer, "state": detour_state})
                request_metadata.append(prefix_actions)

            for prefix_actions, solve_result in zip(
                request_metadata,
                solve_states_in_parallel(
                    requests,
                    planner_path=planner_path,
                    planner_mode=planner_mode,
                    time_limit=planner_time_limit,
                    memory_limit_gb=planner_memory_limit_gb,
                    max_workers=planner_workers,
                ),
            ):
                if solve_result.plan is None:
                    continue
                full_plan = list(prefix_actions) + list(solve_result.plan)
                full_plan_string = plan_to_string(full_plan)
                if any(
                    existing.plan_string == full_plan_string
                    for existing in bundle.plans
                ):
                    continue
                add_plan_if_unique(
                    bundle,
                    StartStatePlan(
                        actions=full_plan,
                        plan_string=full_plan_string,
                        prefix_length=len(prefix_actions),
                        planner_cpu_time=solve_result.cpu_time,
                        planner_mode=solve_result.mode,
                        source="detour_then_planner",
                    ),
                )

    return bundles


def build_examples_for_split(records: Sequence[dict], bundles_by_puzzle_name):
    examples = []
    for record in records:
        puzzle = PushWorldPuzzle(record["puzzle_file_path"])
        for bundle in bundles_by_puzzle_name.get(record["puzzle_name"], []):
            for plan in bundle.plans:
                trajectory = rollout_from_state(puzzle, bundle.state, plan.actions)
                examples.append(
                    {
                        "puzzle_name": record["puzzle_name"],
                        "puzzle_file_path": record["puzzle_file_path"],
                        "trajectory": trajectory,
                        "plan": plan,
                        "start_state": bundle.state,
                        "start_source": bundle.source,
                        "anchor_step": bundle.anchor_step,
                    }
                )
    return examples


def encode_examples(examples, encoder):
    if not examples:
        empty_trajs = np.zeros(
            (0, 0, encoder.grid_size, encoder.grid_size), dtype=np.uint8
        )
        empty_lens = np.zeros((0,), dtype=np.int64)
        return empty_trajs, empty_lens

    encoded_trajectories = []
    lengths = []
    max_length = 0
    for example in examples:
        puzzle = PushWorldPuzzle(example["puzzle_file_path"])
        encoded = np.stack(
            [encoder.encode(puzzle, state) for state in example["trajectory"]]
        )
        encoded_trajectories.append(encoded)
        lengths.append(len(encoded))
        max_length = max(max_length, len(encoded))

    padded = np.zeros(
        (len(encoded_trajectories), max_length, encoder.grid_size, encoder.grid_size),
        dtype=np.uint8,
    )
    for idx, trajectory in enumerate(encoded_trajectories):
        padded[idx, : len(trajectory)] = trajectory

    return padded, np.asarray(lengths, dtype=np.int64)


def to_json_compatible(value):
    if isinstance(value, dict):
        return {
            key: to_json_compatible(inner_value) for key, inner_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(inner_value) for inner_value in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_split(output_path: str, split_name: str, trajectories, lengths, manifest):
    split_path = os.path.join(output_path, split_name)
    os.makedirs(split_path, exist_ok=True)
    with open(
        os.path.join(split_path, f"{split_name}_trajectories.pkl"), "wb"
    ) as trajectory_file:
        pickle.dump(trajectories, trajectory_file)
    with open(os.path.join(split_path, f"{split_name}_lens.pkl"), "wb") as lengths_file:
        pickle.dump(lengths, lengths_file)
    with open(
        os.path.join(split_path, f"{split_name}_manifest.json"), "w"
    ) as manifest_file:
        json.dump(to_json_compatible(manifest), manifest_file, indent=2)


def summarize_bundles(bundles_by_puzzle_name):
    summary = {}
    for puzzle_name, bundles in bundles_by_puzzle_name.items():
        summary[puzzle_name] = {
            "num_start_states": len(bundles),
            "num_trajectories": int(sum(len(bundle.plans) for bundle in bundles)),
            "num_detour_trajectories": int(
                sum(
                    1
                    for bundle in bundles
                    for plan in bundle.plans
                    if plan.source == "detour_then_planner"
                )
            ),
        }
    return summary


def generate_pushworld_dataset_with_cpp_planner(
    output_path,
    planning_results_path,
    puzzles_path,
    test_fraction=0.2,
    seed=0,
    puzzle_names=None,
    grid_size=None,
    center_pad=True,
    encoder_name="categorical_grid",
    num_puzzles=None,
    num_test_puzzles=None,
    states_per_puzzle=64,
    expert_starts_per_puzzle=8,
    solutions_per_state=3,
    start_walk_min_steps=2,
    start_walk_max_steps=8,
    detour_min_steps=1,
    detour_max_steps=6,
    planner_path=RGD_PLANNER_PATH,
    planner_mode="N+RGD",
    planner_workers=None,
    planner_time_limit=120,
    planner_memory_limit_gb=8.0,
    max_state_attempts_multiplier=8,
    max_solution_attempts_multiplier=8,
):
    if not os.path.exists(planner_path):
        raise FileNotFoundError(
            f"Could not find PushWorld planner executable at {planner_path}"
        )

    selected_names = parse_comma_separated_names(puzzle_names)
    records = load_solution_records(
        planning_results_path=planning_results_path,
        puzzles_path=puzzles_path,
        puzzle_names=None if selected_names is None else ",".join(selected_names),
    )
    records = select_records(records, num_puzzles=num_puzzles, seed=seed)

    train_records, test_records = split_records_by_base_puzzle(
        records,
        test_fraction=test_fraction,
        seed=seed,
        num_test_puzzles=num_test_puzzles,
    )

    planner_workers = planner_workers or max(1, os.cpu_count() or 1)
    rng = np.random.default_rng(seed)
    bundles_by_puzzle_name = {}
    for record in records:
        bundles_by_puzzle_name[record["puzzle_name"]] = (
            generate_start_state_bundles_for_record(
                record=record,
                planner_path=planner_path,
                planner_mode=planner_mode,
                planner_workers=planner_workers,
                planner_time_limit=planner_time_limit,
                planner_memory_limit_gb=planner_memory_limit_gb,
                states_per_puzzle=states_per_puzzle,
                expert_starts_per_puzzle=expert_starts_per_puzzle,
                start_walk_min_steps=start_walk_min_steps,
                start_walk_max_steps=start_walk_max_steps,
                solutions_per_state=solutions_per_state,
                detour_min_steps=detour_min_steps,
                detour_max_steps=detour_max_steps,
                max_state_attempts_multiplier=max_state_attempts_multiplier,
                max_solution_attempts_multiplier=max_solution_attempts_multiplier,
                rng=rng,
            )
        )

    encoder = build_observation_encoder(
        records=records,
        encoder_name=encoder_name,
        grid_size=grid_size,
        center_pad=center_pad,
    )

    train_examples = build_examples_for_split(train_records, bundles_by_puzzle_name)
    test_examples = build_examples_for_split(test_records, bundles_by_puzzle_name)
    train_trajectories, train_lengths = encode_examples(train_examples, encoder)
    test_trajectories, test_lengths = encode_examples(test_examples, encoder)

    train_manifest = [
        {
            "puzzle_name": example["puzzle_name"],
            "start_source": example["start_source"],
            "anchor_step": example["anchor_step"],
            "start_state": [list(position) for position in example["start_state"]],
            "plan_string": example["plan"].plan_string,
            "prefix_length": example["plan"].prefix_length,
            "planner_cpu_time": example["plan"].planner_cpu_time,
            "planner_mode": example["plan"].planner_mode,
            "solution_source": example["plan"].source,
            "trajectory_length": len(example["trajectory"]),
        }
        for example in train_examples
    ]
    test_manifest = [
        {
            "puzzle_name": example["puzzle_name"],
            "start_source": example["start_source"],
            "anchor_step": example["anchor_step"],
            "start_state": [list(position) for position in example["start_state"]],
            "plan_string": example["plan"].plan_string,
            "prefix_length": example["plan"].prefix_length,
            "planner_cpu_time": example["plan"].planner_cpu_time,
            "planner_mode": example["plan"].planner_mode,
            "solution_source": example["plan"].source,
            "trajectory_length": len(example["trajectory"]),
        }
        for example in test_examples
    ]

    os.makedirs(output_path, exist_ok=True)
    write_split(output_path, "train", train_trajectories, train_lengths, train_manifest)
    write_split(output_path, "test", test_trajectories, test_lengths, test_manifest)

    selected_puzzles = sorted(record["puzzle_name"] for record in records)
    train_puzzles = sorted(record["puzzle_name"] for record in train_records)
    test_puzzles = sorted(record["puzzle_name"] for record in test_records)
    metadata = {
        **compute_dataset_stats(records),
        **encoder.to_metadata(),
        "generation_mode": "cpp_planner",
        "planning_results_path": planning_results_path,
        "puzzles_path": puzzles_path,
        "selected_puzzles": selected_puzzles,
        "train_puzzles": train_puzzles,
        "test_puzzles": test_puzzles,
        "num_train_trajectories": int(len(train_lengths)),
        "num_test_trajectories": int(len(test_lengths)),
        "states_per_puzzle_target": int(states_per_puzzle),
        "expert_starts_per_puzzle": int(expert_starts_per_puzzle),
        "solutions_per_state_target": int(solutions_per_state),
        "start_walk_range": [int(start_walk_min_steps), int(start_walk_max_steps)],
        "detour_walk_range": [int(detour_min_steps), int(detour_max_steps)],
        "planner_path": planner_path,
        "planner_mode": planner_mode,
        "planner_workers": int(planner_workers),
        "planner_time_limit": planner_time_limit,
        "planner_memory_limit_gb": planner_memory_limit_gb,
        "generation_summary_by_puzzle": summarize_bundles(bundles_by_puzzle_name),
    }
    with open(os.path.join(output_path, "metadata.json"), "w") as metadata_file:
        json.dump(to_json_compatible(metadata), metadata_file, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--planning_results_path", required=True)
    parser.add_argument("--puzzles_path", required=True)
    parser.add_argument(
        "--generation_mode",
        choices=["precomputed", "planner"],
        default="precomputed",
    )
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--puzzle_names", default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--encoder_name", default="categorical_grid")
    parser.add_argument("--rollout_strategies_json", default=None)
    parser.add_argument("--center_pad", action="store_true")
    parser.add_argument("--no_center_pad", action="store_true")
    parser.add_argument("--num_puzzles", type=int, default=None)
    parser.add_argument("--num_test_puzzles", type=int, default=None)
    parser.add_argument("--states_per_puzzle", type=int, default=64)
    parser.add_argument("--expert_starts_per_puzzle", type=int, default=8)
    parser.add_argument("--solutions_per_state", type=int, default=3)
    parser.add_argument("--start_walk_min_steps", type=int, default=2)
    parser.add_argument("--start_walk_max_steps", type=int, default=8)
    parser.add_argument("--detour_min_steps", type=int, default=1)
    parser.add_argument("--detour_max_steps", type=int, default=6)
    parser.add_argument("--planner_path", default=RGD_PLANNER_PATH)
    parser.add_argument("--planner_mode", default="N+RGD")
    parser.add_argument("--planner_workers", type=int, default=None)
    parser.add_argument("--planner_time_limit", type=int, default=120)
    parser.add_argument("--planner_memory_limit_gb", type=float, default=8.0)
    parser.add_argument("--max_state_attempts_multiplier", type=int, default=8)
    parser.add_argument("--max_solution_attempts_multiplier", type=int, default=8)
    args = parser.parse_args()

    center_pad = True
    if args.no_center_pad:
        center_pad = False
    elif args.center_pad:
        center_pad = True

    if args.generation_mode == "precomputed":
        generate_pushworld_dataset(
            output_path=args.output_path,
            planning_results_path=args.planning_results_path,
            puzzles_path=args.puzzles_path,
            test_fraction=args.test_fraction,
            seed=args.seed,
            puzzle_names=args.puzzle_names,
            grid_size=args.grid_size,
            center_pad=center_pad,
            encoder_name=args.encoder_name,
            rollout_strategy_specs_json=args.rollout_strategies_json,
        )
        return

    generate_pushworld_dataset_with_cpp_planner(
        output_path=args.output_path,
        planning_results_path=args.planning_results_path,
        puzzles_path=args.puzzles_path,
        test_fraction=args.test_fraction,
        seed=args.seed,
        puzzle_names=args.puzzle_names,
        grid_size=args.grid_size,
        center_pad=center_pad,
        encoder_name=args.encoder_name,
        num_puzzles=args.num_puzzles,
        num_test_puzzles=args.num_test_puzzles,
        states_per_puzzle=args.states_per_puzzle,
        expert_starts_per_puzzle=args.expert_starts_per_puzzle,
        solutions_per_state=args.solutions_per_state,
        start_walk_min_steps=args.start_walk_min_steps,
        start_walk_max_steps=args.start_walk_max_steps,
        detour_min_steps=args.detour_min_steps,
        detour_max_steps=args.detour_max_steps,
        planner_path=args.planner_path,
        planner_mode=args.planner_mode,
        planner_workers=args.planner_workers,
        planner_time_limit=args.planner_time_limit,
        planner_memory_limit_gb=args.planner_memory_limit_gb,
        max_state_attempts_multiplier=args.max_state_attempts_multiplier,
        max_solution_attempts_multiplier=args.max_solution_attempts_multiplier,
    )


if __name__ == "__main__":
    main()
