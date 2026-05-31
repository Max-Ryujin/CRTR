import json
from typing import Optional

import gin
import numpy as np

from envs.pushworld.data import build_encoder_from_metadata
from envs.pushworld.data import build_observation_encoder
from envs.pushworld.data import load_solution_records
from envs.pushworld.puzzle import Actions
from envs.pushworld.puzzle import NUM_ACTIONS, PushWorldPuzzle


@gin.configurable()
class CustomPushWorldEnv:
    def __init__(
        self,
        planning_results_path,
        puzzles_path,
        metadata_path: Optional[str] = None,
        puzzle_names: Optional[str] = None,
        exclude_puzzle_names: Optional[str] = None,
        grid_size: Optional[int] = None,
        center_pad: bool = True,
        encoder_name: str = "categorical_grid",
        **unused_kwargs,
    ):
        if metadata_path is not None:
            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)
            self.encoder = build_encoder_from_metadata(metadata)
        else:
            records = load_solution_records(
                planning_results_path=planning_results_path,
                puzzles_path=puzzles_path,
                puzzle_names=puzzle_names,
                exclude_puzzle_names=exclude_puzzle_names,
            )
            self.encoder = build_observation_encoder(
                records=records,
                encoder_name=encoder_name,
                grid_size=grid_size,
                center_pad=center_pad,
            )

        self._current_puzzle = None
        self._state_lookup = {}

    def set_problem_context(self, problem_context):
        self._state_lookup = {}
        self._current_puzzle = None

        if problem_context is None:
            return

        puzzle_file_path = problem_context["puzzle_file_path"]
        self._current_puzzle = PushWorldPuzzle(puzzle_file_path)

        initial_state = problem_context.get("initial_raw_state")
        if initial_state is not None:
            encoded_state = self.encoder.flatten(self._current_puzzle, initial_state)
            self._state_lookup[self._state_key(encoded_state)] = tuple(initial_state)

    def reset(self):
        if self._current_puzzle is None:
            return None

        encoded_state = self.encoder.flatten(
            self._current_puzzle, self._current_puzzle.initial_state
        )
        self._state_lookup[self._state_key(encoded_state)] = (
            self._current_puzzle.initial_state
        )
        return encoded_state.astype(np.float32)

    def _state_key(self, state):
        return np.asarray(state, dtype=np.int16).reshape(-1).tobytes()

    def _lookup_raw_state(self, state):
        if self._current_puzzle is None:
            raise RuntimeError("Problem context must be set before stepping PushWorld")

        key = self._state_key(state)
        if key not in self._state_lookup:
            raise KeyError(
                "Encoded PushWorld state is unknown to the environment cache"
            )
        return self._state_lookup[key]

    def step(self, state, action):
        if action not in self.get_all_actions():
            raise ValueError(f"Unsupported PushWorld action: {action}")

        raw_state = self._lookup_raw_state(state)
        next_state = self._current_puzzle.get_next_state(raw_state, action)
        encoded_state = self.encoder.flatten(self._current_puzzle, next_state)
        self._state_lookup[self._state_key(encoded_state)] = next_state
        done = self._current_puzzle.is_goal_state(next_state)
        return encoded_state.astype(np.float32), None, done, None

    def get_all_actions(self):
        return list(range(NUM_ACTIONS))

    def render(self, state, border_width=2, pixels_per_cell=20):
        raw_state = self._lookup_raw_state(state)
        return self._current_puzzle.render(
            raw_state,
            border_width=border_width,
            pixels_per_cell=pixels_per_cell,
        )

    def render_solution(
        self,
        trajectory_actions,
        problem_context,
        border_width=2,
        pixels_per_cell=20,
    ):
        if not trajectory_actions or problem_context is None:
            return []

        puzzle = PushWorldPuzzle(problem_context["puzzle_file_path"])
        state = tuple(
            tuple(position) for position in problem_context["initial_raw_state"]
        )
        frames = [
            puzzle.render(
                state,
                border_width=border_width,
                pixels_per_cell=pixels_per_cell,
            )
        ]

        for action in trajectory_actions:
            if isinstance(action, str):
                action = Actions.FROM_CHAR[action]
            state = puzzle.get_next_state(state, action)
            frames.append(
                puzzle.render(
                    state,
                    border_width=border_width,
                    pixels_per_cell=pixels_per_cell,
                )
            )

        return frames
