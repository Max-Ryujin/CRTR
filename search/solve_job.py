from collections import defaultdict
import time
from copy import deepcopy
import sys

import numpy as np

from joblib import Parallel, delayed
from utils.jax_rand import set_seed
import gin


def solve_problem(solver, problem, jax_seed):
    sys.setrecursionlimit(1000000)
    set_seed(jax_seed)

    time_s = time.time()
    problem_context = None
    if isinstance(problem, dict):
        input_state = problem["input_state"]
        solved_state = problem["solved_state"]
        problem_context = problem.get("problem_context")
    elif len(problem) == 3:
        input_state, solved_state, problem_context = problem
    else:
        input_state, solved_state = problem

    solution, tree_metrics, root, trajectory_actions, additional_info = solver.solve(
        input_state,
        solved_state,
        problem_context=problem_context,
    )
    time_solving = time.time() - time_s
    return dict(
        solution=solution,
        tree_metrics=tree_metrics,
        root=root,
        trajectory_actions=trajectory_actions,
        time_solving=time_solving,
        input_problem=deepcopy(input_state),
        problem_context=deepcopy(problem_context),
        additional_info=additional_info,
    )


@gin.configurable
class SolveJob:
    def __init__(
        self,
        loggers,
        solver_class,
        n_jobs,
        n_parallel_workers,
        batch_size,
        network,
        generate_problems,
        shuffles,
        budget_checkpoints,
        n_actions=None,
        metric=None,
        output_dir=None,
        log_videos=False,
        max_videos=1,
        video_fps=4,
    ):

        self.loggers = loggers
        self.solver_class = solver_class
        self.n_jobs = n_jobs
        self.n_parallel_workers = n_parallel_workers
        self.batch_size = batch_size
        self.budget_checkpoints = budget_checkpoints
        self.shuffles = shuffles
        self.n_actions = n_actions
        self.generate_problems = generate_problems
        self.solution_lengths = []
        self.budget_solved = defaultdict(int)
        self.budget_exp_solved = defaultdict(int)
        self.prefix = (
            f"{self.shuffles}_shuffles_{'search' if self.n_actions != 1 else 'argmax'}"
        )

        self.logged_solutions = 0

        self.solved_boards = 0
        self.all_boards = 0
        self.network = network
        self.metric = metric
        self.log_videos = log_videos
        self.max_videos = max_videos
        self.video_fps = video_fps

    def execute(self, step=0):
        solver = self.solver_class(
            network=self.network,
            metric=self.metric,
            n_actions=self.n_actions,
            shuffles=self.shuffles,
        )
        solver.construct_networks()

        problems_to_solve = self.generate_problems(self.n_jobs, self.shuffles)

        jobs_done = 0
        jobs_to_do = self.n_jobs
        batch_num = 0
        all_batches = self.n_jobs // self.batch_size

        all_results = []

        total_time_start = time.time()
        while jobs_to_do > 0:
            jobs_in_batch = min(jobs_to_do, self.batch_size)
            problems_to_solve_in_batch = problems_to_solve[
                jobs_done : jobs_done + jobs_in_batch
            ]
            print(
                "============================ Batch {:>4}  out  of  {:>4} ============================".format(
                    batch_num + 1, all_batches
                )
            )
            results = Parallel(n_jobs=self.n_parallel_workers, verbose=10000)(
                delayed(solve_problem)(solver, input_problem, 0)
                for input_problem in problems_to_solve_in_batch
            )

            all_results += results

            print(
                "==================================================================================="
            )

            jobs_done += jobs_in_batch
            jobs_to_do -= jobs_in_batch
            batch_num += 1

        if self.log_videos:
            self.log_solution_videos(all_results, solver, step)

        self.log_results(all_results, step)

    def log_solution_videos(self, results, solver, step):
        goal_builder = getattr(solver, "goal_builder", None)
        renderer = getattr(goal_builder, "env", None)
        if renderer is None:
            return

        logged_videos = 0
        for result in results:
            solution = result.get("solution")
            if solution is None:
                continue

            if hasattr(renderer, "render_solution"):
                frames = renderer.render_solution(
                    trajectory_actions=result.get("trajectory_actions"),
                    problem_context=result.get("problem_context"),
                )
            elif hasattr(renderer, "render"):
                problem_context = result.get("problem_context")
                if problem_context is not None and hasattr(
                    renderer, "set_problem_context"
                ):
                    renderer.set_problem_context(problem_context)

                states = [np.asarray(node.state) for node in solution]
                frames = []
                for state in states:
                    frame = renderer.render(state)
                    if frame is None:
                        frames = []
                        break
                    frame = np.asarray(frame)
                    if frame.ndim == 2:
                        frame = np.repeat(frame[..., None], 3, axis=-1)
                    frames.append(frame)
            else:
                return

            if frames:
                self.loggers.log_video(
                    self.get_name(f"solution_video_{logged_videos}"),
                    step,
                    np.stack(frames, axis=0),
                    fps=self.video_fps,
                )
                logged_videos += 1

            if logged_videos >= self.max_videos:
                break

    def get_name(self, name):
        if self.prefix is not None:
            return f"{self.prefix}/{name}"
        return name

    def log_results(self, results, step):
        for log_num, result in enumerate(results):
            # log_scalar_metrics('tree', step+log_num, result['tree_metrics'])
            self.loggers.log_scalar(
                "solution_length",
                step,
                len(result["solution"]) if result["solution"] is not None else -1,
            )
            self.loggers.log_scalar("nodes", step, result["tree_metrics"]["nodes"])
            self.loggers.log_scalar(
                "expanded_nodes", step, result["tree_metrics"]["expanded_nodes"]
            )

            solved = result["solution"] is not None
            if solved:
                self.solved_boards += 1
                self.solution_lengths.append(len(result["solution"]))
                for budget in self.budget_checkpoints:
                    if result["tree_metrics"]["nodes"] < budget:
                        self.budget_solved[budget] += 1
                    if result["tree_metrics"]["expanded_nodes"] < budget:
                        self.budget_exp_solved[budget] += 1

            self.all_boards += 1

        self.loggers.log_scalar(
            self.get_name("solved_rate"), step, self.solved_boards / self.all_boards
        )

        avg_length = (
            0
            if not self.solution_lengths
            else (sum(self.solution_lengths) / len(self.solution_lengths))
        )
        self.loggers.log_scalar(self.get_name("avg_length"), step, avg_length)

        for budget in self.budget_checkpoints:
            self.loggers.log_scalar(
                self.get_name(f"solved_rate_{budget}_nodes"),
                step,
                self.budget_solved[budget] / self.all_boards,
            )
            self.loggers.log_scalar(
                self.get_name(f"solved_rate_{budget}_exp_nodes"),
                step,
                self.budget_exp_solved[budget] / self.all_boards,
            )
