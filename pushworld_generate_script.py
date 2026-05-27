import argparse

from envs.pushworld.data import generate_pushworld_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--planning_results_path", required=True)
    parser.add_argument("--puzzles_path", required=True)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--puzzle_names", default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--encoder_name", default="categorical_grid")
    parser.add_argument("--rollout_strategies_json", default=None)
    parser.add_argument("--center_pad", action="store_true")
    parser.add_argument("--no_center_pad", action="store_true")
    args = parser.parse_args()

    center_pad = True
    if args.no_center_pad:
        center_pad = False
    elif args.center_pad:
        center_pad = True

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


if __name__ == "__main__":
    main()
