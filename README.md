# Contrastive Representations for Temporal Reasoning

Official repository of [Contrastive Representations for Temporal Reasoning](TODO_arxiv_link) (CRTR).  

<p align="center">
  <img src="imgs/figure_1.png" width="85%">
</p>


## Installation
We recommend using Python v3.10. Set up the repository by running:
```bash
pip install -e .
```


## Datasets
### Example datasets
The example datasets for the Rubik's Cube and Sokoban are stored in `example_datasets/{rubik/sokoban}`. You can use them to verify that the code runs correctly. To replicate the results from the paper, please download the full datasets as described below.

### Downloading Full Datasets
#### Install huggingface-cli:
```pip install -U "huggingface_hub[cli]"```
#### Install git lfs (for Ubuntu):
```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
```


### Rubik's Cube
The dataset for the Rubik's Cube requires `12GB` of available storage. Download it by running:
```bash
huggingface-cli download oolongie/rubik_randomly_shuffled --repo-type dataset  --local-dir training_datasets/rubik
```
You can also generate the dataset by running:
```bash
python rubik_generate_script.py <number_of_shuffles (the paper uses 21)> <save_folder>
```

### Sokoban
The dataset for Sokoban requires `14GB` of available storage. Download it by running:
```bash
huggingface-cli download oolongie/sokoban-12-12-4-trajectories --repo-type dataset  --local-dir training_datasets/sokoban
```

### PushWorld
PushWorld is integrated against the upstream PushWorld repo and adapted into CRTR with pluggable observation encoders and rollout strategies.

To generate a PushWorld dataset from solved benchmark plans:
```bash
python pushworld_generate_script.py \
  --output_path training_datasets/pushworld \
  --planning_results_path /path/to/pushworld/benchmark/solutions/level1 \
  --puzzles_path /path/to/pushworld/benchmark/puzzles/level1
```

This writes:
- `training_datasets/pushworld/train/train_trajectories.pkl`
- `training_datasets/pushworld/train/train_lens.pkl`
- `training_datasets/pushworld/test/test_trajectories.pkl`
- `training_datasets/pushworld/test/test_lens.pkl`
- `training_datasets/pushworld/metadata.json`

The default PushWorld encoder is `categorical_grid`, which uses 10 categorical cell types:
- empty
- wall
- agent-only wall
- goal
- movable
- movable on goal
- goal-conditioned movable
- goal-conditioned movable on goal
- agent
- agent on goal

There is also an `object_identity_grid` encoder that allocates separate tokens per movable object instance. This is more expressive than the default categorical grid and can be selected with:
```bash
python pushworld_generate_script.py \
  --output_path training_datasets/pushworld_identity \
  --planning_results_path /path/to/pushworld/benchmark/solutions/level1 \
  --puzzles_path /path/to/pushworld/benchmark/puzzles/level1 \
  --encoder_name object_identity_grid
```

The dataset generator also supports multiple rollout strategies through `--rollout_strategies_json`. Example:
```bash
python pushworld_generate_script.py \
  --output_path training_datasets/pushworld_augmented \
  --planning_results_path /path/to/pushworld/benchmark/solutions/level1 \
  --puzzles_path /path/to/pushworld/benchmark/puzzles/level1 \
  --rollout_strategies_json '[{"name":"expert","count":1},{"name":"solution_suffix","count":2},{"name":"epsilon_plan","count":2,"epsilon":0.1},{"name":"random_walk","count":2,"walk_length":8}]'
```

The currently implemented rollout strategies are:
- `expert`
- `solution_suffix`
- `epsilon_plan`
- `random_walk`


### Eval Boards
Boards that are used for Sokoban evaluation are stored in `example_datasets/sokoban_eval_boards/eval_boards.pkl`.


## Training
To run the training of our method, use the following command:

### Rubik's Cube
For the example dataset:
```bash
python runner.py --config_file configs/train/crtr/rubik.gin
```
For the real dataset:
```bash
python runner.py --config_file configs/train/crtr/rubik.gin --gin_bindings "ContrastiveDataset.path=training_datasets/rubik"
```

### Sokoban
For the example dataset:
```bash
python runner.py --config_file configs/train/crtr/sokoban.gin
```
For the real dataset:
```bash
python runner.py --config_file configs/train/crtr/sokoban.gin --gin_bindings "ContrastiveDataset.path=training_datasets/sokoban/train" "TrainJob.test_path=training_datasets/sokoban/test"
```

### PushWorld
After generating the dataset and copying or binding the benchmark paths into the config, run:
```bash
python runner.py --config_file configs/train/crtr/pushworld.gin
```

If you switch encoders, update `LNConvNet.input_size` to match `num_cell_types` in `metadata.json`.

If your benchmark lives elsewhere, override the benchmark paths at runtime:
```bash
python runner.py --config_file configs/train/crtr/pushworld.gin --gin_bindings \
  "ContrastiveDatasetDiffLen.path='training_datasets/pushworld/train'" \
  "TrainJob.test_path='training_datasets/pushworld/test'" \
  "CustomPushWorldEnv.planning_results_path='/path/to/pushworld/benchmark/solutions/level1'" \
  "CustomPushWorldEnv.puzzles_path='/path/to/pushworld/benchmark/puzzles/level1'" \
  "generate_problems_pushworld.planning_results_path='/path/to/pushworld/benchmark/solutions/level1'" \
  "generate_problems_pushworld.puzzles_path='/path/to/pushworld/benchmark/puzzles/level1'"
```

## Evaluation
Pretrained checkpoints are provided in the folder `example_checkpoints`.
Evaluation of our method on Sokoban can be done by running:
```bash
python runner.py --config_file configs/solve/search/contrastive/sokoban.gin
```
And on the Rubik's Cube by running:
```bash
python runner.py --config_file configs/solve/search/contrastive/rubik.gin
```

PushWorld search evaluation uses:
```bash
python runner.py --config_file configs/solve/search/contrastive/pushworld.gin
```

## Logging 
By default, experiment artifacts and results are stored in directory `result_<timestamp>` 
