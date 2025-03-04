# Brownian PDE Solver 

A Python library for simulating and analyzing Brownian motion with a focus on exit probabilities and boundary value problems.

## Overview

This project provides tools for:
- Simulating Brownian motion processes in rectangular domains
- Collecting and analyzing statistics on exit times and locations
- Training neural networks to approximate boundary value problems using Brownian motion
- Supporting both standard rectangular domains and domains with portals

## Installation

```bash
# Clone the repository
git clone https://github.com/sam-vermeulen/brownian-pde-solver.git
cd brownian-pde-solver 

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Brownian Walks

```bash
python generate_walks.py 0.01 1000 100 --output_dir ./data/my_walks.parquet
```

Arguments:
- `time_step`: Time step for the Brownian motion (e.g., 0.01)
- `num_exits`: Number of exits to collect (e.g., 1000)
- `batch_size`: Number of walkers to simulate simultaneously (e.g., 100)
- `--random_seed`: Optional seed for reproducibility
- `--use_portal`: Enable portal boundaries (default: False)
- `--portal_low`: Lower portal boundary (default: 1/3)
- `--portal_high`: Upper portal boundary (default: 2/3)
- `--output_dir`: Output directory for the parquet file (default: ./data/walks.parquet)

### Train a Model

```bash
python train.py --config configs/default.yaml --seed 42
```

Arguments:
- `--config`: Path to the configuration file (default: configs/default.yaml)
- `--seed`: Random seed for reproducibility

## Configuration

Example configuration file (`configs/default.yaml`):

```yaml
model:
  name: feed-forward
  input_size: 2
  hidden_size: [64, 64, 64, 64] 
  output_size: 1

training:
  batch_size: 64 
  num_exits: 16384 
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0001
  model_dir: ./models
  checkpoints: true
  checkpoint_dir: ./checkpoints

walkers:
  time_step: 0.01

hardware:
  cuda: true
```

## Project Structure

```
.
├── configs/              # Configuration files
├── data/                 # Data storage
├── logs/                 # TensorBoard logs
├── models/               # Saved models
├── scripts/              # Saved models
│   ├── generate_walks.py # CLI for generating random walks 
│   ├── train.py          # CLI for training models 
├── src/
│   ├── domain/           # Domain definitions
│   ├── models/           # Neural network models
│   ├── process/          # Stochastic processes
│   └── utils/            # Utility functions
```

## Dependencies

- PyTorch
- pandas
- tqdm
- PyYAML
- TensorBoard

## License

MIT License
