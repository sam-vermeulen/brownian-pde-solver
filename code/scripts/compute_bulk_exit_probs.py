import torch
import pandas as pd

from src.utils.scoring import construct_scoring_permutations, construct_scoring_tensor

def append_exit_probs(df, permutations, scoring_tensor):
    pass

if __name__ == "__main__":
    seeds = range(0, 10)
    step_sizes = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    permutations = construct_scoring_permutations(2, device)
    scoring_tensor = construct_scoring_tensor(2, device)

    for s in seeds:
        df = pd.read_parquet(f'./data/exit_probabilities-{step_sizes}-{s}.parquet')
        df = append_exit_probs(df, permutations, scoring_tensor)

        

