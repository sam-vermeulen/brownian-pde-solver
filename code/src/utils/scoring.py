import torch
from itertools import product

from src.statistics.brownian_bridge import expected_time_to_hit, prob_to_hit

def construct_scoring_permutations(n_dims, device=None):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.tensor(tuple(product(range(3), repeat=n_dims)), device=device)

def construct_scoring_tensor(n_dims, device=None):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    score = torch.zeros([3] * n_dims)

    for j in construct_scoring_permutations(n_dims):
        if j[0] == 0:
            score[tuple(j)] = 0
        elif j[0] > j[1:].max():
            score[tuple(j)] = 1
        elif j[0] == j[1:].max():
            score[tuple(j)] = 1 / torch.count_nonzero(j == j.max())

    return score

def swap_to_first(a, component):
    temp = a[:, 0, :].clone()
    a[:, 0, :] = a[:, component, :]
    a[:, component, :] = temp

def compute_probs(end, barrier, step_size):
    expected_hit_time = expected_time_to_hit(end, barrier, step_size)
    miss_probs = 1 - prob_to_hit(end, barrier, step_size)

    late_time = step_size

    late_probs = expected_hit_time / late_time
    early_probs = 1 - late_probs - miss_probs
    
    stacked = torch.stack([miss_probs, late_probs, early_probs])
    prob = stacked.permute(1, 2, 0)

    return prob

def estimate_prob_to_hit_first(prob_tensor, n_dims, component, permutations, scoring_tensor, device=None):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if component != 0:
        swap_to_first(prob_tensor, component)

    sum = 0
    for j in permutations:
        sum += prob_tensor[:, torch.arange(n_dims, device=device), tuple(j)].prod(dim=-1) * scoring_tensor[tuple(j)]

    if component != 0:
        swap_to_first(prob_tensor, component)

    return sum 
