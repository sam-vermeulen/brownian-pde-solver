import torch
import pandas as pd
import scipy.integrate as spi

from src.statistics.brownian_bridge import prob_to_hit
from src.utils.scoring import compute_probs, construct_scoring_permutations, construct_scoring_tensor, estimate_prob_to_hit_first

def hitting_time(x, b, s, t, p, q):
    c = 1 / torch.sqrt(2 * torch.pi * (t - s)) * torch.exp((-(q - p)**2) / (2 * (t - s)))
    
    numerator = torch.abs(p - b)

    denominator = 2 * c * torch.pi * (x - s)**(3/2) * (t - x)**(1/2)

    exponential = torch.exp((-(q - b)**2)/(2 * (t - x)) - (p - b)**2/(2 * (x - s)))
    
    return numerator / denominator * exponential

def joint_hitting_time(y, x, x_params, y_params):
    return hitting_time(x, *x_params) * hitting_time(y, *y_params)

def calc_exit_prob(end, start, barrier, time_step):
    s = 0
    t = time_step

    x_params = (barrier[:, 0], s, t, start[:, 0], end[:, 0])
    y_params = (barrier[:, 1], s, t, start[:, 1], end[:, 1])
    
    prob, _ = spi.dblquad(joint_hitting_time, s, t, gfun = lambda x: s, hfun = lambda x: x, args=(x_params, y_params), epsabs=1.5e-6, epsrel=1.5e-6)
    prob_hit_y = prob_to_hit(end[:, 1] - start[:, 1], barrier[:, 1] - start[:, 1], time_step)
    return 1 - prob_hit_y + prob

def calc_new_exit_prob(end, start, barrier, time_step, permutations, scoring_tensor, device) -> torch.Tensor:
    prob_tensor = compute_probs(end - start, barrier - start, time_step)
    prob_hit = estimate_prob_to_hit_first(prob_tensor, 2, 0, permutations, scoring_tensor, device=device)
    return prob_hit

def calc_old_exit_prob(intersection, device) -> torch.Tensor:
    return 1 - torch.where(torch.isclose(intersection[:, 1], torch.tensor(1., dtype=torch.float64, device=device)) | torch.isclose(intersection[:, 1], torch.tensor(0., dtype=torch.float64, device=device)), 1, 0)

def append_exit_probs(df, permutations, scoring_tensor, device):
    new = []
    old = []
    print(df.shape)
    for i, (_, row) in enumerate(df.iterrows()):
        start = torch.tensor(row.old_point, device=device)[None, :]
        end = torch.tensor(row.new_point, device=device)[None, :]
        closest_barrier = torch.tensor(row.barrier, device=device)[None, :]
        intersection = torch.tensor(row.intersection, device=device)[None, :]
        time_step = torch.tensor(row.time_step, device=device)

        
        prob_x_first_new = calc_new_exit_prob(end, start, closest_barrier, time_step, permutations, scoring_tensor, device)
        prob_x_first_old = calc_old_exit_prob(intersection, device)
        #prob_x_first = calc_exit_prob(end, start, closest_barrier, time_step)

        #print("tru: ", prob_x_first)
        new.append(prob_x_first_new.cpu().item())
        old.append(prob_x_first_old.cpu().item())

    df["prob_x_first_new"] = new
    df["prob_x_first_old"] = old

    return df

if __name__ == "__main__":
    seeds = range(0, 20)
    step_sizes = [0.002, 0.003, 0.004, 0.005, 0.007]

    device = torch.device('cpu')
        
    permutations = construct_scoring_permutations(2, device)
    scoring_tensor = construct_scoring_tensor(2, device)

    for ss in step_sizes:
        for s in seeds:
            path = f'./data/exit_probabilities-{ss}-{s}.parquet'
            df = pd.read_parquet(path)
            df = append_exit_probs(df, permutations, scoring_tensor, device)
            df = df[(df.prob_x_first <= 1) & (df.prob_x_first >= 0)]
            df.to_parquet(path)
