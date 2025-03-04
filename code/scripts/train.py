import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from torch.utils.tensorboard.writer import SummaryWriter
from src.domain.domain import RectDomain
from src.models.base_model import SimpleFeedForward
from src.process.collector import BatchCollector
from src.process.gaussian import GaussianIncrements
from src.utils.scoring import compute_probs, construct_scoring_permutations, construct_scoring_tensor, estimate_prob_to_hit_first

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
 
    os.makedirs(config['training']['model_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and config['hardware']['cuda'] else 'cpu')
    print(f"Using device: {device}")

    model = None
    if config['model']['name'] == 'feed-forward':
        model = SimpleFeedForward(config['model']['input_size'], config['model']['hidden_size'], config['model']['output_size'])
    else:
        print('Invalid model name. Valid models are:')
        print('\t\t- "feed-forward"')
        return
    
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    writer = SummaryWriter(log_dir='logs') 
    
    start_epoch = 0
    best_loss = float('inf')

    if config['training']['checkpoints']:
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    time_step = torch.tensor(config['walkers']['time_step'], device=device)
    process = GaussianIncrements(time_step)
    domain = RectDomain() 
    collector = BatchCollector(domain, process, (config['training']['batch_size'],), config['training']['num_exits'])

    model.train()

    permutations = construct_scoring_permutations(2, device)
    scoring_tensor = construct_scoring_tensor(2, device)

    with tqdm(total=collector.max_exits, unit='exits') as pbar:
        for _, (start, end, exited, clipped) in enumerate(collector):

            if torch.any(exited):
                closest_barrier = clipped.round()
                prob_tensor = compute_probs(end - start, closest_barrier - start, time_step)
                prob_hit = estimate_prob_to_hit_first(prob_tensor, 2, 0, permutations, scoring_tensor, device=device)
                est_bv = 1 - prob_hit 

                print(f"START: {start}")
                print(f"END: {end}")
                print(f"PROB: {prob_hit}")
                print(f"BV: {est_bv}")
                print("---")
            
            pbar.update(torch.count_nonzero(exited).item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PyTorch model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed != None:
        torch.manual_seed(args.seed)
    
    train(args.config)
