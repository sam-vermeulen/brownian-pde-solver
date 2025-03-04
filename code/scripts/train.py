import argparse
import os
import torch
import torch.optim as optim
import yaml

from src.domain.domain import RectDomain
from src.models.base_model import FeedForward 
from src.process.collector import BatchCollector
from src.process.gaussian import GaussianIncrements
from src.solver import TDFeynmanSolver

def train(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
 
    os.makedirs(config['training']['model_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and config['hardware']['cuda'] else 'cpu')
    print(f"Using device: {device}")

    model = None
    if config['model']['name'] == 'feed-forward':
        model = FeedForward(config['model']['input_size'], config['model']['hidden_size'], config['model']['output_size'])
    else:
        print('Invalid model name. Valid models are:')
        print('\t\t- "feed-forward"')
        return
    
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    time_step = torch.tensor(config['walkers']['time_step'], device=device)
    process = GaussianIncrements(time_step)
    domain = RectDomain() 
    collector = BatchCollector(domain, process, (config['training']['batch_size'],), config['training']['num_exits'])

    trainer = TDFeynmanSolver(model, optimizer, collector, device) 
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a PyTorch model')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed != None:
        torch.manual_seed(args.seed)
    
    train(args.config)
