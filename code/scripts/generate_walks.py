from src.process.collector import BatchPortalCollector, BatchCollector
from src.process.gaussian import GaussianIncrements
from src.domain import RectDomain, RectPortalDomain

import pandas as pd

from tqdm import tqdm

import argparse

import torch

def generate_walks(batch_size, num_exits, time_step, portal_args, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    process = GaussianIncrements(time_step)

    if portal_args['use_portal']:
        domain = RectPortalDomain(portal_args['portal_low'], portal_args['portal_high'])
        collector = BatchPortalCollector(domain, process, batch_size, num_exits)
    else:
        domain = RectDomain()
        collector = BatchCollector(domain, process, batch_size, num_exits)

    history = {
        'old_point': [],
        'new_point': [],
        'exited': [],
        'intersection': []
    }

    with tqdm(total=collector.max_exits, unit='exits') as pbar:
        for _, (old_point, new_point, exited, intersection) in enumerate(collector):
            history['old_point'].append(old_point.tolist())
            history['new_point'].append(new_point.tolist())
            history['exited'].append(exited.tolist())
            history['intersection'].append(intersection.tolist())

            pbar.update(torch.count_nonzero(exited).item())

    walks = pd.DataFrame(history).explode(['old_point', 'new_point', 'exited', 'intersection'])
    
    walks['time_step'] = time_step
    walks['seed'] = seed

    return walks

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Generate steps of a Brownian walkers.')

    # Define command-line arguments
    parser.add_argument('time_step', type=float, help='time step of diffusion')
    parser.add_argument('num_exits', type=int, help='number of exits from domain')
    parser.add_argument('batch_size', type=int, help='number of walkers at a time')
    parser.add_argument('--random_seed', '-rs', type=int, help='random seed to use')
    parser.add_argument('--use_portal', '-up', type=bool, default=False)
    parser.add_argument('--portal_low', '-pl', type=float, default=1/3)
    parser.add_argument('--portal_high', '-ph', type=float, default=2/3)
    parser.add_argument('--output_dir', '-o', type=str, default='./data/walks.parquet')

    # Parse the command-line arguments
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
   
    print(f"Using device: {device}")

    time_step = args.time_step
    num_exits = args.num_exits
    batch_size = (args.batch_size,)
    seed = args.random_seed

    walks = generate_walks(batch_size, num_exits, time_step, {
        'use_portal': args.use_portal,
        'portal_low': args.portal_low,
        'portal_high': args.portal_high
    }, seed)

    walks.to_parquet(args.output_dir)

    exited = walks[walks.exited == True]['exited']

    print(f'Number of exits {exited.sum()}')

if __name__ == '__main__':
    main()
