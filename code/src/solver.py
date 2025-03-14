import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np

from src.utils.scoring import compute_probs, construct_scoring_permutations, construct_scoring_tensor, estimate_prob_to_hit_first

from pde import CartesianGrid, solve_laplace_equation 
import matplotlib.pyplot as plt

def finite_difference_laplace_soln(points):
  grid = CartesianGrid([(0, 1)] * 2, points)
  bcs = [({'value': 0}, {'value': 0}), ({'value': 1}, {'value': 1})]
  result = solve_laplace_equation(grid, bcs)
  return result

class TDFeynmanSolver():
    def __init__(self, model: nn.Module, optimizer, collector, device=None):
        self.model = model
        self.collector = collector
        self.optimizer = optimizer

        self.permutations = construct_scoring_permutations(2, device)
        self.scoring_tensor = construct_scoring_tensor(2, device)
        
        if device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.points = 100
        self.fd_solution = torch.tensor(finite_difference_laplace_soln(self.points).data.transpose(1,0), device=self.device)

        x = np.linspace(0, 1, self.points, dtype=np.float32) 
        y = np.linspace(0, 1, self.points, dtype=np.float32) 

        xv, yv = np.meshgrid(x, y)
        xv = xv.reshape(-1,)
        yv = yv.reshape(-1,)

        self.eval_coords = torch.tensor(np.stack([xv, yv], axis=-1), device=self.device)
        

    def g_estimate_improved(self, start: torch.Tensor, end: torch.Tensor, clipped: torch.Tensor) -> torch.Tensor | int:
        self.plot_name = "new"
        closest_barrier = clipped.round()
        prob_tensor = compute_probs(end - start, closest_barrier - start, self.collector.process.time_step)
        prob_hit = estimate_prob_to_hit_first(prob_tensor, 2, 0, self.permutations, self.scoring_tensor, device=self.device)
        est_bv = 1 - prob_hit 

        return est_bv

    def g_estimate(self, start: torch.Tensor, end: torch.Tensor, clipped: torch.Tensor) -> torch.Tensor | int:
        self.plot_name = "old"
        est_bv = torch.where(torch.isclose(clipped[:, 1], self.collector.domain.boundaries[1, 1]) | torch.isclose(clipped[:, 1], self.collector.domain.boundaries[1, 0]), 1, 0)
        return est_bv 

    def f_estimate(self, end: torch.Tensor) -> torch.Tensor:
        values = torch.zeros(end.shape[:-1], device=self.device)
        return values

    def loss_fn(self, start: torch.Tensor, end: torch.Tensor, u_end: torch.Tensor) -> torch.Tensor:
        loss = ((1/2) * (u_end - self.model(start) - (1/2)*self.f_estimate(end)[:, None])**2)
        return loss.mean()

    def train(self):
        writer = SummaryWriter(log_dir='logs') 
        # best_loss = float('inf')

        with tqdm(total=self.collector.max_exits, unit='exits') as pbar:
            for i, (start, end, exited, clipped) in enumerate(self.collector):
                self.model.train()
                u_end = torch.where(exited[:, None], self.g_estimate(start, end, clipped)[:, None], self.model(end).detach())

                loss = self.loss_fn(start, end, u_end)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i % 100 == 0:
                    writer.add_scalar("loss", loss, i // 100)

                    self.model.eval()
                    pred = self.model(self.eval_coords).reshape(self.points, self.points).detach()
                    error = (pred-self.fd_solution)**2
                    writer.add_scalar(f"mean_error_{self.plot_name}", error.mean(), i // 100)
                
                pbar.update(torch.count_nonzero(exited).item())

        writer.flush()
        writer.close()

        self.model.eval()
        pred = self.model(self.eval_coords).reshape(self.points, self.points).detach()
        #error = (pred-self.fd_solution)**2

        plt.figure(figsize=(10, 8))
        plt.imshow(pred.cpu(), cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title('100x100 Torch Tensor Visualization')
        plt.tight_layout()
        plt.savefig(f'./plots/{self.plot_name}')
