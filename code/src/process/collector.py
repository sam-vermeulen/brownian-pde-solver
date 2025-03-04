from typing import Generator, Tuple
import torch
from src.domain import Domain, RectPortalDomain
from src.process.gaussian import GaussianIncrements

class BatchCollector():
    '''
    helper function to generate points within domain, simulate Brownian motion, collect information needed to train model
    '''
    def __init__(self, domain: Domain, process: GaussianIncrements, batch_size: Tuple, max_exits: int):
        self.domain = domain
        self.process = process
        self.max_exits = max_exits
        self.batch_size = batch_size
        self.points = domain.sample_points(batch_size)
        self.remaining_walkers = self.max_exits

    def reset(self):
        self.points = self.domain.sample_points(self.batch_size)

    # update step for walkers, yields old positions, new positions, which points have exited domain,
    # and the point of intersection with the domain along the line between new and old points
    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        while self.remaining_walkers > 0:

            old_points = self.points.clone()
            noise = self.process.get_noise(self.points)
            self.points += noise

            exited = ~self.domain.points_inside(self.points)

            if torch.any(exited):
                clipped_points = torch.where(exited[:, None], self.domain.exit_point(old_points, self.points), self.points)
            else:
                clipped_points = self.points

            yield old_points, self.points, exited, clipped_points

            if torch.any(exited):
                num_exited = torch.count_nonzero(exited).item()
                self.remaining_walkers -= num_exited
                self.points = self.domain.resample_points(self.points, where=exited)

class BatchPortalCollector():
    '''
    helper function to generate points within domain, simulate Brownian motion, collect information needed to train model
    '''
    def __init__(self, domain: RectPortalDomain, process: GaussianIncrements, batch_size: Tuple, max_exits: int):
        self.domain = domain
        self.process = process
        self.max_exits = max_exits
        self.batch_size = batch_size
        self.points = domain.sample_points(batch_size)
        self.remaining_walkers = self.max_exits

    def reset(self):
        self.points = self.domain.sample_points(self.batch_size)

    # update step for walkers, yields old positions, new positions, which points have exited domain,
    # and the point of intersection with the domain along the line between new and old points
    def __iter__(self):
        while self.remaining_walkers > 0:

            old_points = self.points.clone()

            # apply noise
            noise = self.process.get_noise(self.points)
            self.points += noise

            # apply portal
            self.points, exit_point = self.domain.get_portal_exit_location(old_points, self.points)

            exited = ~self.domain.points_inside(self.points)

            if torch.any(exited):
                clipped_points = torch.where(exited[:, None], exit_point, self.points)
            else:
                clipped_points = self.points

            yield old_points, self.points, exited, clipped_points

            if torch.any(exited):
                num_exited = torch.count_nonzero(exited).item()
                self.remaining_walkers -= num_exited
                self.points = self.domain.resample_points(self.points, where=exited)
