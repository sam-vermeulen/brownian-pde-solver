from abc import ABC, abstractmethod
from typing import Tuple
import torch

class Domain(ABC):
    def __init__(self, name=''):
        self.name = name

    @abstractmethod
    def sample_points(self, n: Tuple) -> torch.Tensor: 
        """
        Sample random points within the domain

        :param n: number of points to sample
        :return: np.array of points
        """
        pass

    def resample_points(self, points: torch.Tensor, where: torch.Tensor) -> torch.Tensor:
        return torch.where(where[:, None], self.sample_points(points.shape[:-1]), points)

    @abstractmethod
    def points_inside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Checks if points are within the domain

        :param points: the points to check
        :return: torch.Tensor which is true where point are inside the domain 
        """
        pass

    @abstractmethod
    def exit_point(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """
        Returns intersections points of domain boundary given start and end of line

        :param start: the start of the points to check
        :param end: the end of the points to check
        :return: np.array of intersection points
        """
        pass

class RectDomain(Domain):
    def __init__(self, boundaries=[[0, 1], [0, 1]], name='Rectangle', device=None):
        super().__init__(name=name)

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.boundaries = torch.tensor(boundaries, dtype=torch.float32, device=self.device)

    def sample_points(self, n: Tuple) -> torch.Tensor:
        points = torch.rand(size=(*n, 2), device=self.device)

        points = self.boundaries[:, 0] + points * (self.boundaries[:, 1] - self.boundaries[:, 0])
        return points   

    def points_inside(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0]
        y = points[:, 1]

        in_x = (x > self.boundaries[0, 0]) & (x < self.boundaries[0, 1])
        in_y = (y > self.boundaries[1, 0]) & (y < self.boundaries[1, 1])

        return in_x & in_y

    def exit_point(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:

        vec = end - start

        lower_bounds = self.boundaries[:, 0]
        upper_bounds = self.boundaries[:, 1]

        ratio_to_upper = (upper_bounds - start) / (vec+1e-32)
        ratio_to_lower = (lower_bounds - start) / (vec+1e-32)

        ratio = torch.maximum(ratio_to_lower, ratio_to_upper)
        ratio = torch.min(ratio, dim=1)

        intersection = start + ratio.values[:, None] * vec

        return intersection
    
class RectPortalDomain(Domain):
    def __init__(self, portal_min, portal_max, boundaries=[[0, 1], [0, 1]], boundary_values=[[0, 0], [1, 1]], name='Rectangle (Portal)', device=None):
        super().__init__(name=name) 
 
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.boundaries = torch.tensor(boundaries, dtype=torch.float32, device=self.device)
        self.boundary_values = torch.tensor(boundary_values, dtype=torch.float32, device=self.device)
        self.portal_min = portal_min
        self.portal_max = portal_max

    def sample_points(self, n: Tuple) -> torch.Tensor:
        x = self.boundaries[0,0] + torch.rand(size=n) / (self.boundaries[0, 1] - self.boundaries[0, 0])
        y = self.boundaries[1,0] + torch.rand(size=n) / (self.boundaries[1, 1] - self.boundaries[1, 0])

        return torch.column_stack([x, y])
    
    def points_inside(self, points: torch.Tensor) -> torch.Tensor:
        x = points[:, 0]
        y = points[:, 1]

        in_x = (x > self.boundaries[0, 0]) & (x < self.boundaries[0, 1])
        in_y = (y > self.boundaries[1, 0]) & (y < self.boundaries[1, 1])

        return in_x & in_y
    
    def exit_point(self, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:

        vec = end - start

        lower_bounds = self.boundaries[:, 0]
        upper_bounds = self.boundaries[:, 1]

        ratio_to_upper = (upper_bounds - start) / (vec+1e-32)
        ratio_to_lower = (lower_bounds - start) / (vec+1e-32)

        ratio = torch.maximum(ratio_to_lower, ratio_to_upper)
        ratio = torch.min(ratio, dim=1)

        intersection = start + ratio.values[:, None] * vec

        return intersection
    
    def get_portal_exit_location(self, start: torch.Tensor, end: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        exit_point = self.exit_point(start, end)

        # the dimension which hits the boundary will have a value of either 0 or 1, so this one will always be false
        # the other dimension will be a number between 0 and 1. if it's between our portal_min and portal_max
        # then we know the point hit a portal
        hit_portal = torch.any((exit_point < self.portal_max) & (exit_point > self.portal_min), dim=1, keepdim=True)

        # teleport the point to the other side of the domain
        new_end = torch.where(hit_portal, end % 1, end)

        return new_end, exit_point
