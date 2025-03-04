import torch

class GaussianIncrements():
    '''
    generates independant Gaussian noise with specified time step
    '''
    def __init__(self, time_step: torch.Tensor, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.time_step = time_step
        self.scale = torch.sqrt(time_step)

    def get_noise(self, points):
        return self.scale * torch.randn(size=points.shape, device=self.device)
