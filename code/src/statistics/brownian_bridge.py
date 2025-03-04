import torch

def prob_to_hit(new, barriers, time_step):
    '''
    new: the new position of the Brownian walkers centered around the old position (n_walkers, n_dims)
    barriers: the position of the barriers centered around the old position (n_dims, 2)
    '''

    prob = torch.exp((2 * barriers * (new - barriers)) / time_step)
    return torch.where(prob > 1, 1., prob)

def expected_time_to_hit(new, barriers, time_step):
    '''
    new: the new position of the Brownian walkers centered around the old position (n_walkers, n_dims)
    barriers: the position
    '''

    distribution = 1 - torch.special.ndtr((torch.abs(barriers) + torch.abs(new - barriers)) / torch.sqrt(time_step))
    normal_dist = torch.distributions.Normal(loc=0.0, scale=torch.sqrt(time_step))
    density = torch.exp(normal_dist.log_prob(new))

    return torch.abs(barriers) * distribution / density

def prob_to_hit_late(new, barriers, time_step):
    expected_time = expected_time_to_hit(new, barriers, time_step)
    return expected_time / time_step
