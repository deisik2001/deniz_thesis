import torch
from torch.distributions import Normal
import torch.distributions as dist

from scvi.priors.base_prior import BasePrior


class GaussianPrior(BasePrior):
    def __init__(self, n_latent: int):
        super(GaussianPrior, self).__init__()

        self.register_buffer('mean', torch.zeros((n_latent,)))
        self.register_buffer('logvar', torch.zeros((n_latent,)))

    @property
    def distribution(self):
        distribution = Normal(self.mean, torch.exp(0.5 * self.logvar))
        distribution = dist.Independent(distribution, 1)
        return distribution

    def sample(self, n_samples: int):
        return self.distribution.sample((n_samples,))

    def log_prob(self, z):
        return self.distribution.log_prob(z)
