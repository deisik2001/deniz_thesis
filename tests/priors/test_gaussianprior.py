import pytest
import torch

from scvi.priors.gaussianprior import GaussianPrior


@pytest.fixture
def prior():
    return GaussianPrior(n_latent=5)



def test_sample(prior):
    assert prior.sample(10).shape == (10, 5)

def test_log_prob(prior):
    z = torch.zeros((1, 5))
    assert prior.log_prob(z) == torch.tensor()
