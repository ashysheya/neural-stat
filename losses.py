import torch.nn as nn
import torch
import numpy as np
from utils import calculate_kl


def get_loss(opts):

    if opts.nll == 'gaussian':
        nll = NegativeGaussianLogLikelihood()
    else:
        nll = NegativeBernoulliLogLikelihood()

    return {'KL': KLDivergence(),
            'NLL': nll}


class KLDivergence(nn.Module):
    """KL Divergence between two normal distributions."""
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, input_dict):
        # C_D term (equation (9) in paper): KL divergence between context prior and statistic network output
        kl_value_context = calculate_kl(input_dict['logvars_context_prior'],
                                             input_dict['logvars_context'],
                                             input_dict['means_context'],
                                             input_dict['means_context_prior'])
        kl_value_z = 0

        # L_D term (equation (10) in paper): Sum of divergences between z prior and inference network output
        # The expected value is implicit, as the z are generated using the context generated from the statistic network
        for logvar_prior, logvar, mu, mu_prior in zip(input_dict['logvars_latent_z_prior'],
                                                      input_dict['logvars_latent_z'],
                                                      input_dict['means_latent_z'],
                                                      input_dict['means_latent_z_prior']):
            kl_value_z += calculate_kl(logvar_prior, logvar, mu, mu_prior)

        batch_size, sample_size = input_dict['train_data'].size()[:2]

        # Expected value comes in: average KL loss across all samples (16*200 for the default synthetic dataset case)
        return (kl_value_z.sum() + kl_value_context.sum())/(batch_size*sample_size)


    # added in utils function, can be deleted
    # @staticmethod
    # def calculate_kl(logvar_prior, logvar, mu, mu_prior):
    #     kl_val = 0.5 * logvar_prior - 0.5 * logvar
    #     kl_val += (torch.exp(logvar) + (mu - mu_prior) ** 2) / 2 / torch.exp(logvar_prior)
    #     kl_val -= 0.5
    #     return kl_val.sum(dim=-1)


class NegativeGaussianLogLikelihood(nn.Module):
    """Negative Gaussian log likelihood of observations."""
    def __init__(self):
        super(NegativeGaussianLogLikelihood, self).__init__()

    def forward(self, input_dict):
        batch_size, sample_size = input_dict['train_data'].size()[:2]
        observations = input_dict['train_data']
        logvars = input_dict['logvars_x'].view_as(observations)
        means = input_dict['means_x'].view_as(observations)
        # Compute LL loss term R_D from equation (8) in paper.
        # LL = sum_i(-1/2*ln(2*pi*var_i) + (x_i-mu_i)**2/(2*var_i))
        log_likelihood = -0.5*logvars - 0.5 * np.log(2 * np.pi)
        log_likelihood -= (means - observations) ** 2 / 2 / torch.exp(logvars)
        return -log_likelihood.sum()/(batch_size*sample_size)


class NegativeBernoulliLogLikelihood(nn.Module):
    """Negative Bernoulli log likelihood of observations."""
    def __init__(self):
        super(NegativeBernoulliLogLikelihood, self).__init__()

    def forward(self, input_dict):
        batch_size, sample_size = input_dict['train_data'].size()[:2]
        observations = input_dict['train_data']
        probabilities = input_dict['proba_x'].view_as(observations)
        log_likelihood = observations*torch.log(probabilities)
        log_likelihood += (1 - observations)*torch.log(1 - probabilities)
        return -log_likelihood.sum()/(batch_size*sample_size)
