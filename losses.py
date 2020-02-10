import torch.nn as nn
import torch
import numpy as np


def get_loss(opts):
    return {'KL': KLDivergence(), 'NLL': NegativeGaussianLogLikelihood()}


class KLDivergence(nn.Module):
    """KL Divergence between two normal distributions."""
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, input_dict):
        # C_D term (equation (9) in paper): KL divergence between context prior and statistic network output
        kl_value_context = self.calculate_kl(input_dict['logvars_context_prior'],
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
            kl_value_z += self.calculate_kl(logvar_prior, logvar, mu, mu_prior)

        batch_size, sample_size = input_dict['train_data'].size()[:2]

        # Expected value comes in: average KL loss across all samples (16*200 for the default synthetic dataset case)
        return (kl_value_z.sum() + kl_value_context.sum())/(batch_size*sample_size)

    @staticmethod
    def calculate_kl(logvar_prior, logvar, mu, mu_prior):
        # See calculation for diagonal multivariate Gaussian in
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
        # 1/2*(sum_i(ln(var2_i) - ln(var1_i) + 1/var2_i*(mu2_i - mu1_i)**2 + var1_i/var2_i - 1))
        # Here, this is done in different steps:
        # 1. For each i:
        #    a. Compute  1/2*(ln(var_i)- ln(var1_i))
        kl_val = 0.5 * logvar_prior - 0.5 * logvar
        #    b. Compute 1/2*(1/var2_i*(mu2_i - mu1_i)**2 + var1_i/var2_i)
        kl_val += (torch.exp(logvar) + (mu - mu_prior) ** 2) / 2 / (
            torch.exp(logvar_prior))
        #    c. Subtract 1/2
        kl_val -= 0.5
        # 2. Sum over all i
        return kl_val.sum(dim=-1)


class NegativeGaussianLogLikelihood(nn.Module):
    """Negative Gaussian log likelihood of observations."""
    def __init__(self):
        super(NegativeGaussianLogLikelihood, self).__init__()

    def forward(self, input_dict):
        batch_size, sample_size = input_dict['train_data'].size()[:2]  # 16, 200 for the default synthetic dataset case
        observations = input_dict['train_data']  # for youtube, shape is (16, 5, 3, 64, 64)
        # logvars and means are (16*200, 1) for the default synthetic dataset case: make them (16, 200, 1)
        # CHECK HERE: if x were e.g. 3-dimensional, would the result be (16, 200, 3)?
        # For youtube data, these originally have size (16*5, 3, 64, 64) - make them size (16, 5, 3, 64, 64)
        logvars = input_dict['logvars_x'].view_as(observations)
        means = input_dict['means_x'].view_as(observations)
        # Compute LL loss term R_D from equation (8) in paper.
        # LL = sum_i(-1/2*ln(2*pi*var_i) + (x_i-mu_i)**2/(2*var_i))
        log_likelihood = -0.5*logvars - 0.5 * np.log(2 * np.pi)
        log_likelihood -= (means - observations) ** 2 / 2 / (torch.exp(logvars))
        # Sum over all elements, and average over total number of samples for expectation
        return -log_likelihood.sum()/(batch_size*sample_size)
