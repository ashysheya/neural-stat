import numpy as np
import torch

# def preprocess_distribution_parameters(distribution, means, variances):
#     """
#     Auxiliary function that return parameters for numpy
#     functions given mean and variance for the distribution.
#     :param distribution: string, the distribution for which we want to construct dict of parameters
#     :param means: 1d np.array, mean for the distribution
#     :param variances: 1d np.array, variances for the distribution
#     :return: dict, python dictionary with parameters for distribution
#     """
#     if len(means) != len(variances):
#         raise  ValueError("means and variances np.arrays should be of the same size")
#     if distribution == "gaussian":
#         return {"loc": means, "scale": np.sqrt(variances)}
#     elif distribution == "exponential":
#         return {"scale": 1/np.sqrt(variances)}
#     elif distribution == "uniform":
#         return {"low": means - np.sqrt(3*variances),
#                 "high": means + np.sqrt(3*variances)}
#     elif distribution == "laplace":
#         return {"loc": means, "scale": np.sqrt(variances/2)}
#     else:
#         raise ValueError("The distribution entered is not implemented")

# def generate_1d_datasets(num_datasets_per_distr=2500, num_data_per_dataset=200):
#     """
#     Generates 1d datasets from four different distributions for 1d synthetic
#     experiment from the paper.
#     :param num_datasets_per_distr: number of datasets per each distribution
#     :param num_data_per_dataset: number of points in each dataset
#     :return: np.array, generated dataset of size (4*num_datasets_per_distr, num_data_per_dataset
#              list of size 4*num_datasets_per_distr, list of distribution name for each dataset
#     """
#     distributions = [("exponential", np.random.exponential),
#                      ("gaussian", np.random.normal),
#                      ("uniform", np.random.uniform),
#                      ("laplace", np.random.laplace)]

#     means = np.random.uniform(-1, 1, size=num_datasets_per_distr*4)
#     variances = np.random.uniform(0.5, 2, size=num_datasets_per_distr*4)

#     dataset = np.zeros((num_datasets_per_distr*4, num_data_per_dataset))
#     target_distributions = []

#     for i in range(4):
#         distribution = distributions[i]
#         params_dict = preprocess_distribution_parameters(distribution[0],
#             means[num_datasets_per_distr*i: num_datasets_per_distr*(i + 1)],
#             variances[num_datasets_per_distr*i: num_datasets_per_distr*(i + 1)])
#         dataset[num_datasets_per_distr*i: num_datasets_per_distr*(i + 1)] = distribution[1](
#             size=(num_data_per_dataset, num_datasets_per_distr), **params_dict).T

#         target_distributions += [distribution[0]]*num_datasets_per_distr

#     return dataset, target_distributions


def sample_from_normal(mean, logvar):
    """
    Functions that implements reparameterisation trick given mu and log_var
    :param mean: (batch_size, z_dim): means for latent variables
    :param logvar: (batch_size, z_dim): variance for latent variables
    :return: (batch_size, z_dim): samples from the following distributions using rep trick
    """
    noise = torch.FloatTensor(logvar.size()).normal_()
    if logvar.is_cuda:
        noise = noise.cuda()
    return mean + torch.exp(0.5*logvar)*noise

def generate_distribution(distribution):
    m = np.random.uniform(-1, 1)
    v = np.random.uniform(0.5, 2)

    if distribution == 'gaussian':
        samples = np.random.normal(m, v, (200, 1))
        return samples, m, v

    elif distribution == 'exponential':
        samples = np.random.exponential(1, (200, 1))
        return augment_distribution(samples, m, v), m, v

    elif distribution == 'laplace':
        samples = np.random.laplace(m, v, (200, 1))
        return samples, m, v

    elif distribution == 'uniform':
        samples = np.random.uniform(-1, 1, (200, 1))
        return augment_distribution(samples, m, v), m, v


def augment_distribution(samples, m, v):
        aug_samples = samples.copy()
        aug_samples -= np.mean(samples)
        aug_samples /= np.std(samples)
        aug_samples *= v ** 0.5
        aug_samples += m
        return aug_samples


def generate_1d_datasets(num_datasets_per_distr=2500, num_data_per_dataset=200):
    sets = np.zeros((num_datasets_per_distr*4, num_data_per_dataset, 1),
                    dtype=np.float32)
    labels = []
    means = []
    variances = []

    distributions = ['gaussian', 'exponential', 'uniform', 'laplace']

    for i in range(num_datasets_per_distr*4):
        distribution = np.random.choice(distributions)

        x, m, v = generate_distribution(distribution)

        sets[i, :, :] = x
        labels.append(distribution)
        means.append(m)
        variances.append(v)

    return sets, np.array(labels), np.array(means), np.array(variances)
