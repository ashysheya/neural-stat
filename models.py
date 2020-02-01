import torch.nn as nn
import torch
from utils import sample_from_normal

def get_model(opts):
    return NeuralStatistician(opts)


class NeuralStatistician(nn.Module):
    """Class that represents the whole model form the paper"""

    def __init__(self, opts):
        super(NeuralStatistician, self).__init__()
        self.statistic_network = StatisticNetwork(opts.experiment, opts.context_dim, opts.masked)
        self.context_prior = ContextPriorNetwork(opts.context_dim, opts.type_prior)
        self.inference_network = InferenceNetwork(opts.experiment, opts.num_stochastic_layers,
            opts.z_dim, opts.context_dim, opts.x_dim)
        self.latent_decoder_network = LatentDecoderNetwork(opts.experiment, 
            opts.num_stochastic_layers, opts.z_dim, opts.context_dim)
        self.observation_decoder_network = ObservationDecoderNetwork(opts.experiment, 
            opts.num_stochastic_layers, opts.z_dim, opts.context_dim, opts.x_dim)

    def forward(self, datasets):
        outputs = {'train_data': datasets}

        # get context mu, logsigma of size (batch_size, context_dim)
        context_dict = self.statistic_network(outputs)
        outputs.update(context_dict)

        # get prior mu, logsigma for context vectors
        context_prior_dict = self.context_prior(outputs)
        outputs.update(context_prior_dict)

        # get variational approximations for latent z_1, .., z_n
        latent_variables_dict = self.inference_network(outputs)
        outputs.update(latent_variables_dict)

        # get parameters for latent variables prior
        latent_variables_prior_dict = self.latent_decoder_network(outputs)
        outputs.update(latent_variables_prior_dict)

        # get generated samples from decoder network
        observation_dict = self.observation_decoder_network(outputs)
        outputs.update(observation_dict)

        return outputs


class StatisticNetwork(nn.Module):
    """Variational approximation q(c|D)."""
    def __init__(self, experiment, context_dim, masked=False):
        super(StatisticNetwork, self).__init__()
        self.experiment = experiment
        self.masked = masked
        self.context_dim = context_dim

        if experiment == 'synthetic':
            self.before_pooling = nn.Sequential(nn.Linear(1, 128),
                                                nn.ReLU(True),
                                                nn.Linear(128, 128),
                                                nn.ReLU(True),
                                                nn.Linear(128, 128),
                                                nn.ReLU(True))
            self.after_pooling = nn.Sequential(nn.Linear(128, 128),
                                               nn.ReLU(True),
                                               nn.Linear(128, 128),
                                               nn.ReLU(True),
                                               nn.Linear(128, 128),
                                               nn.ReLU(True),
                                               nn.Linear(128, context_dim*2))

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that hold training data -
        torch.FloatTensor of size (batch_size, samples_per_dataset, sample_size)
        :return dictionary of mu, logsigma and context sample for each dataset
        """
        datasets = input_dict['train_data']
        data_size = datasets.size()
        prestat_vector = self.before_pooling(datasets.view(data_size[0]*data_size[1], 
            *data_size[2:]))
        prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1).mean(dim=1)
        outputs = self.after_pooling(prestat_vector)
        samples = sample_from_normal(outputs[:, :self.context_dim],
                                     outputs[:, self.context_dim:])
        samples_expanded = samples[:, None].expand(-1, datasets.size()[1], -1).contiguous()
        samples_expanded = samples_expanded.view(-1, self.context_dim)
        return {'means_context': outputs[:, :self.context_dim],
                'logvars_context': outputs[:, self.context_dim:],
                'samples_context': samples, 
                'samples_context_expanded': samples_expanded}


class ContextPriorNetwork(nn.Module):
    """Prior for c, p(c)."""
    def __init__(self, context_dim, type_prior='standard'):
        """
        :param context_dim: int, dimension of the context vector
        :param type_prior: either neural network-based or standard gaussian
        """
        super(ContextPriorNetwork, self).__init__()
        self.context_dim = context_dim
        self.type_prior = type_prior

        # TODO: add neural-network based type for conditional variant

    def forward(self, input_dict):
        """
        :param input_dict: dict that labels and context for required for prior
        :return: dict of means and log variance for prior
        """
        if self.type_prior == 'standard':
            contexts = input_dict['samples_context']
            means, logvars = torch.zeros_like(contexts), torch.zeros_like(contexts)
            if contexts.is_cuda:
                means = means.cuda()
                logvars = logvars.cuda()
            return {'means_context_prior': means,
                    'logvars_context_prior': logvars}


class InferenceNetwork(nn.Module):
    """Variational approximation q(z_1, ..., z_L|c, x)."""
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim, x_dim):
        """
        :param num_stochastic_layers: number of stochastic layers in the model
        :param z_dim: dimension of each stochastic layer
        :param context_dim: dimension of c
        :param x_dim: dimension of x
        """
        super(InferenceNetwork, self).__init__()
        self.num_stochastic_layers = num_stochastic_layers
        self.z_dim = z_dim    # dimension of each of z_i
        self.experiment = experiment
        self.context_dim = context_dim
        self.x_dim = x_dim
        input_dim = self.context_dim + self.x_dim
        self.model = nn.ModuleList()

        if experiment == 'synthetic':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 128),
                                             nn.Linear(128, 128),
                                             nn.Linear(128, 128),
                                             nn.Linear(128, z_dim*2))]
                # The following stochastic layers also take previous stochastic layer as input
                # TODO: what about z_L? 
                input_dim = self.context_dim + self.z_dim + self.x_dim

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that has
        - context: (batch_size, context_dim) context for each dataset in batch
        - train data: (batch_size, num_datapoints_per_dataset, sample_size) batch of datasets
        :return: dictionary of lists for means, log variances and samples for each stochastic layer
        """
        context_expanded = input_dict['samples_context_expanded']
        datasets = input_dict['train_data']
        datasets_raveled = datasets.view(-1, self.x_dim)
        # Input has dimension (batch_size*num_datapoints_per_dataset, sample_size+context_dim)
        current_input = torch.cat([context_expanded, datasets_raveled], dim=1)

        outputs = {'means_latent_z': [],
                   'logvars_latent_z': [],
                   'samples_latent_z': []}

        for module in self.model:
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = current_output[:, self.z_dim:]
            samples = sample_from_normal(means, logvars)
            # p(z_i|z_{i+1},c) follows normal distribution
            outputs['means_latent_z'] += [means]
            current_input = torch.cat([context_expanded, datasets_raveled, samples], dim=1)    
            outputs['logvars_latent_z'] += [logvars]
            outputs['samples_latent_z'] += [samples]

        return outputs


class LatentDecoderNetwork(nn.Module):
    """Latent prior network p(z_1, ..., z_n|c)."""
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim):
        """
        :param num_stochastic_layers: number of stochastic layers in the model
        :param z_dim: dimension of each stochastic layer
        :param context_dim: dimension of c
        """
        super(LatentDecoderNetwork, self).__init__()
        self.num_stochastic_layers = num_stochastic_layers
        self.z_dim = z_dim
        self.experiment = experiment
        self.context_dim = context_dim
        input_dim = self.context_dim
        self.model = nn.ModuleList()

        if experiment == 'synthetic':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 128),
                                             nn.Linear(128, 128),
                                             nn.Linear(128, 128),
                                             nn.Linear(128, z_dim*2))]
                input_dim = self.context_dim + self.z_dim

    def forward(self, input_dict):
        """
        Given context and samples of z_1, ..., z_{L-1} should return a dictionary with
        parameters mu, log variance for each stochastic layer.
        :param input_dict: dictionary that has
        - context: (batch_size, context_dim) context for each dataset in batch
        - train data: (batch_size, num_datapoints_per_dataset, sample_size) batch of datasets
        :return: dictionary of lists for means, log variances and samples for each stochastic layer
        """
        context_expanded = input_dict['samples_context_expanded']
        samples_latent_z = input_dict['samples_latent_z']

        current_input = context_expanded
        outputs = {'means_latent_z_prior': [],
                   'logvars_latent_z_prior': []}

        for i, module in enumerate(self.model):
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = current_output[:, self.z_dim:]
            outputs['means_latent_z_prior'] += [means]
            outputs['logvars_latent_z_prior'] += [logvars]
            if i < len(self.model) - 1:
                current_input = torch.cat([context_expanded, samples_latent_z[i]], dim=1)
        return outputs


class ObservationDecoderNetwork(nn.Module):
    """
    Network to model p(x|c, z_1, ..., z_n).
    Network that firstly concatenates z_1, ..., z_L, c to produce mu, sigma for x.
    Returns mu_x, sigma_x
    """
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim, x_dim):
        """
        :param num_stochastic_layers: number of stochastic layers in the model
        :param z_dim: dimension of each stochastic layer
        :param context_dim: dimension of c
        :param x_dim: dimension of x
        """
        super(ObservationDecoderNetwork, self).__init__()
        self.experiment = experiment
        self.num_stochastic_layers = num_stochastic_layers
        self.z_dim = z_dim
        self.context_dim = context_dim
        self.x_dim = x_dim

        input_dim = num_stochastic_layers*z_dim + context_dim

        if experiment == 'synthetic':
            self.model = nn.Sequential(nn.Linear(input_dim, 128),
                                       nn.ReLU(True),
                                       nn.Linear(128, 128),
                                       nn.ReLU(True),
                                       nn.Linear(128, 128),
                                       nn.ReLU(True),
                                       nn.Linear(128, x_dim * 2))

    def forward(self, input_dict):
        """
        Given sampled context and sampled latent z_i, should return parameters
        of the distribution for x.
        Check here - input_dict['samples_latent_z']
         has dimensions (n_stochastic_layer, batch_size, z_dim)
        input_dict['samples_context'] has size (batch_size, context_dim)
        What size do we want our input?
        """
        context_expanded = input_dict['samples_context_expanded']
        # z samples from InferenceNetwork
        latent_z = input_dict['samples_latent_z']

        inputs = torch.cat([context_expanded] + latent_z, dim=1)

        outputs = self.model(inputs)

        # n_stochastic_layer = latent_z.size()[0]
        # z_dim = latent_z.size()[2]
        #
        # latent_z_cat = latent_z[0, :, :]
        # # If more than 1 z, concatenate all latent variables
        # for i in range(1, n_stochastic_layer):
        #     latent_z_i = latent_z[i, :, :]
        #     latent_z_cat = torch.cat([latent_z_cat, latent_z_i], dim=1)
        #
        # latent_z_raveled = latent_z_cat.view(-1, n_stochastic_layer*z_dim)
        # outputs = self.model(torch.cat([latent_z_raveled, context], dim=1))


        return {'means_x': outputs[:, :self.x_dim],
                'logvars_x': outputs[:, self.x_dim:]}
