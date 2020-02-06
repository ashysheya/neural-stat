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

        self.apply(self.init_weights)

    def forward(self, datasets):
        outputs = {'train_data': datasets}

        # get context mu, logsigma of size (batch_size, context_dim): q(c|D)
        context_dict = self.statistic_network(outputs)
        outputs.update(context_dict)

        # get prior mu, logsigma for context vectors: p(c)
        context_prior_dict = self.context_prior(outputs)
        outputs.update(context_prior_dict)

        # get variational approximations for latent z_1, .., z_L: q(z_1, .., z_L|c, x)
        latent_variables_dict = self.inference_network(outputs)
        outputs.update(latent_variables_dict)

        # get parameters for latent variables prior: p(z_1, .., z_L|c)
        latent_variables_prior_dict = self.latent_decoder_network(outputs)
        outputs.update(latent_variables_prior_dict)

        # get generated samples from decoder network: p(x|z_1, .., z_L, c)
        observation_dict = self.observation_decoder_network(outputs)
        outputs.update(observation_dict)

        return outputs

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)


class StatisticNetwork(nn.Module):
    """Variational approximation q(c|D)."""
    def __init__(self, experiment, context_dim, masked=False):
        super(StatisticNetwork, self).__init__()
        self.experiment = experiment
        self.masked = masked
        self.context_dim = context_dim

        if experiment == 'synthetic':
            ## CHECK HERE
            ## The 1 is for 1 dataset??
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
        ## For synthetic data this is (16, 200, 1) by default
        :return dictionary of mu, logsigma and context sample for each dataset
        """
        datasets = input_dict['train_data']
        data_size = datasets.size()
        ## Pass all input vectors together: input to NN has size (batch_size*samples_per_dataset, vector_size)
        ## Here the vector_size is 1, that's why the NN input is size 1 so the input has size (16*200, 1).
        ## But if we put larger vectors (e.g. an image), then this will need to be increased - to 28*28 for example.
        prestat_vector = self.before_pooling(datasets.view(data_size[0]*data_size[1],
            *data_size[2:]))
        ## The output prestat_vector is of size (16*200, 128).
        ## Calculate encoding v: this takes the size (16, 200, 128), and calculates the mean along dimension 1: i.e.
        ## along the 200 samples of a given dataset. So first we need to change the prestat_vector view from
        ## (16*200, 128) to (16, 200, 128). After averaging, the encoding v is (16, 1, 128)
        prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1).mean(dim=1)
        ## Output is of (16, size context_dim*2): it has the mean and logvar of the context for each batch
        outputs = self.after_pooling(prestat_vector)
        means = outputs[:, :self.context_dim]
        ## Limit the logvar to reasonable values
        logvars = torch.clamp(outputs[:, self.context_dim:], -10, 10)
        ## Sample the context from the mean and logvar we have just found. It has size (batch_size, c_dim)
        samples = sample_from_normal(means, logvars)
        ## Expand the samples: take the samples for each batch, and copy it for each sample in the dataset. To do so,
        ## one dimension (dim 1) is added to the samples, making it (16, 1, 3), and the values are copied along dim 1
        ## to make it (16, 200, 3) --> this way, the context vector is returned for each input data!
        ## CHECK HERE -- how does contiguous work??
        samples_expanded = samples[:, None].expand(-1, datasets.size()[1], -1).contiguous()
        ## Finally, make samples_expanded of size (16*200, 3)
        samples_expanded = samples_expanded.view(-1, self.context_dim)
        return {'means_context': means,
                'logvars_context': logvars,
                'samples_context': samples, 
                'samples_context_expanded': samples_expanded}
        ## Dict with dimensions means(16, 3), logvars(16, 3), samples(16, 3)-->1 for each D and
        ## samples_expanded(16*200, 3)-->1 for each x (the same copied along each x in each D)


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
            ## The samples_context in the input dict has size (16, 3)--> 1 context vector for each dataset in the batch
            contexts = input_dict['samples_context']
            ## Prior is spherical: make prior mean zeros(16, 3) and logvar zeros(16, 3)-->var ones(16,3)
            means, logvars = torch.zeros_like(contexts), torch.zeros_like(contexts)
            if contexts.is_cuda:
                means = means.cuda()
                logvars = logvars.cuda()
            return {'means_context_prior': means,
                    'logvars_context_prior': logvars}
            ## Dict with means_context_prior(16, 3) and logvars_context_prior(16, 3)


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
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
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
        ## samples_context_expanded have size (16*200, 3), i.e. (batch_size, num_datapoints_per_dataset, context_size)
        context_expanded = input_dict['samples_context_expanded']
        ## datasets has size (16, 200, 1), i.e. (batch_size, num_datapoints_per_dataset, sample_size)
        datasets = input_dict['train_data']
        ## Make datasets of size (16*200, 1)
        datasets_raveled = datasets.view(-1, self.x_dim)
        # Input has dimension (batch_size*num_datapoints_per_dataset, sample_size+context_dim)
        ## Operation below concatenates the (16*200, 3) context samples with the (16*200, 1) samples to make the input
        ## have size (16*200, 4)
        current_input = torch.cat([context_expanded, datasets_raveled], dim=1)

        outputs = {'means_latent_z': [],
                   'logvars_latent_z': [],
                   'samples_latent_z': []}

        ## Iterate over each module in the model: the n_stochastic_layers different networks
        for module in self.model:
            ## current_output has dim (16*200, z_dim*2)
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            ## samples has dim (16*200, z_dim)
            samples = sample_from_normal(means, logvars)
            # p(z_i|z_{i+1},c) follows normal distribution
            ## For the next input, concatenate the previous input with the sample, to get an input of size
            ## (16*200, context_dim+x_dim+z_dim)
            current_input = torch.cat([context_expanded, datasets_raveled, samples], dim=1)
            ## For each dictionary entry, add to the list the values for the current stochastic layer. Each entry added
            ## has dimension (16*200, z_dim)
            outputs['means_latent_z'] += [means]
            outputs['logvars_latent_z'] += [logvars]
            outputs['samples_latent_z'] += [samples]
        ## At the end, each entry in the dictionary is a list of length n_stochastic_layers, with each element in the
        ## list having dimension (16*200, z_dim)
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
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
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
        context_expanded = input_dict['samples_context_expanded'] ## Array of size (16*200, c_dim)
        samples_latent_z = input_dict['samples_latent_z'] ## List of length n_stochastic layer, w/ elem. (16*200, z_dim)

        current_input = context_expanded ## Input for first stochastic layer p(z|c) is just c
        outputs = {'means_latent_z_prior': [],
                   'logvars_latent_z_prior': []}

        ## Comments are same as for the inference network above
        for i, module in enumerate(self.model):
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            outputs['means_latent_z_prior'] += [means]
            outputs['logvars_latent_z_prior'] += [logvars]
            current_input = torch.cat([context_expanded, samples_latent_z[i]], dim=1)
        ## At the end, each entry in the dictionary is a list of length n_stochastic_layers, with each element in the
        ## list having dimension (16*200, z_dim)
        return outputs


class ObservationDecoderNetwork(nn.Module):
    """
    Network to model p(x|c, z_1, ..., z_L).
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
        context_expanded = input_dict['samples_context_expanded'] ## samples_expanded has size (16*200, 3)
        ## z samples from InferenceNetwork: list of length n_stochastic_layers, with each element in the
        ## list having dimension (16*200, z_dim): so essentially it is [(16*200,z_dim), (16*200,z_dim), (16*200,z_dim)]
        latent_z = input_dict['samples_latent_z']
        ## Make a list from the expanded context, and concatenate with the latent z, making the input of size
        ## (16*200, context_dim + num_stochastic_layers*z_dim).
        inputs = torch.cat([context_expanded] + latent_z, dim=1)

        ## Output has size (16*200, x_dim*2)
        outputs = self.model(inputs)

        return {'means_x': outputs[:, :self.x_dim],
                'logvars_x': torch.clamp(outputs[:, self.x_dim:], -10, 10)}
