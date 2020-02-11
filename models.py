import torch
import torch.nn as nn
import numpy as np
from utils import sample_from_normal
import torch.nn.functional as F

def get_model(opts):
    return NeuralStatistician(opts)


class NeuralStatistician(nn.Module):
    """Class that represents the whole model from the paper"""

    def __init__(self, opts):
        super(NeuralStatistician, self).__init__()

        if opts.experiment == 'youtube' or opts.experiment == 'omniglot':
            self.shared_encoder = SharedEncoder(opts.experiment)
            opts.x_dim = 256*4*4  # Dimension of x in shared nets will be that of the encoded input h

        self.statistic_network = StatisticNetwork(opts.experiment, opts.context_dim, opts.x_dim, opts.masked)
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

        # get encoded version of input: x-->h
        if self.shared_encoder:
            encoded_dict = self.shared_encoder(outputs)
            outputs.update(encoded_dict)

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
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)


class SharedEncoder(nn.Module):
    """Shared Encoder x-->h"""
    def __init__(self, experiment):
        super(SharedEncoder, self).__init__()
        self.experiment = experiment

        if self.experiment == 'youtube':
            # Youtube shared encoder is as follows:
            # - 2x{conv2d 32 feature maps with 3x3 kernels and ELU activations}
            # - conv2d 32 feature maps with 3x3 kernels, stride 2 and ELU activations
            # - 2x{conv2d 64 feature maps with 3x3 kernels and ELU activations}
            # - conv2d 64 feature maps with 3x3 kernels, stride 2 and ELU activations
            # - 2x{conv2d 128 feature maps with 3x3 kernels and ELU activations}
            # - conv2d 128 feature maps with 3x3 kernels, stride 2 and ELU activations
            # - 2x{conv2d 256 feature maps with 3x3 kernels and ELU activations}
            # - conv2d 256 feature maps with 3x3 kernels, stride 2 and ELU activations
            # All conv2d layers use batch normalization
            # Input shape is (-1, 3, 64, 64)
            self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=32),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=32),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
                                       nn.BatchNorm2d(num_features=32),
                                       nn.ELU(inplace=True),
                                       # Shape is now (-1, 32, 32, 32)
                                       nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
                                       nn.BatchNorm2d(num_features=64),
                                       nn.ELU(inplace=True),
                                       # Shape is now (-1, 64, 16, 16)
                                       nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=128),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=128),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),
                                       nn.BatchNorm2d(num_features=128),
                                       nn.ELU(inplace=True),
                                       # Shape is now (-1, 128, 8, 8)
                                       nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ELU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                                       nn.BatchNorm2d(num_features=256),
                                       nn.ELU(inplace=True))
                                       # Shape is now (-1, 256, 4, 4)

    def forward(self, input_dict):
        datasets = input_dict['train_data']
        data_size = datasets.size()  # Should be (batch_size, 5, 3, 64, 64)

        # Pass x as (-1, 3, 64, 64), i.e. (batch_size*5, 3, 64, 64)
        h = self.model(datasets.view(data_size[0]*data_size[1], *data_size[2:]))
        # Reshape as (batch_size, num_data_per_dataset, 256, 4, 4), i.e. probably (16, 5, 256, 4, 4)
        h = h.view(data_size[0], data_size[1], 256, 4, 4).contiguous()

        return {'encoded_data': h}  # encoded input has dim (batch_size, num_data_per_dataset, 256, 4, 4)

class StatisticNetwork(nn.Module):
    """Variational approximation q(c|D)."""
    def __init__(self, experiment, context_dim, x_dim, masked=False):
        super(StatisticNetwork, self).__init__()
        self.experiment = experiment
        self.masked = masked
        self.context_dim = context_dim
        self.x_dim = x_dim

        if experiment == 'synthetic':
            self.before_pooling = nn.Sequential(nn.Linear(x_dim, 128),
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

        elif experiment == "youtube":
            # Output of the shared encoder has 256 feature maps, each 4x4: x_dim = 256*4*4=4096
            self.before_pooling = nn.Sequential(nn.Linear(x_dim, 1000),
                                                nn.ELU(inplace=True))
            # Check here: unsure about hidden layer dim and number of FC layers. Also, in paper specifies LINEAR
            # layers, but on repo used a non-linearity. So used ReLU, and same number of layers as in repo.
            self.after_pooling = nn.Sequential(nn.Linear(1000, 1000),
                                               nn.ELU(inplace=True),
                                               nn.Linear(1000, 1000),
                                               nn.ELU(inplace=True),
                                               nn.Linear(1000, context_dim*2))

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that hold training data / encoded data -
        torch.FloatTensor of size (batch_size, samples_per_dataset, sample_dim)
        ## For synthetic data this is (16, 200, 1) by default
        ## For faces data this is (16, 5, 256, 4, 4) after encoder
        :return dictionary of mu, logsigma and context sample for each dataset
        """
        # If encoded data exists, take this - else take non-transformed input data
        datasets = input_dict.get('encoded_data', 'train_data')
        data_size = datasets.size()
        # Pass all input vectors together: input to NN has size (batch_size*n_samples_per_dataset, vector_size)
        # For synthetic data the vector_size is 1, that's why the NN input is size 1 so the input has size (16*200, 1).
        # For the faces dataset, the encoded input size is (batch_size, n_samples_per_dataset, 256, 4, 4), so need to
        # unravel into (batch_size*n_samples_per_dataset, 256*4*4)
        # prestat_vector = self.before_pooling(datasets.view(data_size[0]*data_size[1], np.prod(data_size[2:])))
        prestat_vector = self.before_pooling(datasets.view(-1, self.x_dim))
        # The output prestat_vector is of size (16*200, 128) for synthetic data.
        # Calculate encoding v: this takes the size (16, 200, 128), and calculates the mean along dimension 1: i.e.
        # along the 200 samples of a given dataset. So first we need to change the prestat_vector view from
        # (16*200, 128) to (16, 200, 128). After averaging, the encoding v is (16, 1, 128).
        # Similarly for the youtube dataset, need to change view from (16*5, 1000) to (16, 5, 1000) and average over the
        # samples in the dataset (dim=1):
        prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1).mean(dim=1)
        # Output is of (16, size context_dim*2): it has the mean and logvar of the context for each batch
        outputs = self.after_pooling(prestat_vector)
        means = outputs[:, :self.context_dim]
        # Limit the logvar to reasonable values
        logvars = torch.clamp(outputs[:, self.context_dim:], -10, 10)
        # Sample the context from the mean and logvar we have just found. It has size (batch_size, c_dim)
        samples = sample_from_normal(means, logvars)
        # Expand the samples: take the samples for each batch, and copy it for each sample in the dataset. To do so,
        # one dimension (dim 1) is added to the samples, making it (16, 1, 3), and the values are copied along dim 1
        # to make it (16, 200, 3) for synthetic data --> this way, the context vector is returned for each input data!
        # For faces dataset, it will become (batch_size, 5, context_dim)
        # CHECK HERE -- how does contiguous work??
        samples_expanded = samples[:, None].expand(-1, data_size[1], -1).contiguous()
        # Finally, make samples_expanded of size (16*200, 3) for synthetic data. For faces data, it is (16*5, 3)
        samples_expanded = samples_expanded.view(-1, self.context_dim)
        return {'means_context': means,
                'logvars_context': logvars,
                'samples_context': samples, 
                'samples_context_expanded': samples_expanded}
        # Dict with dimensions means(16, 3), logvars(16, 3), samples(16, 3): 1 for each D and
        # samples_expanded(16*200, 3): 1 for each x (the same copied along each x in each D)
        # For faces data, this is the same, except that sample_size=5 instead of 200 and context_dim=500 by default

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
            # The samples_context in the input dict has size (16, 3)--> 1 context vector for each dataset in the batch
            contexts = input_dict['samples_context']
            # Prior is spherical: make prior mean zeros(16, 3) and logvar zeros(16, 3)-->var ones(16,3)
            means, logvars = torch.zeros_like(contexts), torch.zeros_like(contexts)
            if contexts.is_cuda:
                means = means.cuda()
                logvars = logvars.cuda()
            return {'means_context_prior': means,
                    'logvars_context_prior': logvars}
            # Dict with means_context_prior(16, 3) and logvars_context_prior(16, 3) for synthetic data


class InferenceNetwork(nn.Module):
    """Variational approximation q(z_1, ..., z_L|c, x)."""
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim, x_dim):
        """
        :param num_stochastic_layers: number of stochastic layers in the model
        :param z_dim: dimension of each stochastic layer
        :param context_dim: dimension of c
        :param x_dim: dimension of x (or its encoded version)
        """
        super(InferenceNetwork, self).__init__()
        self.num_stochastic_layers = num_stochastic_layers
        self.z_dim = z_dim    # dimension of each of z_i
        self.experiment = experiment
        self.context_dim = context_dim
        self.x_dim = x_dim  # For faces data, x_dim = 256*4*4

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

        elif experiment == 'youtube':
            for i in range(self.num_stochastic_layers):
                # TODO: Check here: unsure about hidden layer dim and number of FC layers.
                # In repo, instead of concatenating directly the inputs, they created different embeddings for c, h and
                # z_i with one hidden layer for each. Then they concatenate, and apply a nonlinearity. But in paper,
                # state that h and c should be concatenated and THEN use a fully-connected layer, so will use this.
                # Same as paper: FC layer with 1000 units and ReLU, and FC layers to mean and logvar
                self.model += [nn.Sequential(nn.Linear(input_dim, 1000),
                                             nn.ELU(inplace=True),
                                             nn.Linear(1000, z_dim*2))]
                # The following stochastic layers also take previous stochastic layer as input
                input_dim = self.context_dim + self.z_dim + self.x_dim

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that has
        - context: (batch_size, context_dim) context for each dataset in batch
        - train data: (batch_size, num_datapoints_per_dataset, sample_size) batch of datasets
        :return: dictionary of lists for means, log variances and samples for each stochastic layer
        """
        # samples_context_expanded have size (16*200, 3), i.e. (batch_size*num_datapoints_per_dataset, context_size)
        context_expanded = input_dict['samples_context_expanded']
        # For synthetic data, datasets has size (16, 200, 1), i.e. (batch_size, num_datapoints_per_dataset, sample_size)
        # For faces data, it is (16, 5, 256, 4, 4)
        datasets = input_dict.get('encoded_data', 'train_data')
        # Make datasets of size (16*200, 1) for synthetic, (16*5, 256*4*4) for faces
        datasets_raveled = datasets.view(-1, self.x_dim)
        # Input has dimension (batch_size*num_datapoints_per_dataset, sample_size+context_dim)
        # Operation below concatenates the (16*200, 3) context samples with the (16*200, 1) samples to make the input
        # have size (16*200, 4)
        # For faces data, this should be (16*5, 256*4*4 + 500)
        current_input = torch.cat([context_expanded, datasets_raveled], dim=1)

        outputs = {'means_latent_z': [],
                   'logvars_latent_z': [],
                   'samples_latent_z': []}

        # Iterate over each module in the model: the n_stochastic_layers different networks
        for module in self.model:
            # current_output has dim (batch_size*num_datapoints_per_dataset, z_dim*2)
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            # samples has dim (16*200, z_dim)
            samples = sample_from_normal(means, logvars)
            # p(z_i|z_{i+1},c) follows normal distribution
            # For the next input, concatenate the previous input with the sample, to get an input of size
            # (batch_size*num_datapoints_per_dataset, context_dim + x_dim + z_dim)
            current_input = torch.cat([context_expanded, datasets_raveled, samples], dim=1)
            # For each dictionary entry, add to the list the values for the current stochastic layer. Each entry added
            # has dimension (16*200, z_dim)
            outputs['means_latent_z'] += [means]
            outputs['logvars_latent_z'] += [logvars]
            outputs['samples_latent_z'] += [samples]
        # At the end, each entry in the dictionary is a list of length n_stochastic_layers, with each element in the
        # list having dimension (batch_size*num_datapoints_per_dataset, z_dim)
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

        elif experiment == 'youtube':
            for i in range(self.num_stochastic_layers):
                # TODO: Check here again for number of linear layers
                # Same as inference network: different way of doing it than repo.
                # Same as paper: FC layer with 1000 units and ReLU, and FC layers to mean and logvar
                self.model += [nn.Sequential(nn.Linear(input_dim, 1000),
                                             nn.ELU(inplace=True),
                                             nn.Linear(1000, z_dim*2))]
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
        context_expanded = input_dict['samples_context_expanded']  # Array of size (16*200, c_dim)
        samples_latent_z = input_dict['samples_latent_z']  # List of length n_stochastic layer, w/ elem. (16*200, z_dim)

        current_input = context_expanded  # Input for first stochastic layer p(z|c) is just c
        outputs = {'means_latent_z_prior': [],
                   'logvars_latent_z_prior': []}

        # Comments are same as for the inference network above
        for i, module in enumerate(self.model):
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            outputs['means_latent_z_prior'] += [means]
            outputs['logvars_latent_z_prior'] += [logvars]
            current_input = torch.cat([context_expanded, samples_latent_z[i]], dim=1)
        # At the end, each entry in the dictionary is a list of length n_stochastic_layers, with each element in the
        # list having dimension (16*200, z_dim)
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
        :param x_dim: dimension of x or its encoded version
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

        elif experiment == 'youtube':
            # Shared learnable log variance parameter (from https://github.com/conormdurkan/neural-statistician)
            self.logvar = nn.Parameter(torch.randn(1, 3, 64, 64).cuda())

            self.pre_conv = nn.Sequential(nn.Linear(input_dim, 1000),
                                          nn.ELU(inplace=True),
                                          # Used x_dim=256*4*4, but in paper seems to say 256*8*8, which wouldn't work
                                          # with the output image dimensions
                                          nn.Linear(1000, x_dim))
            self.conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                      # Dim is (-1, 256, 4, 4)
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ELU(inplace=True),
                                      # Check here: in repo, didn't specify padding for ConvTranspose2d
                                      nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                      # Dim is now (-1, 256, 8, 8)
                                      nn.BatchNorm2d(num_features=256),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=128),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=128),
                                      nn.ELU(inplace=True),
                                      nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                      # Dim is now (-1, 128, 16, 16)
                                      nn.BatchNorm2d(num_features=128),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=64),
                                      nn.ELU(inplace=True),
                                      nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
                                      # Dim is now (-1, 64, 32, 32)
                                      nn.BatchNorm2d(num_features=64),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ELU(inplace=True),
                                      nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
                                      # Dim is now (-1, 32, 64, 64)
                                      nn.BatchNorm2d(num_features=32),
                                      nn.ELU(inplace=True),
                                      nn.Conv2d(32, 3, kernel_size=1)
                                      # Dim is now (-1, 3, 64, 64)
                                      #nn.Sigmoid(True)
                                      )

    def forward(self, input_dict):
        """
        Given sampled context and sampled latent z_i, should return parameters
        of the distribution for x.
        """
        context_expanded = input_dict['samples_context_expanded']  # samples_expanded has size (16*200, 3)
        # Below: z samples from InferenceNetwork: list of length n_stochastic_layers, with each element in the
        # list having dimension (16*200, z_dim): so essentially it is [(16*200,z_dim), (16*200,z_dim), (16*200,z_dim)]
        latent_z = input_dict['samples_latent_z']
        # Make a list from the expanded context, and concatenate with the latent z, making the input of size
        # (16*200, context_dim + num_stochastic_layers*z_dim).
        inputs = torch.cat([context_expanded] + latent_z, dim=1)

        if self.experiment == 'synthetic':
            # Output has size (16*200, x_dim*2) for synthetic case
            outputs = self.model(inputs)
            x_mean = outputs[:, :self.x_dim],
            x_logvar = torch.clamp(outputs[:, self.x_dim:], -10, 10)

        elif self.experiment == 'youtube':
            pre_conv_output = self.pre_conv(inputs)
            pre_conv_output = pre_conv_output.view(-1, 256, 4, 4)
            x_mean = F.sigmoid(self.conv(pre_conv_output))  # Should be (batch_size*num_data_per_dataset, 3, 64, 64)
            x_logvar = self.logvar.expand_as(x_mean)

        return {'means_x': x_mean, 'logvars_x': x_logvar}
