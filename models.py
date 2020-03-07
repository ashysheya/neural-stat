import torch.nn as nn
import torch
import math
from utils import sample_from_normal
import torch.nn.functional as F

def get_model(opts):
    return NeuralStatistician(opts)

# for summarising points in the mnist experiment
def get_stats(opts):
    return StatisticNetwork(opts.experiment, opts.context_dim, opts.masked)


class NeuralStatistician(nn.Module):
    """Class that represents the whole model from the paper"""

    def __init__(self, opts):
        super(NeuralStatistician, self).__init__()
        self.shared_encoder = SharedEncoder(opts.experiment, opts.n_channels)
        self.statistic_network = StatisticNetwork(opts.experiment, opts.context_dim,
            opts.h_dim, opts.masked, opts.use_labels, opts.num_labels)
        self.context_prior = ContextPriorNetwork(opts.context_dim, opts.use_labels, opts.num_labels)
        self.inference_network = InferenceNetwork(opts.experiment, opts.num_stochastic_layers,
            opts.z_dim, opts.context_dim, opts.h_dim)
        self.latent_decoder_network = LatentDecoderNetwork(opts.experiment, 
            opts.num_stochastic_layers, opts.z_dim, opts.context_dim)
        self.observation_decoder_network = ObservationDecoderNetwork(opts.experiment, 
            opts.num_stochastic_layers, opts.z_dim, opts.context_dim, 
            opts.h_dim, opts.x_dim, opts.n_channels, opts.use_labels, opts.num_labels)

        self.apply(self.init_weights)

    def forward(self, datasets, train=True):
        outputs = {'train_data': datasets['datasets'], 'train': train}
        if 'labels' in datasets:
            outputs['labels'] = datasets['labels']

        # get encoded version of input: x-->h
        shared_encoder_dict = self.shared_encoder(outputs)
        outputs.update(shared_encoder_dict)

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

    def sample_conditional(self, datasets, num_samples_per_dataset, labels=None):
        outputs = {'train_data': datasets, 'train': False, 'labels': labels}

        shared_encoder_dict = self.shared_encoder(outputs)
        outputs.update(shared_encoder_dict)

        context_dict = self.statistic_network.sample(outputs)
        outputs.update(context_dict)

        # get parameters for latent variables prior: p(z_1, .., z_L|c)
        latent_variables_dict = self.latent_decoder_network.sample(outputs,
            num_samples_per_dataset)
        outputs.update(latent_variables_dict)

        # get generated samples from decoder network: p(x|z_1, .., z_L, c)
        observation_dict = self.observation_decoder_network(outputs)
        outputs.update(observation_dict)

        return outputs

    def sample(self, num_samples_per_dataset, num_datasets, labels=None):
        outputs = {'train': False, 'labels': labels}

        context_prior_dict = self.context_prior.sample(num_datasets, labels=labels)
        outputs.update(context_prior_dict)

        latent_variables_dict = self.latent_decoder_network.sample(outputs,
            num_samples_per_dataset)
        outputs.update(latent_variables_dict)

        observation_dict = self.observation_decoder_network(outputs)
        outputs.update(observation_dict)

        return outputs

    def context_params(self, datasets, labels=None):
        outputs = {'train_data': datasets, 'train': False, 'labels': labels}

        shared_encoder_dict = self.shared_encoder(outputs)
        outputs.update(shared_encoder_dict)

        # get context mu, logsigma of size (batch_size, context_dim): q(c|D)
        context_dict = self.statistic_network(outputs)
        outputs.update(context_dict)

        return outputs


    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias.data, 0)


class SharedEncoder(nn.Module):
    """Shared encoder for encoding x -> h."""
    def __init__(self, experiment, n_channels):
        super(SharedEncoder, self).__init__()
        self.experiment = experiment
        self.n_channels = n_channels  # Number of channels (RGB:3, grayscale:1)

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
            self.model = nn.Sequential(nn.Conv2d(self.n_channels, 32, kernel_size=3, padding=1, stride=1),
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

        elif self.experiment == 'omniglot':
            in_channels = 1
            module_list = []
            for i in [64, 128, 256]:
                module_list += [
                    nn.Conv2d(in_channels, i, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True),
                    nn.Conv2d(i, i, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True),
                    nn.Conv2d(i, i, kernel_size=3, padding=1, stride=2),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True)]
                in_channels = i

            self.model = nn.Sequential(*module_list)

    def forward(self, input_dict):
        if self.experiment == 'synthetic' or self.experiment == 'mnist':
            return {'train_data_encoded': input_dict['train_data']}

        elif self.experiment == 'omniglot' or self.experiment == 'youtube':
            datasets = input_dict['train_data']
            data_size = datasets.size()
            encoded = self.model(datasets.view(data_size[0] * data_size[1], *data_size[2:]))
            encoded = encoded.view(data_size[0], data_size[1], -1).contiguous()
            return {'train_data_encoded': encoded}


class StatisticNetwork(nn.Module):
    """Variational approximation q(c|D)."""
    def __init__(self, experiment, context_dim, h_dim, masked=False, use_labels=False, num_labels=7):
        super(StatisticNetwork, self).__init__()
        self.experiment = experiment
        self.masked = masked
        self.h_dim = h_dim
        self.context_dim = context_dim
        self.use_labels = use_labels
        self.num_labels = num_labels

        if experiment == 'synthetic':
            self.before_pooling = nn.Sequential(nn.Linear(h_dim, 128),
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

        elif experiment == 'youtube':
            self.before_pooling = nn.Sequential(nn.Linear(h_dim, 1000),
                                                nn.ELU(inplace=True))
            self.after_pooling = nn.Sequential(nn.Linear(1000 + num_labels if use_labels else 1000, 1000),
                                               nn.ELU(inplace=True),
                                               nn.Linear(1000, 1000),
                                               nn.ELU(inplace=True),
                                               nn.Linear(1000, context_dim*2))

        elif experiment == 'mnist':
          # in mnist experiment h_dim = 2
          # CHECK HERE: it doesn't work when use h_dim instead of 2
          self.before_pooling = nn.Sequential(nn.Linear(2, 256),
                                              nn.ReLU(True),
                                              nn.Linear(256, 256),
                                              nn.ReLU(True),
                                              nn.Linear(256, 256),
                                              nn.ReLU(True))
          self.after_pooling = nn.Sequential(nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, context_dim*2))


        elif experiment == 'omniglot':
            self.before_pooling = nn.Sequential(nn.Linear(h_dim, 256),
                                                nn.ELU(inplace=True))
            self.after_pooling = nn.Sequential(nn.Linear(256 + int(masked), 256),
                                               nn.ELU(inplace=True),
                                               nn.Linear(256, 256),
                                               nn.ELU(inplace=True),
                                               nn.Linear(256, context_dim * 2))

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that hold training data / encoded data -
        torch.FloatTensor of size (batch_size, samples_per_dataset, sample_dim)
        :return dictionary of mu, logsigma and context sample for each dataset
        """

        # If encoded data exists, take this - else take non-transformed input data
        datasets = input_dict['train_data_encoded']
        data_size = datasets.size()
        #print(data_size)
        prestat_vector = self.before_pooling(datasets.view(data_size[0]*data_size[1], 
            *data_size[2:]))

        # CHECK HERE： not sure what mask is doing
        if self.experiment == 'mnist':
            prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1).mean(dim=1)
        else:
            if not self.masked:
                prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1).mean(dim=1)
            else:
                mask_first = torch.ones((data_size[0], 1, 1)).cuda()
                if data_size[1] - 1 > 0:
                    p = 0.8 if input_dict['train'] else 1.0
                    mask = torch.bernoulli(p*torch.ones((data_size[0], data_size[1] - 1, 1))).cuda()
                    mask = torch.cat([mask_first, mask], 1)
                else:
                    mask = mask_first

                prestat_vector = prestat_vector.view(data_size[0], data_size[1], -1)
                prestat_vector = prestat_vector*mask.expand_as(prestat_vector)

                extra_feature = torch.sum(mask, 1)
                prestat_vector = torch.sum(prestat_vector, 1)
                prestat_vector /= extra_feature.expand_as(prestat_vector)

                prestat_vector = torch.cat([prestat_vector, extra_feature], 1)

        if self.use_labels:
            prestat_vector = torch.cat([prestat_vector, input_dict['labels']], dim=1)

        outputs = self.after_pooling(prestat_vector)
        means = outputs[:, :self.context_dim]
        logvars = torch.clamp(outputs[:, self.context_dim:], -10, 10)
        samples = sample_from_normal(means, logvars)
        samples_expanded = samples[:, None].expand(-1, data_size[1], -1).contiguous()
        samples_expanded = samples_expanded.view(-1, self.context_dim)

        output_dict = {'means_context': means,
            'logvars_context': logvars,
            'samples_context': samples, 
            'samples_context_expanded': samples_expanded}

        if self.use_labels:
            labels_expanded = input_dict['labels'][:, None].expand(-1, data_size[1], -1).contiguous()
            labels_expanded = labels_expanded.view(-1, self.num_labels)
            output_dict['labels_expanded'] = labels_expanded

        return output_dict

    def sample(self, input_dict):
        output_dict = self.forward(input_dict)
        return {'samples_context': output_dict['means_context']}


class ContextPriorNetwork(nn.Module):
    """Prior for c, p(c)."""
    def __init__(self, context_dim, use_labels=False, num_labels=7):
        """
        :param context_dim: int, dimension of the context vector
        :param type_prior: either neural network-based or standard gaussian
        """
        super(ContextPriorNetwork, self).__init__()
        self.context_dim = context_dim
        self.use_labels = use_labels
        self.num_labels = num_labels

        if self.use_labels:
            self.prior_network = nn.Sequential(nn.Linear(num_labels, 128),
                nn.ReLU(True),
                nn.Linear(128, 128),
                nn.ReLU(True),
                nn.Linear(128, 2*context_dim))

        # TODO: add neural-network based type for conditional variant

    def forward(self, input_dict):
        """
        :param input_dict: dict that labels and context for required for prior
        :return: dict of means and log variance for prior
        """
        if not self.use_labels:
            contexts = input_dict['samples_context']
            means, logvars = torch.zeros_like(contexts), torch.zeros_like(contexts)
            if contexts.is_cuda:
                means = means.cuda()
                logvars = logvars.cuda()

        else:
            labels = input_dict['labels']
            outputs = self.prior_network(labels)
            means = outputs[:, :self.context_dim]
            logvars = torch.clamp(outputs[:, self.context_dim:], -10, 10)

        return {'means_context_prior': means,
                'logvars_context_prior': logvars}


    def sample(self, num_datasets, labels=None):
        if not self.use_labels:
            means = torch.zeros((num_datasets, self.context_dim)).cuda()
            logvars = torch.zeros_like(means).cuda()
        else:
            outputs = self.prior_network(labels)
            means = outputs[:, :self.context_dim]
            logvars = torch.clamp(outputs[:, self.context_dim:], -10, 10)
        return {'samples_context': sample_from_normal(means, logvars)}

class InferenceNetwork(nn.Module):
    """Variational approximation q(z_1, ..., z_L|c, x)."""
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim, h_dim):
        """
        :param num_stochastic_layers: number of stochastic layers in the model
        :param z_dim: dimension of each stochastic layer
        :param context_dim: dimension of c
        :param h_dim: dimension of (encoded) input
        """
        super(InferenceNetwork, self).__init__()
        self.num_stochastic_layers = num_stochastic_layers
        self.z_dim = z_dim    # dimension of each of z_i
        self.experiment = experiment
        self.context_dim = context_dim
        self.h_dim = h_dim
        input_dim = self.context_dim + self.h_dim
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
                input_dim = self.context_dim + self.z_dim + self.h_dim

        elif experiment == 'youtube':
            for i in range(self.num_stochastic_layers):
                # In repo, instead of concatenating directly the inputs, they created different embeddings for c, h and
                # z_i with one hidden layer for each. Then they concatenate, and apply a nonlinearity. But in paper,
                # state that h and c should be concatenated and THEN use a fully-connected layer, so will use this.
                # Do same as paper: FC layer with 1000 units and ReLU, and FC layers to mean and logvar
                self.model += [nn.Sequential(nn.Linear(input_dim, 1000),
                                             nn.ELU(inplace=True),
                                             nn.Linear(1000, z_dim*2))]
                input_dim = self.context_dim + self.z_dim + self.h_dim

        elif experiment == 'mnist':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, z_dim*2))]
                input_dim = self.context_dim + self.z_dim + self.h_dim

        elif experiment == 'omniglot':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, z_dim*2))]
                input_dim = self.context_dim + self.z_dim + self.h_dim

    def forward(self, input_dict):
        """
        :param input_dict: dictionary that has
        - context: (batch_size, context_dim) context for each dataset in batch
        - train data: (batch_size, num_datapoints_per_dataset, sample_size) batch of datasets
        :return: dictionary of lists for means, log variances and samples for each stochastic layer
        """
        context_expanded = input_dict['samples_context_expanded']
        datasets = input_dict['train_data_encoded']
        datasets_raveled = datasets.view(-1, self.h_dim)
        current_input = torch.cat([context_expanded, datasets_raveled], dim=1)

        outputs = {'means_latent_z': [],
                   'logvars_latent_z': [],
                   'samples_latent_z': []}

        # Iterate over each module in the model: the n_stochastic_layers different networks
        for module in self.model:
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            samples = sample_from_normal(means, logvars)

            current_input = torch.cat([context_expanded, datasets_raveled, samples], dim=1)

            outputs['means_latent_z'] += [means]
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
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
                                             nn.Linear(128, 128),
                                             nn.ReLU(True),
                                             nn.Linear(128, z_dim*2))]
                input_dim = self.context_dim + self.z_dim

        elif experiment == 'youtube':
            for i in range(self.num_stochastic_layers):
                # Same as inference network: different way of doing it than repo.
                # Same as paper: FC layer with 1000 units and ReLU, and FC layers to mean and logvar
                self.model += [nn.Sequential(nn.Linear(input_dim, 1000),
                                             nn.ELU(inplace=True),
                                             nn.Linear(1000, z_dim*2))]
                input_dim = self.context_dim + self.z_dim

        elif experiment == 'mnist':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, 256),
                                             nn.ReLU(True),
                                             nn.Linear(256, z_dim*2))]
                input_dim = self.context_dim + self.z_dim

        if experiment == 'omniglot':
            for i in range(self.num_stochastic_layers):
                self.model += [nn.Sequential(nn.Linear(input_dim, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, 256),
                                             nn.ELU(inplace=True),
                                             nn.Linear(256, z_dim * 2))]
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

        current_input = context_expanded  # Input for first stochastic layer p(z|c) is just c
        outputs = {'means_latent_z_prior': [],
                   'logvars_latent_z_prior': []}

        for i, module in enumerate(self.model):
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            outputs['means_latent_z_prior'] += [means]
            outputs['logvars_latent_z_prior'] += [logvars]
            current_input = torch.cat([context_expanded, samples_latent_z[i]], dim=1)

        return outputs

    def sample(self, input_dict, num_samples_per_dataset):
        context = input_dict['samples_context']
        context_expanded = context[:, None].expand(-1, num_samples_per_dataset, -1).contiguous()
        context_expanded = context_expanded.view(-1, self.context_dim)

        current_input = context_expanded

        outputs = {'samples_latent_z': [], 
                   'samples_context_expanded': context_expanded}

        if input_dict['labels'] is not None:
            labels = input_dict['labels']
            labels_expanded = labels[:, None].expand(-1, num_samples_per_dataset, -1).contiguous()
            labels_expanded = labels_expanded.view(-1, input_dict['labels'].size()[1])
            outputs['labels_expanded'] = labels_expanded

        for i, module in enumerate(self.model):
            current_output = module.forward(current_input)
            means = current_output[:, :self.z_dim]
            logvars = torch.clamp(current_output[:, self.z_dim:], -10, 10)
            samples = sample_from_normal(means, logvars)

            current_input = torch.cat([context_expanded, samples], dim=1)
            outputs['samples_latent_z'] += [samples]

        return outputs


class ClampLayer(nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max
        self.kwargs = {}
        if min is not None:
            self.kwargs['min'] = min
        if max is not None:
            self.kwargs['max'] = max

    def forward(self, input):
        return torch.clamp(input, **self.kwargs)


class ReshapeLayer(nn.Module):
    def __init__(self, num_channels=256):
        super(ReshapeLayer, self).__init__()
        self.num_channels = num_channels

    def forward(self, input):
        spacial_dim = int(math.sqrt(input.size()[-1]/self.num_channels))
        return input.view(-1, self.num_channels, spacial_dim, spacial_dim)


class ObservationDecoderNetwork(nn.Module):
    """
    Network to model p(x|c, z_1, ..., z_L).
    Network that firstly concatenates z_1, ..., z_L, c to produce mu, sigma for x.
    Returns mu_x, sigma_x
    """
    def __init__(self, experiment, num_stochastic_layers, z_dim, context_dim, h_dim, x_dim, n_channels,
     use_labels=False, num_labels=7):
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
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.n_channels = n_channels
        self.use_labels = use_labels
        self.num_labels = num_labels

        input_dim = num_stochastic_layers*z_dim + context_dim

        if self.use_labels:
            input_dim += self.num_labels

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
            #self.logvar = nn.Parameter(torch.randn(1, self.n_channels, 64, 64).cuda())
            self.logvar = nn.Parameter(torch.randn(1, self.n_channels, 64, 64))

            self.pre_conv = nn.Sequential(nn.Linear(input_dim, 1000),
                                          nn.ELU(inplace=True),
                                          # Used h_dim=256*4*4, but in paper seems to say 256*8*8, which wouldn't work
                                          # with the output image dimensions
                                          nn.Linear(1000, h_dim))

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
                                      nn.Conv2d(32, self.n_channels, kernel_size=1),
                                      # Dim is now (-1, n_channels, 64, 64)
                                      ClampLayer(-10, 10)
                                      )

        elif experiment == 'mnist':
            self.model = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.ReLU(True),
                                       nn.Linear(256, 256),
                                       nn.ReLU(True),
                                       nn.Linear(256, 256),
                                       nn.ReLU(True),
                                       nn.Linear(256, x_dim*2))


        if experiment == 'omniglot':

            module_list = [nn.Linear(input_dim, 4*4*256),
                           nn.ELU(inplace=True),
                           ReshapeLayer(256)]

            in_channels = 256

            for i in [256, 128, 64]:
                module_list += [
                    nn.Conv2d(in_channels, i, kernel_size=3, padding=1, stride=1),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True),
                    nn.Conv2d(i, i, kernel_size=3, padding=0 if i == 64 else 1, stride=1),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True),
                    nn.ConvTranspose2d(i, i, kernel_size=2, stride=2),
                    nn.BatchNorm2d(num_features=i),
                    nn.ELU(inplace=True)]
                in_channels = i

            module_list += [nn.Conv2d(in_channels, 1, kernel_size=1),
                            ClampLayer(-10, 10),
                            nn.Sigmoid()]

            self.model = nn.Sequential(*module_list)

    def forward(self, input_dict):
        """
        Given sampled context and sampled latent z_i, should return parameters
        of the distribution for x.
        """
        context_expanded = input_dict['samples_context_expanded']
        if self.use_labels:
            context_expanded = torch.cat([context_expanded, input_dict['labels_expanded']], dim=1)

        latent_z = input_dict['samples_latent_z']
        inputs = torch.cat([context_expanded] + latent_z, dim=1)

        if self.experiment == 'youtube':
            pre_conv_output = self.pre_conv(inputs)
            pre_conv_output = pre_conv_output.view(-1, 256, 4, 4)
            x_mean = F.sigmoid(self.conv(pre_conv_output))
            x_logvar = self.logvar.expand_as(x_mean)
            return {'means_x': x_mean,
                    'logvars_x': x_logvar}

        outputs = self.model(inputs)

        if self.experiment == 'synthetic' or self.experiment == 'mnist':
            return {'means_x': outputs[:, :self.x_dim],
                    'logvars_x': torch.clamp(outputs[:, self.x_dim:], -10, 10)}

        else:
            return {'proba_x': outputs}
