import argparse
import importlib
import torch
import tqdm
import os
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
from losses import get_loss
from logs import get_logger
from torchvision.utils import make_grid
from PIL import Image
from losses import KLDivergence

parser = argparse.ArgumentParser(description='Arguments for test procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='omniglot',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

# Dataloaders options
parser.add_argument('--batch_size', type=int, default=20, help='size of batch')

parser.add_argument('--num_data_per_dataset', type=int, default=1,
    help='number of samples per dataset')

parser.add_argument('--num_classes', type=int, default=5)

parser.add_argument('--test_mnist', action='store_true', help='whether to test on mnist')

# Architecture options
parser.add_argument('--context_dim', type=int, default=512, help='context dimension')

parser.add_argument('--masked', action='store_true',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=1,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=16,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=1, help='dimension of input')
parser.add_argument('--h_dim', type=int, default=4096, help='dimension of input')

# Logging options
parser.add_argument('--model_name', type=str, default='omniglot_10:02:2020_22:24:41/last')
parser.add_argument('--model_dir', type=str, default='model_params')
parser.add_argument('--result_dir', type=str, default='results')


opts = parser.parse_args()

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment, 'accuracy']))

test_dataset = dataset_module.get_dataset(opts)
test_dataset.sample_experiment()

test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

model = get_model(opts).cuda()

# Load model parameters

print(opts.model_name)
model.load_state_dict(torch.load(f'{opts.model_dir}/{opts.model_name}'))

model.eval()

os.makedirs(f'{opts.result_dir}/{opts.model_name}', exist_ok=True)

dataset_name = 'mnist' if opts.test_mnist else 'omniglot'

with torch.no_grad():

    accuracies = []

    for _ in tqdm.tqdm(range(100)):

        test_dataset.set_train()

        means_classes = []
        logvars_classes = []
        labels_classes = []

        for i, data_dict in enumerate(test_dataloader):
            data = data_dict['datasets'].cuda()
            output_dict = model.context_params(data)
            means_classes += [output_dict['means_context'].cpu()]
            logvars_classes += [output_dict['logvars_context'].cpu()]
            labels_classes += [data_dict['targets']]

        means_classes = torch.cat(means_classes, dim=0)
        logvars_classes = torch.cat(logvars_classes, dim=0)
        labels_classes = torch.cat(labels_classes, dim=0)

        test_dataset.set_test()

        means = []
        logvars = []
        labels = []

        for i, data_dict in enumerate(test_dataloader):
            data = data_dict['datasets'].cuda()
            output_dict = model.context_params(data)
            means += [output_dict['means_context'].cpu()]
            logvars += [output_dict['logvars_context'].cpu()]
            labels += [data_dict['targets']]

        means = torch.cat(means, dim=0)
        logvars = torch.cat(logvars, dim=0)
        labels = torch.cat(labels, dim=0)

        # calculate KL

        means = means[:, None].expand(-1, means_classes.size()[0], -1)
        logvars = logvars[:, None].expand(-1, logvars_classes.size()[0], -1)
        means_classes = means_classes[None].expand(means.size()[0], -1, -1)
        logvars_classes = logvars_classes[None].expand(logvars.size()[0], -1, -1)

        kls = KLDivergence.calculate_kl(logvars, logvars_classes, means_classes, means).numpy()

        argmins = np.argmin(kls, axis=1)

        chosen_labels = labels_classes.numpy()[argmins]

        true_labels = labels.numpy()

        accuracy = (true_labels == chosen_labels).sum()/len(true_labels)

        accuracies += [accuracy]

        test_dataset.sample_experiment()

accuracy = np.array(accuracies, dtype=np.float).mean()

print(f'accuracy for num_classes = {opts.num_classes}, \
train samples = {opts.num_data_per_dataset}, mnist = {opts.test_mnist}, accuracy = {accuracy}')
