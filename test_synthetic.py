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

parser = argparse.ArgumentParser(description='Arguments for training procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='synthetic',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

# Dataloaders options
parser.add_argument('--test_num_datasets_per_distr', type=int, default=250,
    help='number of test datasets per distribution for synthetic experiment')

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=16, help='size of batch')

# Architecture options
parser.add_argument('--context_dim', type=int, default=3, help='context dimension')

parser.add_argument('--masked', action='store_true',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=1,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=32,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=1, help='dimension of input')
parser.add_argument('--h_dim', type=int, default=1, help='dimension of input')


# Logging options
parser.add_argument('--model_name', type=str, default='synthetic_04:02:2020_20:40:45/98')
parser.add_argument('--model_dir', type=str, default='model_params')
parser.add_argument('--result_dir', type=str, default='results')


opts = parser.parse_args()

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

test_dataset = dataset_module.get_dataset(opts, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
	shuffle=True, num_workers=10)

model = get_model(opts).cuda()

# Load model parameters
model.load_state_dict(torch.load(f'{opts.model_dir}/{opts.model_name}'))

with torch.no_grad():

    contexts_test = []
    labels_test = []
    means_test = []
    variances_test = []

    for data_dict in test_dataloader:
        data = data_dict['datasets'].cuda()
        
        labels_test.append(data_dict['targets'].numpy())

        means_test.append(data_dict['means'].numpy())

        variances_test.append(data_dict['variances'].numpy())

        output_dict = model.forward(data)

        contexts_test.append(output_dict['means_context'].cpu().numpy())

contexts_test = np.concatenate(contexts_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
means_test = np.concatenate(means_test, axis=0)
variances_test = np.concatenate(variances_test, axis=0)

os.makedirs(f'{opts.result_dir}/{opts.model_name}', exist_ok=True)
for name, ar in zip(['context', 'labels', 'means', 'variances'], 
                    [contexts_test, labels_test, means_test, variances_test]):
    np.save(f'{opts.result_dir}/{opts.model_name}/{name}.npy', ar)

