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

parser = argparse.ArgumentParser(description='Arguments for test procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='omniglot',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

# Dataloaders options
parser.add_argument('--batch_size', type=int, default=16, help='size of batch')

parser.add_argument('--num_data_per_dataset', type=int, default=5,
    help='number of samples per dataset')

parser.add_argument('--num_samples_per_dataset', type=int, default=5)

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
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

test_dataset = dataset_module.get_dataset(opts, split='test')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True)

model = get_model(opts).cuda()

# Load model parameters
model.load_state_dict(torch.load(f'{opts.model_dir}/{opts.model_name}'))

model.eval()

os.makedirs(f'{opts.result_dir}/{opts.model_name}', exist_ok=True)

dataset_name = 'mnist' if opts.test_mnist else 'omniglot'

with torch.no_grad():

    for i, data_dict in enumerate(test_dataloader):
        data = data_dict['datasets'].cuda()
        output_dict = model.sample_conditional(data, opts.num_samples_per_dataset)

        samples = output_dict['proba_x'].cpu()
        data = data.view_as(samples).cpu()

        data_gen = make_grid(samples, nrow=opts.num_samples_per_dataset)
        data_real = make_grid(data, nrow=opts.num_data_per_dataset)

        image = np.concatenate([data_real, data_gen], axis=-1)
        im = Image.fromarray(np.uint8(image.transpose((1, 2, 0))*255))
        im.save(f'{opts.result_dir}/{opts.model_name}/{dataset_name}_{i}.png')
