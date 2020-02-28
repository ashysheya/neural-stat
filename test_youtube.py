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
from logs import get_logger, normalize_img
from torchvision.utils import make_grid
from PIL import Image

parser = argparse.ArgumentParser(description='Arguments for test procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='omniglot',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

# Dataloaders options
parser.add_argument('--batch_size', type=int, default=16, help='size of batch')

parser.add_argument('--num_samples_per_dataset', type=int, default=5,
    help='number of samples per dataset')

parser.add_argument('--num_data_per_dataset', type=int, default=5,
    help='number of data to input per dataset')

parser.add_argument('--test_mnist', action='store_true', help='whether to test on mnist')

# Path for data directory if using the youtube experiment
parser.add_argument('--data_dir', type=str, default=None, help='location of sampled youtube data')

parser.add_argument('--train_num_persons', type=int, default=1395,
    help='number of persons in the training datasets for youtube experiment')

parser.add_argument('--test_num_persons', type=int, default=100,
    help='number of persons in the testing datasets for youtube experiment')

parser.add_argument('--total_num_persons', type=int, default=1595,
    help='total number of persons in the datasets - set to 25 for the poses experiment')

# For youtube, can either sample conditioned on data, or sample from the context distribution
parser.add_argument('--test_conditioned', action='store_true', help='whether to test conditioned on samples')

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
parser.add_argument('--n_channels', type=int, default=3, help='number of channels in the image (=3 for RGB)')

# Logging options
parser.add_argument('--model_name', type=str, default='youtube_10:02:2020_22:24:41/last')
parser.add_argument('--model_dir', type=str, default='model_params')
parser.add_argument('--result_dir', type=str, default='results')


opts = parser.parse_args()

if opts.test_conditioned:
    dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))
    test_dataset = dataset_module.get_dataset(opts, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True)

model = get_model(opts).cuda()

# Load model parameters
model.load_state_dict(torch.load(f'{opts.model_dir}/{opts.model_name}'))

model.eval()

os.makedirs(f'{opts.result_dir}/{opts.model_name}', exist_ok=True)

dataset_name = 'youtube'

with torch.no_grad():
    if opts.test_conditioned:
        for i, data_dict in enumerate(test_dataloader):
            data = data_dict['datasets'].cuda()
            output_dict = model.sample_conditional(data, opts.num_samples_per_dataset)

            samples = output_dict['means_x'].data.cpu()
            data = data.view_as(samples).data.cpu()

            data_gen = make_grid(samples, nrow=opts.num_samples_per_dataset)
            data_real = make_grid(data, nrow=opts.num_data_per_dataset)

            image = np.concatenate([data_real, data_gen], axis=-1)
            im = Image.fromarray(np.uint8(normalize_img(image.transpose((1, 2, 0)))*255))
            im.save(f'{opts.result_dir}/{opts.model_name}/{dataset_name}_{i}_conditioned.png')

    else:
        output_dict = model.sample(opts.num_samples_per_dataset, opts.batch_size)
        samples = output_dict['means_x'].data.cpu()

        data_gen = make_grid(samples, nrow=opts.num_samples_per_dataset)

        im = Image.fromarray(np.uint8(normalize_img(data_gen.numpy().transpose((1, 2, 0)))*255))
        im.save(f'{opts.result_dir}/{opts.model_name}/{dataset_name}_unseen.png')
