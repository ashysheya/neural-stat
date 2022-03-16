import argparse
import importlib
import torch
import tqdm
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from models import get_model
from losses import get_loss
from logs import get_logger
from utils_mnist import summarize_batch
import pickle

import matplotlib.pyplot as plt

from utils import sample_from_normal

# configuration for 'mnist' experienment
# --experiment 'mnist' --num_epochs 100 --context_dim 64 --num_stochastic_layers 3 --z_dim 2 --x_dim 2 --h_dim 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Arguments for training procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='synthetic',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

parser.add_argument('--nll', type=str, default='gaussian', help='type of loglikelihood')

# Dataloaders options
parser.add_argument('--train_num_datasets_per_distr', type=int, default=2500,
    help='number of training datasets per distribution for synthetic experiment')

parser.add_argument('--test_num_datasets_per_distr', type=int, default=500,
    help='number of test datasets per distribution for synthetic experiment')

parser.add_argument('--train_num_persons', type=int, default=1395,
    help='number of persons in the training datasets for youtube experiment')

parser.add_argument('--test_num_persons', type=int, default=100,
    help='number of persons in the testing datasets for youtube experiment')

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=16, help='size of batch') #16

# Path for data directory if using the youtube experiment
parser.add_argument('--data_dir', type=str, default=None, help='location of sampled youtube data')

parser.add_argument('--test_mnist', action='store_true', help='whether to test on mnist')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')

# beta1 is the exponential decay rate for the first moment estimates (e.g. 0.9), used in the Adam optimizer.
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for optimizer')

parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')

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

parser.add_argument('--h_dim', type=int, default=1, help='dimension of h after shared encoder')

# Logging options
parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--save_dir', type=str, default='model_params')
parser.add_argument('--save_freq', type=int, default=20)  # 20

opts = parser.parse_args()

# If using youtube dataset, check that a data directory is specified
if opts.experiment == 'youtube' and opts.data_dir is None:
    exit("Must specify a directory for the youtube dataset")

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

train_dataset = dataset_module.get_dataset(opts, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=6)

test_dataset = dataset_module.get_dataset(opts, split='val')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=6)
test_batch = next(iter(test_dataloader))

# Initialize the Neural Statistician model in models.py
model = get_model(opts).to(device)

loss_dict = get_loss(opts)
logger = get_logger(opts)
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

alpha = 1.0

vlb_per_epoch = []
recon_loss_per_epoch = []
kl_per_epoch = []

iters_per_epoch = (train_dataset.spatial.shape[0] // opts.batch_size)

for epoch in tqdm.tqdm(range(opts.num_epochs)):
    model.train()

    epoch_vlb = 0
    epoch_recon_loss = 0
    epoch_kl= 0

    for data_dict in train_dataloader:
        data = data_dict['datasets'].to(device)

        optimizer.zero_grad()

        output_dict = model.forward(data, train=True)

        losses = {}

        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)

        # Can weigh the contribution from each term
        losses['sum'] = (1 + alpha)*losses['NLL'] + losses['KL']/(1 + alpha)

        # Compute gradients, and take step backward
        losses['sum'].backward()

        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)

        optimizer.step()

        # Save model outputs and losses from the training.
        logger.log_data(output_dict, losses)

        epoch_vlb += (-losses['NLL'] - losses['KL']).item()
        epoch_recon_loss += -losses['NLL'].item()
        epoch_kl += losses['KL'].item()

    epoch_vlb /= iters_per_epoch
    epoch_recon_loss /= iters_per_epoch
    epoch_kl /= iters_per_epoch

    vlb_per_epoch.append(epoch_vlb)
    recon_loss_per_epoch.append(epoch_recon_loss)
    kl_per_epoch.append(epoch_kl)

    alpha *= 0.5

    if epoch % opts.save_freq == 0:

        with torch.no_grad():
            model.eval()

            # for data_dict in test_dataloader:

            #     data = data_dict['datasets'].to(device)

            #     output_dict = model.sample_conditional(data, num_samples_per_dataset=50)
            #     output_dict = model.forward(data, train=False)
            #     losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

            #     logger.log_data(output_dict, losses, split='test')
                    
            #     input_plot = data
            #     sample_plot = sample_from_normal(output_dict['means_x'], output_dict['logvars_x'])
            #     # sample_plot = output_dict['means_x']  
            #     print("Summarizing...")
            #     summaries = summarize_batch(opts, input_plot, output_size=6)
            #     print("Summary complete!")     
            #     break

            # logger.grid(input_plot, sample_plot, summaries=summaries, ncols=10, mode = 'summary')

            for data_dict in train_dataloader:

                data = data_dict['datasets'].to(device)

                output_dict = model.sample_conditional(data, num_samples_per_dataset=200)
                # output_dict = model.forward(data, train=False)
                losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

                logger.log_data(output_dict, losses, split='test')

                input_plot = data
                sample_plot = sample_from_normal(output_dict['means_x'], output_dict['logvars_x'])
                # sample_plot = output_dict['means_x']  
                print("Summarizing...")
                summaries = summarize_batch(opts, input_plot, output_size=50)
                print("Summary complete!")     
                break
            
            logger.grid(input_plot, sample_plot, summaries=summaries, ncols=5, mode = 'summary')

            logger.save_model(model, str(epoch))

logger.save_model(model, 'last')


def plot_loss(save_path, **kwarg_losses):
    plt.figure()
    for loss_name, loss in kwarg_losses.items():
        plt.plot(loss[2:], label=loss_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


pickle.dump(vlb_per_epoch, open("vlb.pkl", "wb"))
pickle.dump(recon_loss_per_epoch, open("recon_loss.pkl", "wb"))
pickle.dump(kl_per_epoch, open("kl.pkl", "wb"))

plot_loss("./losses.png", vlb=vlb_per_epoch, recon=recon_loss_per_epoch, kl=kl_per_epoch)
