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

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=16, help='size of batch')

parser.add_argument('--test_mnist', action='store_true', help='whether to test on mnist')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')

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
parser.add_argument('--save_freq', type=int, default=20)

opts = parser.parse_args()

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

train_dataset = dataset_module.get_dataset(opts, split='train')
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)
test_dataset = dataset_module.get_dataset(opts, split='val')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)

model = get_model(opts).cuda()
loss_dict = get_loss(opts)
logger = get_logger(opts)
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

alpha = 0.0

for epoch in tqdm.tqdm(range(opts.num_epochs)):
    model.train()
    for data_dict in train_dataloader:
        data = data_dict['datasets'].cuda()

        optimizer.zero_grad()

        output_dict = model.forward(data, train=True)

        losses = {}

        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)
            
        losses['sum'] = (1 + alpha)*losses['NLL'] + losses['KL']/(1 + alpha)

        losses['sum'].backward()

        optimizer.step()

        logger.log_data(output_dict, losses)

    if opts.experiment == 'omniglot':
        	logger.log_image(output_dict, 'train')

    alpha *= 0.5

    with torch.no_grad():
        model.eval()
        if opts.experiment == 'synthetic':
            contexts_test = []
            labels_test = []
            means_test = []
            variances_test = []

        for data_dict in test_dataloader:

            data = data_dict['datasets'].cuda()
            output_dict = model.forward(data, train=False)
            losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

            logger.log_data(output_dict, losses, split='test')

            if opts.experiment == 'synthetic':

                labels_test.append(data_dict['targets'].numpy())
                means_test.append(data_dict['means'].numpy())
                variances_test.append(data_dict['variances'].numpy())
                contexts_test.append(output_dict['means_context'].cpu().numpy())

        if opts.experiment == 'omniglot':
        	logger.log_image(output_dict, 'test')
    
    if opts.experiment == 'synthetic':    
        contexts_test = np.concatenate(contexts_test, axis=0)
        labels_test = np.concatenate(labels_test, axis=0)
        means_test = np.concatenate(means_test, axis=0)
        variances_test = np.concatenate(variances_test, axis=0)

    if epoch%opts.save_freq == 0:
        if opts.experiment == 'synthetic':
            logger.log_embedding(contexts_test, labels_test, means_test, variances_test)
        logger.save_model(model, str(epoch))

if opts.experiment == 'synthetic':
    logger.log_embedding(contexts_test, labels_test, means_test, variances_test)
logger.save_model(model, 'last')
