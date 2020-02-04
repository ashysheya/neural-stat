import argparse
import importlib
import torch
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from models import get_model
from losses import get_loss
from logs import get_logger
from plot_synthetic import scatter_context

import time
time_stamp = time.strftime("%d-%m-%Y-%H:%M:%S")

parser = argparse.ArgumentParser(description='Arguments for training procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='synthetic',
    help='options that tells what experiment to replicate: synthetic|mnist|omniglot|youtube')

# Dataloaders options
parser.add_argument('--train_num_datasets_per_distr', type=int, default=2500,
    help='number of training datasets per distribution for synthetic experiment')

parser.add_argument('--test_num_datasets_per_distr', type=int, default=500,
    help='number of test datasets per distribution for synthetic experiment')

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=30, help='size of batch')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')

parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for optimizer')

parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')

# Architecture options
parser.add_argument('--context_dim', type=int, default=3, help='context dimension')

parser.add_argument('--masked', action='store_false',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=1,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=32,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=1, help='dimension of input')

# Logging options
parser.add_argument('--tensorboard', action='store_false', help='whether to use tensorboard')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--save_dir', type=str, default='model_params')
parser.add_argument('--save_freq', type=int, default=20)


opts = parser.parse_args()



#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))

train_dataset = dataset_module.get_dataset(opts, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=10)
test_dataset = dataset_module.get_dataset(opts, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=10)

model = get_model(opts).cuda()
loss_dict = get_loss(opts)
logger = get_logger(opts)
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

for epoch in tqdm.tqdm(range(opts.num_epochs)):
    for data, targets in train_dataloader:
        data = data.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()

        output_dict = model.forward(data)
        losses = {'sum': 0}

        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)
            losses['sum'] += losses[key]

        losses['sum'].backward()
        optimizer.step()

        logger.log_data(output_dict, losses)

    with torch.no_grad():
        means_context = {'data': [], 'labels': []}  # For plotting contexts
        for data, targets in test_dataloader:
            data = data.cuda()
            targets = targets.cuda()

            output_dict = model.forward(data)

            losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

            logger.log_data(output_dict, losses, split='test')

            # If synthetic experiment, save mean contexts to be plotted later, along with targets for colour labelling
            if opts.experiment == 'synthetic':
                means_context['data'] += [output_dict['means_context'].cpu().numpy()]  # (batch_size, context_dim)
                means_context['labels'] += [targets.cpu().numpy()]
        
        # Plot if synthetic experiment
        if opts.experiment == 'synthetic' and opts.context_dim == 3:
            path = '/figures/' + time_stamp + '-{}.pdf'.format(epoch + 1)
            
            scatter_context(means_context, savepath = path) # still problem with it
            # scatter_context(means_context)
        

    if epoch%opts.save_freq == 0:
        logger.save_model(model, str(epoch))

logger.save_model(model, 'last')
