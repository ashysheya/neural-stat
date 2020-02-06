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

# Dataloaders options
parser.add_argument('--train_num_datasets_per_distr', type=int, default=2500,
    help='number of training datasets per distribution for synthetic experiment')

parser.add_argument('--test_num_datasets_per_distr', type=int, default=500,
    help='number of test datasets per distribution for synthetic experiment')

parser.add_argument('--num_data_per_dataset', type=int, default=200,
    help='number of samples per dataset')

parser.add_argument('--batch_size', type=int, default=16, help='size of batch')

# Optimization options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizer')

## beta1 is the exponential decay rate for the first moment estimates (e.g. 0.9), used in the Adam optimizer.
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for optimizer')

parser.add_argument('--num_epochs', type=int, default=100, help='number of training epochs')

# Architecture options
parser.add_argument('--context_dim', type=int, default=3, help='context dimension')

## action='store_true' stores the value True when --masked is specified
parser.add_argument('--masked', action='store_true',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=1,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=32,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=1, help='dimension of input')

# Logging options
parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--save_dir', type=str, default='model_params')
parser.add_argument('--save_freq', type=int, default=1)

opts = parser.parse_args()

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))  # imports dataset_sythetic.py

## Initialise an object of type SyntheticDataset, which can be used as input to DataLoader. It has members .dataset,
## which contains datasets for each distribution type and is of dimension (4*2500, 200, 1), .targets of size (1, 4*2500)
## which contains the corresponding targets, and .means and .variances which are of size (1, 4*2500) each and contain
## the distribution means and variances respectively.
train_dataset = dataset_module.get_dataset(opts, train=True)
## The DataLoader shuffles the data and creates batches for the dataset
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)
test_dataset = dataset_module.get_dataset(opts, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)

## Initialize the Neural Statistician model in models.py
model = get_model(opts).cuda()
loss_dict = get_loss(opts)  # Returns dict with {'KL': KLDivergence(), 'NLL': NegativeGaussianLogLikelihood()}
logger = get_logger(opts)  ## For saving results: creates a directory for saving logs / models, specifying saving freq.
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

alpha = 0

for epoch in tqdm.tqdm(range(opts.num_epochs)):
    ## Calls __getitem__ in dataset_synthetic: takes one batch of data, and returns a dictionary with 'datasets'
    ## (16, 200, 1), 'targets' (1, 16), 'means' (1, 16) and 'variances' (1, 16)
    for data_dict in train_dataloader:
        data = data_dict['datasets'].cuda()  ## Size is (16, 200, 1)

        optimizer.zero_grad()

        ## Make a forward pass through model. The output is a dictionary with:
        ## - q(c|D): 'means_context': means (16, c_dim),
        ##           'logvars_context': logvars (16, c_dim),
        ##           'samples_context': samples (16, c_dim),
        ##           'samples_context_expanded' (16*200, c_dim): context samples, copied for all x in the dataset
        ## - p(c): 'means_context_prior': means (16, c_dim),
        ##         'logvars_context_prior': logvars (16, c_dim)
        ## - q(z_1, .., z_L|c, x): 'means_latent_z': list of means for each n_stochastic_layers, each is (16*200, z_dim)
        ##                         'logvars_latent_z': list of logvar for each n_stochastic_layers, each (16*200, z_dim)
        ##                         'samples_latent_z': list of samp. for each n_stochastic_layers, each (16*200, z_dim)
        ## - p(z_1, .., z_L|c): 'means_latent_z_prior': list of means for each n_stochastic_layers, each (16*200, z_dim)
        ##                      'logvars_latent_z_prior': list of var for each n_stochastic_layers, each (16*200, z_dim)
        ## - p(x|z_1, .., z_L, c): 'means_x': means (16*200, x_dim)
        ##                         'logvars_x': logvars (16*200, x_dim)

        output_dict = model.forward(data)

        losses = {}

        ## Compute each type of loss - see losses.py for comments on how each of these is computed
        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)

        ## Can weigh the contribution from each term
        losses['sum'] = (1 + alpha)*losses['NLL'] + losses['KL']/(1 + alpha)

        ## Compute gradients, and take step backward
        losses['sum'].backward()

        optimizer.step()

        ## Save model outputs and losses from the training.
        logger.log_data(output_dict, losses)

    ## Compute the same for the test data
    with torch.no_grad():

        if opts.experiment == 'synthetic':
            contexts_test = []
            labels_test = []
            means_test = []
            variances_test = []

        for data_dict in test_dataloader:

            data = data_dict['datasets'].cuda()
            output_dict = model.forward(data)
            losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

            logger.log_data(output_dict, losses, split='test')

            ## Save input labels, means and variances for plotting, as well as output context means.
            if opts.experiment == 'synthetic':
                labels_test.append(data_dict['targets'].numpy()) ## (16, )
                means_test.append(data_dict['means'].numpy()) ## (16, )
                variances_test.append(data_dict['variances'].numpy()) ## (16, )
                contexts_test.append(output_dict['means_context'].cpu().numpy()) ## (16, c_dim)
    
    if opts.experiment == 'synthetic':    
        contexts_test = np.concatenate(contexts_test, axis=0)  ## (500*4, 3)
        labels_test = np.concatenate(labels_test, axis=0)  ## (500*4, )
        means_test = np.concatenate(means_test, axis=0)  ## (500*4, )
        variances_test = np.concatenate(variances_test, axis=0)  ## (500*4, )

    if epoch%opts.save_freq == 0:
        if opts.experiment == 'synthetic':
            ## Save a plot of the context means
            logger.log_embedding(contexts_test, labels_test, means_test, variances_test)
        logger.save_model(model, str(epoch))

if opts.experiment == 'synthetic':
    logger.log_embedding(contexts_test, labels_test, means_test, variances_test)
logger.save_model(model, 'last')
