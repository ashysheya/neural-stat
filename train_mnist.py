import argparse
import importlib
import torch
import tqdm
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from models_mnist import get_model, get_stats
from losses import get_loss
from logs import get_logger
from itertools import combinations

parser = argparse.ArgumentParser(description='Arguments for training procedure')

# Experiment options
parser.add_argument('--experiment', type=str, default='mnist',
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

parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for optimizer')

parser.add_argument('--num_epochs', type=int, default=2, help='number of training epochs')

# Architecture options
parser.add_argument('--context_dim', type=int, default=64, help='context dimension')

parser.add_argument('--masked', action='store_true',
    help='whether to use masking during training')

parser.add_argument('--type_prior', type=str, default='standard',
    help='either use standard gaussian prior or prior conditioned on labels')

parser.add_argument('--num_stochastic_layers', type=int, default=3,
    help='number of stochastic layers')

parser.add_argument('--z_dim', type=int, default=2,
    help='dimension of latent variables')

parser.add_argument('--x_dim', type=int, default=2, help='dimension of input')

# Logging options
parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--save_dir', type=str, default='model_params')
parser.add_argument('--save_freq', type=int, default=1)

# if mnist
parser.add_argument('--data_dir', type=str, default='mnist')

opts = parser.parse_args()

#import dataset module
dataset_module = importlib.import_module('_'.join(['dataset', opts.experiment]))


train_dataset = dataset_module.get_dataset(opts, split = 'train')
train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)
test_dataset = dataset_module.get_dataset(opts, split = 'test')
test_dataloader = DataLoader(test_dataset, batch_size=opts.batch_size, 
    shuffle=True, num_workers=10)
test_batch = next(iter(test_dataloader))

model = get_model(opts).cuda()
loss_dict = get_loss(opts)
logger = get_logger(opts)
optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))

alpha = 0


stats = get_stats(opts).cuda()


# https://github.com/conormdurkan/neural-statistician/blob/master/spatial/spatialmodel.py
def summarize_batch(inputs, output_size=6):
    summaries = []
    for dataset in tqdm.tqdm(inputs):
        summary = summarize(dataset, output_size=output_size)
        summaries.append(summary)
    summaries = torch.cat(summaries)
    return summaries

def calculate_kl(logvar_prior, logvar, mu, mu_prior):
    kl_val = 0.5 * logvar_prior - 0.5 * logvar
    kl_val += (torch.exp(logvar) + (mu - mu_prior) ** 2) / 2 / (
        torch.exp(logvar_prior))
    kl_val -= 0.5
    return kl_val.sum(dim=-1)

def summarize(dataset, output_size=6):

    # cast to torch Cuda Variable and reshape
    sample_size = 50
    dataset = dataset.view(1, sample_size, 2)
    # get approximate posterior over full dataset
    c_mean_full, c_logvar_full, _, _ = stats({'train_data': dataset, 'train': True})

    # iteratively discard until dataset is of required size
    while dataset.size(1) != output_size:
        kl_divergences = []
        # need KL divergence between full approximate posterior and all
        # subsets of given size
        subset_indices = list(combinations(range(dataset.size(1)), dataset.size(1) - 1))

        for subset_index in subset_indices:
            # pull out subset, numpy indexing will make this much easier
            ix = Variable(torch.LongTensor(subset_index).cuda())
            subset = dataset.index_select(1, ix)

            # calculate approximate posterior over subset
            c_mean, c_logvar, _, _ = stats({'train_data': subset, 'train': True})

            # kl_input = {'logvars_context_prior': c_logvar_full,
            #          'logvars_context': c_logvar,
            #          'means_context': c_mean, 
            #          'means_context_prior': c_mean_full}

            # kl = loss_dict['KL'].forward(kl_input)
            
            ##### TypeError: can't multiply sequence by non-int of type 'float'
            kl = calculate_kl(c_logvar_full, c_logvar, c_mean, c_mean_full)
            kl_divergences.append(kl.data[0])

        # determine which sample we want to remove
        best_index = kl_divergences.index(min(kl_divergences))

        # determine which samples to keep
        to_keep = subset_indices[best_index]
        to_keep = Variable(torch.LongTensor(to_keep).cuda())

        # keep only desired samples
        dataset = dataset.index_select(1, to_keep)

    # return pruned dataset
    return dataset




for epoch in tqdm.tqdm(range(opts.num_epochs)):
    model.train()
    for data_dict in train_dataloader:
        data = data_dict['datasets'].cuda()

        optimizer.zero_grad()

        output_dict = model.forward(data, train = True)

        # print('length of proba_x: ' + str(len(output_dict['proba_x'])))
        # length: 800

        losses = {}

        for key in loss_dict:
            losses[key] = loss_dict[key].forward(output_dict)
            
        losses['sum'] = (1 + alpha)*losses['NLL'] + losses['KL']/(1 + alpha)

        losses['sum'].backward()

        optimizer.step()

        logger.log_data(output_dict, losses)

    with torch.no_grad():
        model.eval()
        count = 0

        for data_dict in test_dataloader:
            data = data_dict['datasets'].cuda()
            output_dict = model.forward(data, train = False)
            losses = {'NLL': loss_dict['NLL'].forward(output_dict)}

            logger.log_data(output_dict, losses, split='test')

            # plot examples in the first batch
            if count == 0:
                input_plot = data
                sample_plot = output_dict['means_x']       

            count += 1

        # check opts.save_freq
        if epoch%opts.save_freq == 0:
            logger.grid(input_plot, sample_plot, ncols = 10, mode = 'test')
            logger.save_model(model, str(epoch))

logger.save_model(model, 'last')


### TO DO: plot summary at the end & highlighted dots, also summary for inputs!
### TO DO: double check the intermediate plots of lines! are they normal/ind sampled? check prior of c used for sampling
    
count = 0
# summarise test batch at end of training
for data_dict in test_dataloader:
    if count != 0:
        break
    else:
        n = 10  # number of datasets to summarize
        inputs = Variable(data_dict['datasets'].cuda())
        print(inputs.shape)
        print("Summarizing...")
        summaries = summarize_batch(inputs[:n], output_size=6)
        print("Summary complete!")

        # plot summarized datasets
        samples = model.forward(inputs, train = False)
        grid(inputs, samples['means_x'], summaries=summaries, ncols=n, mode = 'summary')

        count += 1
            


