import numpy as np
import torch
from torch.autograd import Variable
from itertools import combinations
from models import get_stats
from losses import KLDivergence
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# adapt from https://github.com/conormdurkan/neural-statistician/blob/master/spatial/spatialmodel.py
def summarize_batch(opts, inputs, output_size=6):
    summaries = []
    for dataset in tqdm.tqdm(inputs[:5]):
        summary = summarize(opts, dataset, output_size=output_size)
        summaries.append(summary)
    summaries = torch.cat(summaries)
    return summaries


def summarize(opts, dataset, output_size=6):
    stats = get_stats(opts).to(device)

    # cast to torch Cuda Variable and reshape
    sample_size = 200
    dataset = dataset.view(1, sample_size, 2)
    # get approximate posterior over full dataset
    output = stats.forward({'train_data_encoded': dataset})
    c_mean_full = output['means_context']
    c_logvar_full = output['logvars_context']

    # iteratively discard until dataset is of required size
    while dataset.size(1) != output_size:
        kl_divergences = []
        # need KL divergence between full approximate posterior and all
        # subsets of given size
        subset_indices = list(combinations(range(dataset.size(1)), dataset.size(1) - 1))

        for subset_index in subset_indices:
            # pull out subset, numpy indexing will make this much easier
            ix = Variable(torch.LongTensor(subset_index).to(device))
            subset = dataset.index_select(1, ix)

            # calculate approximate posterior over subset
            output = stats.forward({'train_data_encoded': subset})
            c_mean = output['means_context']
            c_logvar = output['logvars_context']

            
            kl = KLDivergence.calculate_kl(c_logvar_full, c_logvar, c_mean, c_mean_full)
            kl_divergences.append(kl.data[0])

        # determine which sample we want to remove
        best_index = kl_divergences.index(min(kl_divergences))

        # determine which samples to keep
        to_keep = subset_indices[~best_index]
        to_keep = Variable(torch.LongTensor(to_keep).to(device))

        # keep only desired samples
        dataset = dataset.index_select(1, to_keep)

    # return pruned dataset
    return dataset
