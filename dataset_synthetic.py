import torch
import numpy as np
from torch.utils.data import Dataset
from utils import generate_1d_datasets


def get_dataset(opts, split='train'):
    """Function to get instance of SyntheticDataset given training options."""
    if split == 'train':
        return SyntheticDataset(opts.train_num_datasets_per_distr, opts.num_data_per_dataset)
    else:
        return SyntheticDataset(opts.test_num_datasets_per_distr, opts.num_data_per_dataset)


class SyntheticDataset(Dataset):
    """Dataset for 1d synthetic data experiment."""

    def __init__(self, num_datasets_per_distr=2500, num_data_per_dataset=200):
        """
        :param num_datasets_per_distr: int, number of datasets per distribution to generate
        :param num_data_per_dataset: int, number of datapoints per dataset
        """
        ## gen_data is a list containing the dataset, targets, and distribution means and variances
        gen_data = generate_1d_datasets(num_datasets_per_distr, num_data_per_dataset)
        ## datasets has dimensions (2500*4, 200, 1) for full training data
        self.datasets = gen_data[0]
        ## targets has size (4*2500, ) for training data and (4*500, ) for testing data
        targets = gen_data[1]
        ## means has size (4*2500, )
        self.means = gen_data[2]
        ## variances has size (4*2500, )
        self.variances = gen_data[3]
        # Convert strings to numeric labels
        self.targets = np.zeros_like(targets, dtype=np.int)
        self.idx_dict = {0: 'exponential', 1: 'gaussian', 2: 'uniform', 3: 'laplace'}
        for key in self.idx_dict:
            self.targets[targets == self.idx_dict[key]] = key

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return {'datasets': torch.FloatTensor(self.datasets[idx]), 
                'targets': self.targets[idx], 
                'means': self.means[idx], 
                'variances': self.variances[idx]} 
