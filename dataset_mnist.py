import gzip
import numpy as np
import os
import pickle

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from torchvision.datasets import MNIST
import torch.nn.functional as F


def get_dataset(opts, split = 'train'):
    """Function to get instance of SyntheticDataset given training options."""
    if split == 'train':
        return MnistDataset(opts.data_dir, mode = 'train')
    else:
        return MnistDataset(opts.data_dir, mode = 'test')


class MnistDataset(Dataset):
    """Data set for mnist experiment."""
    def __init__(self, data_dir, mode = 'train'):
        data_train = MNIST('mnist', download=True, train=True)
        x_train, y_train = data_train.train_data.float(), data_train.train_labels
        data_test = MNIST('mnist', download=True, train=False)
        x_test, y_test = data_test.test_data.float(), data_test.test_labels

        x_train_resize = x_train.view(x_train.shape[0],-1).data.numpy()
        y_train_onehot = F.one_hot(y_train, num_classes = 10).data.numpy()
        x_test_resize = x_test.view(x_test.shape[0],-1).data.numpy()
        y_test_onehot = F.one_hot(y_test, num_classes = 10).data.numpy()

        if mode == 'train':
            # to speed up debugging
            spatial = create_spatial_highlight(x_train_resize[:1000], y_train_onehot[:1000], sample_size=50, sample_size_highlight=6, plot=False)
            #spatial = create_spatial_highlight(x_train_resize, y_train_onehot, sample_size=50, sample_size_highlight=6, plot=False)
            self.spatial = np.array(spatial).astype(np.float32)
            # to speed up debugging
            self.labels = y_train_onehot[:1000]
            #self.labels = y_train_onehot
        elif mode == 'test':
            # to speed up debugging
            spatial = create_spatial_highlight(x_test_resize[:500], y_test_onehot[:500], sample_size=50, sample_size_highlight=6, plot=False)
            #spatial = create_spatial_highlight(x_test_resize, y_test_onehot, sample_size=50, sample_size_highlight=6, plot=False)
            self.spatial = np.array(spatial).astype(np.float32)
            # to speed up debugging
            self.labels = y_test_onehot[:500]
            #self.labels = y_test_onehot


        #ix = self.labels[:, 1] != 1
        #print(ix.shape)
        #self.spatial = self.spatial[ix]
        #self.labels = self.labels[ix]

        assert len(self.spatial) == len(self.labels)

    def __len__(self):
        return len(self.spatial) 

    def __getitem__(self, idx):
        return {'datasets': torch.FloatTensor(self.spatial[idx]),  # have shape (num_img, 50, 2), where 2 is [a,b] refers to location
                'targets': self.labels[idx]}                       # have shape (num_img, 10), where 10 is one-hot encodding


def create_spatial_highlight(images, labels, sample_size=50, sample_size_highlight=6, plot=False):
    spatial = np.zeros([images.shape[0], sample_size, 2])
    spatial_highlight = np.zeros([images.shape[0], sample_size_highlight, 2])
    grid = np.array([[i, j] for j in range(27, -1, -1) for i in range(28)])
    for i, image in enumerate(tqdm(images)):
        replace = True if (sum(image > 0) < sample_size) else False
        ix = np.random.choice(range(28*28), size=sample_size,
                              p=image/sum(image), replace=replace)
        ix_highlight = np.random.choice(range(28*28), size=sample_size_highlight,
                              p=image/sum(image), replace=replace)
        spatial[i, :, :] = grid[ix] + np.random.uniform(0, 1, (sample_size, 2))
        spatial_highlight[i, :, :] = grid[ix_highlight] + np.random.uniform(0, 1, (sample_size_highlight, 2))

    # sanity check
    if plot:
        sample = spatial[:10]
        sample_highlight = spatial_highlight[:10]
        fig, axs = plt.subplots(1, 10, figsize=(8, 8))
        axs = axs.flatten()
        for i in range(10):
            axs[i].scatter(sample[i, :, 0], sample[i, :, 1], s=2)
            axs[i].scatter(sample_highlight[i, :, 0], sample_highlight[i, :, 1], s=2, c='red')
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_xlim([0, 27])
            axs[i].set_ylim([0, 27])
            axs[i].set_aspect('equal', adjustable='box')
        plt.show()

    return spatial
