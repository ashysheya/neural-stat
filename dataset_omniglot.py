import torch
import numpy as np
import PIL.ImageOps    
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, Omniglot
from skimage.morphology import dilation
from skimage.transform import resize


def get_dataset(opts, split='train'):
    """Function to get instance of SyntheticDataset given training options."""
    return OmniglotDataset(opts.num_data_per_dataset, split, opts.test_mnist)


class OmniglotDataset(Dataset):
    """Dataset for omniglot data experiment."""

    def __init__(self, num_data_per_dataset, split, mnist):
        """
        :param num_data_per_dataset: int, number of data per dataset
        :param split: str, type of dataset
        :param mnist: boolean, whether to test on mnist
        """
        self.split = split
        self.num_data_per_dataset = num_data_per_dataset
        self.mnist = mnist

        if self.split == 'test' and self.mnist:
            data = MNIST('mnist', download=True, train=True)
            x_train, y_train = data.data.float()/255, data.targets
            data = MNIST('mnist', download=True, train=False)
            x_test, y_test = data.data.float()/255, data.targets
            self.x = torch.cat([x_train.round(), x_test.round()], dim=0)[:, None]
            self.y = torch.cat([y_train, y_test], dim=0)

        else:
            self.dataset_background = Omniglot('omniglot', download=True)
            self.dataset_eval = Omniglot('omniglot', download=True, background=False)

            images = []
            labels = []

            for image, label in self.dataset_background:
                image = PIL.ImageOps.invert(image)
                images += [(np.asarray(image.resize((28, 28)))/255)[None]]
                labels += [label]

            for image, label in self.dataset_eval:
                image = PIL.ImageOps.invert(image)
                images += [(np.asarray(image.resize((28, 28)))/255)[None]]
                labels += [label]

            images = np.stack(images, axis=0)
            labels = np.array(labels, dtype=np.int)

            if self.split == 'test':
                self.images = images[1300*20:].round()
                self.labels = labels[1300*20:].round()
            elif self.split == 'val':
                self.images = images[1200*20:1300*20].round()
                self.labels = labels[1200*20:1300*20].round()
            else:
                self.images = images[:1200*20]
                self.labels = labels[:1200*20]

            self.sample_data()

    def sample_data(self):
        if self.split == 'test' or self.split == 'val':
            datasets = []
            targets = []
            num_datasets_per_class = 20 // self.num_data_per_dataset

            for i in range(0, len(self.labels), 20):
                for j in range(num_datasets_per_class):
                    datasets += [self.images[i + j*self.num_data_per_dataset: i + (
                        j + 1)*self.num_data_per_dataset]]
                    targets += [self.labels[i]]

            self.x = datasets
            self.y = targets

        else:
            datasets = []
            targets = []
            num_datasets_per_class = 20 // self.num_data_per_dataset

            for i in range(0, len(self.labels), 20):
                label = self.labels[i]
                for degs in [0, 1, 2]:
                    idxs = np.random.permutation(20) + i
                    current_images = self.images[idxs]
                    if degs == 1:
                        current_images = current_images.transpose((0, 1, 3, 2))[:, :, :, ::-1]
                    if degs == 2:
                        current_images = current_images[:, :, ::-1, ::-1]
                    for j in range(num_datasets_per_class):
                        dataset = []
                        for k in range(j, j + self.num_data_per_dataset):
                            if np.random.uniform(0, 1) < 0.5:
                                img = dilation(current_images[k]).astype(current_images.dtype)
                            else:
                                img = np.copy(current_images[k])

                            dataset += [np.random.binomial(1, p=img, size=img.shape)]

                        dataset = np.stack(dataset, axis=0)
                        datasets += [dataset]
                        targets += [label]

            self.x = datasets
            self.y = targets


    def __len__(self):
        if self.mnist:
            return 1000
        return len(self.x)

    def __getitem__(self, idx):
        if self.mnist:
            label = np.random.randint(0, 10)
            data_label = self.x[self.y == label]
            x_chosen_idx = np.random.choice(len(data_label),
                size=self.num_data_per_dataset, replace=False)

            return {'datasets': data_label[x_chosen_idx],
                    'targets': label}

        return {'datasets': torch.FloatTensor(self.x[idx]),
                'targets': self.y[idx]}
