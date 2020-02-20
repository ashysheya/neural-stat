import torch
import numpy as np
import PIL.ImageOps    
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, Omniglot
from skimage.morphology import binary_dilation, disk
from skimage.transform import resize

### add non-binary dilation blabla
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

    def __len__(self):
        if self.mnist:
            return 1000
        elif self.split == 'train':
            return 1200*(20//self.num_data_per_dataset)
        elif self.split == 'test':
            return 323*(20//self.num_data_per_dataset)
        else:
            return 100*(20//self.num_data_per_dataset)

    def __getitem__(self, idx):
        if self.mnist and self.split == 'test':
            label = np.random.randint(0, 10)
            data_label = self.x[self.y == label]
            x_chosen_idx = np.random.choice(len(data_label),
                size=self.num_data_per_dataset, replace=False)

            return {'datasets': data_label[x_chosen_idx],
                    'targets': label}
        elif self.split == 'train':
            if len(self.dataset_background)//self.num_data_per_dataset <= idx:
                idx = idx - len(self.dataset_background)//self.num_data_per_dataset
                dataset = self.dataset_eval
            else:
                dataset = self.dataset_background

            rotation_angle = np.random.uniform(0, 360)
            images = []
            for i in range(self.num_data_per_dataset):
                image, label = dataset[idx*self.num_data_per_dataset + i]
                image = PIL.ImageOps.invert(image)
                images += [np.asarray(image.resize((28, 28)).rotate(rotation_angle))/255]

            images = np.stack(images, axis=0)
            if np.random.uniform(0, 1) < 0.5:
                images = images[:, :, ::-1]
            if np.random.uniform(0, 1) < 0.5:
                images = images[:, ::-1]
            images = np.random.binomial(1, p=images, size=images.shape).astype(np.bool)

            selem = np.random.randint(0, 2)
            # selem = 0

            if selem > 0:
                for i in range(len(images)):
                    images[i] = binary_dilation(images[i], selem=disk(selem))

            perm = np.random.permutation(self.num_data_per_dataset)

            return {'datasets': torch.FloatTensor(images[perm][:, None]),
                    'targets': label}

        elif self.split == 'test':
            start_index = 1300*(20//self.num_data_per_dataset)
            start_index -= len(self.dataset_background)//self.num_data_per_dataset
            idx = start_index + idx
            dataset = self.dataset_eval

        elif self.split == 'val':
            start_index = 1200 * (20//self.num_data_per_dataset)
            start_index -= len(self.dataset_background)//self.num_data_per_dataset
            idx = start_index + idx
            dataset = self.dataset_eval

        images = []
        for i in range(self.num_data_per_dataset):
            image, label = dataset[idx*self.num_data_per_dataset + i]
            image = PIL.ImageOps.invert(image)
            images += [(np.asarray(image.resize((28, 28)))/255).round()]
        images = np.stack(images, axis=0)

        return {'datasets': torch.FloatTensor(images[:, None]),
                'targets': label}

