from torch.utils.data import Dataset
from utils import generate_1d_datasets


def get_dataset(opts):
    """Function to get instance of SyntheticDataset given training options."""
    return SyntheticDataset(opts.num_datasets_per_distr, opts.num_data_per_dataset)


class SyntheticDataset(Dataset):
    """Dataset for 1d synthetic data experiment."""

    def __init__(self, num_datasets_per_distr=2500, num_data_per_dataset=200):
        """
        :param num_datasets_per_distr: int, number of datasets per distribution to generate
        :param num_data_per_dataset: int, number of datapoints per dataset
        """
        self.datasets, self.targets = generate_1d_datasets(num_datasets_per_distr,
                                                           num_data_per_dataset)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx], self.targets[idx]
