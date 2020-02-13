import importlib
import tqdm
from torch.utils.data import DataLoader

dataset_module = importlib.import_module('_'.join(['dataset', 'faces']))  # imports dataset_sythetic.py

## Initialise an object of type SyntheticDataset, which can be used as input to DataLoader. It has members .dataset,
## which contains datasets for each distribution type and is of dimension (4*2500, 200, 1), .targets of size (1, 4*2500)
## which contains the corresponding targets, and .means and .variances which are of size (1, 4*2500) each and contain
## the distribution means and variances respectively.
train_dataset = dataset_module.get_dataset(split='train')
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
print(train_dataloader)
print(train_dataset.dataset.shape)

for epoch in tqdm.tqdm(range(2)):
    ## Calls __getitem__ in dataset_synthetic: takes one batch of data, and returns a dictionary with 'datasets'
    ## (16, 200, 1), 'targets' (1, 16), 'means' (1, 16) and 'variances' (1, 16)
    for data_dict in train_dataloader:
        data = data_dict['datasets']  ## Size is (16, 200, 1)
        print(data.shape)
