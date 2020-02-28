import os
import torch
import numpy as np
import tqdm
from skimage.io import imread
from torch.utils.data import Dataset

def get_dataset(opts, split='train'):
    """Function to get instance of YoutubeDataset given training options."""

    splits = {'train': slice(0, opts.train_num_persons),
              'test': slice(opts.train_num_persons, opts.train_num_persons + opts.test_num_persons),
              'val': slice(opts.train_num_persons + opts.test_num_persons, opts.total_num_persons)
             }

    return YoutubeDataset(opts.data_dir, splits[split], opts.num_data_per_dataset)


class YoutubeDataset(Dataset):
    """Dataset for youtube faces experiment."""

    def __init__(self, data_dir, persons_split=slice(0, 1595), num_data_per_dataset=5):
        """
        :param data_dir: string, directory for the cropped and scaled images of the youtube dataset
        :param split: slice, selection of the persons for the dataset
        :param num_data_per_dataset: int, number of frames per dataset
        """
        self.data_dir = data_dir
        self.persons = os.listdir(self.data_dir)[persons_split]
        self.num_data_per_dataset = num_data_per_dataset

        videos = []  # name for each video
        datasets = []
        # Take each video for each person in dataset, and save both unique video name and path
        for person in tqdm.tqdm(self.persons, desc="{Loading data}"):
            for video in os.listdir(os.path.join(self.data_dir, person)):
                videos.append(person + "_" + video)
                video_path = os.path.join(self.data_dir, person, video)

                # Sample num_data_per_dataset frames from the given video, without replacement.
                dataset_frames = np.random.choice(os.listdir(os.path.join(self.data_dir, person, video)),
                                                  size=self.num_data_per_dataset, replace=False)
                dataset = []
                for frame in dataset_frames:
                    frame_path = os.path.join(video_path, frame)
                    img = np.array(imread(frame_path).transpose(2, 0, 1)).astype(np.float32) / 255

                    # Image must be size (num_channels, w, h) --> transpose, and append to list of images
                    dataset.append(img)

                dataset = np.array(dataset)  # Shape is (num_data_per_dataset, 3, 64, 64)
                datasets.append(dataset)

        self.videos = videos
        self.dataset = np.array(datasets)  # Shape is (num_datasets, num_data_per_dataset, 3, 64, 64)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return {'datasets': torch.FloatTensor(self.dataset[idx])}
