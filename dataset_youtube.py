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

    return YoutubeDataset(opts.data_dir, splits[split], opts.num_data_per_dataset, 
    	opts.n_channels, opts.use_labels, opts.num_labels)


class YoutubeDataset(Dataset):
    """Dataset for youtube faces experiment."""

    def __init__(self, data_dir, persons_split=slice(0, 1595), num_data_per_dataset=5, 
    	n_channels=3, use_labels=False, num_labels=8):
        """
        :param data_dir: string, directory for the cropped and scaled images of the youtube dataset
        :param split: slice, selection of the persons for the dataset
        :param num_data_per_dataset: int, number of frames per dataset
        """
        self.data_dir = data_dir
        self.use_labels = use_labels
        self.num_labels = num_labels
        # define labels dir:
        if self.use_labels:
            parts = self.data_dir.split('/')
            parts = parts[:-1] if parts[-1] == '' else parts
            parts[-1] = 'emotions_labels'
            self.labels_dir = '/'.join(parts)

        self.persons = os.listdir(self.data_dir)[persons_split]
        self.num_data_per_dataset = num_data_per_dataset
        self.n_channels = n_channels

        videos = []  # name for each video
        datasets = []
        if self.use_labels:
            labels = []
        # Take each video for each person in dataset, and save both unique video name and path
        for person in tqdm.tqdm(self.persons, desc="{Loading data}"):
            for video in os.listdir(os.path.join(self.data_dir, person)):
                videos.append(person + "_" + video)
                video_path = os.path.join(self.data_dir, person, video)

                if self.use_labels:
                    label_path = os.path.join(self.labels_dir, person, video)
                    label_file = os.listdir(label_path)[0]
                    with open(os.path.join(label_path, label_file), 'r') as f:
                        current_label = int(float(f.read()))
                        label = np.zeros(self.num_labels)
                        label[current_label] = 1.0

                # Sample num_data_per_dataset frames from the given video, without replacement.
                if len(os.listdir(os.path.join(self.data_dir, person, video))) < self.num_data_per_dataset:
                    print("Not enough frames in folder ", os.path.join(self.data_dir, person, video))

                dataset_frames = np.random.choice(os.listdir(os.path.join(self.data_dir, person, video)),
                                                  size=self.num_data_per_dataset, replace=False)
                dataset = []
                for frame in dataset_frames:
                    frame_path = os.path.join(video_path, frame)
                    img = imread(frame_path)
                    img = np.array(img.reshape(img.shape[0], img.shape[1], self.n_channels)
                                   .transpose(2, 0, 1)).astype(np.float32) / 255
                    # Image must be size (num_channels, w, h) --> transpose, and append to list of images
                    dataset.append(img)

                dataset = np.array(dataset)  # Shape is (num_data_per_dataset, 3, 64, 64)
                datasets.append(dataset)
                if self.use_labels:
                    labels.append(label)

        self.videos = videos
        self.dataset = np.array(datasets)  # Shape is (num_datasets, num_data_per_dataset, 3, 64, 64)
        if self.use_labels:
            self.labels = np.array(labels)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.use_labels:
            return {'datasets': torch.FloatTensor(self.dataset[idx]), 
                    'labels': torch.FloatTensor(self.labels[idx])}
        return {'datasets': torch.FloatTensor(self.dataset[idx])}
