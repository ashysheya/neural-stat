import numpy as np
import os
import cv2
from torch.utils.data import Dataset


def get_dataset(opts, split='train'):
    """Function to get instance of FacesDataset given training options."""
    splits = {'train': slice(0, opts.train_num_persons),
              'test': slice(opts.train_num_persons, opts.train_num_persons + opts.test_num_persons),
              'valid': slice(opts.train_num_persons + opts.test_num_persons, 1595)
             }

    return FacesDataset(opts.data_dir, splits[split], opts.num_frames_per_dataset)


class FacesDataset(Dataset):
    """Dataset for youtube faces experiment."""

    def __init__(self, data_dir, persons_split=slice(0, 1595), num_frames_per_dataset=5):
        """
        :param data_dir: string, directory for the cropped and scaled images of the youtube dataset
        :param split: slice, selection of the persons for the dataset
        :param num_frames_per_dataset: int, number of frames per dataset
        """
        self.data_dir = data_dir
        self.persons = os.listdir(self.data_dir)[persons_split]
        self.num_frames_per_dataset = num_frames_per_dataset

        self.videos = []  # name for each video
        self.video_paths = []  # path for each video
        # Take each video for each person in dataset, and save both unique video name and path
        for person in self.persons:
            for video in os.listdir(os.path.join(self.data_dir, person)):
                self.videos.append(person + "_" + video)
                self.video_paths.append(os.path.join(self.data_dir, person, video))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Select num_frames_per_dataset from the given video, without replacement.
        dataset_frames = np.random.choice(os.listdir(video_path), size=self.num_frames_per_dataset, replace=False)
        dataset = []
        for frame in dataset_frames:
            img_scaled = cv2.imread(os.path.join(video_path, frame), cv2.IMREAD_UNCHANGED).astype(np.float64)/255
            # Image must be size (num_channels, w, h) --> transpose, and append to list of images
            dataset.append(img_scaled.transpose(2, 0, 1))

        # Return dataset as array of size (n_frames, num_channels, 64, 64)
        return np.array(dataset)
