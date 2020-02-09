import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

##
#train_num_persons = 1595-200
#test_num_persons = 100
#num_frames_per_dataset=5
#data_dir = "C:/Users/Victor/Documents/Cambridge/Lent/MLMI4/Project/YouTubeFaces/resized_images_DB/"
##

#def get_dataset(split='train'):
def get_dataset(opts, split='train'):
    """Function to get instance of FacesDataset given training options."""
    #splits = {'train': slice(0, train_num_persons),
    #          'test': slice(train_num_persons, train_num_persons + test_num_persons),
    #          'valid': slice(train_num_persons + test_num_persons, 1595)
    #         }

    splits = {'train': slice(0, opts.train_num_persons),
              'test': slice(opts.train_num_persons, opts.train_num_persons + opts.test_num_persons),
              'valid': slice(opts.train_num_persons + opts.test_num_persons, 1595)
             }

#    return FacesDataset(data_dir, slice(0, 30), num_frames_per_dataset)
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
        datasets = []
        # Take each video for each person in dataset, and save both unique video name and path

        show_im = True
        for person in self.persons:
            for video in os.listdir(os.path.join(self.data_dir, person)):
                self.videos.append(person + "_" + video)
                video_path = os.path.join(self.data_dir, person, video)

                # Sample num_frames_per_dataset frames from the given video, without replacement.
                dataset_frames = np.random.choice(os.listdir(os.path.join(self.data_dir, person, video)),
                                                  size=self.num_frames_per_dataset, replace=False)
                dataset = []
                for frame in dataset_frames:
                    frame_path = os.path.join(video_path, frame)

                    img_scaled = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED).astype(np.float64)/255
                    # Image must be size (num_channels, w, h) --> transpose, and append to list of images
                    dataset.append(img_scaled.transpose(2, 0, 1))

                dataset = np.array(dataset)  # Shape is (num_frames_per_dataset, 3, 64, 64)
                datasets.append(dataset)  # Shape is (num_datasets, num_frames_per_dataset, 3, 64, 64)

        self.dataset = np.array(datasets)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return {'datasets': torch.FloatTensor(self.dataset[idx])}
