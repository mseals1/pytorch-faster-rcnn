from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


class AphyllaDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transf=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.bbox_coords_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transf

    def __len__(self):
        return len(self.bbox_coords_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.bbox_coords_frame.iloc[idx, 0])
        print(img_name)
        image = io.imread(img_name)
        bbox_coords = self.bbox_coords_frame.iloc[idx, 4:8].as_matrix()
        bbox_coords = bbox_coords.astype('float').reshape(-1, 2)
        bbox_coords = np.vstack((bbox_coords, [bbox_coords[0][0], bbox_coords[1][1]]))
        bbox_coords = np.vstack((bbox_coords, [bbox_coords[1][0], bbox_coords[0][1]]))
        sample = {'image': image, 'bbox_coords': bbox_coords}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, bbox_coords = sample['image'], sample['bbox_coords']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        bbox_coords = bbox_coords * [new_w / w, new_h / h]

        return {'image': img, 'bbox_coords': bbox_coords}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, bbox_coords = sample['image'], sample['bbox_coords']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        bbox_coords = bbox_coords - [left, top]

        return {'image': image, 'bbox_coords': bbox_coords}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbox_coords = sample['image'], sample['bbox_coords']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbox_coords': torch.from_numpy(bbox_coords)}


transformed_dataset = AphyllaDataset(csv_file='aphylla/image_annotations.csv',
                                     root_dir='aphylla/annotated_images_only/',
                                     transf=transforms.Compose([ToTensor()]))

for i in range(len(transformed_dataset)):
    s = transformed_dataset[i]

    print(i, s['image'].size(), '\n', s['bbox_coords'])

    if i == 3:
        break
