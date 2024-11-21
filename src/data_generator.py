import json
import os

from keras import utils
from pathlib import Path
import numpy as np

import pickle

class DataGenerator(utils.Sequence):
    def __init__(self, image_paths, labels, for_fitting=True, batch_size=32, dim=(256, 256),
                 n_channels=1, n_classes=10, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.indices = np.arange(len(self.image_paths))
        self.for_fitting = for_fitting
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x = self._load_batch_images(indices)

        if self.for_fitting:
            y = self._load_batch_labels(indices)
            return x, y
        else:
            return x

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_batch_images(self, indices):
        images = []
        for image_path in self.image_paths[indices]:
            images.append(self._load_image(image_path))

    def _load_batch_labels(self, indices):
        return self.labels[indices]

    def _load_image(self, image_path):
        return utils.img_to_array(utils.load_img(image_path))