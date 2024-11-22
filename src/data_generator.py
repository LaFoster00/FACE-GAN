import json
import os

from keras import utils
from pathlib import Path
import numpy as np
import threading
import queue
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import pickle

class DataGenerator(utils.Sequence):
    def __init__(self,
                 image_paths,
                 labels,
                 label_structure,
                 for_fitting=True,
                 batch_size=32,
                 shuffle=True,
                 prefetch_batches=64,
                 max_workers=None,
                 dim=(224, 224)):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.label_structure = label_structure
        self.indices = np.arange(len(self.image_paths))
        self.for_fitting = for_fitting
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dim = dim
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Prefetching setup
        self.prefetch_queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_event = threading.Event()
        self.current_index = 0

        # Concurrent loading configuration
        self.max_workers = max_workers or cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Start prefetching thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

        self.on_epoch_end()

    def _prefetch_worker(self):
        """Background thread for batch prefetching with parallel image loading"""
        while not self.stop_event.is_set():
            try:
                # Check if we can add more batches to the queue
                if self.prefetch_queue.qsize() < self.prefetch_queue.maxsize:
                    # Calculate batch indices
                    start = self.current_index
                    end = start + self.batch_size

                    # Wrap around if needed
                    if end > len(self.image_paths):
                        if self.shuffle:
                            np.random.shuffle(self.indices)
                        self.current_index = 0
                        start = 0
                        end = self.batch_size

                    # Get batch indices
                    batch_indices = self.indices[start:end]

                    # Parallel image loading
                    futures = []
                    for index, image_path in enumerate(self.image_paths[batch_indices]):
                        futures.append(
                            self.executor.submit(self._load_image, image_path, index)
                        )

                    # Collect images
                    images = [None] * len(batch_indices)
                    for future in as_completed(futures):
                        image, index = future.result()
                        images[index] = image

                    x = np.array(images)

                    # Load labels if fitting
                    if self.for_fitting:
                        y = self._load_batch_labels(batch_indices)
                        self.prefetch_queue.put((x, y), block=True)
                    else:
                        self.prefetch_queue.put(x, block=True)

                    # Update current index
                    self.current_index = end
                else:
                    # Prevent busy waiting
                    threading.Event().wait(0.1)
            except Exception as e:
                print(f"Prefetch error: {e}")
                break

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Retrieve prefetched batch
        return self.prefetch_queue.get()

    def on_epoch_end(self):
        # Clear existing queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except queue.Empty:
                break

        # Reset current index
        self.current_index = 0

        # Restart prefetching thread
        self.stop_event.set()
        if hasattr(self, 'prefetch_thread'):
            self.prefetch_thread.join()

        # Reset stop event and restart thread
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def _load_batch_images(self, indices):
        """Load a batch of images"""
        images = []
        for image_path in self.image_paths[indices]:
            images.append(self._load_image(image_path))
        return np.array(images)

    def _load_batch_labels(self, indices):
        """Load batch labels"""
        if self.label_structure is None:
            return self.labels[indices]
        else:
            labels = self.labels[indices]
            split_labels = []

            # Split the labels into their separate slices
            for index, label_key in enumerate(self.label_structure):
                split_labels.append(labels[:, index])

            # Add the labels to the labels dictionary
            mapped_labels = {}
            for index, label_key in enumerate(self.label_structure):
                mapped_labels[label_key] = split_labels[index]
            return mapped_labels

    def _load_image(self, image_path, index):
        """Load and preprocess single image"""
        return utils.img_to_array(utils.load_img(image_path, target_size=self.dim, interpolation="lanczos"), dtype='uint8'), index

    def __del__(self):
        """Clean up resources"""
        self.stop_event.set()
        if hasattr(self, 'prefetch_thread'):
            self.prefetch_thread.join()
        self.executor.shutdown(wait=True)