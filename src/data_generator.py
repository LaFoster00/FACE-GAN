import json
import os

from keras import utils
from pathlib import Path
import numpy as np

import pickle

def generate_labels(image_paths):
    # Check if the jsons already exist in a serialized form
    serialized_feature_json_dir = Path(__file__).parent / '../data/tmp'
    serialized_feature_json_dir.mkdir(parents=True, exist_ok=True)
    serialized_feature_json_id = "features_json.pickle"
    print(f"Trying to deserialize feature json from {serialized_feature_json_dir/serialized_feature_json_id}")
    if os.path.exists(serialized_feature_json_dir / serialized_feature_json_id):
        feature_jsons = pickle.load(open(serialized_feature_json_dir / serialized_feature_json_id, "rb"))
        print("Successfully deserialized feature json")
    else:
        print("Serialized feature json not found, loading from source.")
        feature_jsons_path = Path(__file__).parent / '../third_party/ffhq-features-dataset/json'
        if not os.path.exists(feature_jsons_path):
            raise Exception(f"{feature_jsons_path} does not exist! Make sure to init and update all submodules." )
        feature_json_ids = os.listdir(feature_jsons_path)
        feature_jsons = {}
        missing_jsons = 0
        for feature_json_id in feature_json_ids:
            with open(feature_jsons_path / feature_json_id, 'r') as f:
                feature_json = json.load(f)
                if len(feature_json) > 0:
                    feature_jsons[int(Path(feature_json_id).stem)] = feature_json
                else:
                    missing_jsons += 1
                    print(f"{missing_jsons}: {feature_json_id} has no features.")

    print("Serializing features json")
    pickle.dump(
        feature_jsons,
        open(serialized_feature_json_dir / serialized_feature_json_id, "wb"),
        protocol=pickle.HIGHEST_PROTOCOL)

    print("Mapping features from jsons to png name/number")
    mapped_labels = {}
    for feature_json_id, feature_json in feature_jsons.items():
        face_attributes = feature_json[0]['faceAttributes']
        mapped_labels[feature_json_id] = (1, int(face_attributes['gender'] == 'female'), face_attributes['age'])

    labels = []
    for image_path in image_paths:
        image_id = int(Path(image_path).stem)
        labels.append(mapped_labels[image_id])

    return labels

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