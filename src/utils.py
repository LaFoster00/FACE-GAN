import json
import pickle
import random
from pathlib import Path
import os
import numpy as np
from keras import callbacks, models, ops
import keras
import matplotlib.pyplot as plt
import wandb

def append_list(a, b):
    """
    Appends the contents of list `b` to list `a`.

    Parameters:
    - a: list, the target list to be appended to.
    - b: list, the list to append.
    """
    for i in range(len(b)):
        a.append(b[i])

def generate_ffhq_labels(image_paths):
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


    if not os.path.exists(serialized_feature_json_dir / serialized_feature_json_id):
        print("Serializing features json")
        pickle.dump(
            feature_jsons,
            open(serialized_feature_json_dir / serialized_feature_json_id, "wb"),
            protocol=pickle.HIGHEST_PROTOCOL)

    print("Mapping features from jsons to png name/number")
    mapped_labels = {}
    for feature_json_id, feature_json in feature_jsons.items():
        face_attributes = feature_json[0]['faceAttributes']
        mapped_labels[feature_json_id] = [face_attributes['age'], int(face_attributes['gender'] == 'female')]

    image_paths = [image_path for image_path in image_paths if int(Path(image_path).stem) in mapped_labels]
    labels = [mapped_labels[int(Path(image_path).stem)] for image_path in image_paths]

    return image_paths, labels

def load_ffhq_data(path):
    image_paths = [os.path.join(path, image) for image in os.listdir(path) if Path(image).suffix == '.png']
    image_paths.sort()
    image_paths, labels = generate_ffhq_labels(image_paths)
    return image_paths, labels

def load_utk_face_data(path):
    image_paths = [os.path.join(path, image) for image in os.listdir(path) if Path(image).suffix == '.jpg']
    image_paths.sort()

    labels = []
    invalid_images = []

    # If no serialized data, load images manually
    for index, file_path in enumerate(image_paths):
        filename = Path(file_path).name
        if not filename.endswith('.jpg'):
            continue

        # Split the filename to extract age and gender
        parts = filename.split('_')
        if len(parts) < 4:
            invalid_images.append(index)
            continue

        # Extract age and gender from filename
        try:
            age = int(parts[0])
        except ValueError:
            print(f"Age {parts[0]} is not a valid number. File '{filename}'")
            invalid_images.append(index)
            continue

        try:
            gender = int(parts[1])
        except ValueError:
            print(f"Gender {parts[1]} is not a valid number. File '{filename}'")
            invalid_images.append(index)
            continue

        labels.append([age, gender])  # Label '1' for face images

    # Remove invalid images
    for invalid_image in invalid_images:
        image_paths.pop(invalid_image)

    return image_paths, labels

def load_face_data(utk_face_path, ffhq_path, with_ffhq = True, with_utk = True):
    if with_ffhq:
        images, labels = load_ffhq_data(ffhq_path)
    else:
        images, labels = [], []

    if with_utk:
        utk_images, utk_labels = load_utk_face_data(utk_face_path)

        append_list(images, utk_images)
        append_list(labels, utk_labels)

    return np.array(images), np.array(labels, dtype=np.uint8)

def getGeneratorInputData(latent_dim, age=None, gender=None, numImages=1):
    data = []
    if age is None:
        age = random.randint(0, 100)
    if gender is None:
        gender = random.randint(0, 1)

    for i in range(numImages):
        noise = keras.random.normal(shape=(latent_dim,))
        noise_and_labels = ops.concatenate([noise, np.array([age]), np.array([gender])])
        data.append(noise_and_labels)

    return np.array(data)

def getGeneratorSimpleData(latent_dim, nunImages=1):
    data = []
    for i in range(nunImages):
        noise = keras.random.normal(shape=(latent_dim,))
        data.append(noise)
    return np.array(data)


class GeneratorTestCallback(callbacks.Callback):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        generator : models.Model = self.model.generator
        results = generator.predict(getGeneratorSimpleData(self.latent_dim))
        for result in results:
            plt.axis('off')
            plt.imshow(result)
            plt.show()
        try:
            wandb.log({
                'generator_images': [wandb.Image(img, caption=f'Image {i} from epoch {epoch}')
                                     for i, img in enumerate(results)]
            })
        except Exception:
            pass

def log2(x):
    return int(np.log2(x))