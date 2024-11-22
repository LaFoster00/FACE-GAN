import json
import pickle
from pathlib import Path
import os
import numpy as np

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
        mapped_labels[feature_json_id] = [1, face_attributes['age'], int(face_attributes['gender'] == 'female')]

    image_paths = [image_path for image_path in image_paths if int(Path(image_path).stem) in mapped_labels]
    labels = [mapped_labels[int(Path(image_path).stem)] for image_path in image_paths]

    return image_paths, labels

def load_ffhq_data(path):
    image_paths = [os.path.join(path, image) for image in os.listdir(path) if Path(image).suffix == '.png']
    image_paths, labels = generate_ffhq_labels(image_paths)
    return np.array(image_paths), np.array(labels)

if __name__ == '__main__':
    image_paths, labels = load_ffhq_data('../data/ffhq/images256x256')