import json
from pathlib import Path
import os

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

    print("Mapping features from jsons to png name/number")
    mapped_labels = {}
    for feature_json_id, feature_json in feature_jsons.items():
        face_attributes = feature_json[0]['faceAttributes']
        mapped_labels[feature_json_id] = [1, face_attributes['age'], int(face_attributes['gender'] == 'female')]

    image_paths = [image_path for image_path in image_paths if int(Path(image_path).stem) in mapped_labels]
    labels = [mapped_labels[int(Path(image_path).stem)] for image_path in image_paths]

    return image_paths, labels
