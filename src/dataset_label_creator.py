import os
from operator import contains
from pathlib import Path

import click

from utils import generate_ffhq_labels

import json

@click.command()
@click.option('--target', help='Directory or archive name for target dataset', required=True, metavar='PATH')
def create_dataset_labels(
        target: str):
    print("Generating dataset labels from ffhq-features-dataset")
    target = Path(target)
    images = os.listdir(target)
    if len(images) == 0 or any([os.path.isdir(image) for image in images]):
        "Recursive or empty image directories not supported. Make sure to specify a folder with all the images in it."
        exit(1)
    images = [os.path.join(target, x) for x in images if Path(x).suffix == ".png"]
    images_mapped, labels = generate_ffhq_labels(images)
    images_mapped_set = set(images_mapped)

    images_without_label = [image for image in images if not contains(images_mapped_set, image)]
    for image in images_without_label:
     os.remove(image)

    dataset_dict = {}
    mapped_labels = []
    for index, image in enumerate(images_mapped):
        image_labels = labels[index]
        mapped_labels.append([Path(image).name, [image_labels[1], image_labels[2]]])
    dataset_dict["labels"] = mapped_labels
    with open(target / "dataset.json", "w", encoding='utf-8') as f:
        json.dump(dataset_dict, f, ensure_ascii=False, indent=4)
    print(f"Exported dataset labels to {target}/dataset.json.")


if __name__ == '__main__':
    create_dataset_labels()
