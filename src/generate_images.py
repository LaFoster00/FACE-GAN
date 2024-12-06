import pickle
from pathlib import Path

import click
import torch
from PIL import Image
import numpy as np

def generate_images(generator, labels, batch_size=32):
    generated_images = []
    print(f"Generating {len(labels)} images with batch size {batch_size}. Total batches {len(labels)//batch_size}.")
    # Iterate through chunks of size 32
    for i in range(0, len(labels), batch_size):
        print(f"Generating images for batch {i//batch_size}")
        labels_batch = labels[i:i + batch_size]
        labels_batch_tensor = torch.tensor(labels_batch, dtype=torch.int32).cuda()
        latent_batch_tensor = torch.randn([len(labels_batch), generator.z_dim]).cuda()
        images = generator(latent_batch_tensor, labels_batch_tensor)
        images = images.permute(0, 2, 3, 1)
        images_cpu = torch.clamp(((images + 1.0) / 2.0) * 255, min=0.0, max=255.0).cpu()
        for index, image_cpu in enumerate(images_cpu):
            image_cpu = image_cpu.numpy().astype(np.uint8)
            generated_images.append((labels_batch[index], image_cpu))
    return generated_images

@click.command()
@click.option(
    '--network-path',
    type=click.Path(exists=True, dir_okay=True, path_type=Path),
    default=Path(__file__).parent / "../models",
    help="Path to the trained network pickle file or directory (alphabetically last)."
)
@click.option(
    '--num-images',
    type=int,
    default=24000,
    help="Total number of images to generate."
)
@click.option(
    '--min-age',
    type=int,
    default=0,
    help="Minimum age (inclusive) for generated images."
)
@click.option(
    '--max-age',
    type=int,
    default=80,
    help="Maximum age (exclusive) for generated images."
)
@click.option(
    '--output-path',
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=Path(__file__).parent / "../generated_images/",
    help="Directory where generated images will be saved."
)
def main(network_path=Path(__file__).parent / "../models",
         num_images=24000,
         min_age=0, #Inclusive
         max_age=80, #Exlusive
         output_path=Path(__file__).parent / "../generated_images/"):
    if network_path.is_dir():
        models = [model for model in network_path.rglob("*.pkl")]
        models.sort()
        network_path = Path(models[-1])
        print(f"Network path: {network_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(network_path, "rb") as f:
        g = pickle.load(f)["G_ema"].cuda()

    labels = []
    num_images_per_age = num_images // (max_age - min_age)
    for age in range(min_age, max_age):
        for i in range(num_images_per_age//2):
            labels.append([age, 0]) # Male
            labels.append([age, 1]) # Female

    images = generate_images(g, labels, batch_size=32)
    for index, label_image in enumerate(images):
        label, image = label_image
        Image.fromarray(image).save(output_path / f"{label[0]}_{label[1]}__{index}.jpg", format="JPEG", quality=90)


if __name__ == "__main__":
    main()
