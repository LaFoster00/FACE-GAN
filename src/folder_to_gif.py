import click
import os

from utils import images_to_gif, get_image_paths

@click.command()
@click.option('--folder', type=click.Path(exists=True), required=True)
@click.option('--fps', type=int, default=2)
def folder_to_gif(folder, fps):
    images_to_gif(get_image_paths(folder), os.path.join(folder, "all"), 1000.0 / fps, sort_func=lambda x: x.split('.')[0])


if __name__ == "__main__":
    folder_to_gif()