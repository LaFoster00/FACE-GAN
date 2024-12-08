import click
import wandb
from PIL import Image
from pathlib import Path
import os
import concurrent.futures
from tqdm import tqdm


def images_to_gif(image_fnames, fname, duration):
    image_fnames.sort(key=lambda x: int(x.split('_')[-2]))  # sort by step
    frames = [Image.open(image) for image in image_fnames]
    frame_one = frames[0]
    frame_one.save(f'{fname}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=duration, loop=1)

def download_file(file, root):
    if file.name.endswith('.png'):
        file.download(root=root, exist_ok=True)

def get_image_paths(directory):
    image_extensions = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(Path(directory).rglob(ext))

    return [str(path) for path in image_paths]

@click.command()
@click.option('--run-path', required=True, type=str, default=None, help='Path to wandb run, from wandb run info.')
@click.option('--fps', type=int, default=2, help='Number of frames per second in the output gif.')
def run_to_gif(run_path: str = None,
               fps: int = 2):
    if run_path is None or run_path == '':
        raise ValueError("run_path cannot be empty")

    api = wandb.Api()
    run = api.run(run_path)

    run_files_path = Path(__file__).parent / "../wandb_run_data" / run_path

    files = list(run.files())
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a progress bar
        with tqdm(total=len(files), desc="Downloading files") as pbar:
            # Submit tasks to the executor
            futures = {
                executor.submit(download_file, file, run_files_path): file
                for file in files
            }

            # Update the progress bar as each task completes
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

    image_paths = get_image_paths(run_files_path)
    images_to_gif(image_paths, run_files_path / "all", 1000.0 / fps)
    print(f"Gif saved to {run_files_path.absolute() / 'all.gif'}")

if __name__ == "__main__":
    run_to_gif()