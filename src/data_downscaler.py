import os
from multiprocessing import Pool, Manager

from pathlib import Path
from PIL import Image
import tqdm


def scale_down_file(input_path, input_file, output_path, resolution):
    try:
        original_image = Image.open(os.path.join(input_path, input_file))
        resized_image = original_image.resize((resolution, resolution), Image.Resampling.LANCZOS)
        resized_image.save(os.path.join(output_path, input_file))
    except FileNotFoundError as e:
        print(f"File {input_file} not found. Could not scale down image. \n{e}")

if __name__ == '__main__':
    download_dir = Path(__file__).parent / '..' / 'data' / 'ffhq' / 'images1024x1024'
    download_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(__file__).parent / '..' / 'data' / 'ffhq' / 'images256x256'
    output_dir.mkdir(parents=True, exist_ok=True)

    files = os.listdir(download_dir)
    resolution = 256

    # Prepare the arguments as a list of tuples
    args_list = [(download_dir, file, output_dir, resolution) for file in files if Path(file).suffix == '.png']

    with Pool(16) as p:
        result = p.starmap(scale_down_file, args_list)

    print('\nProcessing complete!')
