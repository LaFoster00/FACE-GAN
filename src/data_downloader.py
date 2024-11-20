import subprocess
import sys
from pathlib import Path
import kagglehub
import shutil
import os

if __name__ == '__main__':
    download_dir = Path(__file__).parent / '..' / 'data' / 'ffhq'
    print(download_dir)
    download_dir.mkdir(exist_ok=True, parents=True)

    # Download latest version
    path = kagglehub.dataset_download("gibi13/flickr-faces-hq-dataset-ffhq")

    print("Path to dataset files:", path)

    print(f"Moving files from \n\t'{path}'\nto\n\t'{download_dir}'")
    folder_names = os.listdir(path)
    for folder_name in folder_names:
        shutil.move(os.path.join(path, folder_name), download_dir)