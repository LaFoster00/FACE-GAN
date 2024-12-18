import argparse
import shutil
import os
from pathlib import Path

import kagglehub

def download_ffhq():
    download_dir = Path(__file__).parent / '..' / 'data' / 'ffhq'
    if not os.path.exists(download_dir):
        download_dir.mkdir(exist_ok=True, parents=True)

        print("Downloading FFHQ images")
        # Download latest version
        path = kagglehub.dataset_download("gibi13/flickr-faces-hq-dataset-ffhq")

        print("Path to dataset files:", path)

        print(f"Moving files from \n\t'{path}'\nto\n\t'{download_dir}'")
        folder_names = os.listdir(path)
        for folder_name in folder_names:
            shutil.move(os.path.join(path, folder_name), download_dir)
    else:
        print("FFHQ Data already downloaded. ")

if __name__ == '__main__':
    download_ffhq()
