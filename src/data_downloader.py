import argparse
import shutil
import tarfile
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

def download_utk():
    download_dir = Path(__file__).parent / ".." / "data" / "utk-face"

    if not os.path.exists(download_dir):
        download_dir.mkdir(exist_ok=True, parents=True)

        print("Downloading UTK Face images")
        # Download latest version
        path = kagglehub.dataset_download("jangedoo/utkface-new")

        print("Path to dataset files:", path)

        print(f"Moving files from \n\t'{path}'\nto\n\t'{download_dir}'")
        folder_names = os.listdir(path)
        for folder_name in folder_names:
            shutil.move(os.path.join(path, folder_name), download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='data_downloader',
        description='Downloads the ffhq and utk-face dataset for training and testing'
    )

    download_ffhq()
