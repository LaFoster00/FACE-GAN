import argparse
import shutil
import gdown
import tarfile
import os
from pathlib import Path

import kagglehub

def download_and_extract(file_id, output_folder):
    """
    Downloads a file from Google Drive and extracts it if it's a tar.gz file.

    Args:
    file_id (str): The Google Drive file ID to download.
    output_folder (str): The folder to extract the files into.
    """
    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Temporary location for the downloaded file
    temp_file = output_folder / "temp.tar.gz"

    try:
        # Download the file from Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(temp_file), quiet=False)

        # Extract if it's a tar.gz file
        if temp_file.exists() and temp_file.suffix == '.gz':
            with tarfile.open(str(temp_file), 'r:gz') as tar:
                # Extract all files to the output folder
                tar.extractall(path=str(output_folder))
            print(f"Successfully extracted to {output_folder}")

    except Exception as e:
        print(f"Error processing file: {e}")

    finally:
        # Clean up the temporary file
        if temp_file.exists():
            temp_file.unlink()

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
    # Argument parsing for various options
    parser.add_argument('-noutkface', action='store_true', help='Do not download the utk-face dataset')
    parser.add_argument('--noffhq', action='store_true', help='Do not download the ffhq dataset')
    parser.print_usage()
    args = parser.parse_args()

    # Download UTK Face dataset if not skipped
    if not args.noutkface:
        download_utk()

    if not args.noffhq:
        download_ffhq()
