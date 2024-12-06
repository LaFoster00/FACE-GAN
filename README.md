# FACE-GAN
FACE-GAN: Facial Appearance Creation &amp; Enhancement using GANs.

## Requirements
### FACE-GAN
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 (Should be part of the conda environment) and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later (Should be part of the conda environment).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 or Anaconda to create and activate your FACE-GAN Python environment:
  - `conda env create -f environment.yml`
  - `conda activate FACE-GAN`

### Stylegan3
You will also need stylegan3 to generate the images.
1. Clone stylegan3 to another directory 
* Export stylegans dnnlib and torch_utils directories to PYTHONPATH
  * `export PYTHONPATH="PathToStylegan3":${PYTHONPATH}`
* In case you use the conda environemnt activate it and add the dependencies with this command
  * `conda-develop "PathToStylegan3"`

## Getting started
1. Download the ffhq dataset using the ```data_downloader.py``` script.
2. Create the dataset.json containing the labels for age and gender using the ```dataset_label_creator.py``` script.
   - `python3 ./src/dataset_label_creator.py --target="YourProjectDir"/data/ffhq/images1024x1024`
3. Create the dataset for training using the ```dataset_tool.py```
   - `python3 "PathToStylegan3"/dataset_tool.py --source="YourProjectDir"/data/ffhq/images1024x1024 --dest="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --resolution=128x128`
4. Start training using this command, change the parameters to your liking:
   - `python3 "PathToStylegan3"/train.py --outdir="YourProjectDir"/training-runs --cfg=stylegan3-r --data="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --cond=True --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10 --mirror=1 --kimg=25000 --metrics=none`

From here on refer to the stylegan3 documentation on how to proceed.

## Generating images
To generate images with an even distribution of ages use `src/generate_images.py`
* `python script_name.py --network-path /path/to/network.pkl --num-images 10000 --min-age 18 --max-age 60 --output-path /path/to/output`
