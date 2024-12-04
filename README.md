# FACE-GAN
FACE-GAN: Facial Appearance Creation &amp; Enhancement using GANs.

## Requirements
### FACE-GAN
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate FACE-GAN`

### Stylegan3
You will also need stylegan3 to generate the images.
* Clone stylegan3 to another directory
* Export stylegans dnnlib and torch_utils directories to PYTHONPATH
  * `export PYTHONPATH="PathToStylegan3"/dnnlib:${PYTHONPATH}`
  * `export PYTHONPATH="PathToStylegan3"/torch_utils:${PYTHONPATH}`
* In case you use the conda environemnt activate it and add the dependencies with this command
  * `conda-develop "PathToStylegan3"/dnnlib`
  * `conda-develop ""PathToStylegan3"/torch_utils`

## Getting started
1. Download the ffhq dataset using the ```data_downloader.py``` script.
2. Create the dataset.json containing the labels for age and gender using the ```dataset_label_creator.py``` script.
   - `python3 ./src/dataset_label_creator.py --target="YourProjectDir"/data/ffhq/images1024x1024`

### 3. Switch to the stylegan3 environment
4. Create the dataset for training using the ```dataset_tool.py```
   - `python3 dataset_tool.py --source="YourProjectDir"/data/ffhq/images1024x1024 --dest="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --resolution=128x128`
5. Start training using this command, change the parameters to your liking:
   - `python3 train.py --outdir="YourProjectDir"/training-runs --cfg=stylegan3-r --data="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --cond=True --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10 --mirror=1 --kimg=25000`

From here on refer to the stylegan3 documentation on how to proceed further.
