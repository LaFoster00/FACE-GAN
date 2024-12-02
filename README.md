# FACE-GAN
FACE-GAN: Facial Appearance Creation &amp; Enhancement using GANs.

## Requirements
* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later.  (Why is a separate CUDA toolkit installation required?  See [Troubleshooting](./docs/troubleshooting.md#why-is-cuda-toolkit-installation-necessary)).
* GCC 7 or later (Linux) or Visual Studio (Windows) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your StyleGAN3 Python environment:
  - `conda env create -f environment.yml`
  - `conda activate stylegan3`
* Docker users:
  - Ensure you have correctly installed the [NVIDIA container runtime](https://docs.docker.com/config/containers/resource_constraints/#gpu).
  - Use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

See [Troubleshooting](./docs/troubleshooting.md) for help on common installation and run-time problems.

## Getting started
1. Download the ffhq dataset using the ```data_downloader.py``` script.
2. Create the dataset.json containing the labels for age and gender using the ```dataset_label_creator.py``` script.
   - `python3 ./src/dataset_label_creator.py --target="YourProjectDir"/data/ffhq/images1024x1024`
3. Create the dataset for training using the ```dataset_tool.py``` from third_party/stylegan3 using this command
   - `python3 ./third_party/stylegan3/dataset_tool.py --source="YourProjectDir"/data/ffhq/images1024x1024 --dest="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --resolution=128x128`
4. Start training using this command, change the parameters to your liking:
   - `python3 ./third_party/stylegan3/train.py --outdir="YourProjectDir"/training-runs --cfg=stylegan3-r --data="YourProjectDir"/datasets/ffhq_cond_stylegan3_128x128.zip --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10 --mirror=1 --kimg=25000`

From here on refer to the stylegan3 documentation on how to proceed further.
