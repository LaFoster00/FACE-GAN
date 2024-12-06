# FACE-GAN
FACE-GAN: Facial Appearance Creation &amp; Enhancement using GANs.

FACE-GAN uses nvidia's stylegan3 model to train a conditional gan that can generate face images with a specific age and gender.

For an example of how to generate images using a finished model look at `src/generate_images.py` 

## Requirements
### FACE-GAN
* Linux is supported.
* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* 64-bit Python 3.8 (Should be part of the conda environment) and PyTorch 1.9.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.1 or later (Should be part of the conda environment).
* GCC 7 or later (Linux) compilers.  Recommended GCC version depends on CUDA version, see for example [CUDA 11.4 system requirements](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-installation-guide-linux/index.html#system-requirements).
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 or Anaconda to create and activate your FACE-GAN Python environment:
  - `conda env create -f environment.yml`
  - `conda activate FACE-GAN`

### Stylegan3
You will also need stylegan3 to generate the images.
1. Init, update all submodules. Stylegan3 will be downloaded now.
2. Export stylegans dnnlib and torch_utils directories to the python environment using this command
   * `conda-develop third_party/stylegan3`

## Getting started
1. Download the ffhq dataset using [data_downloader.py](./src/data_downloader.py).
2. Create the dataset.json containing the labels for age and gender using the [dataset_label_creator.py](./src/dataset_label_creator.py) script.
   - `python src/dataset_label_creator.py --target=data/ffhq/images1024x1024`
3. Create the dataset for training using the [dataset_tool.py](./third_party/stylegan3/dataset_tool.py). Change the resolution to whatever your target might be.
   - `python  third_party/stylegan3/dataset_tool.py --source=data/ffhq/images1024x1024 --dest=datasets/ffhq_cond_stylegan3_128x128.zip --resolution=128x128`
4. Start training using [train.py](./third_party/stylegan3/train.py), change the parameters to your liking (--cfg=stylegan2 will train much faster):
   - `python third_party/stylegan3/train.py --outdir=training-runs --cfg=stylegan3-r --data=datasets/ffhq_cond_stylegan3_128x128.zip --cond=True --gpus=1 --batch=32 --gamma=0.5 --batch-gpu=16 --snap=10 --mirror=1 --kimg=25000 --metrics=none`
   - **Note:** It is important to update the gamma value to an appropriate one for the output resolution you are going for. See [configs.md](./third_party/stylegan3/docs/configs.md)

From here on refer to the stylegan3 documentation on how to proceed.

## Generating images
After training your model till the desired quality is reached, generate images with an even distribution of ages and genders using [generate_images.py](./src/generate_images.py)
* `python script_name.py --network-path /path/to/network.pkl --num-images 10000 --min-age 18 --max-age 60 --output-path /path/to/output`
* Alternatively if you want to use the latest model (alphabetical sorting) just specify the directory with the models in it, e.g.
  * `python script_name.py --network-path /path/to/models --num-images 10000 --min-age 18 --max-age 60 --output-path /path/to/output`

### Using networks from Python

You can use pre-trained networks in your own Python code as follows:

```.python
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()                                              # torch.nn.Module
z = torch.randn([1, G.z_dim]).cuda()                                                # latent codes
c = torch.tensor([[23, 1]]).cuda()                                                  # class labels (23 years old, female)
images = G(z, c)                                                                    # NCHW, float32, dynamic range [-1, +1], no truncation
images = images.permute(0, 2, 3, 1)                                                 # Reorder channels to NHWC (channels_last)
images_cpu = torch.clamp(((images + 1.0) / 2.0) * 255, min=0.0, max=255.0).cpu()    # Dynamic range [0, 255], copy data to ram
for index, image_cpu in enumerate(images_cpu):
    image_cpu = (image_cpu.numpy()).astype(np.uint8)                                # Tensor to numpy array of type uint8
    plt.imshow(image_cpu)
    plt.show()
```