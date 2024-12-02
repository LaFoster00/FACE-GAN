# FACE-GAN
FACE-GAN: Facial Appearance Creation &amp; Enhancement using GANs.

## Requirements
* Install the requirements using `requirements.txt` and pip.
  * `pip install -r requirements.txt`
* Download stylegan3 and set it up.

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
