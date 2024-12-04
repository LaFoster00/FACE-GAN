import pickle

import torch

with open("/home/lasse/Git/FACE-GAN/training-runs/00001-stylegan3-r-ffhq_cond_stylegan3_128x128-gpus1-batch32-gamma0.5/network-snapshot-002440.pkl", "rb") as f:
    G = pickle.load(f)["G_ema"].cuda()
z = torch.randn([1, G.z_dim]).cuda()
c = [1, 35]
img = G(z, c)
print(img)