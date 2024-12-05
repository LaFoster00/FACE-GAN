import pickle
from pathlib import Path

import torch
from matplotlib import pyplot

network_path = Path(__file__).parent / "../models/network-snapshot-004840.pkl"

with open(network_path, "rb") as f:
    G = pickle.load(f)["G_ema"].cuda()
z = torch.randn([1, G.z_dim]).cuda()
z = z.repeat(2,1)
c = torch.tensor([[1, 1], [1, 0]], dtype=torch.int32).cuda()
img = G(z, c)
img = (img.permute(0, 2, 3, 1).cpu() + 1) / 2
print(img[0])
pyplot.imshow(img[0])

pyplot.show()
pyplot.imshow(img[1])
pyplot.show()