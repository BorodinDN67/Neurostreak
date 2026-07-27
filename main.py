from torch.utils.data import DataLoader
from src.data import StreakImageDataset
from src.data import DataPrefetcher
from src.data import minmax_scaling, fft, scene_without_background, mad,smooth
import matplotlib.pyplot as plt
import numpy as np

dataset = StreakImageDataset(data_dir='data/clean_water_10m', cache=False, groundtruth=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


batch = next(iter(dataloader))
ne_batch = smooth(batch[0])

fig, ax = plt.subplots(1,2)
ax[0].imshow(batch[0][0,:,:], cmap = 'gray')
ax[1].imshow(ne_batch[0], cmap = 'gray')
plt.show()