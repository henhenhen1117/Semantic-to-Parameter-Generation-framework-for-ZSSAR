from Dataset import AdapterDataset as DataSet
import torch
from tqdm import tqdm
import numpy as np
dataset= DataSet("/media/zzf/ljn/wsx/PGFA/PGFA-main/checkpoint",dim_per_token=768,
                              )
data_loader = dict()
data_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=1,
    num_workers=16,
    shuffle=True,
    drop_last=True)
loader =data_loader
for data, label in tqdm(loader):
    print(data)
    print(label)
print(dataset)
# data=torch.load("/media/zzf/ljn/wsx/PGFA/PGFA-main/output/diffusion_dataset/diffusion_training_dataset.npy")
# for it in data.items():
#     print(it)