import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file,transform=None, target_transform=None):
        # Read the annotations file line by line
        self.img_labels = []
        self.img_dir = []
        with open(annotations_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()  # Assuming space-separated values
                self.img_labels.append(label)
                self.img_dir.append(path)
        
        self.transform = transforms.Compose([
            transforms.Resize(size=(292, 968)), # Resize images to a fixed size
        ])
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_dir[idx])
        label = self.img_labels[idx]
        img_dir = self.img_dir[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_dir



dataset = CustomImageDataset(annotations_file="/root/gaurav/UPSC/train.txt")
print(dataset[0][0].shape)
print(dataset[0][1])
print(dataset[0][2])


dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in dataloader:
    print("image tensor", batch)
    break