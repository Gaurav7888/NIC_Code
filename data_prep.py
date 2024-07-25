import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from combined_features import FeatureFusion

from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig

# Load a sample image
# image_path = 'path_to_your_image.jpg'
# image = Image.open(image_path)

image = torch.ones([3,456,768])

# Define transforms for ResNet input
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 pixels
    #transforms.ToTensor(),  # Convert PIL Image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize tensor
])

# Apply transforms to image for ResNet
resnet_image_tensor = resnet_transform(image)


# Define feature extractor for ViT input
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Apply feature extraction for ViT
vit_image_tensor = vit_feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig
from combined_features import FeatureFusion  # Make sure this import is correct based on your project structure

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = []
        self.img_dir = []
        with open(annotations_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.img_labels.append(label)
                self.img_dir.append(path)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def vit_features(self, image_tensor):
        feature_fusion = FeatureFusion(image_tensor)
        combined_features = feature_fusion.process_and_combine()
        return combined_features

    def __getitem__(self, idx):
        image = read_image(self.img_dir[idx]).float()
        label = self.img_labels[idx]
        img_dir = self.img_dir[idx]

        # Apply any transformations defined in the constructor
        if self.transform:
            image = self.transform(image)
        
        # Generate features using ResNet and ViT
        image = self.vit_features(image)

        # Optionally, apply target transformations
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_dir



dataset = CustomImageDataset(annotations_file="/root/gaurav/UPSC/train.txt")
print(dataset[0][0].shape)
print(dataset[0][1])
print(dataset[0][2])

# values1 = []
# values2 = []
# values3 = []
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# for batch in  tqdm(dataloader, desc="Processing Batches"):
#     values1.append(batch[0])
#     values2.append(batch[1])
#     values3.append(batch[2])

# data_dict = {'Value1': values1, 'Value2': values2, 'Value3': values3}

# # Create a DataFrame from the dictionary
# df_train = pd.DataFrame(data_dict)
# df_train.to_csv('train_data.csv', index=False)

