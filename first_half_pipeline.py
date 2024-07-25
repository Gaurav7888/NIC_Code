import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig
from PIL import Image
from torchsummary import summary

model_Res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
#print("model_Res",model_Res)

#print(list(model_Res.children()))

layers = list(model_Res.children())

model_Res_modified = nn.Sequential(*layers[:-1])

#print(list(model_Res_modified.children()))
# count = 0
# for layer in model_Res_modified.children():
#     count+=1
#     if (count > 6):
#         for param in layer.parameters():
#             print("dsa")
#             param.requires_grad = True

count = 0
for child in model_Res_modified.children():
    count += 1
    if count < 8:
        for param in child.parameters():
            param.requires_grad = False

for name, param in model_Res_modified.named_parameters():
    print(f'Layer: {name} | Shape: {param.shape} | Requires Grad: {param.requires_grad}')

model_trans = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
count = 0
for child in model_trans.children():
    count += 1
    if count < 4:
        for param in child.parameters():
            param.requires_grad = False

layers_trans = list(model_trans.children()) # Get all the layers from the Transformer model
model_trans_top = nn.Sequential(*layers_trans[:-2]) # Remove the normalization layer and pooler layer
trans_layer_norm = list(model_trans.children())[2] # Get the normalization layer

class model_final(nn.Module):
    def __init__(self, model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3):
        super().__init__()
        # All the trans model layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.trans_linear = nn.Linear(150528, 2048)

        # All the ResNet model
        self.model_Res = model_Res

        # Merge the result and pass the
        self.dropout = nn.Dropout(dp_rate)
        self.linear1 = nn.Linear(4096, 1000)
        self.linear2 = nn.Linear(1000,500)

    def forward(self, trans_b, res_b):
        # Get intermediate outputs using hidden layer
        result_trans = self.model_trans_top(trans_b)
        patch_state = result_trans.last_hidden_state[:,1:,:] # Remove the classification token and get the last hidden state of all patchs
        result_trans = self.trans_layer_norm(patch_state)
        result_trans = self.trans_flatten(patch_state)
        result_trans = self.dropout(result_trans)
        result_trans = self.trans_linear(result_trans)

        result_res = self.model_Res(res_b)
        # result_res = result_res.squeeze() # Batch size cannot be 1
        result_res = torch.reshape(result_res, (result_res.shape[0], result_res.shape[1]))

        result_merge = torch.cat((result_trans, result_res),1)
        result_merge = self.dropout(result_merge)
        result_merge = self.linear1(result_merge)
        result_merge = self.dropout(result_merge)
        result_merge = self.linear2(result_merge)

        return result_merge

model = model_final(model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3)

print(model())

class petDataset_pred(Dataset):
    def __init__(self, dataframe, trans_transform=None, res_transform=None):
        self.images = dataframe["file_path"]
        self.trans_transform = trans_transform
        self.res_transform = res_transform

    def __len__ (self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)

        image_trans = self.trans_transform(image)

        image_res = self.res_transform(image)
        return image_trans, image_res

trans_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
res_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


test_ds = petDataset_pred(df, trans_transform=trans_transform, res_transform=res_transform)
test_dl = DataLoader(test_ds, batch_size=2, shuffle=False)