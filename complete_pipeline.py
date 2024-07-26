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
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
import pickle 
import os
os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs
image = torch.ones([3,456,768])
device = torch.device("hpu")

# Ensure all tensors are moved to HPU
def move_to_hpu(tensor):
    return tensor.to(device)

# Define feature extractor for ViT input
vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
# Apply feature extraction for ViT
vit_image_tensor = vit_feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig, AutoConfig
# from combined_features import FeatureFusion 
from combined_features_vit import FeatureFusion

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
        image = image.to(device)
        # Apply any transformations defined in the constructor
        if self.transform:
            image = self.transform(image)
        
        # Generate features using ResNet and ViT
        image = self.vit_features(image)

        # Optionally, apply target transformations
        if self.target_transform:
            label = self.target_transform(label)

        return image.to(device), label, img_dir

dataset = CustomImageDataset(annotations_file="/root/gaurav/UPSC/train.txt")
print(dataset[0][0].shape)
print(dataset[0][1])
print(dataset[0][2])

vit_features = []
texts = []
img_path = []
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
for batch in tqdm(dataloader, desc="Processing Batches"):
    vit_features.append(batch[0])
    texts.append(batch[1])
    img_path.append(batch[2])

# Save the lists to a file
with open('data_vit.pkl', 'wb') as f:
    pickle.dump({'vit_features': vit_features, 'texts': texts, 'img_path': img_path}, f)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
base_model = AutoModel.from_pretrained('xlm-roberta-base')

class data1(Dataset):
    def __init__(self, vit_features, texts, tokenizer, max_length=512):
        self.vit_features = vit_features
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        text = text[0]
        if not isinstance(text, str):
            raise ValueError(f"Expected text to be a string, got {type(text)}")
        vit_feature = self.vit_features[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True
        )

        return {
            'input_ids': move_to_hpu(torch.tensor(inputs['input_ids'], dtype=torch.long)),
            'attention_mask': move_to_hpu(torch.tensor(inputs['attention_mask'], dtype=torch.long)),
            'token_type_ids': move_to_hpu(torch.tensor(inputs['token_type_ids'], dtype=torch.long)),
            'vit_features': move_to_hpu(vit_feature.view(-1))
        }

class bert(nn.Module):
    def __init__(self, base_model, vit_feature_size):
        super(bert, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(base_model.config.hidden_size, vit_feature_size)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        regressed_output = self.regressor(dropout_output)
        return regressed_output

model = bert(base_model, vit_feature_size=151296).to(device)

dataset = data1(vit_features, texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(500):
    print(epoch)
    for batch in dataloader:
    
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        vit_features = batch['vit_features']

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        loss = loss_fn(outputs, vit_features.squeeze(1))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        htcore.mark_step()
        optimizer.step()
        htcore.mark_step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")