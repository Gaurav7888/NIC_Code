import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW

vit_features = [torch.rand(1, 3,224,224) for _ in range(10)]
texts = ["ఇది"] * 10

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
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'vit_features': vit_feature.view(-1)
        }

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
base_model = AutoModel.from_pretrained('xlm-roberta-base')

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

model = bert(base_model, vit_feature_size=150528)

dataset = data1(vit_features, texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(2):
    print(epoch)
    for batch in dataloader:
        optimizer.zero_grad()
        
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
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
