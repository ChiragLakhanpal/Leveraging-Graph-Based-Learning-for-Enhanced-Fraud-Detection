import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader, HeteroData
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, precision_score, recall_score, confusion_matrix
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Load data from the folder

df = pd.read_csv("Tabsformer_5000 (1).csv")

# Target encoding the columns

from category_encoders import TargetEncoder
target_enc = TargetEncoder()

df['Merchant City'] = target_enc.fit_transform(df['Merchant City'], df['Is Fraud?'])
df['Merchant State'] = target_enc.fit_transform(df['Merchant State'], df['Is Fraud?'])
df['Zip'] = target_enc.fit_transform(df['Zip'], df['Is Fraud?'])
df['MCC'] = target_enc.fit_transform(df['MCC'], df['Is Fraud?'])

print(df.head())

# Mapping cardID and merchantID to unique indices

card_id_to_index = {card_id: i for i, card_id in enumerate(df['card_id'].unique())}
merchant_id_to_index = {merchant_id: i + len(card_id_to_index) for i, merchant_id in enumerate(df['merchant_id'].unique())}

# Preparing edge index tensor

edge_index = torch.tensor([
    [card_id_to_index[card_id] for card_id in df['card_id']],
    [merchant_id_to_index[merchant_id] for merchant_id in df['merchant_id']]
], dtype=torch.long)
#edge_index.shape()


# Define labels

labels = df['Is Fraud?'].values
#print(labels)

# Create edge features and convert to tensor

edge_features = df.drop(["card_id","merchant_id","Is Fraud?"], axis=1).copy()
edge_features = torch.tensor(edge_features.values, dtype=torch.float)

# Create a PyG Geometric Dataset

data = Data(edge_index=edge_index, edge_attr=edge_features, y=torch.tensor(labels, dtype=torch.long))
print(data)

# Create a GCN Model Class

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(18, 16)
        self.conv2 = GCNConv(16, 2)  # Assuming binary classification

    def forward(self, data):
        x, edge_index = data.edge_attr, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Instantiate the model

model = GCN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
