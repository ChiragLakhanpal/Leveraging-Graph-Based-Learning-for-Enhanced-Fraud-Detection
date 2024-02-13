# %%
import pandas as pd
from category_encoders import TargetEncoder
target_enc = TargetEncoder()

def preprocess(df):
    df['card_id'] = df['User'].astype('str') + df['Card'].astype('str')
    df = df.drop(['User', 'Card'], axis=1)

    df.rename({'Merchant Name':'merchant_id'}, axis=1, inplace=True)
    df['merchant_id'] = df['merchant_id'].astype('str')
    df['merchant_id'] = df['merchant_id'].apply(lambda x: x.replace('-', ''))

    df['Zip'] = df['Zip'].astype('category')
    df['MCC'] = df['MCC'].astype('category')

    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['minute'] = df['Time'].dt.minute
    df.drop('Time', axis=1, inplace=True)
    
    df['Amount'] = df['Amount'].str.replace('$', '')
    df['Amount'] = df['Amount'].astype('float')

    df = pd.get_dummies(df, columns=['Errors?'], dtype=int)
    df = pd.get_dummies(df, columns=['Use Chip'], dtype=int)
    
    df['Is Fraud?'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})
    
    df = df.dropna(subset=['Zip', 'MCC'])       

    df['Merchant City'] = target_enc.fit_transform(df['Merchant City'], df['Is Fraud?'])
    df['Merchant State'] = target_enc.fit_transform(df['Merchant State'], df['Is Fraud?'])
    df['Zip'] = target_enc.fit_transform(df['Zip'], df['Is Fraud?'])
    df['MCC'] = target_enc.fit_transform(df['MCC'], df['Is Fraud?'])
    
    return df

# %%
df_raw = pd.read_csv('credit_card/card_transaction.v1.csv')

df = df_raw.sample(n=5000)

df = preprocess(df)

df

# %%
"""
## Approach 1.
"""

# %%
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, precision_score, recall_score, confusion_matrix

import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv

# %%
import pandas as pd
import torch
from torch_geometric.data import Data


card_id_to_index = {card_id: i for i, card_id in enumerate(df['card_id'].unique())}
merchant_id_to_index = {merchant_id: i + len(card_id_to_index) for i, merchant_id in enumerate(df['merchant_id'].unique())}

edge_index = torch.tensor([
    [card_id_to_index[card_id] for card_id in df['card_id']],
    [merchant_id_to_index[merchant_id] for merchant_id in df['merchant_id']]
], dtype=torch.long)

edge_features_columns = df.columns.drop(['card_id', 'merchant_id'])
edge_features = torch.tensor(df[edge_features_columns].values, dtype=torch.float)

data = Data(edge_index=edge_index, edge_attr=edge_features)

data


# %%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GNNModel(torch.nn.Module):
    def __init__(self, num_nodes, embedding_dim, hidden_channels):
        super(GNNModel, self).__init__()
        self.node_embeddings = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, edge_index, edge_attr):
        x = self.node_embeddings.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

num_nodes = data.edge_index.max().item() + 1
model = GNNModel(num_nodes=num_nodes, embedding_dim=24, hidden_channels=64)


# %%
import torch.optim as optim
from torch_geometric.utils import negative_sampling
from torch.nn.functional import cosine_similarity

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

def train(data):
    model.train()
    optimizer.zero_grad()

    node_embeddings = model(data.edge_index, data.edge_attr)

    neg_edge_index = negative_sampling(edge_index=data.edge_index,
                                       num_nodes=num_nodes,
                                       num_neg_samples=data.edge_index.size(1))

    edge_index_all = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    
    labels = torch.cat([torch.ones(data.edge_index.size(1)), 
                        torch.zeros(neg_edge_index.size(1))], dim=0).unsqueeze(-1)

    src, dest = edge_index_all
    predictions = cosine_similarity(node_embeddings[src], node_embeddings[dest]).unsqueeze(-1)

    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()

    return loss.item()



for epoch in range(200):
    loss = train(data)
    print(f'Epoch {epoch+1}: Loss {loss}')


# %%
from sklearn.metrics import roc_auc_score

def evaluate(data):
    model.eval()  
    with torch.no_grad(): 
        node_embeddings = model(data.edge_index, data.edge_attr)
        
        eval_edge_index, eval_labels = create_eval_samples(data)
        src, dest = eval_edge_index
        predictions = cosine_similarity(node_embeddings[src], node_embeddings[dest]).unsqueeze(-1)
   
        probabilities = torch.sigmoid(predictions).squeeze()
        auc_score = roc_auc_score(eval_labels.cpu().numpy(), probabilities.cpu().numpy())
    return auc_score


# %%
"""
## Approch 2
"""

# %%
import torch
from torch_geometric.data import Data
import pandas as pd


card_id_mapping = {card_id: i for i, card_id in enumerate(df['card_id'].unique())}
merchant_id_mapping = {merchant_id: i + len(card_id_mapping) for i, merchant_id in enumerate(df['merchant_id'].unique())}

node_mapping = {**card_id_mapping, **merchant_id_mapping}

edges = [(card_id_mapping[row['card_id']], merchant_id_mapping[row['merchant_id']]) for _, row in df.iterrows()]

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

edge_features = torch.tensor(df[['Amount']].values, dtype=torch.float).view(-1, 1)

card_avg_amount = df.groupby('card_id')['Amount'].mean().to_dict()

merchant_avg_amount = df.groupby('merchant_id')['Amount'].mean().to_dict()

num_nodes = len(card_id_mapping) + len(merchant_id_mapping)
node_features = np.zeros((num_nodes, 1)) 

for card_id, index in card_id_mapping.items():
    node_features[index, 0] = card_avg_amount.get(card_id, 0)

for merchant_id, index in merchant_id_mapping.items():
    node_features[index, 0] = merchant_avg_amount.get(merchant_id, 0)

node_features = torch.tensor(node_features, dtype=torch.float)



import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

num_nodes = len(node_mapping)

# labels = torch.cat([torch.ones(data.edge_index.size(1)), 
#                     torch.zeros(neg_edge_index.size(1))], dim=0).unsqueeze(-1)

data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

num_classes = 2  

model = GCN(num_node_features=1, num_classes=num_classes)

from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

model = GCN(num_node_features=1, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    logits, accs = model(data), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

for epoch in range(200):
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


# %%
test_record_Y = df_raw[df_raw['Is Fraud?'] == 'Yes'].sample(n=5)
test_record_N = df_raw[df_raw['Is Fraud?'] == 'No'].sample(n=5)

test_record = pd.concat([test_record_Y, test_record_N])
test_record = preprocess(test_record)

model.eval()

test_node_indices = []

for _, row in test_record.iterrows():
    card_idx = card_id_mapping.get(row['card_id'], None)
    merchant_idx = merchant_id_mapping.get(row['merchant_id'], None)
    if card_idx is not None:
        test_node_indices.append(card_idx)
    if merchant_idx is not None:
        test_node_indices.append(merchant_idx)

test_node_indices = torch.tensor(test_node_indices, dtype=torch.long)

with torch.no_grad():
    logits = model(data)
    test_logits = logits[test_node_indices]
    test_probs = torch.softmax(test_logits, dim=1)
    predicted_classes = test_probs.argmax(dim=1)
    
    print(f"Predicted classes for test nodes: {predicted_classes.tolist()}")
    print(f"Fraud probabilities for test nodes: {test_probs[:, 1].tolist()}")

