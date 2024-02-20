
import torch
from torch_geometric.data import Data
import pandas as pd
from category_encoders import TargetEncoder
from torch_geometric.data import DataLoader
import os

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import LabelEncoder

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
try:
    df = pd.read_csv('Week 1/credit_card/card_transaction.v1.csv')
except:
    df = pd.read_csv('/home/ec2-user/Capstone/Working_dir/data/raw_data/card_transaction_sample_5000')
    
df = preprocess(df)

unique_cards = df['card_id'].unique()
unique_merchants = df['merchant_id'].unique()

card_mapping = {c:i for i,c in enumerate(unique_cards)}
merchant_mapping = {m:i for i,m in enumerate(unique_merchants)}

edges = []
for _, row in df.iterrows():
    src = card_mapping[row['card_id']]
    dst = merchant_mapping[row['merchant_id']]
    edges.append([src, dst])

edge_feat = df[[
    'Year', 'Month', 'Day', 'Amount', 
    'Merchant City', 'Merchant State', 'Zip', 'MCC',
    'hour', 'minute', 
    'Errors?_Bad CVV,', 'Errors?_Bad Card Number,', 
    'Errors?_Bad Expiration,', 'Errors?_Bad PIN,',
    'Errors?_Insufficient Balance,', 'Errors?_Technical Glitch,'
]].values

edge_label = df['Is Fraud?'].values
edge_type = df[[
    'Use Chip_Chip Transaction', 
    'Use Chip_Online Transaction',
    'Use Chip_Swipe Transaction'  
]].values

data = Data(edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous())

data.num_nodes = len(set(df['card_id']) | set(df['merchant_id']))
data.edge_feat = torch.tensor(edge_feat, dtype=torch.float) 
data.edge_label = torch.tensor(edge_label, dtype=torch.long)
data.edge_type = torch.tensor(edge_type, dtype=torch.long) 

data.num_edge_features = edge_feat.shape[1]
data.num_edge_types = edge_type.shape[1]

feature_size = 16
node_features = torch.ones((data.num_nodes, feature_size)) 
data.x = node_features

card_ids = df['card_id'].unique()
merchant_ids = df['merchant_id'].unique()

print(data)
print("Number of nodes: ", data.num_nodes)
print('Number of card_ids: ', len(card_ids))
print('Number of merchant_ids: ', len(merchant_ids))
print("Number of edges: ", data.num_edges)
print("Number of features per node: ", data.num_node_features)
print("Number of features per edge: ", data.num_edge_features)
print("Number of edge types: ", data.num_edge_types)
# print("Number of classes: ", data.num_classes)
print("Sample edge: ", data.edge_index[:, 0], data.edge_feat[0], data.edge_label[0], data.edge_type[0])
print("Shape of edge_index: ", data.edge_index.shape)
print("Shape of edge_feat: ", data.edge_feat.shape)
print("Shape of edge_label: ", data.edge_label.shape)
print("Shape of edge_type: ", data.edge_type.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classif = nn.Linear(hidden_channels, 2)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classif(x)
        return F.log_softmax(x, dim=1)
        
model = GCN(num_node_features= feature_size, hidden_channels=4406)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

def train():
    model.train()

    optimizer.zero_grad()  
    out = model(data.x, data.edge_index) 
    loss = F.nll_loss(out[data.edge_label], data.edge_label) 
    loss.backward()    
    optimizer.step()
    
for epoch in range(10):
    
    optimizer.zero_grad()
    
    out = model(data.edge_feat, data.edge_index)
    loss = F.nll_loss(out[data.edge_label], data.edge_label)
    loss.backward()
    
    optimizer.step()
    
    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
    
node_embeddings = model(data.edge_feat, data.edge_index)

print(node_embeddings.shape)