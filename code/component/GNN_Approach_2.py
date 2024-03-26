import pickle

import numpy as np
import pandas as pd
from prettytable import prettytable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc, roc_curve
from sklearn.model_selection import train_test_split
import category_encoders as ce
import uuid

import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import GATConv, to_hetero

#from torch_sparse import SparseTensor
from class_GNN import GCN,GAT,GnnTrainer,MetricManager
from preprocess import preprocess_data
import torch.nn.functional as F

import argparse
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed = 42
np.random.seed(seed)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv("Sampled Dataset.csv")
print(data.head())

data.rename(columns={'User':'user',
                         'Card':'card',
                         'Year':'year',
                         'Month':'month',
                         'Day':'day',
                         'Time':'time',
                         'Amount':'amount',
                         'Use Chip':'use_chip',
                         'Merchant Name':'merchant_name',
                         'Merchant City':'merchant_city',
                         'Merchant State':'merchant_state',
                         'Zip':'zip',
                         'Errors?':'errors',
                         'Is Fraud?':'is_fraud'}, inplace=True)

print(data.info())

data['id'] = [hash(uuid.uuid4()) for _ in range(len(data))]
print(data['id'].dtype)

# To make 'id' the first column, you can use DataFrame reindex with columns sorted to your preference
cols = ['id'] + [col for col in data.columns if col != 'id']
data = data[cols]


# Create card_id (Node)
data['card_id'] = data['user'] + data['card']
data = data.drop(['user', 'card','errors','merchant_state','zip'], axis=1)

# Create merchant_id (Node)
data.rename(columns={'merchant_name': 'merchant_id'}, inplace=True)

# Dealing with Date and time variables
data['time'] = pd.to_datetime(data['time'], format='%H:%M')
data['hour'] = data['time'].dt.hour
data['minute'] = data['time'].dt.minute
data.drop('time', axis=1, inplace=True)

# Clean Amount column
data['amount'] = data['amount'].str.replace('$', '')
data['amount'] = data['amount'].astype('float')

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['use_chip'], dtype=int)

# Convert is_fraud to binary
data['is_fraud'] = data['is_fraud'].map({'Yes': 1, 'No': 0})

# # Define categorical columns
# categorical_columns = ['merchant_id', 'card_id']
# for column in categorical_columns:
#     data[column] = data[column].astype('category')

# target encoding high cardinality columns
high_cardinality = ['merchant_city']
for column in high_cardinality:
    target_encoder = ce.TargetEncoder()
    data[column] = target_encoder.fit_transform(data[column], data['is_fraud'])

print(data.head())
print(data.info())

data_processed = data

# Creating non target node types

non_target_node_types = ['card_id','MCC']

get_cat_map = lambda vals: {val: idx for idx, val in enumerate(vals)}


def get_edge_list(df, identifier):
    # Find number of unique categories for this node type
    unique_entries = df[identifier].drop_duplicates().dropna()
    # Create a map of category to value
    entry_map = get_cat_map(unique_entries)
    # Create edge list mapping transaction to node type
    edge_list = [[], []]

    for idx, transaction in data.iterrows():
        node_type_val = transaction[identifier]
        # Don't create nodes for NaN values
        if pd.isna(node_type_val):
            continue
        edge_list[0].append(idx)
        edge_list[1].append(entry_map[node_type_val])
    return torch.tensor(edge_list, dtype=torch.long)

edge_dict = {
    node_type: get_edge_list(data, node_type)
    for node_type in non_target_node_types
}
print(edge_dict)

transaction_feats = data.drop(['id','card_id','is_fraud','merchant_id'],axis =1)
classified_idx = transaction_feats.index
print(transaction_feats)
scaler = StandardScaler()
transaction_feats = scaler.fit_transform(transaction_feats)
#transaction_feats = transaction_feats.to_numpy()
transaction_feats = torch.tensor(transaction_feats,dtype=torch.float)
print(transaction_feats.shape)


# Creating a graph data object

data = HeteroData()

data["transaction"].num_nodes = len(data_processed)
data["transaction"].x = transaction_feats
data["transaction"].y = torch.tensor(data_processed["is_fraud"], dtype=torch.float)

for node_type in non_target_node_types:
    data["transaction", "to", node_type].edge_index = edge_dict[node_type]
    data[node_type].num_nodes = edge_dict[node_type][1].max() + 1
    # Create dummy features for the non-transaction node types
    data[node_type].x = torch.zeros((edge_dict[node_type][1].max() + 1, 1))

print(data)

import torch_geometric.transforms as T

data = T.ToUndirected()(data)
data.to(device)
data = T.AddSelfLoops()(data)
data = T.NormalizeFeatures()(data)

print(data)

# Creating a train test split
train_idx, test_idx = train_test_split(classified_idx.values, random_state=42,
                                       test_size=0.2, stratify=data_processed['is_fraud'])
#train_idx, valid_idx = train_test_split(train_idx, test_size=0.2,random_state=42)
print("train_idx size {}".format(len(train_idx)))
#print("valid_idx size {}".format(len(valid_idx)))
print("test_idx size {}".format(len(test_idx)))



# Add in the train and valid idx
data.train_idx = train_idx
#data.valid_idx = valid_idx
data.test_idx = test_idx

# Set training arguments
args = {"epochs": 100, 'lr': 0.0001, 'weight_decay': 5e-4, 'heads': 2, 'hidden_dim': 128, 'dropout': 0.3}
from torch_geometric.nn import GATConv, Linear, to_hetero

# class GAT(torch.nn.Module):
#     def __init__(self, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
#         self.lin1 = Linear(-1, hidden_channels)
#         self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
#         self.lin2 = Linear(-1, out_channels)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index) + self.lin1(x)
#         x = x.relu()
#         x = self.conv2(x, edge_index) + self.lin2(x)
#         return F.sigmoid(x)

class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv((-1,-1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1,-1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index,adj=None):
        x = self.conv1(x, edge_index)+ self.lin1(x)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        #x = F.relu(x)
        x = self.conv2(x, edge_index)+ self.lin2(x)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        return F.sigmoid(x)


model = GAT(args['hidden_dim'], 1).to(device)
#model = GAT(hidden_channels=64, out_channels=1).to(device)
model = to_hetero(model, data.metadata(), aggr='sum').to(device)



def train_val():
    for epoch in range(args['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        for key in out:
            out[key] = out[key].squeeze()
        out = torch.stack([tensor for tensor in out['transaction']])
        loss = criterion(out[data.train_idx], data['transaction'].y[data.train_idx])
        target_labels = data['transaction'].y[data.train_idx].detach().cpu().numpy()
        pred_scores = out[data.train_idx].detach().cpu().numpy()
        threshold=0.5
        pred_labels = pred_scores>threshold


        accuracy = accuracy_score(target_labels, pred_labels)
        f1 = f1_score(target_labels, pred_labels)
        recall = recall_score(target_labels, pred_labels)
        precision = precision_score(target_labels, pred_labels)

        loss.backward()
        # clipping_value = 0.5  # arbitrary value of your choosing
        # torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
        optimizer.step()

        if epoch % 5 == 0:
            print(
                "epoch: {} - loss: {:.4f} - accuracy train: {:.4f} - recall train: {:.4f}  - precision train: {:.4f}  - f1_score train: {:.4f}".format(
                    epoch, loss.item(),accuracy, recall, precision,f1))

        # Validation

        model.eval()
        target_labels = data['transaction'].y[data.test_idx].detach().cpu().numpy()
        pred_scores = out[data.test_idx].detach().cpu().numpy()
        threshold = 0.5
        pred_labels = pred_scores > threshold


        accuracy = accuracy_score(target_labels, pred_labels)
        f1 = f1_score(target_labels, pred_labels)
        recall = recall_score(target_labels, pred_labels)
        precision = precision_score(target_labels, pred_labels)

        if epoch % 5 == 0:
            print(
                "epoch: {} - loss: {:.4f} - accuracy val: {:.4f} - recall val: {:.4f}  - precision val: {:.4f}  - f1_score val: {:.4f}".format(
                    epoch, loss.item(),accuracy, recall, precision,f1))

def predict():
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)
    for key in out:
        out[key] = out[key].squeeze()
    out = torch.stack([tensor for tensor in out['transaction']])
    target_labels = data['transaction'].y[data.test_idx].detach().cpu().numpy()
    pred_scores = out[data.test_idx].detach().cpu().numpy()
    threshold = 0.5
    pred_labels = pred_scores > threshold
    accuracy = accuracy_score(target_labels, pred_labels)
    f1 = f1_score(target_labels, pred_labels)
    recall = recall_score(target_labels, pred_labels)
    precision = precision_score(target_labels, pred_labels)
    results = prettytable.PrettyTable(title="Metric Table")
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    print(results)
    cm = confusion_matrix(target_labels, pred_labels)
    print(cm)
    print("Testing Statistics: ")
    print(np.sum(pred_labels == 1))
    print(np.sum(target_labels == 1))
    # Count occurrences of unique elements
    unique_elements, counts = np.unique(pred_scores, return_counts=True)
    # Create a dictionary mapping each unique element to its count
    element_counts = dict(zip(unique_elements, counts))
    print("Element Counts:", element_counts)
    print(len(pred_labels))
    print(len(target_labels))

# Setup training settings
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001,weight_decay=args['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=1, verbose=True)
criterion = torch.nn.BCELoss()

# Training the data
gnn_trainer_gat = train_val()
gnn_predict_gat = predict()
