import numpy as np
import pandas as pd
import polars as pl
import prettytable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import uuid
import pickle
import time

import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

from preprocess import preprocess_data

import argparse
import warnings
warnings.filterwarnings("ignore")

# Set random seed

seed = 42

np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Set device as gpu if available

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_gpus = 0
if num_gpus:
    cuda = True
    device = torch.device('cuda:0')
else:
    cuda = False
    device = torch.device('cpu')


# Read the data and use the preprocess function for preprocessing

data = pl.read_csv('/home/ec2-user/Capstone/Data/Sampled Dataset.csv')

data = preprocess_data(data)

data = data.to_pandas()

data_processed = data

# Create transaction IDs for transaction node in graph

data['id'] = [hash(uuid.uuid4()) for _ in range(len(data))]

cols = ['id'] + [col for col in data.columns if col != 'id']
data = data[cols]

# Create transaction feature tensors

transaction_feats = data.drop(['id','card_id','is_fraud','merchant_id'],axis =1)
classified_idx = transaction_feats.index

scaler = StandardScaler()
transaction_feats = scaler.fit_transform(transaction_feats)
transaction_feats = torch.tensor(transaction_feats,dtype=torch.float)

print(transaction_feats.shape)

# Create edge indexes for different edge types

transaction_to_card = data[['id','card_id']].astype(int)
transaction_to_merchant = data[['id','merchant_id']].astype(int)

nodes = transaction_to_card['id'].unique()
map_id = {j: i for i, j in enumerate(nodes)}

nodes_1 = transaction_to_card['card_id'].unique()
map_id_1 = {j: i for i, j in enumerate(nodes_1)}

nodes_2 = transaction_to_merchant['merchant_id'].unique()
map_id_2 = {j: i for i, j in enumerate(nodes_2)}

nodes_3 = transaction_to_merchant['id'].unique()
map_id_3 = {j: i for i, j in enumerate(nodes_3)}

card_edges = transaction_to_card.copy()
card_edges.card_id = card_edges.card_id.map(map_id_1)
card_edges.id = card_edges.id.map(map_id)

merchant_edges = transaction_to_merchant.copy()
merchant_edges.merchant_id = merchant_edges.merchant_id.map(map_id_2)
merchant_edges.id = merchant_edges.id.map(map_id)

# Create DGL graph

graph_data = {
   ('card_id', 'card_id<>transaction', 'transaction'): (card_edges['card_id'], card_edges['id']),
   ('merchant_id', 'merchant_id<>transaction', 'transaction'): (merchant_edges['merchant_id'], merchant_edges['id']),
   ('transaction', 'self_relation', 'transaction'): (card_edges['id'], card_edges['id']),
   ('transaction', 'transaction<>card_id', 'card_id'): (card_edges['id'], card_edges['card_id']),
   ('transaction', 'transaction<>merchant_id', 'merchant_id'): (merchant_edges['id'], merchant_edges['merchant_id'])
}

g = dgl.heterograph(graph_data)

g.nodes['transaction'].data['y'] = torch.tensor(data_processed["is_fraud"], dtype=torch.float)
g.to(device)
print(g)

# Print graph info

print("Graph properties: ")
print("Total number of nodes in graph: ", g.num_nodes())
ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
print("Number of nodes for each node type:",ntype_dict)
print("Edge Dictionary for different edge types: ")
print("Card_To_Transaction Edges: ",g.edges(etype='card_id<>transaction'))
print("Merchant_To_Transaction Edges: ",g.edges(etype='merchant_id<>transaction'))
print("Transaction_Self_Loop: ",g.edges(etype='self_relation'))
print("Transaction_To_Card Edges: ",g.edges(etype='transaction<>card_id'))
print("Transaction_To_Merchant Edges: ",g.edges(etype='transaction<>merchant_id'))

# Creating a train test split

train_idx, test_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2, stratify=data_processed['is_fraud'])
print("train_idx size {}".format(len(train_idx)))
print("test_idx size {}".format(len(test_idx)))

# Creating a HeteroRGCN model class
class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        self.weight = nn.ModuleDict({
                name: nn.Linear(in_size, out_size) for name in etypes
            })
    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            if srctype in feat_dict:
                Wh = self.weight[etype](feat_dict[srctype])
                G.nodes[srctype].data['Wh_%s' % etype] = Wh
                funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        G.multi_update_all(funcs, 'sum')
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes if 'h' in G.nodes[ntype].data}

class HeteroRGCN(nn.Module):
    def __init__(self, ntype_dict, etypes, in_size, hidden_size, out_size, n_layers):
        super(HeteroRGCN, self).__init__()
        embed_dict = {ntype: nn.Parameter(torch.Tensor(num_nodes, in_size))
                      for ntype, num_nodes in ntype_dict.items() if ntype != 'transaction'}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(in_size, hidden_size, etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_size, hidden_size, etypes))
        self.lin = nn.Linear(hidden_size,out_size)

    def forward(self, g, features):
        x_dict = {ntype: emb for ntype, emb in self.embed.items()}
        x_dict['transaction'] = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                x_dict = {k: F.leaky_relu(x) for k, x in x_dict.items()}
            x_dict = layer(g, x_dict)
        return self.lin(x_dict['transaction'])


# Defining model parameters

in_size = transaction_feats.shape[1]
hidden_size = 16
out_size = 2
n_layers = 3
ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
labels = g.nodes['transaction'].data['y'].long()
learning_rate = 0.01
num_epochs = 300

# Calling the model

model = HeteroRGCN(ntype_dict,g.etypes, in_size, hidden_size, out_size, n_layers)
model.to(device)
print(model)

# Set up the optimizer and loss function

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 5e-4)
criterion = nn.CrossEntropyLoss()

# Defining the train function
def train_val():

    total_loss = 0
    for epoch in range(num_epochs):

        start_time = time.time()

        model.train()
        optimizer.zero_grad()
        out = model(g,transaction_feats)
        pred = out
        pred_c = out.argmax(1)
        loss = criterion(pred[train_idx], labels[train_idx])
        target = labels[train_idx]
        pred_scores = pred_c[train_idx]


        threshold = 0.5
        pred = pred_scores > threshold

        accuracy = accuracy_score(target, pred)
        f1 = f1_score(target, pred)
        recall = recall_score(target, pred)
        precision = precision_score(target, pred)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        end_time = time.time()

        duration = end_time - start_time

        if (epoch) % 5 == 0:
                print(
                    "Epoch: {} - Duration: {:.2f} - Loss: {:.2f}  - Accuracy Train: {:.4f} - Recall Train: {:.2f}  - Precision Train: {:.2f}  - F1_Score Train: {:.2f}".format(
                        epoch, duration, loss.item(), accuracy, recall, precision, f1))

        start_time = time.time()
        model.eval()
        with torch.no_grad():
            target = labels[test_idx]
            pred_scores = pred_c[test_idx]
            threshold = 0.5
            pred = pred_scores > threshold

            accuracy = accuracy_score(target, pred)
            f1 = f1_score(target, pred)
            recall = recall_score(target, pred)
            precision = precision_score(target, pred)

        end_time = time.time()

        duration = end_time - start_time

        if (epoch) % 5 == 0:
                print(
                    "Epoch: {} - Duration: {:.2f} - Loss: {:.2f} - Accuracy Val: {:.4f} - Recall Val: {:.2f}  - Precision Val: {:.2f}  - F1_score Val: {:.2f}".format(
                        epoch, duration,loss.item(), accuracy, recall, precision, f1))

train_val()

# Testing the data and getting the metrics
def predict():
    model.eval()
    with torch.no_grad():
        out = model(g, transaction_feats)
        target_labels = labels[test_idx]
        pred_c = out.argmax(1)
        pred_scores = pred_c[test_idx]
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
        print("Confusion Matrix")
        print(cm)

predict()


