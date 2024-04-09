import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import ReLU
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, GATConv, BatchNorm
from torch_geometric.utils import to_undirected
import pandas as pd
import polars as pl
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_data
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import prettytable

torch.manual_seed(42)

df = pd.read_csv('/home/ec2-user/Capstone/code/component/processed_data_inductiveGRL.csv')
#df = preprocess_data(df)
# df = df.to_pandas()


data = HeteroData()

card_id_encoder = LabelEncoder()
merchant_name_encoder = LabelEncoder()
df['card_id'] = card_id_encoder.fit_transform(df['card_id'])
df['merchant_id'] = merchant_name_encoder.fit_transform(df['merchant_id'])

# Add nodes
data['card_id'].x = torch.tensor(df['card_id'].unique(), dtype=torch.long)
data['merchant_id'].x = torch.tensor(df['merchant_id'].unique(), dtype=torch.long)

transaction_features = df[[col for col in df.columns if col not in ['card_id', 'merchant_id', 'is_fraud']]].values

data['transaction'].x = torch.tensor(transaction_features, dtype=œ.float).contiguous()
data['merchant_id'].x = torch.randn(data['merchant_id'].num_nodes, transaction_features.shape[1])
data['card_id'].x = torch.randn(data['card_id'].num_nodes, transaction_features.shape[1])

card_transaction_edges = torch.tensor([df['card_id'].values, df.index.values], dtype=torch.long)
data['card_id', 'transaction'].edge_index = card_transaction_edges

merchant_transaction_edges = torch.tensor([df['merchant_id'].values, df.index.values], dtype=torch.long)
data['merchant_id', 'transaction'].edge_index = merchant_transaction_edges

data['transaction'].y = torch.tensor(df['is_fraud'].values, dtype=torch.long)


def split_nodes(data: HeteroData, node_type: str, train_ratio=0.7, val_ratio=0.15):
    assert node_type in data.node_types and 'y' in data[node_type], "Node type must exist and have labels"
    
    num_nodes = data[node_type].y.size(0)  
    labels = data[node_type].y

    indices = torch.randperm(num_nodes)
    
    num_train = int(num_nodes * train_ratio)
    num_val = int(num_nodes * val_ratio)
    num_test = num_nodes - num_train - num_val
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:num_train]] = True
    val_mask[indices[num_train:num_train+num_val]] = True
    test_mask[indices[num_train+num_val:]] = True
    
    data[node_type].train_mask = train_mask
    data[node_type].val_mask = val_mask
    data[node_type].test_mask = test_mask

    return data

node_type = 'transaction'  
split = split_nodes(data, node_type)

data = T.ToUndirected()(data)
data = T.NormalizeFeatures()(data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)

print(data)

train_size = data['transaction'].train_mask.sum().item()
val_size = data['transaction'].val_mask.sum().item()
test_size = data['transaction'].test_mask.sum().item()

train_input_nodes = ('transaction', data['transaction'].train_mask)
val_input_nodes = ('transaction', data['transaction'].val_mask)
test_input_nodes = ('transaction', data['transaction'].test_mask)


print(f"Training Data Size: {train_size} nodes")
print(f"Validation Data Size: {val_size} nodes")
print(f"Test Data Size: {test_size} nodes")

print(f"Number of records in train set: {train_size}")
print(f"Number of records in validation set: {val_size}")  
print(f"Number of records in test set: {test_size}")

print("Number for fraud and non-fraud transactions in train set:" , data['transaction'].y[data['transaction'].train_mask].bincount())
print("Number for fraud and non-fraud transactions in validation set:" , data['transaction'].y[data['transaction'].val_mask].bincount())
print("Number for fraud and non-fraud transactions in test set:" , data['transaction'].y[data['transaction'].test_mask].bincount())

kwargs = {'batch_size': 1024, 'num_workers': 8, 'persistent_workers': True}

train_loader = NeighborLoader(data, num_neighbors={key: [64] for key in data.edge_types}, shuffle=True, input_nodes=train_input_nodes, **kwargs)
val_loader = NeighborLoader(data, num_neighbors={key: [64] for key in data.edge_types}, input_nodes=val_input_nodes, **kwargs)
test_loader = NeighborLoader(data, num_neighbors={key: [64] for key in data.edge_types}, input_nodes=test_input_nodes, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.nn import GCNConv, Linear, Sequential, to_hetero
from torch.nn import ReLU, BatchNorm1d, Dropout

from torch_geometric.nn import SAGEConv, Linear, Sequential, to_hetero, GATConv, GCNConv, TransformerConv
from torch.nn import ReLU, BatchNorm1d, Dropout, LeakyReLU, ELU

# model = Sequential('x, edge_index', [
#     (SAGEConv(-1, 128), 'x, edge_index -> x1'),
#     ReLU(inplace=True),
#     (BatchNorm1d(128), 'x1 -> x1'),
#     (Dropout(0.2), 'x1 -> x1'),

#     (SAGEConv(128, 128), 'x1, edge_index -> x2'),
#     ReLU(inplace=True),
#     (BatchNorm1d(128), 'x2 -> x2'),
#     (Dropout(0.2), 'x2 -> x2'),

#     (SAGEConv(128, 64), 'x2, edge_index -> x3'),
#     ReLU(inplace=True),
#     (BatchNorm1d(64), 'x3 -> x3'),
#     (Dropout(0.2), 'x3 -> x3'),

#     (lambda x1, x3: torch.cat([x1, x3], dim=-1), 'x1, x3 -> x4'),  # Skip connection

#     (Linear(128 + 64, 32), 'x4 -> x5'),
#     ReLU(inplace=True),
#     (BatchNorm1d(32), 'x5 -> x5'),
#     (Dropout(0.2), 'x5 -> x5'),

#     (Linear(32, 1), 'x5 -> x'),  # Change output size to 1
#     (torch.sigmoid, 'x -> x'),  # Apply sigmoid activation
# ])

model = Sequential('x, edge_index', [
    (SAGEConv(-1, 128), 'x, edge_index -> x1'),
    ReLU(inplace=True),
    (BatchNorm1d(128), 'x1 -> x1'),
    (Dropout(0.2), 'x1 -> x1'),

    (GATConv(128, 128, heads=4, add_self_loops=False), 'x1, edge_index -> x2'),
    ELU(inplace=True),
    (BatchNorm1d(512), 'x2 -> x2'),
    (Dropout(0.2), 'x2 -> x2'),

    (TransformerConv(512, 128, heads=4), 'x2, edge_index -> x3'),
    LeakyReLU(inplace=True),
    (BatchNorm1d(512), 'x3 -> x3'),
    (Dropout(0.2), 'x3 -> x3'),

    (SAGEConv(512, 64), 'x3, edge_index -> x4'),
    ReLU(inplace=True),
    (BatchNorm1d(64), 'x4 -> x4'),
    (Dropout(0.2), 'x4 -> x4'),

    (Linear(512, 64), 'x3 -> x5'),
    ReLU(inplace=True),
    (BatchNorm1d(64), 'x5 -> x5'),
    (Dropout(0.2), 'x5 -> x5'),

    (lambda x5, x4: torch.cat([x5, x4], dim=-1), 'x5, x4 -> x6'),

    (Linear(64 + 64, 128), 'x6 -> x7'),
    ReLU(inplace=True),
    (BatchNorm1d(128), 'x7 -> x7'),
    (Dropout(0.2), 'x7 -> x7'),

    (Linear(128, 64), 'x7 -> x8'),
    ReLU(inplace=True),
    (BatchNorm1d(64), 'x8 -> x8'),
    (Dropout(0.2), 'x8 -> x8'),

    (Linear(64, 1), 'x8 -> x'),
    (torch.sigmoid, 'x -> x'),
])

model = to_hetero(model, data.metadata(), aggr='sum').to(device)


@torch.no_grad()
def init_params():
    batch = next(iter(train_loader))
    batch = batch.to(device, 'edge_index')
    model(batch.x_dict, batch.edge_index_dict)


def train():
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch_size = batch['transaction'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['transaction'][:batch_size]
        loss = F.binary_cross_entropy(out.view(-1), batch['transaction'].y[:batch_size].float())
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    y_true = []
    y_pred = []

    for batch in tqdm(loader):
        batch_size = batch['transaction'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict)['transaction'][:batch_size]
        pred = (out > 0.6).float() 

        y_true.extend(batch['transaction'].y[:batch_size].tolist())
        y_pred.extend(pred.view(-1).tolist())

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, f1, recall, precision, specificity, cm, y_pred


init_params()  # Initialize parameters.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.0001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0000001, momentum=0.9)
#optimizer = torch.optim.ASGD(model.parameters(), lr=0.0000001, weight_decay=0.0001)

best_f1 = 0
best_model_state = None

for epoch in range(1, 56):
    loss = train()
    val_acc, val_f1, val_recall, val_precision, val_specificity, val_cm, _ = test(val_loader)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    print(f'Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val Recall: {val_recall:.4f}, Val Precision: {val_precision:.4f}, Val Specificity: {val_specificity:.4f}')

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = model.state_dict()
        print(f'Saving best model with F1: {best_f1:.4f}')

model.load_state_dict(best_model_state)

test_acc, test_f1, test_recall, test_precision, test_specificity, test_cm, y_pred = test(test_loader)
print("Unique Predictions: ", set(y_pred))

results = prettytable.PrettyTable(title='GNN Results')
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", test_acc])
results.add_row(["F1 Score", test_f1])
results.add_row(["Recall", test_recall])
results.add_row(["Precision", test_precision])
results.add_row(["Specificity", test_specificity])
print("Confusion Matrix: ", test_cm)
print(results)