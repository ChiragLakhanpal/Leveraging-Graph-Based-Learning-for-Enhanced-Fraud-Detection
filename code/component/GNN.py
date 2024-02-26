import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv,GATConv

import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score

# Read the csv data into a dataframe
df_raw = pd.read_csv("card_transaction.v1.csv", nrows = 100000)
# Create card_id from User and Card columns
df_raw['card_id'] = df_raw['User'].astype('str') + df_raw['Card'].astype('str')

# Convert negative merchant_id to positive
df_raw.rename({'Merchant Name': 'merchant_id'}, axis=1, inplace=True)
df_raw['merchant_id'] = df_raw['merchant_id'].astype('str')
df_raw['merchant_id'] = df_raw['merchant_id'].apply(lambda x: x.replace('-', ''))

# Convert Zip and MCC to categorical
df_raw['Zip'] = df_raw['Zip'].astype('category')
df_raw['MCC'] = df_raw['MCC'].astype('category')

# Split Time column into hour and minute
df_raw['Time'] = pd.to_datetime(df_raw['Time'])
df_raw['hour'] = df_raw['Time'].dt.hour
df_raw['minute'] = df_raw['Time'].dt.minute
df_raw.drop('Time', axis=1, inplace=True)

# Strip $ from Amount in front
df_raw['Amount'] = df_raw['Amount'].str.replace('$', '')
df_raw['Amount'] = df_raw['Amount'].astype('float')

# Select one row with target variable 'Yes'
yes_row = df_raw[df_raw['Is Fraud?'] == 'Yes'].iloc[:5]

# Select one row with target variable 'No'
no_row = df_raw[df_raw['Is Fraud?'] == 'No'].iloc[:5]

df = pd.concat([yes_row, no_row])

print(df)

# One hot encode use_chip and errors
df = pd.get_dummies(df, columns=['Errors?'], dtype=int)
df = pd.get_dummies(df, columns=['Use Chip'], dtype=int)

# Label encode the Is Fraud? column
df['Is Fraud?'] = df['Is Fraud?'].map({'No': 0, 'Yes': 1})

# Drop NA values in Zip and MCC
# df = df.dropna(subset=['Zip', 'MCC'])

# Target encode ["merchant_city", "merchant_state", "zip", "mcc"] columns
from category_encoders import TargetEncoder

target_enc = TargetEncoder()

df['Merchant City'] = target_enc.fit_transform(df['Merchant City'], df['Is Fraud?'])
df['Merchant State'] = target_enc.fit_transform(df['Merchant State'], df['Is Fraud?'])
df['Zip'] = target_enc.fit_transform(df['Zip'], df['Is Fraud?'])
df['MCC'] = target_enc.fit_transform(df['MCC'], df['Is Fraud?'])

print(df.head())


# Create a dataframe for edges
df_edges = df[["card_id","merchant_id"]].copy()

# Create a dataframe merge with node features ids and classes

df_card = df.drop(["merchant_id","User","Card"],axis=1)
df_card = df_card.rename(columns = {"card_id":"id"})
#df_card[["Merchant City","Merchant State","Zip","MCC"]] == 0

df_merchant = df.drop(["card_id","User","Card"],axis=1)
df_merchant = df_merchant.rename(columns = {"merchant_id":"id"})
#df_merchant[["Year","Month","Day","Amount","hour","minute"]] == 0

df_merge = pd.concat([df_card,df_merchant]).reset_index(drop=True)
print(df.merge)

# Setup transaction ID to node ID mapping
nodes = df_merge["id"].values

# Mapping nodes to indexes
map_id = {j: i for i, j in enumerate(nodes)}

# Create edge dataframe that has transaction ID mapped to nodeIDs
edges = df_edges.copy()
edges.card_id = edges.card_id.map(map_id)
edges.merchant_id = edges.merchant_id.map(map_id)
edges = edges.astype(int)

# Create an edge_index tensor
edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()

print("Shape of edge index is {}".format(edge_index.shape))
print("Edge index tensor")
print(edge_index)

# Create weights tensor with same shape of edge_index
# weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.float)
# print("Shape of edge weight tensor: ", weights.shape)
# print("Edge weights tensor")
# print(weights)

# Define labels
labels = df_merge['Is Fraud?'].values
print("Unique Labels", np.unique(labels))
print("Labels Array")
print(labels)

# Mapping txIds to corresponding indices, to pass node features to the model
node_features = df_merge.drop(['id'], axis=1).copy()
print(node_features)
# node_features[0] = node_features[0].map(map_id) # Convert transaction ID to node ID \
print("Unique classes in node features=", node_features["Is Fraud?"].unique())

# Retain known vs unknown IDs
classified_idx = node_features['Is Fraud?'].loc[node_features['Is Fraud?'] !=2].index

# Filter on illicit and licit labels
classified_illicit_idx = node_features['Is Fraud?'].loc[node_features['Is Fraud?'] == 1].index
classified_licit_idx = node_features['Is Fraud?'].loc[node_features['Is Fraud?'] == 0].index

# Drop unwanted columns, 0 = transID, 1=time period, class = labels
node_features = node_features.drop(columns=['Is Fraud?'])

# Convert to tensor
node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double), dtype=torch.float)
print("Node features tensor")
print(node_features_t)

# Train val splits
train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.4)
print("train_idx size {}".format(len(train_idx)))
print("test_idx size {}".format(len(valid_idx)))

# Creating a PyG Dataset
data_train = Data(x=node_features_t, edge_index=edge_index,
                               y=torch.tensor(labels, dtype=torch.float))

import torch_geometric.transforms as T

data_train = T.ToUndirected()(data_train)
data_train

print(data_train)

# Add in the train and valid idx
data_train.train_idx = train_idx
data_train.valid_idx = valid_idx


# Defining model classes

# GCNConv Class
class GCN(torch.nn.Module):
    def __init__(self,num_nodes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_nodes, 8)
        self.conv2 = GCNConv(8, 2)
        self.classifier = Linear(2, 1)

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h1 = self.conv2(h, edge_index)
        embeddings = h1.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        # return out, embeddings
        return F.sigmoid(out)


# Creating a model trainer object

class GnnTrainer(object):

    def __init__(self, model):
        self.model = model
        self.metric_manager = MetricManager(modes=["train", "val"])

    def train(self, data_train, optimizer, criterion, scheduler, args):

        self.data_train = data_train
        for epoch in range(args['epochs']):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data_train)

            out = out.reshape((data_train.x.shape[0]))
            loss = criterion(out[data_train.train_idx], data_train.y[data_train.train_idx])
            ## Metric calculations
            # train data
            target_labels = data_train.y.detach().cpu().numpy()[data_train.train_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.train_idx]
            train_acc, train_f1, train_f1macro, train_aucroc, train_recall, train_precision, train_cm = self.metric_manager.store_metrics(
                "train", pred_scores, target_labels)

            ## Training Step
            loss.backward()
            optimizer.step()

            # validation data
            self.model.eval()
            target_labels = data_train.y.detach().cpu().numpy()[data_train.valid_idx]
            pred_scores = out.detach().cpu().numpy()[data_train.valid_idx]
            val_acc, val_f1, val_f1macro, val_aucroc, val_recall, val_precision, val_cm = self.metric_manager.store_metrics(
                "val", pred_scores, target_labels)

            if epoch % 5 == 0:
                print(
                    "epoch: {} - loss: {:.4f} - accuracy train: {:.4f} -accuracy valid: {:.4f}  - val roc: {:.4f}  - val f1micro: {:.4f}".format(
                        epoch, loss.item(), train_acc, val_acc, val_aucroc, val_f1))

    # To predict labels
    def predict(self, data=None, unclassified_only=True, threshold=0.5):
        # evaluate model:
        self.model.eval()
        if data is not None:
            self.data_train = data

        out = self.model(self.data_train)
        out = out.reshape((self.data_train.x.shape[0]))

        if unclassified_only:
            pred_scores = out.detach().cpu().numpy()[self.data_train.test_idx]
        else:
            pred_scores = out.detach().cpu().numpy()

        pred_labels = pred_scores > threshold

        return {"pred_scores": pred_scores, "pred_labels": pred_labels}

        # To save metrics

    def save_metrics(self, save_name, path="./save/"):
        file_to_store = open(path + save_name, "wb")
        pickle.dump(self.metric_manager, file_to_store)
        file_to_store.close()

        # To save model

    def save_model(self, save_name, path="./save/"):
        torch.save(self.model.state_dict(), path + save_name)


class MetricManager(object):
    def __init__(self, modes=["train", "val"]):

        self.output = {}

        for mode in modes:
            self.output[mode] = {}
            self.output[mode]["accuracy"] = []
            self.output[mode]["f1micro"] = []
            self.output[mode]["f1macro"] = []
            self.output[mode]["aucroc"] = []
            # new
            self.output[mode]["precision"] = []
            self.output[mode]["recall"] = []
            self.output[mode]["cm"] = []

    def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):

        # calculate metrics
        pred_labels = pred_scores > threshold
        accuracy = accuracy_score(target_labels, pred_labels)
        f1micro = f1_score(target_labels, pred_labels, average='micro')
        f1macro = f1_score(target_labels, pred_labels, average='macro')
        aucroc = roc_auc_score(target_labels, pred_scores)
        # new
        recall = recall_score(target_labels, pred_labels)
        precision = precision_score(target_labels, pred_labels)
        cm = confusion_matrix(target_labels, pred_labels)

        # Collect results
        self.output[mode]["accuracy"].append(accuracy)
        self.output[mode]["f1micro"].append(f1micro)
        self.output[mode]["f1macro"].append(f1macro)
        self.output[mode]["aucroc"].append(aucroc)
        # new
        self.output[mode]["recall"].append(recall)
        self.output[mode]["precision"].append(precision)
        self.output[mode]["cm"].append(cm)

        return accuracy, f1micro, f1macro, aucroc, recall, precision, cm

    # Get best results
    def get_best(self, metric, mode="val"):

        # Get best results index
        best_results = {}
        i = np.array(self.output[mode][metric]).argmax()

        # Output
        for m in self.output[mode].keys():
            best_results[m] = self.output[mode][m][i]

        return best_results

# Training and validation

# Set training arguments, set prebuild=True to use builtin PyG models otherwise False
args= {"epochs":100, 'lr':0.01, 'weight_decay':1e-5, 'prebuild':True, 'heads':2, 'hidden_dim': 128, 'dropout': 0.5}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_nodes = node_features_t.shape[1]
net = "GCN"

if net == "GCN":
    model = GCN(num_nodes = num_nodes).to(device)

# Push data to GPU
data_train = data_train.to(device)

# Setup training settings
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = torch.nn.BCELoss()

# Train
gnn_trainer_gat = GnnTrainer(model)
gnn_trainer_gat.train(data_train, optimizer, criterion, scheduler, args)
