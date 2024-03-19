import pickle
import numpy as np
import polars as pl
import pandas as pd
import torch
import uuid
import prettytable

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv,GATv2Conv
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler, DataLoader, NeighborLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score
from preprocess import preprocess_data

import argparse
import warnings
warnings.filterwarnings("ignore")

# def CustomDataLoader(data):
#
#     train_loader = NeighborLoader(data,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)
#
#     val_loader = NeighborLoader(data,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)
#
#     test_loader = NeighborLoader(data,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)
#
#     return train_loader, val_loader, test_loader

# GCNConv Class
class GCN(torch.nn.Module):
    def __init__(self,num_nodes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_nodes, 8)
        self.conv2 = GCNConv(8,2)
        self.classifier = Linear(2, 1)
    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h1 = self.conv2(h, edge_index)
        embeddings = F.relu(h1)
        out = self.classifier(embeddings)
        return F.sigmoid(out)

# GATConv Class
class GAT(torch.nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,args):
        super(GAT, self).__init__()
        #use our gat message passing
        self.conv1 = GATConv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATConv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])
        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout'] ),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        x = self.post_mp(x)
        return F.sigmoid(x)
    
class GATv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GATv2, self).__init__()
        # use our gat message passing
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=args['heads'])
        self.conv2 = GATv2Conv(args['heads'] * hidden_dim, hidden_dim, heads=args['heads'])

        self.post_mp = nn.Sequential(
            nn.Linear(args['heads'] * hidden_dim, hidden_dim), nn.Dropout(args['dropout']),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, data, adj=None):
        x, edge_index = data.x, data.edge_index
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.dropout(F.relu(x), p=args['dropout'], training=self.training)
        # MLP output
        x = self.post_mp(x)
        return F.sigmoid(x)


# Creating a model trainer object
class GnnTrainer(object):

    def __init__(self, model):
        self.model = model
        self.metric_manager = MetricManager(modes=["train", "val"])

    def train(self,data_train,optimizer, criterion, scheduler,args):

        #self.data_train = train_loader
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
            #scheduler.step()

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


    def predict(self,data_train,unclassified_only = True,threshold=0.5):
        # evaluate model:
        self.model.eval()
        
        out = self.model(data_train)
        out = out.reshape((data_train.x.shape[0]))
        target_labels = data_train.y.detach().cpu().numpy()[data_train.test_idx]

        if unclassified_only:

            pred_scores = out.detach().cpu().numpy()[data_train.test_idx]
        else:
            pred_scores = out.detach().cpu().numpy()

        pred_labels = (pred_scores > threshold).astype(int)

        return pred_labels, pred_scores, target_labels

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
            self.output[mode]["precision"] = []
            self.output[mode]["recall"] = []
            self.output[mode]["cm"] = []

    def store_metrics(self, mode, pred_scores, target_labels, threshold=0.5):

        # calculate metrics
        pred_labels = pred_scores>threshold
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

    # def test_metrics(self,y_test,y_pred,y_score):
    #     accuracy = accuracy_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred)
    #     recall = recall_score(y_test, y_pred)
    #     f1 = f1_score(y_test, y_pred)
    #     cm = confusion_matrix(y_test, y_pred)
    #     roc_auc = roc_auc_score(y_test, y_score)
    #     precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    #     aucpr = auc(recall_curve, precision_curve)
    #
    #     results = prettytable.PrettyTable(title="Metric Table")
    #     results.field_names = ["Metric", "Value"]
    #     results.add_row(["Accuracy", accuracy])
    #     results.add_row(["Precision", precision])
    #     results.add_row(["Recall", recall])
    #     results.add_row(["F1 Score", f1])
    #     results.add_row(["ROC AUC", roc_auc])
    #     results.add_row(["AUCPR", aucpr])
    #     print(results)
    #
    #     return accuracy, precision, recall, f1, cm, roc_auc, aucpr

args= {"epochs":500, 'lr':0.001, 'weight_decay':1e-5, 'heads':2, 'hidden_dim': 128, 'dropout': 0.5}



