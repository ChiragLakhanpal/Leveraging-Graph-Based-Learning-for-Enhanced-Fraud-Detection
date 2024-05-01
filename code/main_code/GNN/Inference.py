import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import uuid

import dgl
import torch
from torch import nn
import torch.optim as optim
import prettytable
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
import os
import sys

sys.path.append('../../component')

from preprocess import *
from gnn.utils import *
from gnn.model import *

import argparse
import warnings
warnings.filterwarnings("ignore")

# Set random seed   

seed = 42

np.random.seed(seed)

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
parent_dir = os.getcwd()
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(script_dir, '..','..', '..', 'Data/Raw Data/data.csv')

data_path = os.path.normpath(data_path)


device = torch.device('cpu')

m_name = "GNN"
model_path = "models/"+f'model_{m_name}.pt'


def generate_aucpr_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=f'{model_name} (AUC={average_precision_score(y_test, y_prob):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('plots/pr_curve.png')
    plt.show()

def generate_aucroc_plot(y_test, y_prob_dict):
    plt.figure(figsize=(8, 6))
    for model_name, y_prob in y_prob_dict.items():
        if y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]  
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc_score(y_test, y_prob):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('plots/roc_curve.png')
    plt.show()

def main():
    args = arg_parser()


    data = read_data(data_path=data_path)

    data = preprocess_data(dataframe=data,
                           detect_binary=True,
                           numeric_dtype=False,
                           one_hot=True,
                           na_cleaner_mode="mode",
                           normalize=False,
                           balance=False,
                           sample=True,
                           sample_size=0.6,
                           stratify_column='Is Fraud?',
                           datetime_columns=['Time'],
                           clean_columns=['Amount'],
                           remove_columns=[],
                           consider_as_categorical=['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC',
                                                    'Errors?'],
                           target='Is Fraud?',
                           verbose=False)

    # Create transaction IDs for transaction node in graph

    data['id'] = [hash(uuid.uuid4()) for _ in range(len(data))]

    cols = ['id'] + [col for col in data.columns if col != 'id']
    data = data[cols]

    # Create card IDs for card node in graph
    data['Card_id'] = data['Card'].astype(str) + data['User'].astype(str)
    data.drop(columns=['Card', 'User'], inplace=True)

    # rename Merchant Name to Merchant_id
    data.rename(columns={'Merchant_Name': 'Merchant_id'}, inplace=True)

    # Create transaction feature tensors

    transaction_feats = data.drop(['id', 'Card_id', 'Is_Fraud', 'Merchant_id'], axis=1)
    classified_idx = transaction_feats.index

    scaler = StandardScaler()
    transaction_feats = scaler.fit_transform(transaction_feats)
    transaction_feats = torch.tensor(transaction_feats, dtype=torch.float)

    # Create edge indexes for different edge types

    transaction_to_card = data[['id', 'Card_id']].astype(int)
    transaction_to_merchant = data[['id', 'Merchant_id']].astype(int)

    nodes = transaction_to_card['id'].unique()
    map_id = {j: i for i, j in enumerate(nodes)}

    nodes_1 = transaction_to_card['Card_id'].unique()
    map_id_1 = {j: i for i, j in enumerate(nodes_1)}

    nodes_2 = transaction_to_merchant['Merchant_id'].unique()
    map_id_2 = {j: i for i, j in enumerate(nodes_2)}

    card_edges = transaction_to_card.copy()
    card_edges.Card_id = card_edges.Card_id.map(map_id_1)
    card_edges.id = card_edges.id.map(map_id)

    merchant_edges = transaction_to_merchant.copy()
    merchant_edges.Merchant_id = merchant_edges.Merchant_id.map(map_id_2)
    merchant_edges.id = merchant_edges.id.map(map_id)

    # Create DGL graph

    graph_data = {
        ('Card_id', 'Card_id<>transaction', 'transaction'): (card_edges['Card_id'], card_edges['id']),
        ('Merchant_id', 'Merchant_id<>transaction', 'transaction'): (merchant_edges['Merchant_id'], merchant_edges['id']),
        ('transaction', 'self_relation', 'transaction'): (card_edges['id'], card_edges['id']),
        ('transaction', 'transaction<>Card_id', 'Card_id'): (card_edges['id'], card_edges['Card_id']),
        ('transaction', 'transaction<>Merchant_id', 'Merchant_id'): (merchant_edges['id'], merchant_edges['Merchant_id'])
    }

    g = dgl.heterograph(graph_data)

    g.nodes[args.target_node].data['y'] = torch.tensor(data[args.target], dtype=torch.float)
    g.to(device)

    # Creating a train test split

    train_idx, test_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2,
                                           stratify=data[args.target])
    train_idx, valid_idx = train_test_split(train_idx, random_state=42, test_size=0.2)
    

    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    labels = g.nodes[args.target_node].data['y'].long()
    in_size = transaction_feats.shape[1]

    model = HeteroRGCN(ntype_dict, g.etypes, in_size, args.hidden_size, args.out_size, args.n_layers,args.target_node)
    model.to(device)

    # Calling the trainer object to test data

    trainer = GNN_Trainer(model)
    accuracy, precision, recall, f1, roc_auc, auc_pr, targets, outputs, probs = trainer.predict(g, transaction_feats, test_idx, model_path, m_name, args.target_node, labels)

    # Display the evaluation results
    results_dict = {
        m_name: {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC AUC': roc_auc,
            'PR AUC': auc_pr
        }
    }

    y_prob_dict = {m_name: probs}
    
    generate_aucpr_plot(targets, y_prob_dict)

    # Generate AUCROC plot
    generate_aucroc_plot(targets, y_prob_dict)
    
    return targets, outputs, probs

if __name__ == "__main__":
    targets, outputs, probs = main()





