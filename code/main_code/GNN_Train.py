import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import uuid

import dgl
import torch
from torch import nn
import torch.optim as optim

import sys
sys.path.append('../component')

from preprocess import preprocess_data
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

# Set device as cpu 

device = torch.device('cpu')

m_name = "GNN"

def main():

    args = arg_parser()

    # Read the data and run the preprocess function

    data = pd.read_csv('../../Data/card_transaction.v1.csv')

    data = preprocess_data(dataframe=data,
                    detect_binary=True,
                    numeric_dtype=False,
                    one_hot=True,
                    na_cleaner_mode="mode",
                    normalize=False,
                    balance=False,
                    sample=True,
                    sample_size = 0.6,
                    stratify_column = 'Is Fraud?',
                    datetime_columns = ['Time'],
                    clean_columns = ['Amount'],
                    remove_columns = [],
                consider_as_categorical = ['Use Chip', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?'],
                               target = 'Is Fraud?',
                    verbose = True)
    
    # Create transaction IDs for transaction node in graph
    
    data['id'] = [hash(uuid.uuid4()) for _ in range(len(data))]
    
    cols = ['id'] + [col for col in data.columns if col != 'id']
    data = data[cols]
    
    # Create card IDs for card node in graph

    data['Card_id'] = data['Card'].astype(str) + data['User'].astype(str)
    data.drop(columns = ['Card', 'User'], inplace = True)
    
    # Rename Merchant Name to Merchant_id
    data.rename(columns={'Merchant_Name':'Merchant_id'},inplace=True)
    
    # Create transaction feature tensors

    transaction_feats = data.drop(['id','Card_id','Is_Fraud','Merchant_id'],axis =1)
    classified_idx = transaction_feats.index
    
    scaler = StandardScaler()
    transaction_feats = scaler.fit_transform(transaction_feats)
    transaction_feats = torch.tensor(transaction_feats,dtype=torch.float)

    # Create edge indexes for different edge types
    
    transaction_to_card = data[['id','Card_id']].astype(int)
    transaction_to_merchant = data[['id','Merchant_id']].astype(int)
    
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
    
    print_graph_info(g)
    
    # Creating a train test split
    
    train_idx, test_idx = train_test_split(classified_idx.values, random_state=42, test_size=0.2, stratify=data[args.target])
    train_idx, valid_idx = train_test_split(train_idx, random_state=42, test_size=0.2)
    

    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    labels = g.nodes[args.target_node].data['y'].long()
    in_size = transaction_feats.shape[1]
    
    model = HeteroRGCN(ntype_dict,g.etypes, in_size, args.hidden_size, args.out_size, args.n_layers,args.target_node)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay = args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Calling the trainer object to train and validate data

    trainer = GNN_Trainer(model)
    trainer.train_val(g,transaction_feats,args.epochs, train_idx, valid_idx,optimizer, criterion, args.best_val_f1, m_name,labels,args.target_node)

if __name__ == "__main__":
    main()





