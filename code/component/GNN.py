import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
import uuid

import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler,DataLoader
from class_GNN import GCN,GAT,GnnTrainer,MetricManager
from preprocess import preprocess_data

import argparse
import warnings
warnings.filterwarnings("ignore")

def main():
    data = pl.read_csv('/home/ec2-user/Capstone/card_transaction.v1.csv')

    data_processed = preprocess_data(data)

    data_processed = data_processed.to_pandas()

    data_processed = data_processed.sample(n=50000).reset_index(drop=True)

    # %%
    # Assign a unique id to each row
    data_processed['id'] = [uuid.uuid4() for _ in range(len(data_processed))]

    # To make 'id' the first column, you can use DataFrame reindex with columns sorted to your preference
    cols = ['id'] + [col for col in data_processed.columns if col != 'id']
    data_processed = data_processed[cols]

    # %%
    # Create label dataset for GNN model with id and label columns
    df_classes = data_processed[['id', 'is_fraud']]

    # %%
    # Create features dataset for GNN model with id and features columns

    df_features = data_processed.drop(columns=['is_fraud', 'merchant_id', 'card_id'])

    # %%
    # Create edges dataset for GNN model with source and target columns
    user_to_transactions = data_processed.groupby('card_id')['id'].apply(list).to_dict()

    # %%
    edges_list = []

    for transactions in user_to_transactions.values():
        if len(transactions) > 1:
            for i in range(len(transactions)):
                for j in range(i + 1, len(transactions)):
                    # Append both directions of the edge to the list
                    edges_list.append({'source': transactions[i], 'target': transactions[j]})
                    edges_list.append({'source': transactions[j], 'target': transactions[i]})

    # Create the DataFrame from the list of edge dictionaries
    df_edges = pd.DataFrame(edges_list)

    # %%
    df_merge = df_features.merge(df_classes, how='left', on='id')
    df_merge = df_merge.sort_values('id').reset_index(drop=True)
    df_merge.head()

    # %%
    import torch
    import torch_geometric

    # Setup trans ID to node ID mapping
    nodes = df_merge['id'].values

    map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

    # Create edge df that has transID mapped to nodeIDs
    edges = df_edges.copy()
    edges.source = edges.source.map(map_id)  # get nodes idx1 from edges list and filtered data
    edges.target = edges.target.map(map_id)

    edges = edges.astype(int)

    edge_index = np.array(edges.values).T  # convert into an array
    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()  # create a tensor

    print("shape of edge index is {}".format(edge_index.shape))
    edge_index

    # %%
    # Define labels
    labels = df_merge['is_fraud'].values
    print("lables", np.unique(labels))
    labels

    # %%
    # mapping txIds to corresponding indices, to pass node features to the model

    node_features = df_merge.drop(['id'], axis=1).copy()
    # node_features[0] = node_features[0].map(map_id) # Convert transaction ID to node ID \
    print("unique=", node_features["is_fraud"].unique())

    # Retain known vs unknown IDs
    classified_idx = node_features.index

    #classified_illicit_idx = node_features['is_fraud'].loc[node_features['is_fraud'] == 1].index  # filter on illicit labels
    #classified_licit_idx = node_features['is_fraud'].loc[node_features['is_fraud'] == 0].index  # filter on licit labels

    # Drop unwanted columns, 0 = transID, 1=time period, class = labels
    node_features = node_features.drop(columns=['is_fraud'])

    # Convert to tensor
    node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double),
                                   dtype=torch.float)  # drop unused columns
    print(node_features_t)

    # %%
    from preprocess import split_data

    # Create a known vs unknown mask
    train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.15)
    print("train_idx size {}".format(len(train_idx)))
    print("test_idx size {}".format(len(valid_idx)))

    # %%
    from torch_geometric.data import Data
    import torch_geometric.transforms as T

    data_train = Data(x=node_features_t, edge_index=edge_index,
                      y=torch.tensor(labels, dtype=torch.float))
    print(data_train)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train = T.ToUndirected()(data_train)
    data_train = data_train.to(device)

    # Add in the train and valid idx
    data_train.train_idx = train_idx
    data_train.valid_idx = valid_idx

    # %%
    from class_GNN import GCN, GAT, GnnTrainer, MetricManager

    # Set training arguments
    args = {"epochs": 20, 'lr': 0.01, 'weight_decay': 1e-5, 'heads': 2, 'hidden_dim': 128, 'dropout': 0.5}
    num_nodes = node_features_t.shape[1]

    # # Initialize the argument parser
    # parser = argparse.ArgumentParser(description='Specify network architecture')

    # # Add argument for specifying the network architecture
    # parser.add_argument('--net', type=str, default='GCN', choices=['GCN', 'GAT'],
    #                     help='Network architecture (GCN or GAT)')

    # # Parse the command-line arguments
    # arg = parser.parse_args()
    # # Now you can access the selected network architecture using args.net
    # if arg.net == "GCN":
    #     model = GCN(num_nodes=num_nodes).to(device)
    # elif arg.net == "GAT":
    #     model = GAT(data_train.num_node_features, args['hidden_dim'], 1, args).to(device)

    model = GCN(num_nodes=num_nodes).to(device)
    # Setup training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.BCELoss()

    # Train
    gnn_trainer_gat = GnnTrainer(model)
    gnn_trainer_gat.train(data_train, optimizer, criterion, scheduler, args)

if __name__ == '__main__':
    main()
