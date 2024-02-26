import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from class_GNN import GCN,GAT,GnnTrainer,MetricManager
from preprocess import preprocess_data

import warnings
warnings.filterwarnings("ignore")

def main():
    df_raw = pd.read_csv("card_transaction.v1.csv", nrows=1000000)

    # Preprocess the data
    df = preprocess_data(df_raw)
    print(df)

    # Create a dataframe for edges
    df_edges = df[["card_id", "merchant_id"]].copy()

    # Create a dataframe merge with node features ids and classes

    df_card = df.drop(["merchant_id"], axis=1)
    df_card = df_card.rename(columns={"card_id": "id"})

    df_merchant = df.drop(["card_id"], axis=1)
    df_merchant = df_merchant.rename(columns={"merchant_id": "id"})

    df_merge = pd.concat([df_card, df_merchant]).reset_index(drop=True)
    print(df.merge)

    # Setup transaction ID to node ID mapping
    nodes = df_merge["id"].values

    # Mapping nodes to indexes
    map_id = {j: i for i, j in enumerate(nodes)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # Define labels
    labels = df_merge['is_fraud'].values
    print("Unique Labels", np.unique(labels))
    print("Labels Array")
    print(labels)

    # Mapping txIds to corresponding indices, to pass node features to the model
    node_features = df_merge.drop(['id'], axis=1).copy()
    print(node_features)
    print("Unique classes in node features=", node_features["is_fraud"].unique())

    # Get node features index
    classified_idx = node_features.index

    # Drop unwanted columns
    node_features = node_features.drop(columns=['is_fraud'])

    # Convert to tensor
    node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double), dtype=torch.float)
    print("Node features tensor")
    print(node_features_t)

    # Train val splits
    train_idx, valid_idx = train_test_split(classified_idx.values, test_size=0.2)
    print("train_idx size {}".format(len(train_idx)))
    print("test_idx size {}".format(len(valid_idx)))

    # Creating a PyG Dataset
    data_train = Data(x=node_features_t, edge_index=edge_index, y=torch.tensor(labels, dtype=torch.float))
    data_train = T.ToUndirected()(data_train)
    data_train = data_train.to(device)
    print(data_train)

    print("Number of nodes: ", data_train.num_nodes)
    print("Number of edges: ", data_train.num_edges)
    print("Number of features per node: ", data_train.num_node_features)
    print("Number of edge types: ", data_train.num_edge_types)
    print("Shape of edge_index: ", data_train.edge_index.shape)

    # Add in the train and valid idx
    data_train.train_idx = train_idx
    data_train.valid_idx = valid_idx

    # Set training arguments
    args= {"epochs":100, 'lr':0.01, 'weight_decay':1e-5, 'heads':2, 'hidden_dim': 128, 'dropout': 0.5}
    num_nodes = node_features_t.shape[1]
    net = "GCN"

    if net == "GAT":
        model = GCN(num_nodes = num_nodes).to(device)
    else:
        model = GAT(data_train.num_node_features, args['hidden_dim'], 1, args).to(device)

    # Setup training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.BCELoss()

    # Train
    gnn_trainer_gat = GnnTrainer(model)
    gnn_trainer_gat.train(data_train, optimizer, criterion, scheduler, args)

if __name__ == '__main__':
    main()
