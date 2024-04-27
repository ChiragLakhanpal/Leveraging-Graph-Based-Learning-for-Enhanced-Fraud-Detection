import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score, auc, precision_recall_curve
import uuid
import prettytable
import itertools

import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborSampler,DataLoader, NeighborLoader

from class_GNN import CustomDataLoader,GCN,GAT,GATv2,GnnTrainer,MetricManager#,print_metrics
from preprocess import preprocess_data,split_data

import argparse
import warnings
warnings.filterwarnings("ignore")

# Set random seed
seed = 42
np.random.seed(seed)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def main():

    # Reading the data
    data = pl.read_csv('/home/ec2-user/Capstone/Data/Sampled Dataset.csv')

    # Applying the preprocess function
    data_processed = preprocess_data(data)
    print("Shape of data: ", data_processed.shape)
    data_processed = data_processed.to_pandas()

    data_processed = data_processed.sample(n=200000, random_state=42).reset_index(drop=True)
    print("Is Fraud Value Counts", data_processed['is_fraud'].value_counts())

    # Assigning a unique id to each row
    data_processed['id'] = [hash(uuid.uuid4()) for _ in range(len(data_processed))]

    # Making id the first column in dataframe
    cols = ['id'] + [col for col in data_processed.columns if col != 'id']
    data_processed = data_processed[cols]

    data_processed['edge'] = data_processed['card_id'].astype(str)+data_processed['merchant_id'].astype(str)+data_processed['hour'].astype(str)
    print(data_processed['edge'].nunique())
    # Create label dataset for GNN model
    df_classes = data_processed[['id', 'is_fraud']]

    # Create features dataset for GNN model with id and features columns

    df_features = data_processed.drop(columns=['is_fraud', 'card_id','edge','zip'])




    # Group transactions by card_id
    user_to_transactions = data_processed.groupby('edge')['id'].apply(list).to_dict()

    # Initialize edges list
    edges_list = []

    # Iterate over transactions for each user
    for transactions in user_to_transactions.values():
        if len(transactions) > 1:
            # Generate pairs of transactions using itertools.combinations
            pairs = itertools.combinations(transactions, 2)
            # Append both directions of the edge to the list
            for pair in pairs:
                edges_list.append({'source': pair[0], 'target': pair[1]})
                #edges_list.append({'source': pair[1], 'target': pair[0]})

    # Create the DataFrame from the list of edge dictionaries
    df_edges = pd.DataFrame(edges_list)

    df_merge = df_features.merge(df_classes, how='left', on='id')
    df_merge = df_merge.sort_values('id').reset_index(drop=True)
    df_merge.head()

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

    print("Shape of edge index is {}".format(edge_index.shape))

    # Define labels
    labels = df_merge['is_fraud'].values
    print("Unique Labels", np.unique(labels))

    # Mapping txIds to corresponding indices, to pass node features to the model

    node_features = df_merge.drop(['id'], axis=1).copy()

    # Retain known vs unknown IDs
    classified_idx = node_features.index

    # Drop unwanted columns
    node_features = node_features.drop(columns=['is_fraud'])

    # Convert to tensor
    node_features_t = torch.tensor(np.array(node_features.values, dtype=np.double),
                                   dtype=torch.float)  # drop unused columns
    print(node_features_t)

    # Creating a train test split
    train_idx, test_idx = train_test_split(classified_idx.values, random_state=42,
                                           test_size=0.2, stratify=data_processed['is_fraud'])
    train_idx, valid_idx = train_test_split(train_idx, test_size=0.2,random_state=42,
                                            stratify=data_processed.iloc[train_idx]['is_fraud'])
    print("train_idx size {}".format(len(train_idx)))
    print("valid_idx size {}".format(len(valid_idx)))
    print("test_idx size {}".format(len(test_idx)))

    data_train = Data(x=node_features_t, edge_index=edge_index,
                      y=torch.tensor(labels, dtype=torch.float))
    print(data_train)

    #train_loader, val_loader, test_loader = CustomDataLoader(data_train)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_train = T.ToUndirected()(data_train)
    data_train = data_train.to(device)

    # Add in the train and valid idx
    data_train.train_idx = train_idx
    data_train.valid_idx = valid_idx
    data_train.test_idx = test_idx

    print("Node features shape: ", data_train.x.shape)
    print("Edge index shape: ", data_train.edge_index.shape)
    print("Labels shape: ", data_train.y.shape)
    print("Number of nodes: ", data_train.num_nodes)
    print("Number of edges: ", data_train.num_edges)
    print("Number of features: ", data_train.num_features)
    print("Dataset Shape: ", data_train.x.shape)
    print("df_merge shape: ", df_merge.shape)
    print("df_features shape: ", df_features.shape)
    print("df_classes shape: ", df_classes.shape)
    print("df_edges shape: ", df_edges.shape)
    print("df_merge columns: ", df_merge.columns)
    print("df_features columns: ", df_features.columns)
    print("df_classes columns: ", df_classes.columns)
    print("df_edges columns: ", df_edges.columns)

    # train_loader = NeighborLoader(data_train,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)
    #
    # val_loader = NeighborLoader(data_train,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)
    #
    # test_loader = NeighborLoader(data_train,num_neighbors=[15,10], batch_size=128,directed=False, shuffle=True)

    # Set training arguments
    args = {"epochs": 100, 'lr': 0.001, 'weight_decay': 5e-4, 'heads': 2, 'hidden_dim': 128, 'dropout': 0.5}
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

    model = GAT(data_train.num_node_features, args['hidden_dim'], 1,args).to(device)
    #model = GAT(args['hidden_dim'],1,args).to(device)
    #model = GCN(num_nodes=num_nodes).to(device)

    # Setup training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1,weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=1, verbose=True)
    criterion = torch.nn.BCELoss()

    # Training the data
    gnn_trainer_gat = GnnTrainer(model)
    gnn_trainer_gat.train(data_train,optimizer, criterion, scheduler,args)

    # Evaluation on test set
    y_pred, y_score, y_test = gnn_trainer_gat.predict(data_train)

    # Testing statistics
    print("Testing Statistics: ")
    print(np.sum(y_pred == 1))
    print(np.sum(y_test == 1))
    # Count occurrences of unique elements
    unique_elements, counts = np.unique(y_score, return_counts=True)
    # Create a dictionary mapping each unique element to its count
    element_counts = dict(zip(unique_elements, counts))
    print("Element Counts:", element_counts)
    print(len(y_pred))
    print(len(y_test))

    #print_metrics(y_test,y_pred,y_score)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    roc_auc = roc_auc_score(y_test, y_score)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    aucpr = auc(recall_curve, precision_curve)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)

    results = prettytable.PrettyTable(title="Metric Table")
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)

if __name__ == '__main__':
    main()
