import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc

import prettytable
def arg_parser():
    """
       Defines and parses the command-line arguments for the script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=int, default=0.01)
    parser.add_argument('--out-size', type=int, default=2)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--target', type=str, default="Is_Fraud")
    parser.add_argument('--weight-decay', type=int, default = 5e-4)
    parser.add_argument('--target-node', type=str, default="transaction")
    parser.add_argument('--best-val-f1', type=int, default=-np.inf)
    return parser.parse_known_args()[0]

def print_graph_info(g):
    """
           Prints graph properties and information
    """

    print(g)
    print("Graph properties: ")
    print("Total number of nodes in graph: ", g.num_nodes())
    ntype_dict = {n_type: g.number_of_nodes(n_type) for n_type in g.ntypes}
    print("Number of nodes for each node type:", ntype_dict)
    print("Edge Dictionary for different edge types: ")
    print("Card_To_Transaction Edges: ", g.edges(etype='Card_id<>transaction'))
    print("Merchant_To_Transaction Edges: ", g.edges(etype='Merchant_id<>transaction'))
    print("Transaction_Self_Loop: ", g.edges(etype='self_relation'))
    print("Transaction_To_Card Edges: ", g.edges(etype='transaction<>Card_id'))
    print("Transaction_To_Merchant Edges: ", g.edges(etype='transaction<>Merchant_id'))

def get_metrics(y_true,y_pred):
    """
       Computes various evaluation metrics for binary classification.

       :param y_true : Ground truth labels.
       :param y_pred : Predicted labels.
       :returns: accuracy, precision, recall, f1, roc_auc, aucpr and cm metrics

    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    aucpr = auc(recall_curve, precision_curve)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, aucpr, cm

def print_metrics(accuracy, precision, recall, f1, roc_auc, aucpr, m_name):
    """
    Prints evaluation metrics in a tabular format.

    :param accuracy (float): Accuracy score.
    :param precision (float): Precision score.
    :param recall (float): Recall score.
    :param f1 (float): F1 score.
    :param roc_auc (float): ROC AUC score.
    :param aucpr (float): AUCPR score.

    """
    results = prettytable.PrettyTable(title=f'{m_name} Results')
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)


class GNN_Trainer(object):
    def __init__(self, model):
        """
        Initializes the GNN_Trainer object.

        :param model (nn.Module): The PyTorch model to be trained and evaluated.

        """
        self.model = model

    def save_model(self, m_name, save_dir="models"):
        """
        Saves the model's state dictionary to a file.

        :param m_name (str): Name of the model.
        :param save_dir (str, optional): Directory to save the model. Default is "models".

        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'model_{m_name}.pt')
        torch.save(self.model.state_dict(), save_path)

    def train_val(self, g, transaction_feats,num_epochs, train_idx, val_idx,optimizer, criterion, best_val_f1, m_name,labels,target_node):
        """
       Trains and evaluates the model on the training and validation sets.

       :param g (DGLHeteroGraph): The heterogeneous graph.
       :param transaction_feats (torch.Tensor): Input features for the transaction nodes.
       :param num_epochs (int): Number of training epochs.
       :param train_idx (dict): Dictionary of node indices for the training set.
       :param val_idx (dict): Dictionary of node indices for the validation set.
       :param optimizer (torch.optim.Optimizer): Optimizer for training.
       :param criterion (nn.Module): Loss function.
       :param best_val_f1 (float): Best validation F1 score so far.
       :param m_name (str): Name of the model.
       :param labels (torch.Tensor): Ground truth labels.
       :param target_node (str): Name of the target node type.

       """
        total_loss = 0
        for epoch in range(num_epochs):
    
            start_time = time.time()

            self.model.train()
            optimizer.zero_grad()
            out = self.model(g,transaction_feats,target_node)
            pred = out
            pred_c = out.argmax(1)
            loss = criterion(pred[train_idx], labels[train_idx])
            target = labels[train_idx]
            pred_scores = pred_c[train_idx]


            threshold = 0.5
            pred = pred_scores > threshold

            accuracy, precision, recall, f1, _, _, _ = get_metrics(target, pred)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            end_time = time.time()

            duration = end_time - start_time


            print(
                "Epoch: {} - Duration: {:.2f} - Loss: {:.4f}  - Accuracy Train: {:.4f} - Recall Train: {:.2f}  - Precision Train: {:.2f}  - F1_Score Train: {:.2f}".format(
                    f'{epoch+1}/{num_epochs}', duration, loss.item(), accuracy, recall, precision, f1))

        

            start_time = time.time()
            self.model.eval()
            with torch.no_grad():
                target = labels[val_idx]
                pred_scores = pred_c[val_idx]
                threshold = 0.5
                pred = pred_scores > threshold

                accuracy, precision, recall, f1, _, _, _ = get_metrics(target, pred)

                if f1 > best_val_f1:
                    best_val_f1 = f1
                    self.save_model(m_name, save_dir="models")
                    print('Model Saved !!')

            end_time = time.time()

            duration = end_time - start_time


            print(
                    "Epoch: {} - Duration: {:.2f} - Loss: {:.4f} - Accuracy Val: {:.4f} - Recall Val: {:.2f}  - Precision Val: {:.2f}  - F1_score Val: {:.2f}".format(
                        f'{epoch+1}/{num_epochs}', duration,loss.item(), accuracy, recall, precision, f1))

    def predict(self, g, transaction_feats,test_idx,model_path, m_name, target_node,labels):
        """
           Evaluates the model on the test set and prints the evaluation metrics.

           :param g (DGLHeteroGraph): The heterogeneous graph.
           :param transaction_feats (torch.Tensor): Input features for the transaction nodes.
           :param test_idx (dict): Dictionary of node indices for the test set.
           :param model_path (str): Path to the saved model.
           :param m_name (str): Name of the model.
           :param target_node (str): Name of the target node type.
           :param labels (torch.Tensor): Ground truth labels.
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            out = self.model(g, transaction_feats,target_node)
            target = labels[test_idx]
            pred_c = out.argmax(1)
            pred_scores = pred_c[test_idx]
            threshold = 0.5
            pred = pred_scores > threshold

            targets = target.cpu().numpy()
            outputs = pred.cpu().numpy()
            probs = F.softmax(out[test_idx], dim=1).cpu().numpy()

            accuracy, precision, recall, f1, roc_auc, auc_pr, _ = get_metrics(targets, outputs)
            print_metrics(accuracy, precision, recall, f1, roc_auc, auc_pr, m_name)

            return accuracy, precision, recall, f1, roc_auc, auc_pr, targets, outputs, probs


    