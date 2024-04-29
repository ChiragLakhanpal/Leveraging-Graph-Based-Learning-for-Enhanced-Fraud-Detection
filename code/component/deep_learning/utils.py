import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, auc

import prettytable

def arg_parser():
    """
           Defines and parses the command-line arguments for the script.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--best-val-f1', type=int, default= -np.inf)
    parser.add_argument('--learning-rate', type=int, default=0.0001)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--hidden-size-2', type=int, default=256)
    parser.add_argument('--output-dim', type=int, default=1)
    parser.add_argument('--target', type=str, default="Is Fraud?")
    parser.add_argument('--cnn-input-size', type=int, default=1)
    parser.add_argument('--cnn-hidden-size', type=int, default=16)
    parser.add_argument('--cnn-out-size', type=int, default=32)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--kernel-size-pool', type=int, default=2)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--lstm-input-size', type=int, default=14)
    parser.add_argument('--lstm-n-layers', type=int, default=1)
    parser.add_argument('--cnn-lstm-out-size', type=int, default=64)
    parser.add_argument('--sample-size', type=int, default=0.6)
    
    return parser.parse_known_args()[0]

    
def save_test_data(X_test, y_test, data_path):
    """
    Saves the test data as a CSV file.

    :param X_test: Input features for the test data.
    :param y_test: Target labels for the test data.
    :param data_path: Path to the directory where the test data should be saved.
    """

    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)

    test_df = pd.concat([X_test_df, y_test_df], axis=1)

    test_file_path = os.path.join(data_path, 'test_data.csv')
    test_df.to_csv(test_file_path, index=False)
    
def dataset_stats(data, X_train, X_val, X_test, y_train, y_val, y_test):
    """
   Display statistics of the dataset.

   :param data: Complete dataset.
   :param X_train: Features of the training set.
   :param X_val: Features of the validation set.
   :param X_test: Features of the test set.
   :param y_train: Labels of the training set.
   :param y_val: Labels of the validation set.
   :param y_test: Labels of the test set.
   """
    table = prettytable.PrettyTable()

    # Display dataset stats
    print("Dataset Stats")

    table.field_names = ["Data", "Rows", "Columns", "Frauds", "Non-Frauds", "Fraud Percentage"]
    table.add_row(
        ["Complete Dataset", data.shape[0], data.shape[1], data['Is_Fraud'].sum(), data.shape[0] - data['Is_Fraud'].sum(),
         f"{round(data['Is_Fraud'].sum() / data.shape[0] * 100, 2)}%"])
    table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum(),
                   f"{round(y_train.sum() / y_train.shape[0] * 100, 2)}%"])
    table.add_row(["Validation", X_val.shape[0], X_val.shape[1], y_val.sum(), y_val.shape[0] - y_val.sum(),
                   f"{round(y_val.sum() / y_val.shape[0] * 100, 2)}%"])
    table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum(),
                   f"{round(y_test.sum() / y_test.shape[0] * 100, 2)}%"])
    print(table)

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

class DL_Trainer(object):
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
        
    def train_val(self,train_loader, val_loader, epochs, optimizer,criterion,best_val_f1, m_name):
        """
        Trains and validates the model.

        :param train_loader: DataLoader for training data.
        :param val_loader: DataLoader for validation data.
        :param epochs (int): Number of epochs for training.
        :param optimizer: Optimizer for training.
        :param criterion: Loss function criterion.
        :param best_val_f1 (float): Best F1 score achieved so far.
        :param m_name (str): Name of the model.
        """
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            all_targets = []
            all_outputs = []
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}') as pbar:
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_outputs.extend(outputs.cpu().detach().numpy())
    
                    pbar.update(1)
    
            targets = np.array(all_targets)
            outputs = np.array(all_outputs) > 0.5
            accuracy, precision, recall, f1, _, _, _ = get_metrics(targets, outputs)
    
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}, Train_accuracy: {accuracy:.4f}, Train_precision: {precision:.4f}, Train_recall: {recall:.4f}, Train_f1_score: {f1:.4f}")
    
            # Validation
            self.model.eval()
            all_val_targets = []
            all_val_outputs = []
            with torch.no_grad():
                with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1}') as pbar:
                    for inputs, targets in val_loader:
                        val_outputs = self.model(inputs)
                        val_loss = criterion(val_outputs, targets.unsqueeze(1))
                        all_val_targets.extend(targets.cpu().numpy())
                        all_val_outputs.extend(val_outputs.cpu().detach().numpy())
    
                        pbar.update(1)
    
                targets = np.array(all_val_targets)
                outputs = np.array(all_val_outputs) > 0.5
                accuracy, precision, recall, f1, _, _, _ = get_metrics(targets,outputs)
    
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {val_loss.item():.4f}, Val_accuracy: {accuracy:.4f}, Val_precision: {precision:.4f}, Val_recall: {recall:.4f}, Val_f1_score: {f1:.4f}")
    
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    self.save_model(m_name,save_dir="models")
                    print('Model Saved !!')

    def predict(self,test_loader, model_path,m_name):
        """
        Predicts using the saved trained model.

        :param test_loader: DataLoader for test data.
        :param model_path (str): Path to the saved model.
        :param m_name (str): Name of the model.
        """
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        all_test_targets = []
        all_test_outputs = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                test_outputs = self.model(inputs)
                all_test_targets.extend(targets.cpu().numpy())
                all_test_outputs.extend(test_outputs.cpu().detach().numpy())


        targets = np.array(all_test_targets)
        outputs = np.array(all_test_outputs) > 0.5
        accuracy, precision, recall, f1, roc_auc, auc_pr, _ = get_metrics(targets, outputs)

        print_metrics(accuracy, precision, recall, f1, roc_auc, auc_pr,m_name)
                