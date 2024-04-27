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
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--best-val-f1', type=int, default= -np.inf)

    return parser.parse_known_args()[0]
    
def save_test_data(X_test, y_test):

    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)

    test_df = pd.concat([X_test_df, y_test_df], axis=1)

    test_df.to_csv('test_data.csv', index=False)
def get_metrics(y_true,y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
    aucpr = auc(recall_curve, precision_curve)
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, roc_auc, aucpr, cm

def print_metrics(accuracy, precision, recall, f1, roc_auc, aucpr):
    """
    Prints evaluation metrics in a tabular format.

    :param accuracy (float): Accuracy score.
    :param precision (float): Precision score.
    :param recall (float): Recall score.
    :param f1 (float): F1 score.
    :param roc_auc (float): ROC AUC score.
    :param aucpr (float): AUCPR score.

    """
    results = prettytable.PrettyTable(title='CNN Results')
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
        self.model = model
        
    def save_model(self,m_name):
        torch.save(self.model.state_dict(), f'model_{m_name}.pt')
    def train_val(self,train_loader, val_loader, epochs, optimizer,criterion,best_val_f1, m_name):
    
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
                    self.save_model(m_name)
                    print('Model Saved !!')

    def predict(self,test_loader, model_path):
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

        print_metrics(accuracy, precision, recall, f1, roc_auc, auc_pr)
                