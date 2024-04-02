
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
# necessary imports of this part
from inductiveGRL.graphconstruction import GraphConstruction
from inductiveGRL.hinsage import HinSAGE_Representation_Learner
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from stellargraph.mapper import HinSAGENodeGenerator



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import prettytable
# optimzer 
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

df = pd.read_csv("/home/ec2-user/Capstone/processed_data_inductiveGRL.csv")
# 
# Global parameters:
embedding_size = 256
add_additional_data = True
batch_size = 128
ephocs = 1
threshold = 0

#GraphSAGE parameters
num_samples = [2,32]
embedding_node_type = "transaction"

# we will take 70% of our dataset as traindata
cutoff = round(0.7*len(df)) 
train_data = df.head(cutoff)
inductive_data = df.tail(len(df)-cutoff)

print('The distribution of fraud for the train data is:\n', train_data['Is Fraud?'].value_counts())
print('The distribution of fraud for the inductive data is:\n', inductive_data['Is Fraud?'].value_counts())

transaction_node_data = train_data.drop("card_id", axis=1).drop("Merchant Name", axis=1).drop('Is Fraud?', axis=1)
client_node_data = pd.DataFrame([1]*len(train_data["card_id"].unique())).set_index(train_data["card_id"].unique())
merchant_node_data = pd.DataFrame([1]*len(train_data["Merchant Name"].unique())).set_index(train_data["Merchant Name"].unique())

nodes = {"client":train_data.card_id, "merchant":train_data["Merchant Name"], "transaction":train_data.index}
edges = [zip(train_data.card_id, train_data.index),zip(train_data["Merchant Name"], train_data.index)]
features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

graph = GraphConstruction(nodes, edges, features)
S = graph.get_stellargraph()
print(S.info())

hinsage = HinSAGE_Representation_Learner(embedding_size, num_samples, embedding_node_type)

trained_hinsage_model, train_emb = hinsage.train_hinsage(S, 
                                                         list(train_data.index), 
                                                         train_data['Is Fraud?'], 
                                                         batch_size=batch_size, 
                                                         epochs=ephocs)


# # Prepare the test data
# test_data = inductive_data.reset_index(drop=True)
# test_transaction_node_data = test_data.drop("card_id", axis=1).drop("Merchant Name", axis=1).drop('Is Fraud?', axis=1)
# test_client_node_data = pd.DataFrame([1]*len(test_data["card_id"].unique())).set_index(test_data["card_id"].unique())
# test_merchant_node_data = pd.DataFrame([1]*len(test_data["Merchant Name"].unique())).set_index(test_data["Merchant Name"].unique())
# test_nodes = {"client":test_data.card_id, "merchant":test_data["Merchant Name"], "transaction":test_data.index}
# test_edges = [zip(test_data.card_id, test_data.index),zip(test_data["Merchant Name"], test_data.index)]
# test_features = {"transaction": test_transaction_node_data, 'client': test_client_node_data, 'merchant': test_merchant_node_data}
# test_graph = GraphConstruction(test_nodes, test_edges, test_features)
# test_S = test_graph.get_stellargraph()

# print(test_S.info())

# # Generate embeddings for test data
# test_gen = HinSAGENodeGenerator(test_S, batch_size=batch_size, num_samples=num_samples, head_node_type=embedding_node_type)
# test_emb = trained_hinsage_model.predict(test_gen.flow(list(test_data.index)))

# # Make predictions on test data
# y_pred_probs = test_emb[:, 0]  # Assuming the positive class probability is in the first column
# y_true = test_data['Is Fraud?'].values

# print('test_emb:', test_emb)
# # Apply a threshold to convert probabilities to binary predictions

# y_pred = (y_pred_probs > threshold).astype(int)

# print("y_true shape:", y_true.shape)
# print("y_pred shape:", y_pred.shape)
# print("y_true unique values:", np.unique(y_true))
# print("y_pred unique values:", np.unique(y_pred))

# # Calculate metrics
# accuracy = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
# aucpr = average_precision_score(y_true, y_pred_probs)
# roc_auc = roc_auc_score(y_true, y_pred_probs)
# cm = confusion_matrix(y_true, y_pred)

# results = prettytable.PrettyTable(title='GNN (3 Nodes) Results')
# results.field_names = ["Metric", "Value"]
# results.add_row(["Accuracy", accuracy])
# results.add_row(["Precision", precision])
# results.add_row(["Recall", recall])
# results.add_row(["F1 Score", f1])
# results.add_row(["ROC AUC", roc_auc])
# results.add_row(["AUCPR", aucpr])
# print(cm)
# print(results)

pd.options.mode.chained_assignment = None

train_data['index'] = train_data.index
inductive_data['index'] = inductive_data.index
inductive_graph_data = pd.concat((train_data,inductive_data))
inductive_graph_data = inductive_graph_data.set_index(inductive_graph_data['index']).drop("index",axis = 1)

transaction_node_data = inductive_graph_data.drop("card_id", axis=1).drop("Merchant Name", axis=1).drop("Is Fraud?", axis=1)
client_node_data = pd.DataFrame([1]*len(inductive_graph_data['card_id'].unique())).set_index(inductive_graph_data['card_id'].unique())
merchant_node_data = pd.DataFrame([1]*len(inductive_graph_data["Merchant Name"].unique())).set_index(inductive_graph_data["Merchant Name"].unique())

nodes = {"client":inductive_graph_data.card_id, "merchant":inductive_graph_data["Merchant Name"], "transaction":inductive_graph_data.index}
edges = [zip(inductive_graph_data.card_id, inductive_graph_data.index),zip(inductive_graph_data["Merchant Name"], inductive_graph_data.index)]
features = {"transaction": transaction_node_data, 'client': client_node_data, 'merchant': merchant_node_data}

graph = GraphConstruction(nodes, edges, features)
S = graph.get_stellargraph()
print(S.info())

inductive_emb = hinsage.inductive_step_hinsage(S, trained_hinsage_model, inductive_data.index, batch_size=5)

from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=100, objective='binary:logistic')

train_labels = train_data['Is Fraud?']

if add_additional_data is True:
    train_emb = pd.merge(train_emb, train_data.loc[train_emb.index].drop('Is Fraud?', axis=1), left_index=True, right_index=True)
    inductive_emb = pd.merge(inductive_emb, inductive_data.loc[inductive_emb.index].drop('Is Fraud?', axis=1), left_index=True, right_index=True)
    
    baseline_train = train_data.drop('Is Fraud?', axis=1)
    baseline_inductive = inductive_data.drop('Is Fraud?', axis=1)

    # Drop the 'card_id' column from the baseline DataFrames
    baseline_train = baseline_train.drop('card_id', axis=1)
    baseline_inductive = baseline_inductive.drop('card_id', axis=1)

    classifier.fit(baseline_train, train_labels)
    baseline_predictions_prob = classifier.predict_proba(baseline_inductive)
    baseline_predictions_pred = classifier.predict(baseline_inductive)

train_emb['card_id'] = train_data["card_id"].factorize()[0]
inductive_emb['card_id'] = inductive_data["card_id"].factorize()[0]

classifier.fit(train_emb, train_labels)
predictions_prob = classifier.predict_proba(inductive_emb)
predictions_pred = classifier.predict(inductive_emb)
    
from inductiveGRL.evaluation import Evaluation
inductive_labels = df.loc[inductive_emb.index]['Is Fraud?']

graphsage_evaluation = Evaluation(predictions_prob, inductive_labels, "GraphSAGE+features") 
graphsage_evaluation.pr_curve()

if add_additional_data is True:
    baseline_evaluation = Evaluation(baseline_predictions_prob, inductive_labels, "Baseline")
    baseline_evaluation.pr_curve()
    
# Calculate metrics
accuracy = accuracy_score(inductive_labels, predictions_pred)
precision = precision_score(inductive_labels, predictions_pred)
recall = recall_score(inductive_labels, predictions_pred)
f1 = f1_score(inductive_labels, predictions_pred)
aucpr = average_precision_score(inductive_labels, predictions_prob[:,1])
roc_auc = roc_auc_score(inductive_labels, predictions_prob[:,1])
cm = confusion_matrix(inductive_labels, predictions_pred)

# Base model results
base_accuracy = accuracy_score(inductive_labels, baseline_predictions_pred)
base_precision = precision_score(inductive_labels, baseline_predictions_pred)
base_recall = recall_score(inductive_labels, baseline_predictions_pred)
base_f1 = f1_score(inductive_labels, baseline_predictions_pred)
base_aucpr = average_precision_score(inductive_labels, baseline_predictions_prob[:,1])
base_roc_auc = roc_auc_score(inductive_labels, baseline_predictions_prob[:,1])
baseline_cm = confusion_matrix(inductive_labels, baseline_predictions_pred)

results = prettytable.PrettyTable(title='GNN (3 Nodes) Results vs XGBoost Baseline')
results.field_names = ["Metric", "GNN (3 Nodes)", "XGBoost Baseline"]
results.add_row(["Accuracy", accuracy, base_accuracy])
results.add_row(["Precision", precision, base_precision])
results.add_row(["Recall", recall, base_recall])
results.add_row(["F1 Score", f1, base_f1])
results.add_row(["ROC AUC", roc_auc, base_roc_auc])
results.add_row(["AUCPR", aucpr, base_aucpr])
print(cm)
print(baseline_cm)
print(results)
