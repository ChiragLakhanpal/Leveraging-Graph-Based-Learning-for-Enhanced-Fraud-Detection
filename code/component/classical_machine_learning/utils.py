import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, confusion_matrix
import xgboost as xgb
import prettytable
import matplotlib.pyplot as plt
import category_encoders as ce


def evaluate_model(model, X_test, y_test):

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred)
    aucpr = auc(recall_curve, precision_curve)
    return y_pred, accuracy, precision, recall, f1, cm, roc_auc, aucpr

    
def display_results(model, X_test, y_test, title = None):
    _, accuracy, precision, recall, f1, cm, roc_auc, aucpr = evaluate_model(model, X_test, y_test)
    
    results = prettytable.PrettyTable(title=title)
    results.field_names = ["Metric", "Value"]
    results.add_row(["Accuracy", accuracy])
    results.add_row(["Precision", precision])
    results.add_row(["Recall", recall])
    results.add_row(["F1 Score", f1])
    results.add_row(["ROC AUC", roc_auc])
    results.add_row(["AUCPR", aucpr])
    print(results)
    print("Confusion Matrix")
    print(cm)

def generate_aucpr_plot(y_test, y_prob):
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    aucpr = auc(recall_curve, precision_curve)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, marker='.', label='AUCPR = {:.2f}'.format(aucpr))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    
def dataset_stats(data, target, X_train,y_train,  X_val = None, X_test = None, y_val = None, y_test = None):
    table = prettytable.PrettyTable()

    print("Dataset Stats")

    table.field_names = ["Data", "Rows", "Columns", "Target True", "Target False", "Target %"]
    table.add_row(
        ["Complete Dataset", data.shape[0], data.shape[1], data[target].sum(), data.shape[0] - data[target].sum(),
         f"{round(data[target].sum() / data.shape[0] * 100, 2)}%"])
    
    table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum(),
                   f"{round(y_train.sum() / y_train.shape[0] * 100, 2)}%"])
    if X_test is not None and y_test is not None:

        table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum(),
                    f"{round(y_test.sum() / y_test.shape[0] * 100, 2)}%"])
    
    if X_val is not None and y_val is not None:

        table.add_row(["Validation", X_val.shape[0], X_val.shape[1], y_val.sum(), y_val.shape[0] - y_val.sum(),
                       f"{round(y_val.sum() / y_val.shape[0] * 100, 2)}%"])
    print(table)