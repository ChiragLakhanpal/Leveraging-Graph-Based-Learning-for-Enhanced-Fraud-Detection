import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from preprocess import read_data, preprocess_data, split_data
import seaborn as sns
import prettytable

seed = 42
np.random.seed(seed)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

data = pl.read_csv('/home/ec2-user/DS_Capstone/Data/Sampled Dataset.csv')

data = preprocess_data(data)

data = data.to_pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

table = prettytable.PrettyTable()

# Display dataset stats
print("Dataset Stats")

table.field_names = ["Data", "Rows", "Columns", "Frauds", "Non-Frauds", "Fraud Percentage"]
table.add_row(
    ["Complete Dataset", data.shape[0], data.shape[1], data['is_fraud'].sum(), data.shape[0] - data['is_fraud'].sum(),
     f"{round(data['is_fraud'].sum() / data.shape[0] * 100, 2)}%"])
table.add_row(["Train", X_train.shape[0], X_train.shape[1], y_train.sum(), y_train.shape[0] - y_train.sum(),
               f"{round(y_train.sum() / y_train.shape[0] * 100, 2)}%"])
table.add_row(["Test", X_test.shape[0], X_test.shape[1], y_test.sum(), y_test.shape[0] - y_test.sum(),
               f"{round(y_test.sum() / y_test.shape[0] * 100, 2)}%"])
print(table)

X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(160, 128)  # Adjust input size based on kernel_size
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)

input_dim = 28
output_dim = 1

model = CNN(input_dim, output_dim).to(device)
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 50
batch_size = 128
best_val_loss = np.inf
best_val_f1 = 0.0
best_model = None
trial_log = []
train_losses = []
val_losses = []
f1_scores = []

# Training 
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        inputs = X_train[i:i + batch_size]
        targets = y_train[i:i + batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    #scheduler.step(f1)

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test.unsqueeze(1)).item()
        y_pred = (val_outputs > 0.5).float()
        y_true = y_test.unsqueeze(1)

        precision = precision_score(y_true.cpu(), y_pred.cpu())
        recall = recall_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        trial_log.append({
            'epoch': epoch + 1,
            'train_loss': total_loss / (X_train.shape[0] // batch_size),
            'val_loss': val_loss,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

        train_losses.append(total_loss / (X_train.shape[0] // batch_size))
        val_losses.append(val_loss)
        f1_scores.append(f1)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = model.state_dict()
            print('Model updated!')

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

model.load_state_dict(best_model)

# Evaluation 
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    y_pred = (test_outputs > 0.5).float().cpu().numpy()
    y_true = y_test.cpu().numpy()

# Metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
aucpr = average_precision_score(y_true, y_pred)

# Print metrics
results = prettytable.PrettyTable(title='MLP Results')
results.field_names = ["Metric", "Value"]
results.add_row(["Accuracy", accuracy])
results.add_row(["Precision", precision])
results.add_row(["Recall", recall])
results.add_row(["F1 Score", f1])
results.add_row(["ROC AUC", roc_auc])
results.add_row(["AUCPR", aucpr])
print(results)

plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend(loc='best')

# Plot F1 Score
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), f1_scores, 'g-', label='F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score over Epochs')
plt.legend(loc='best')

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt=',d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')

# Visualize the ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, marker='.', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Visualize the Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred)
plt.figure(figsize=(8, 8))
plt.plot(recall_curve, precision_curve, marker='.', label=f'Precision-Recall curve (area = {aucpr:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()