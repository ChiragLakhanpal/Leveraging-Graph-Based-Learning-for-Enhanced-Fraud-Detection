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

print("Training and testing data shape: ")
print("Training Shape: ", X_train.shape, y_train.shape)
print("Testing Shape: ", X_test.shape, y_test.shape)


X_train_tensors = torch.reshape(X_train,   (X_train.shape[0], 1, X_train.shape[1]))
X_test_tensors = torch.reshape(X_test,  (X_test.shape[0], 1, X_test.shape[1]))

X_train_tensors.to(device)
X_test_tensors.to(device)

print("Training and testing data shape after adding time dimension for LSTM: ")
print("Training Shape: ", X_train_tensors.shape, y_train.shape)
print("Testing Shape: ", X_test_tensors.shape, y_test.shape)

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device)
        output, (hn, cn) = self.lstm(x.unsqueeze(1), (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = F.relu(hn)
        out = self.fc(out)
        return F.sigmoid(out)


input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 1

model = LSTM1(num_classes, input_size, hidden_size,num_layers, X_train_tensors.shape[1]).to(device)

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
results = prettytable.PrettyTable(title='LSTM Results')
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