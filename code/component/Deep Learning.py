import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
from preprocess import read_data, preprocess_data, split_data
import seaborn as sns

data = pl.read_csv('Sampled Dataset.csv')

data = preprocess_data(data)

data = data.to_pandas()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Separate the features (X) and target variable (y)
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.values, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test.values, dtype=torch.float32, device=device)

# Modify the model with additional layers or regularization as needed
class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetectionModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), 
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32), 
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

model = FraudDetectionModel(X_train.shape[1]).to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  


# Train the model
epochs = 1
batch_size = 32
early_stopping_patience = 5
best_val_loss = np.inf
best_val_f1 = 0.0
best_model = None
trial_log = []
train_losses = []
val_losses = []
f1_scores = []

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test.unsqueeze(1)).item()
        y_pred = (val_outputs > 0.5).float()
        y_true = y_test.unsqueeze(1)

        # Metrics
        precision = precision_score(y_true.cpu(), y_pred.cpu())
        recall = recall_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        # Log trial
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

        # Check for the best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = model.state_dict()  # Save the best model

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# To ensure the best model is used for evaluation
model.load_state_dict(best_model)

# Evaluation on test set
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

# Results
print("Confusion Matrix:\n", conf_matrix)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"AUCPR: {aucpr:.4f}")

# After the end of the training loop
plt.figure(figsize=(12, 5))

# Plot Training and Validation Losses
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, epochs+1), val_losses, 'r-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend(loc='best')

# Plot F1 Score
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), f1_scores, 'g-', label='F1 Score')
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
