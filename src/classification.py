import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -- Load and preprocess --

df = pd.read_csv("data/processed/cleaned_T_ONTIME_MARKETING.csv")
df = df[df["CANCELLED"] == 0]
df["ARR_DEL15"] = (df["ARR_DELAY"] > 15).astype(int)

features = ["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "CRS_DEP_TIME"]
target = "ARR_DEL15"

X = df[features].copy()
y = df[target].values

X["FL_DATE"] = pd.to_datetime(X["FL_DATE"])
X["DAY_OF_WEEK"] = X["FL_DATE"].dt.dayofweek
X.drop("FL_DATE", axis=1, inplace=True)

numeric_features = ["CRS_DEP_TIME", "DAY_OF_WEEK"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Fit preprocessor on full data, then transform
X_processed = preprocessor.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray() if hasattr(X_train, "toarray") else X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define Dataset
class FlightDelayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FlightDelayDataset(X_train_tensor, y_train_tensor)
test_dataset = FlightDelayDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the Neural Network
class FlightDelayNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]
model = FlightDelayNN(input_dim)

# Handle class imbalance with weighted loss
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Using logits => no sigmoid in final layer

# Since we want logits in loss, update model last layer to raw output (remove sigmoid)
class FlightDelayNNLogits(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)  # No sigmoid here
        )
    def forward(self, x):
        return self.model(x)

model = FlightDelayNNLogits(input_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    return total_loss / len(dataloader.dataset)

# Evaluation loop
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y_batch.numpy())
    return np.vstack(all_preds), np.vstack(all_targets)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 20
for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Predictions on test set
y_proba, y_true = evaluate(model, test_loader, device)
y_pred = (y_proba >= 0.5).astype(int)

# Metrics
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print("ROC AUC:", roc_auc_score(y_true, y_proba))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"PyTorch NN (AUC = {roc_auc_score(y_true, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()