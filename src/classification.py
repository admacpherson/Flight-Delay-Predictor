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
from torch.utils.tensorboard import SummaryWriter
import datetime

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

# Convert HHMM to minutes since midnight
X["DEP_HOUR"] = X["CRS_DEP_TIME"] // 100
X["DEP_MINUTE"] = X["CRS_DEP_TIME"] % 100
X["DEP_TIME_SIN"] = np.sin(2 * np.pi * (X["DEP_HOUR"]*60 + X["DEP_MINUTE"]) / (24*60))
X["DEP_TIME_COS"] = np.cos(2 * np.pi * (X["DEP_HOUR"]*60 + X["DEP_MINUTE"]) / (24*60))
X.drop("CRS_DEP_TIME", axis=1, inplace=True)

numeric_features = ["DEP_TIME_SIN", "DEP_TIME_COS", "DAY_OF_WEEK"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Fit preprocessor on full data, then transform
X_processed = preprocessor.fit_transform(X)

# Train/test split

# First split: train+val and test
X_temp, X_test, y_temp, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# Second split: train and val
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)


# Convert to PyTorch tensors
# Convert to tensors
def to_tensor(data):
    return torch.tensor(data.toarray() if hasattr(data, "toarray") else data, dtype=torch.float32)

X_train_tensor = to_tensor(X_train)
X_val_tensor   = to_tensor(X_val)
X_test_tensor  = to_tensor(X_test)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tensor   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Datasets and loaders
train_dataset = FlightDelayDataset(X_train_tensor, y_train_tensor)
val_dataset   = FlightDelayDataset(X_val_tensor, y_val_tensor)
test_dataset  = FlightDelayDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)


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
        # 4 layers x 128 neurons
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

input_dim = X_train_tensor.shape[1]
model = FlightDelayNN(input_dim)

# Handle class imbalance with weighted loss
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Using logits so no sigmoid in final layer

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

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

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric):
        score = val_metric  # e.g. ROC AUC

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


log_dir = f"runs/flight_delay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

# Learning rate settings
early_stopper = EarlyStopping(patience=5)
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = train(model, train_loader, criterion, optimizer, device)

    y_val_proba, y_val_true = evaluate(model, val_loader, device)
    val_auc = roc_auc_score(y_val_true, y_val_proba)

    best_model_state = model.state_dict()

    if val_auc > early_stopper.best_score:
        best_model_state = model.state_dict()  # keep best version

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("AUC/val", val_auc, epoch)

    early_stopper(val_auc)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

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


# Save PyTorch model
model_path = "models/pytorch_flight_delay_model.pth"
torch.save(model.state_dict(), model_path)

# Save preprocessor
import joblib
preprocessor_path = "models/preprocessor.joblib"
joblib.dump(preprocessor, preprocessor_path)

print(f"Saved model to {model_path} and preprocessor to {preprocessor_path}")