import pandas as pd
import numpy as np
import joblib
import os
import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Load fully preprocessed CSV
df = pd.read_csv("data/processed/cleaned_T_ONTIME_MARKETING.csv")

# Select features and target (already engineered)
features = [
    "DAY_OF_WEEK", "DEP_TIME_SIN", "DEP_TIME_COS",
    "OP_UNIQUE_CARRIER", "ORIGIN", "DEST"
]
target = "ARR_DEL15"

X = df[features]
y = df[target].values

numeric_features = ["DAY_OF_WEEK", "DEP_TIME_SIN", "DEP_TIME_COS"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_processed = preprocessor.fit_transform(X)

# Train/Val/Test split: 60% train, 20% val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

def to_tensor(data):
    return torch.tensor(data.toarray() if hasattr(data, "toarray") else data, dtype=torch.float32)

X_train_tensor = to_tensor(X_train)
X_val_tensor = to_tensor(X_val)
X_test_tensor = to_tensor(X_test)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class FlightDelayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(FlightDelayDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(FlightDelayDataset(X_val_tensor, y_val_tensor), batch_size=64)
test_loader = DataLoader(FlightDelayDataset(X_test_tensor, y_test_tensor), batch_size=64)

class FlightDelayNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train_tensor.shape[1]
model = FlightDelayNN(input_dim).to(device)

pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopper = EarlyStopping(patience=5)

log_dir = f"runs/flight_delay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds.append(probs)
            targets.append(y_batch.cpu().numpy())
    return np.vstack(preds), np.vstack(targets)

best_model_state = None
epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
    train_loss = running_loss / len(train_loader.dataset)

    y_val_proba, y_val_true = evaluate(model, val_loader)
    val_auc = roc_auc_score(y_val_true, y_val_proba)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("AUC/val", val_auc, epoch)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    if val_auc > (early_stopper.best_score or 0):
        best_model_state = model.state_dict()

    early_stopper(val_auc)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_model_state)

y_proba, y_true = evaluate(model, test_loader)
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print("Test ROC AUC:", roc_auc_score(y_true, y_proba))

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/pytorch_flight_delay_model.pth")
joblib.dump(preprocessor, "models/preprocessor.joblib")

print("Model and preprocessor saved.")