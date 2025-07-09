import pandas as pd
import numpy as np
import joblib
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ======================
# Load and preprocess data
# ======================

def load_data():
    df = pd.read_csv("data/processed.csv")
    df = df.drop(columns=["flight_date", "airline", "origin_airport", "destination_airport", "tail_number", "wheels_on", "actual_elapsed_time", "wheels_off"])

    # Drop rows with missing target
    df = df.dropna(subset=["delayed"])
    y = df["delayed"]
    X = df.drop(columns=["delayed"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = encoder.fit_transform(X[categorical_cols])
    X_cat = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X[numerical_cols]), columns=numerical_cols)

    X_processed = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)

    # Save preprocessor
    preprocessor = {"encoder": encoder, "scaler": scaler, "columns": X_processed.columns}

    return X_processed, y, preprocessor

# ======================
# Dataset
# ======================
class FlightDelayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ======================
# Model
# ======================
class FlightDelayNN(nn.Module):
    def __init__(self, input_dim):
        super(FlightDelayNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ======================
# EarlyStopping
# ======================
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ======================
# Evaluate
# ======================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    return np.array(y_true), np.array(y_pred)

# ======================
# Training Script
# ======================
def main():
    # Load data
    X, y, preprocessor = load_data()

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val.values,   dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test.values,  dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_tensor   = torch.tensor(y_val.values,   dtype=torch.float32).unsqueeze(1)
    y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(FlightDelayDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    val_loader   = DataLoader(FlightDelayDataset(X_val_tensor,   y_val_tensor),   batch_size=64)
    test_loader  = DataLoader(FlightDelayDataset(X_test_tensor,  y_test_tensor),  batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlightDelayNN(X_train.shape[1]).to(device)

    pos_weight = torch.tensor([(y_train_tensor == 0).sum() / (y_train_tensor == 1).sum()], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopper = EarlyStopping(patience=3)

    writer = SummaryWriter(log_dir="runs/flight_delay")
    best_model_state = None

    for epoch in range(1, 21):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/20"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        y_val_true, y_val_proba = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(y_val_true, y_val_proba)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val AUC = {val_auc:.4f}")

        if val_auc > (early_stopper.best_score or 0):
            best_model_state = model.state_dict()
        early_stopper.step(val_auc)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # Test set evaluation
    y_test_true, y_test_proba = evaluate(model, test_loader, device)
    y_test_pred = (y_test_proba > 0.5).astype(int)

    print(confusion_matrix(y_test_true, y_test_pred))
    print(classification_report(y_test_true, y_test_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test_true, y_test_proba))

    # Save model and preprocessor
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/flight_delay_nn.pth")
    joblib.dump(preprocessor, "model/preprocessor.pkl")
    writer.close()

if __name__ == "__main__":
    main()