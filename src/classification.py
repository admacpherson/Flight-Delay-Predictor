import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

# Load preprocessed data
df = pd.read_csv("data/processed/cleaned_T_ONTIME_MARKETING.csv")
df = df[df["CANCELLED"] == 0]
df["ARR_DEL15"] = (df["ARR_DELAY"] > 15).astype(int)

# Feature engineering
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek

# Convert CRS_DEP_TIME (HHMM) to cyclical features
df["DEP_HOUR"] = df["CRS_DEP_TIME"] // 100
df["DEP_MINUTE"] = df["CRS_DEP_TIME"] % 100
df["DEP_TIME_SIN"] = np.sin(2 * np.pi * (df["DEP_HOUR"]*60 + df["DEP_MINUTE"]) / (24*60))
df["DEP_TIME_COS"] = np.cos(2 * np.pi * (df["DEP_HOUR"]*60 + df["DEP_MINUTE"]) / (24*60))

# Select features
numeric_features = ["DEP_TIME_SIN", "DEP_TIME_COS", "DAY_OF_WEEK"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

# Label encode categorical features
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X_num = df[numeric_features].values
X_cat = df[categorical_features].values
y = df["ARR_DEL15"].values

# Scale numeric features
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Train/val/test split
X_num_temp, X_num_test, X_cat_temp, X_cat_test, y_temp, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num_temp, X_cat_temp, y_temp, test_size=0.25, random_state=42
)

# PyTorch dataset with embeddings
class FlightDelayDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

train_dataset = FlightDelayDataset(X_num_train, X_cat_train, y_train)
val_dataset = FlightDelayDataset(X_num_val, X_cat_val, y_val)
test_dataset = FlightDelayDataset(X_num_test, X_cat_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model with embeddings
class FlightDelayNNEmbeddings(nn.Module):
    def __init__(self, num_numeric_feats, cat_cardinalities, embedding_dims):
        super().__init__()
        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(cat_cardinalities, embedding_dims)
        ])
        self.embedding_output_dim = sum(embedding_dims)
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.embedding_output_dim + num_numeric_feats, 128),
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
    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [x_num], dim=1)
        return self.fc(x)

# Cardinalities and embedding dimensions
cat_cardinalities = [len(label_encoders[col].classes_) for col in categorical_features]
embedding_dims = [min(50, (card + 1) // 2) for card in cat_cardinalities]  # heuristic

input_num_feats = X_num_train.shape[1]
model = FlightDelayNNEmbeddings(input_num_feats, cat_cardinalities, embedding_dims)

# Use device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer with class imbalance weight
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

# Training and evaluation functions
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_num_batch, X_cat_batch, y_batch in dataloader:
        X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_num_batch, X_cat_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_num_batch.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_num_batch, X_cat_batch, y_batch in dataloader:
            X_num_batch, X_cat_batch = X_num_batch.to(device), X_cat_batch.to(device)
            outputs = model(X_num_batch, X_cat_batch)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y_batch.numpy())
    return np.vstack(all_preds), np.vstack(all_targets)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# TensorBoard writer
log_dir = f"runs/flight_delay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=log_dir)

epochs = 50
early_stopper = EarlyStopping(patience=5)

best_model_state = None

for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_proba, val_true = evaluate(model, val_loader, device)
    val_auc = roc_auc_score(val_true, val_proba)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("AUC/val", val_auc, epoch)

    scheduler.step(val_auc)

    if best_model_state is None or val_auc > early_stopper.best_score:
        best_model_state = model.state_dict()

    early_stopper(val_auc)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# Load best model weights
model.load_state_dict(best_model_state)

# Test evaluation
y_proba, y_true = evaluate(model, test_loader, device)
y_pred = (y_proba >= 0.5).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print("Test ROC AUC:", roc_auc_score(y_true, y_proba))

# Save model, scaler, label encoders
torch.save(model.state_dict(), "models/pytorch_flight_delay_embeddings.pth")
import joblib
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(label_encoders, "models/label_encoders.joblib")

print("Saved model and preprocessors.")