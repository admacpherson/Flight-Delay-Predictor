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
import joblib
import os

import seaborn as sns
import io
from PIL import Image
import torchvision.transforms as transforms

# -------- Data Loading and Setup --------

# Load preprocessed data
df = pd.read_csv("data/processed/cleaned_T_ONTIME_MARKETING.csv")

# Specify numeric and categorical features
numeric_features = ["DEP_TIME_SIN", "DEP_TIME_COS", "DAY_OF_WEEK", "ORIGIN_FLIGHT_COUNT", "IS_HOLIDAY"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

# Label encode categorical features to integers (for embedding)
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate numerical & categorical feature matrices and target vector as numpy arrays
X_num = df[numeric_features].values
X_cat = df[categorical_features].values
y = df["ARR_DEL15"].values

# Standardize numeric features to zero mean and unit variance
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)

# Train/val/test split (60/20/20)
X_num_temp, X_num_test, X_cat_temp, X_cat_test, y_temp, y_test = train_test_split(
    X_num, X_cat, y, test_size=0.2, random_state=42
)
X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
    X_num_temp, X_cat_temp, y_temp, test_size=0.25, random_state=42
)

# -------- PyTorch Dataset --------

# Define PyTorch dataset with embeddings
class FlightDelayDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
    # Convert numpy arrays to torch tensors
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

# Create train/test/val datasets
train_dataset = FlightDelayDataset(X_num_train, X_cat_train, y_train)
val_dataset = FlightDelayDataset(X_num_val, X_cat_val, y_val)
test_dataset = FlightDelayDataset(X_num_test, X_cat_test, y_test)

# Create DataLoaders to iterate batches during training and evaluation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# -------- Model Definition --------

# Define neural network with embeddings
class FlightDelayNNEmbeddings(nn.Module):
    def __init__(self, num_numeric_feats, cat_cardinalities, embedding_dims):
        super().__init__()
        # Create embedding layers for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dim)
            for cardinality, dim in zip(cat_cardinalities, embedding_dims)
        ])
        # Sum of all embedding dimensions (to concatenate with numeric features)
        self.embedding_output_dim = sum(embedding_dims)
        # Fully connected NN layers
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
            nn.Linear(32, 1) # Single output for binary classification (logit)
        )

    # Forward pass
    def forward(self, x_num, x_cat):
        # Embed each categorical variable separately
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        # Concatenate embeddings and numerical features
        x = torch.cat(embedded + [x_num], dim=1)
        # Forward pass through fully connected layers
        return self.fc(x)

# Determine number of categories for each categorical feature (for embeddings)
cat_cardinalities = [len(label_encoders[col].classes_) for col in categorical_features]
# Heuristic for embedding dimension size (up to 50, or half the category size)
embedding_dims = [min(50, (card + 1) // 2) for card in cat_cardinalities]
# Number of numeric features (input size)
input_num_feats = X_num_train.shape[1]

# -------- Training and Evaluation --------

# Instantiate model
model = FlightDelayNNEmbeddings(input_num_feats, cat_cardinalities, embedding_dims)

# Choose device (Use GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the device
model.to(device)

# Compute positive class weight to handle class imbalance in loss function
pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], dtype=torch.float32).to(device)
# Use binary cross entropy with logits loss (more stable than sigmoid + BCE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# Adam optimizer with fixed LR
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# Learning rate scheduler that reduces LR on plateau of validation AUC
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# Training loop for each epoch
def train(model, dataloader, criterion, optimizer, device):
    # Set model to training mode
    model.train()
    total_loss = 0
    for X_num_batch, X_cat_batch, y_batch in dataloader:
        # Move batch to device
        X_num_batch, X_cat_batch, y_batch = X_num_batch.to(device), X_cat_batch.to(device), y_batch.to(device)
        # Clear previous gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_num_batch, X_cat_batch)
        # Compute loss
        loss = criterion(outputs, y_batch)
        # Backpropogate gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        # Sum loss weighted by batch size
        total_loss += loss.item() * X_num_batch.size(0)
    # Return average loss
    return total_loss / len(dataloader.dataset)

# Evaluation function for model predictions and true labels
def evaluate(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad(): # Disable gradient calculation
        for X_num_batch, X_cat_batch, y_batch in dataloader:
            X_num_batch, X_cat_batch = X_num_batch.to(device), X_cat_batch.to(device)
            outputs = model(X_num_batch, X_cat_batch)
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y_batch.numpy())
    # Stack all batch predictions and targets vertically
    return np.vstack(all_preds), np.vstack(all_targets)

# -------- Helper functions --------

# Update README with performance metrics
def update_readme_with_metrics(metrics):
    readme_path = "README.md"
    today = datetime.date.today().isoformat()

    # Set up table
    header = (
        "## Model Performance\n\n <i>(Automatically updated during run)<i>"
        "| Date | Accuracy | Precision (0) | Recall (0) | F1 (0) | Precision (1) | Recall (1) | F1 (1) | ROC AUC |\n"
        "|------|----------|----------------|------------|--------|----------------|------------|--------|---------|\n"
    )

    # Create new row with metrics
    new_row = f"| {today} | {metrics['accuracy']:.2f} | {metrics['precision_0']:.2f} | {metrics['recall_0']:.2f} | {metrics['f1_0']:.2f} | {metrics['precision_1']:.2f} | {metrics['recall_1']:.2f} | {metrics['f1_1']:.2f} | {metrics['roc_auc']:.3f} |\n"

    # Iterate through the existing README
    with open(readme_path, "r") as f:
        lines = f.readlines()
    # Find start of model performance section (if already present)
    start_idx = next((i for i, line in enumerate(lines) if line.strip() == "# Results (Auto Generated)"), None)

    # Clear previous entries
    if start_idx is not None:
        lines = lines[:start_idx]

    # Add new section with updated content
    with open(readme_path, "w") as f:
        f.writelines(lines)
        f.write("\n" + header + new_row)

# Early stopping to halt training if validation metric doesn't improve
class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience # How many epochs to wait
        self.delta = delta # Minimum improvement to reset patience
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

# Initialize TensorBoard summary writer with timestamped log directory
log_dir = f"runs/flight_delay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Helper function to create a confusion matrix image for TensorBoard visualization
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    # Save plot to in memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    # Convert PIL image to tensor for TensorBoard
    return transforms.ToTensor()(image)

# -------- Main training pipeline --------

# Number of epochs to train
epochs = 25
# Instantiate early stopping mechanism with patience of 5 epochs
early_stopper = EarlyStopping(patience=5)

# Initialize variable to track best performing model weights
best_model_state = None

for epoch in range(epochs):
    # Train epoch and get average training loss
    train_loss = train(model, train_loader, criterion, optimizer, device)
    # Evaluate on validation set, get predicted probabilities and true labels
    val_proba, val_true = evaluate(model, val_loader, device)
    # Compute validation ROC AUC score
    val_auc = roc_auc_score(val_true, val_proba)

    # Print progress info
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Log training loss and validation AUC to TensorBoard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("AUC/val", val_auc, epoch)

    # Step learning rate scheduler based on validation AUC
    scheduler.step(val_auc)

    # Save model weights if validation AUC improved
    if best_model_state is None or val_auc > early_stopper.best_score:
        best_model_state = model.state_dict()

    # Check early stopping condition and break if triggered
    early_stopper(val_auc)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# Load best model weights after training is complete
model.load_state_dict(best_model_state)

# Evaluate final model on test set
y_proba, y_true = evaluate(model, test_loader, device)

# Flatten arrays for metric calculations
y_true = y_true.ravel()
y_proba = y_proba.ravel()

# Apply threshold to probabilities for predicted classes
y_pred = (y_proba >= 0.2).astype(int)

# Generate classification report dictionary with precision, recall, f1, etc.
report_raw = classification_report(y_true, y_pred, output_dict=True, labels=[0, 1])
# Convert keys to strings
report = {str(k): v for k, v in report_raw.items()}

# Compute confusion matrix and ROC AUC on test data
conf_matrix = confusion_matrix(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_proba)

# Print results
print(conf_matrix)
print(classification_report(y_true, y_pred))
print("Test ROC AUC:", roc_auc)

# Save the trained PyTorch model state dictionary
torch.save(model.state_dict(), "models/pytorch_flight_delay_embeddings.pth")
# Save preprocessing objects for later use (scaler and label encoders)
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(label_encoders, "models/label_encoders.joblib")
print("Saved model and preprocessors.")

# Prepare dictionary of key metrics to update README file
metrics_dict = {
    "accuracy": report["accuracy"],
    "precision_0": report["0"]["precision"],
    "recall_0": report["0"]["recall"],
    "f1_0": report["0"]["f1-score"],
    "precision_1": report["1"]["precision"],
    "recall_1": report["1"]["recall"],
    "f1_1": report["1"]["f1-score"],
    "roc_auc": roc_auc,
}

# Update README with current model performance metrics
update_readme_with_metrics(metrics_dict)

# Generate confusion matrix image tensor and add it to TensorBoard
conf_tensor = plot_confusion_matrix(conf_matrix)
writer.add_image("ConfusionMatrix", conf_tensor)
# Log test accuracy and ROC AUC metrics to TensorBoard
writer.add_scalar("Test/Accuracy", report["accuracy"])
writer.add_scalar("Test/ROC_AUC", roc_auc)
# Log test accuracy and ROC AUC metrics to TensorBoard
writer.close()