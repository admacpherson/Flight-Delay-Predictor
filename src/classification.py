import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Load preprocessed dataset
df = pd.read_csv("data/processed/cleaned_flight_data.csv")

# Drop cancelled flights
df = df[df["CANCELLED"] == 0]

# Create binary target: was arrival delayed > 15 minutes
df["ARR_DEL15"] = (df["ARR_DELAY"] > 15).astype(int)

# Define features and target
features = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME"
]
target = "ARR_DEL15"

X = df[features].copy()
y = df[target]

# Feature engineering on date
X["FL_DATE"] = pd.to_datetime(X["FL_DATE"])
X["DAY_OF_WEEK"] = X["FL_DATE"].dt.dayofweek
X.drop("FL_DATE", axis=1, inplace=True)

# Preprocessing
numeric_features = ["CRS_DEP_TIME", "DAY_OF_WEEK"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# XGBoost classifier with class weight scaling
pos_weight = (y == 0).sum() / (y == 1).sum()  # Balance the positive class
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=pos_weight,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb_clf)
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(clf_pipeline, "models/xgb_classifier.joblib")

# Evaluate
y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc_score(y_test, y_proba):.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Feature importance
xgb_model = clf_pipeline.named_steps["classifier"]
importances = xgb_model.feature_importances_
feature_names = clf_pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
num_names = numeric_features
all_names = np.concatenate((num_names, feature_names))

feat_imp_df = pd.DataFrame({
    "Feature": all_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df.head(20), x="Importance", y="Feature")
plt.title("Top 20 Feature Importances (XGBoost)")
plt.tight_layout()
plt.show()