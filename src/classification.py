import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load preprocessed dataset (update this path if needed)
df = pd.read_csv("data/processed/cleaned_T_ONTIME_MARKETING.csv")

# Drop cancelled flights
df = df[df["CANCELLED"] == 0]

# Create target variable for classification
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

# Convert FL_DATE to datetime and extract useful parts
X["FL_DATE"] = pd.to_datetime(X["FL_DATE"])
X["DAY_OF_WEEK"] = X["FL_DATE"].dt.dayofweek
X.drop("FL_DATE", axis=1, inplace=True)

# Preprocessing
numeric_features = ["CRS_DEP_TIME", "DAY_OF_WEEK"]
categorical_features = ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Build pipeline
clf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
clf_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf_pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))