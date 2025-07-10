# ✈️ PyTorch Flight Delay Predictor

A neural network model built with PyTorch to predict flight delays using historical U.S. airline performance data. The model classifies whether a flight will be delayed at departure based on key operational features.

## 📊 Overview

Delays in commercial aviation have wide-reaching impacts. This project uses supervised learning on real-world flight data to predict delays before they happen. The model is trained on cleaned data from the U.S. Department of Transportation, using PyTorch for model definition and training and TensorBoard for logging.

## 🧠 Features Used

The model uses the following **processed and engineered features** as inputs:

* `DEP_TIME_SIN` and `DEP_TIME_COS`: Sinusoidal encoding of scheduled departure time (from `CRS_DEP_TIME`) to represent time cyclicality
* `DEP_DELAY`: Departure delay in minutes
* `DAY_OF_WEEK`: Day of the week extracted from flight date
* `ORIGIN_FLIGHT_COUNT`: Number of flights from the origin airport on that date (proxy for congestion)
* `IS_HOLIDAY`: Binary flag indicating whether the flight date is a U.S. holiday
* Encoded categorical variables:

  * `OP_UNIQUE_CARRIER` (airline code)
  * `ORIGIN` (origin airport)
  * `DEST` (destination airport)

**Target variable:**

* `ARR_DEL15`: Binary label indicating if arrival delay exceeds 15 minutes (`1 = delayed`, `0 = on time`)

*Note:* Raw columns like `CRS_DEP_TIME`, `FL_DATE`, `OP_UNIQUE_CARRIER`, `ORIGIN`, and `DEST` are transformed and encoded during preprocessing before being fed into the model.

## 🗂️ Project Structure
```text
Flight-Delay-Predictor/
├── data/
│   ├── raw/                # Too large for GitHub - download on user machine
│   └── processed/
│       └── cleaned_T_ONTIME_MARKETING.csv
├── preprocessed.py         # Data cleaning and encoding
├── classification.py       # PyTorch model training & evaluation
├── models/
│   └── model.pt            # Saved PyTorch model
├── tensorboard_logs/
│   └── run1/
├── notebooks/
│   └── exploration.ipynb   # Optional EDA and prototyping
├── requirements.txt
└── README.md
```

## ⚙️ How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess the Data

```bash
python preprocessed.py
```

### 3. Train the Model

```bash
python classification.py
```

### 4. View Training Logs with TensorBoard

```bash
tensorboard --logdir=tensorboard_logs
```

## 📈 Model Performance

The model is evaluated using:

* Accuracy
* Precision, Recall, F1-Score
* Confusion Matrix
* TensorBoard loss/accuracy curves

Class imbalance is addressed using weighted loss functions or resampling techniques.

### Results *(Auto Generated)*

| Date | Accuracy | Precision (0) | Recall (0) | F1 (0) | Precision (1) | Recall (1) | F1 (1) | ROC AUC |
|------|----------|----------------|------------|--------|----------------|------------|--------|---------|
| 2025-07-10 | 0.65 | 0.87 | 0.67 | 0.76 | 0.28 | 0.56 | 0.37 | 0.666 |

Confusion matrix:

```text
[[ TN  FP ]
 [ FN  TP ]]
```

## 🚀 Future Enhancements

* Integrate weather and holiday data
* Add embedding layers for categorical features
* Web dashboard for live predictions

**Data Source:**
[U.S. DOT Bureau of Transportation Statistics](https://www.transtats.bts.gov/)
