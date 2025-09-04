import pandas as pd
import numpy as np
from nn.model import HeartRiskNN

# -------------------------
# Load dataset
# -------------------------
file_path = r"C:\Users\Sreehari S J\Desktop\AI Projects\Heart_Risk_Dectector_NN\data\heart_data.csv"
data = pd.read_csv(file_path)

# -------------------------
# Preprocess dataset (train/val/test split BEFORE fitting transforms)
# -------------------------
numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# Set seed and shuffle indices
seed = 42
rng = np.random.default_rng(seed)
indices = np.arange(len(data))
rng.shuffle(indices)

# 70/15/15 split
n_total = len(indices)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

train_df = data.iloc[train_idx].reset_index(drop=True)
val_df = data.iloc[val_idx].reset_index(drop=True)
test_df = data.iloc[test_idx].reset_index(drop=True)

# Fit normalization on train, apply to val/test
train_min = train_df[numeric_cols].min()
train_max = train_df[numeric_cols].max()
epsilon = 1e-8
train_df[numeric_cols] = (train_df[numeric_cols] - train_min) / (train_max - train_min + epsilon)
val_df[numeric_cols] = (val_df[numeric_cols] - train_min) / (train_max - train_min + epsilon)
test_df[numeric_cols] = (test_df[numeric_cols] - train_min) / (train_max - train_min + epsilon)

# One-hot encode using train-fitted columns; align val/test to train columns
train_df = pd.get_dummies(train_df, columns=categorical_cols)
val_df = pd.get_dummies(val_df, columns=categorical_cols)
test_df = pd.get_dummies(test_df, columns=categorical_cols)

# Align columns to train
train_cols = train_df.columns
val_df = val_df.reindex(columns=train_cols, fill_value=0)
test_df = test_df.reindex(columns=train_cols, fill_value=0)

# Ensure everything is float32 for numpy
train_df = train_df.astype(np.float32)
val_df = val_df.astype(np.float32)
test_df = test_df.astype(np.float32)

# Features and target
X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values.reshape(-1,1)
X_val = val_df.drop('target', axis=1).values
y_val = val_df['target'].values.reshape(-1,1)
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values.reshape(-1,1)

# -------------------------
# Initialize model (with regularization and dropout)
# -------------------------
model = HeartRiskNN(
    input_size=X_train.shape[1],
    hidden_sizes=[16,8],
    lr=0.01,
    l2_lambda=5e-3,
    dropout_rate=0.3,
    seed=seed,
)

# -------------------------
# Train model with validation and early stopping
# -------------------------
print("Training model...")
model.train(
    X_train,
    y_train,
    epochs=600,
    batch_size=32,
    X_val=X_val,
    y_val=y_val,
    patience=50,
    verbose_every=100,
)

# -------------------------
# Evaluate
# -------------------------
train_preds = model.predict(X_train)
train_acc = (train_preds == y_train).mean()
print(f"Training Accuracy: {train_acc:.2f}")

val_preds = model.predict(X_val)
val_acc = (val_preds == y_val).mean()
print(f"Validation Accuracy: {val_acc:.2f}")

test_preds = model.predict(X_test)
test_acc = (test_preds == y_test).mean()
print(f"Test Accuracy: {test_acc:.2f}")

# Optional: show predictions
print("Sample predictions:", test_preds[:10].T)
