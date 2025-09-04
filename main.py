import pandas as pd
import numpy as np
from nn.model import HeartRiskNN

# -------------------------
# Load dataset
# -------------------------
file_path = r"C:\Users\Sreehari S J\Desktop\AI Projects\Heart_Risk_Dectector_NN\data\heart_data.csv"
data = pd.read_csv(file_path)

# -------------------------
# Preprocess dataset
# -------------------------
numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# Normalize numeric
data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())

# One-hot encode categorical
data = pd.get_dummies(data, columns=categorical_cols)

# Ensure everything is float32 for numpy
data = data.astype(np.float32)

# Features and target
X = data.drop('target', axis=1).values
y = data['target'].values.reshape(-1,1)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------
# Initialize model
# -------------------------
model = HeartRiskNN(input_size=X_train.shape[1], hidden_sizes=[32,16], lr=0.01)

# -------------------------
# Train model
# -------------------------
print("Training model...")
model.train(X_train, y_train, epochs=2000, batch_size=32)

# -------------------------
# Evaluate
# -------------------------
train_preds = model.predict(X_train)
train_acc = (train_preds == y_train).mean()
print(f"Training Accuracy: {train_acc:.2f}")

test_preds = model.predict(X_test)
test_acc = (test_preds == y_test).mean()
print(f"Test Accuracy: {test_acc:.2f}")

# Optional: show predictions
print("Sample predictions:", test_preds[:10].T)
