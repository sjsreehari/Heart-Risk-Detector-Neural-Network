# Heart Risk Detector Neural Network

A feed-forward neural network implemented from scratch using NumPy to predict heart disease risk from tabular clinical data.

## Features
- NumPy-only forward/backward propagation
- ReLU hidden layers, sigmoid output
- Mini-batch training with cross-entropy loss
- L2 weight decay and dropout
- Early stopping with validation monitoring
- Robust preprocessing: randomized train/val/test split, train-fitted normalization and one-hot alignment

## How it works (Backpropagation and training details)
- **Architecture**: Input → Dense(16, ReLU) → Dropout → Dense(8, ReLU) → Dropout → Dense(1, Sigmoid)
- **Forward pass**: Matrix multiplies with biases (`Z = A_prev @ W + b`), ReLU in hidden layers, sigmoid for output probabilities.
- **Loss**: Binary cross-entropy with optional L2 penalty: `BCE + (λ/2m) * (||W1||² + ||W2||² + ||W3||²)`.
- **Backpropagation**:
  - Output layer: `dZ3 = A3 - y`, `dW3 = A2ᵀ @ dZ3 / m`, `db3 = sum(dZ3)/m`.
  - Hidden layers: propagate gradients through ReLU using its derivative; if dropout is active, mask/scaled gradients are applied consistently to the corresponding layers.
- **Optimizer**: Vanilla SGD with learning rate `lr` (no momentum/Adam for clarity).
- **Initialization**: He/Xavier-style scaling (`np.sqrt(2/fan_in)`) for stable gradients.
- **Mini-batching**: Shuffles each epoch, trains on batches for efficiency and noise injection.
- **Regularization**: L2 weight decay on weights, inverted-dropout during training; dropout disabled during evaluation.
- **Early stopping**: Tracks validation loss; restores best weights after `patience` epochs without improvement.
- **Prediction**: Thresholds sigmoid probabilities at 0.5 to produce class labels.

## Project structure
```
Heart_Risk_Dectector_NN/
  data/
    heart_data.csv
  nn/
    activations.py
    layers.py
    model.py
    __init__.py
  main.py
  README.md
```

## Requirements
- Python 3.9+
- numpy
- pandas

Install with:
```bash
pip install numpy pandas
```

## Dataset
Place `heart_data.csv` under `data/`. The script expects the target column named `target`, numeric columns `['age','trestbps','chol','thalach','oldpeak']`, and categorical columns `['sex','cp','fbs','restecg','exang','slope','ca','thal']`.

## How to run
From the project root:
```bash
python main.py
```
This will:
- Randomly split data into 70% train, 15% validation, 15% test (seed=42)
- Normalize numeric features using train stats only and align one-hot columns
- Train `HeartRiskNN` with L2, dropout, and early stopping
- Print train/val/test accuracies and sample predictions

## Key model settings (in `main.py`)
```python
model = HeartRiskNN(
    input_size=X_train.shape[1],
    hidden_sizes=[16,8],
    lr=0.01,
    l2_lambda=5e-3,
    dropout_rate=0.3,
    seed=42,
)

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
```

## Overfitting fixes applied
- Randomized train/val/test split with fixed seed
- Train-set-only normalization and one-hot alignment
- L2 regularization and dropout on hidden layers
- Early stopping on validation loss
- Reduced capacity (hidden sizes `[16, 8]`)

## Example results
Example run (will vary by split/seed):
```
Training model...
Epoch 0: Train Loss=0.72, Acc=0.50 | Val Loss=0.70, Acc=0.52
...
Training Accuracy: ~0.86
Validation Accuracy: ~0.85
Test Accuracy: ~0.89
```

## Reproducibility
We set `seed=42` for data split and NumPy RNG used by dropout.

## Notes
- Adjust `numeric_cols`/`categorical_cols` in `main.py` if your CSV schema differs.
