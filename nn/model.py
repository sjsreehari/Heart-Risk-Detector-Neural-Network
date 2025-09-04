import numpy as np

# -------------------------
# Activation functions
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# -------------------------
# Neural Network class
# -------------------------
class HeartRiskNN:
    def __init__(self, input_size, hidden_sizes=[32,16], output_size=1, lr=0.01, l2_lambda=0.0, dropout_rate=0.0, seed=42):
        self.lr = lr
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(seed)

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2 / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2 / hidden_sizes[1])
        self.b3 = np.zeros((1, output_size))

    def forward(self, X, training=True):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        # Dropout after first hidden layer
        if training and self.dropout_rate > 0.0:
            self.D1 = (self.rng.random(self.A1.shape) > self.dropout_rate).astype(self.A1.dtype)
            self.A1 = (self.A1 * self.D1) / (1.0 - self.dropout_rate)
        else:
            self.D1 = None

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        # Dropout after second hidden layer
        if training and self.dropout_rate > 0.0:
            self.D2 = (self.rng.random(self.A2.shape) > self.dropout_rate).astype(self.A2.dtype)
            self.A2 = (self.A2 * self.D2) / (1.0 - self.dropout_rate)
        else:
            self.D2 = None

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)

        return self.A3

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        data_loss = -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))
        l2_loss = 0.0
        if self.l2_lambda > 0.0:
            l2_loss = (self.l2_lambda/2.0) * (
                np.sum(self.W1 * self.W1) +
                np.sum(self.W2 * self.W2) +
                np.sum(self.W3 * self.W3)
            ) / y_true.shape[0]
        return data_loss + l2_loss

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        dZ3 = y_pred - y_true
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        # Apply dropout mask gradient for layer 2 if used
        if self.D2 is not None:
            dZ2 = (dZ2 * self.D2) / (1.0 - self.dropout_rate)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        # Apply dropout mask gradient for layer 1 if used
        if self.D1 is not None:
            dZ1 = (dZ1 * self.D1) / (1.0 - self.dropout_rate)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights
        if self.l2_lambda > 0.0:
            self.W3 -= self.lr * (dW3 + (self.l2_lambda/m) * self.W3)
            self.W2 -= self.lr * (dW2 + (self.l2_lambda/m) * self.W2)
            self.W1 -= self.lr * (dW1 + (self.l2_lambda/m) * self.W1)
        else:
            self.W3 -= self.lr * dW3
            self.W2 -= self.lr * dW2
            self.W1 -= self.lr * dW1
        self.b3 -= self.lr * db3
        self.b2 -= self.lr * db2
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=2000, batch_size=32, X_val=None, y_val=None, patience=100, verbose_every=100):
        best_val_loss = np.inf
        best_params = None
        epochs_without_improve = 0
        n = X.shape[0]
        for epoch in range(epochs):
            perm = self.rng.permutation(n)
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch, y_pred)

            # Evaluate
            if (epoch % verbose_every) == 0 or epoch == epochs - 1:
                y_pred_train = self.forward(X, training=False)
                train_loss = self.compute_loss(y, y_pred_train)
                train_acc = ((y_pred_train>0.5)==y).mean()
                if X_val is not None:
                    y_pred_val = self.forward(X_val, training=False)
                    val_loss = self.compute_loss(y_val, y_pred_val)
                    val_acc = ((y_pred_val>0.5)==y_val).mean()
                    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f} | Val Loss={val_loss:.4f}, Acc={val_acc:.2f}")
                else:
                    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.2f}")

            # Early stopping
            if X_val is not None:
                y_pred_val = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, y_pred_val)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_params = (
                        self.W1.copy(), self.b1.copy(),
                        self.W2.copy(), self.b2.copy(),
                        self.W3.copy(), self.b3.copy(),
                    )
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        if best_params is not None:
                            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = best_params
                        print(f"Early stopping at epoch {epoch} (best val loss {best_val_loss:.4f})")
                        break

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)
