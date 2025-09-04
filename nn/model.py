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
    def __init__(self, input_size, hidden_sizes=[32,16], output_size=1, lr=0.01):
        self.lr = lr
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2 / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2 / hidden_sizes[1])
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)

        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)

        return self.A3

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        loss = -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))
        return loss

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        dZ3 = y_pred - y_true
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=2000, batch_size=32):
        for epoch in range(epochs):
            # Mini-batch gradient descent
            perm = np.random.permutation(X.shape[0])
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, y_pred)

            if epoch % 100 == 0:
                y_pred_all = self.forward(X)
                loss = self.compute_loss(y, y_pred_all)
                acc = ((y_pred_all>0.5)==y).mean()
                print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.2f}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)
