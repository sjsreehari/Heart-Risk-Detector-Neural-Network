import numpy as np
from nn.layers import forward_layer, backward_layer
from nn.utils import compute_loss, accuracy

class HeartRiskNN:
    def __init__(self, input_size, hidden_sizes=[8,6,4], output_size=1, lr=0.01):
        self.lr = lr
        self.sizes = [input_size] + hidden_sizes + [output_size]
        np.random.seed(42)
        
        # Initialize weights and biases
        self.params = {}
        for i in range(len(self.sizes)-1):
            self.params[f"W{i+1}"] = np.random.randn(self.sizes[i], self.sizes[i+1]) * 0.01
            self.params[f"b{i+1}"] = np.zeros((1, self.sizes[i+1]))

    def forward(self, X):
        cache = {"A0": X}
        L = len(self.sizes)-1
        A_prev = X
        for l in range(1, L+1):
            activation = "sigmoid" if l==L else "relu"
            Z, A = forward_layer(A_prev, self.params[f"W{l}"], self.params[f"b{l}"], activation)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A
            A_prev = A
        return A_prev, cache

    def backward(self, y, cache):
        grads = {}
        L = len(self.sizes)-1
        dA = cache[f"A{L}"] - y
        
        for l in reversed(range(1, L+1)):
            activation = "sigmoid" if l==L else "relu"
            dA, dW, db = backward_layer(dA, cache[f"Z{l}"], cache[f"A{l-1}"], self.params[f"W{l}"], activation)
            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db
        return grads

    def update_params(self, grads):
        L = len(self.sizes)-1
        for l in range(1, L+1):
            self.params[f"W{l}"] -= self.lr * grads[f"dW{l}"]
            self.params[f"b{l}"] -= self.lr * grads[f"db{l}"]

    def train(self, X, y, epochs=1000):
        for i in range(epochs):
            y_hat, cache = self.forward(X)
            loss = compute_loss(y, y_hat)
            grads = self.backward(y, cache)
            self.update_params(grads)
            if i % 100 == 0:
                acc = accuracy(y, y_hat)
                print(f"Epoch {i}: Loss={loss:.4f}, Accuracy={acc:.2f}")

    def predict(self, X):
        y_hat, _ = self.forward(X)
        return (y_hat > 0.5).astype(int)
