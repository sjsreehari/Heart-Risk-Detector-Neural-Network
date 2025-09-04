import numpy as np
from nn.activations import relu, relu_derivative, sigmoid, sigmoid_derivative

def forward_layer(X, W, b, activation="relu"):
    Z = np.dot(X, W) + b
    if activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    return Z, A

def backward_layer(dA, Z, A_prev, W, activation="relu"):
    m = A_prev.shape[0]
    if activation == "relu":
        dZ = dA * relu_derivative(Z)
    elif activation == "sigmoid":
        dZ = dA * sigmoid_derivative(dA)
    dW = (1/m) * np.dot(A_prev.T, dZ)
    db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)
    return dA_prev, dW, db
