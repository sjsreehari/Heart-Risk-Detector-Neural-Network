import numpy as np

def compute_loss(y, y_hat):
    m = y.shape[0]
    loss = - (1/m) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))
    return loss

def accuracy(y, y_hat):
    predictions = (y_hat > 0.5).astype(int)
    return np.mean(predictions == y)
