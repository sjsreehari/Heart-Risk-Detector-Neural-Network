import numpy as np
from nn.model import HeartRiskNN


# Synthetic Data
np.random.seed(42)
X = np.random.rand(100, 5)
y = (np.sum(X, axis=1) > 2.5).astype(int).reshape(-1, 1)


# Train the model
model = HeartRiskNN(input_size=5)
model.train(X, y, epochs=1000)


# Test Predictions
predictions = model.predict(X)
accuracy = np.mean(predictions == y)
print("Final Accuracy:", accuracy)
print("Predictions:\n", predictions.T)
