import numpy as np
from typing import List, Tuple


class NeuralNetwork:
    """Simple feedforward neural network implementation."""
    
    def __init__(self, layer_sizes: List[int], activation_type: str = "relu", learning_rate: float = 0.01):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of neurons in each layer (e.g., [784, 128, 64, 10])
            activation_type: "relu", "sigmoid", or "tanh"
            learning_rate: Learning rate for optimization
        """
        self.architecture = layer_sizes
        self.activation_type = activation_type
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def activation(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_type == "relu":
            return np.maximum(0, z)
        elif self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_type == "tanh":
            return np.tanh(z)
        return z
    
    def activation_derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation_type == "relu":
            return (z > 0).astype(float)
        elif self.activation_type == "sigmoid":
            return a * (1 - a)
        elif self.activation_type == "tanh":
            return 1 - a ** 2
        return np.ones_like(z)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        self.activations = [X]
        self.z_values = []
        
        a = X
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            self.z_values.append(z)
            a = self.activation(z)
            self.activations.append(a)
        
        return a
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> float:
        """Backward pass and weight update."""
        m = X.shape[0]
        
        # Output layer error
        deltas = [self.activations[-1] - y]
        
        # Backpropagate errors
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.activation_derivative(
                self.z_values[i - 1], self.activations[i]
            )
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
        
        # Compute loss
        loss = np.mean((self.activations[-1] - y) ** 2)
        return loss
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> List[float]:
        """Train the network."""
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                self.forward(X_batch)
                loss = self.backward(X_batch, y_batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / (n_samples // batch_size)
            losses.append(avg_loss)
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate network on test data."""
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        return mse, mae
