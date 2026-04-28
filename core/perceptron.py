import numpy as np 

class perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.bais = None
        self.weights = None
        self.error_per_epoch = []

    def precict(self, x):
        return 1 / (1 + np.exp(- np.dot(x, self.weights) +  self.bais)))

    def train(self, x):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bais = 0.0 
        for i in range(self.epochs):
            errors = 0 
            for xi, target in zip(x ,y):
                linear_output = np.dot(xi, self.weights) + self.bais
                y_pred = 1 if linear_output >= 0 else 0 
                update = self.lr * (target - y_pred)
                self.weights += update * xi 
                self.bais += update 

