import numpy as np


class LinearRegression():
    def __init__(self, iteration, learning_rate):
        self.iteration = iteration
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        self.rows, self.features = X.shape
        self.weight = np.zeros(self.features)
        self.cost = 0
        self.X = X
        self.Y = Y

        for i in range(self.iteration):
            self.gradient_descent()

    def gradient_descent(self):
        y_predict = self.predict(self.X)

        del_weight = -(2 * (self.X.T).dot(self.Y - y_predict)) / self.rows
        del_cost = -2 * np.sum(self.Y - y_predict) / self.rows

        self.weight = self.weight - self.learning_rate * del_weight
        self.cost = self.cost - self.learning_rate * del_cost

    def predict(self, X):
        return X.dot(self.weight) + self.cost
