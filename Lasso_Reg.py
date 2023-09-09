import numpy as np


class Lasso:
    def __init__(self, iteration, learning_rate, lamb):
        self.iteration = iteration
        self.learning_rate = learning_rate
        self.lamb = lamb

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
        dw = np.zeros(self.features)

        for i in range(self.features):
            if self.weight[i] >0:
                dw[i] = (-(2*(self.X[:,i]).dot(self.Y - y_predict)) + self.lamb) / self.rows
            else:

                dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - y_predict)) - self.lamb) / self.rows
        db = - 2 * np.sum(self.Y - y_predict) / self.rows
        self.weight = self.weight - self.learning_rate * dw

        self.cost = self.cost - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weight) + self.cost

mod = Lasso(1000,0.01,0.01)