import numpy as np


class SVM_classifier:
    def __init__(self, iteration, rate, lamb):
        self.iteration = iteration
        self.rate = rate
        self.lamb = lamb

    def fit(self, X, Y):
        self.rows, self.features = X.shape
        self.weight = np.zeros(self.features)
        self.cost = 0
        self.X = X
        self.Y = Y
        # self.y_cap = np.where(self.Y <= 0, -1, 1)
        for i in range(self.iteration):
            self.gradient_descent()

    def gradient_descent(self):
        y_label = np.where(self.Y <= 0, -1, 1)

        for idx, value in enumerate(self.X):
            if (np.dot(y_label[idx], (np.dot(self.weight, value) - self.cost)) >= 1).any():
                dw = 2 * self.lamb * self.weight
                db = 0
            else:
                dw = 2 * self.lamb * self.weight - np.dot(y_label[idx], value)
                db = y_label[idx]

            self.weight = self.weight - self.rate * dw

            self.cost = self.cost - self.rate * db

    def predict(self, X):
        op = np.dot(X, self.weight) - self.cost
        pred_lables = np.sign(op)
        y_cap = np.where(pred_lables <= -1, 0, 1)

        return y_cap
