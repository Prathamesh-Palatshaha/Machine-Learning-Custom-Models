import numpy as np


class SVM():
    def __init__(self, iteration, rate, l):
        self.iteration = iteration
        self.rate = rate
        self.l = l

    def fit(self, X, Y):
        self.rows, self.features = X.shape
        self.weight = np.zeros(self.features)
        self.cost = 0
        self.X = X
        self.Y = Y

        for i in range(self.iteration):
            self.gradient_descent()

    def gradient_descent(self):

        y_cap = np.where(self.Y <= 0, -1, 1)

        if y_cap.dot(self.X.dot(self.weight) - self.cost) >= 1:
            dw = 2 * self.l * self.weight
            db = 0
        else:
            dw = 2 * self.l * self.weight - y_cap.dot(self.weight)
            db = y_cap

        self.weight = self.weight - self.rate * dw
        self.cost = self.cost - self.rate * db

    def predict(self,X):
        op = np.dot(X, self.weight) - self.cost
        pred_lables = np.sign(op)
        y_cap = np.where(pred_lables <=-1, 0, 1)

        return y_cap