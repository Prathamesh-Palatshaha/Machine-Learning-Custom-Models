import math

import numpy as np
class Logistic_Regression():
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
        y_predict = 1/(1 + np.exp( - (self.X.dot(self.weight) + self.cost)))
        # y_predict = self.predict(z)
        dw = (1/self.rows)*np.dot(self.X.T, (y_predict-self.Y))
        db = np.sum(y_predict - self.Y)*(1/self.rows)

        self.weight = self.weight - self.learning_rate * dw
        self.cost = self.cost - self.learning_rate * db


    def predict(self,X):

        return np.where(1/(1 + np.exp( - (X.dot(self.weight) + self.cost))) > 0.5, 1, 0 )
