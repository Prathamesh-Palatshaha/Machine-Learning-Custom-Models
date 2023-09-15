import statistics

import numpy as np


class KNN:
    def __init__(self, metric):
        self.metric = metric

    def distance(self, training_data_point, test_dp):
        dist = 0
        if self.metric == 'eucledian':
            for i in range(len(training_data_point) - 1):
                dist += (training_data_point[i] - test_dp[i]) ** 2
            eucld = np.sqrt(dist)
            return eucld
        elif self.metric == 'manhattan':
            for i in range(len(training_data_point) - 1):
                dist += abs(training_data_point[i] - test_dp[i])
            man = dist
            return man

    def nearest_neighbour(self, X_train, test_data, k):
        neigh_dist_list = []

        for train_d in X_train:
            distance = self.distance(train_d, test_data)
            neigh_dist_list.append((train_d, distance))

        neigh_dist_list.sort(key=lambda x: x[1])
        neigh_list = []
        for i in range(k):
            neigh_list.append(neigh_dist_list[i][0])

        return neigh_list

    def predict(self, X_train, test_data,k):

        neighbours = self.nearest_neighbour(X_train, test_data, k)

        label = []
        for data in neighbours:
            label.append(data[-1])

        predicted_class = statistics.mode(label)

        return predicted_class


