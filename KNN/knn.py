"""
Implementation of k-nearest neighbours classifier
"""

from cProfile import label
from turtle import distance, rt
import numpy as np

import utils
from utils import euclidean_dist_squared


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):

        n = X_hat.shape[0]
        preds = np.zeros(n)

        distances = utils.euclidean_dist_squared(X_hat, self.X)

        for i in range(n):
            k_closest_points = np.argsort(distances[i])[:self.k]
            most_common_label = utils.mode(self.y[k_closest_points])
            preds[i] = most_common_label
        
        return preds