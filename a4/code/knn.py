import numpy as np
from scipy import stats
import utils


class KNN:
    # or just use scikit-learn's...
    def __init__(self, k, X=None, y=None):
        self.k = k
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, Xtest):
        sqdists = utils.euclidean_dist_squared(Xtest, self.X)
        closest_k = np.argpartition(sqdists, self.k, axis=1)[:, : self.k]
        return stats.mode(self.y[closest_k], axis=1, keepdims=False).mode
