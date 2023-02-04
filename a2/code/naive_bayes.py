import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans


class NaiveNaiveBayes:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        # assumes X and y values are all {0, 1}
        n, d = X.shape

        self.theta_y = np.mean(y)

        # we would fit a Bernoulli for each X variable independently
        # but...there's no need to actually do it,
        # because it's not going to affect our predictions

        # self.p_x = np.mean(X, axis=1)

    def predict(self, X):
        t, d = X.shape

        # in our "naive naive bayes", y is totally independent of X
        # so no reason to even look at the data we get!
        return np.repeat(1 if self.theta_y > 0.5 else 0, t)


class NaiveBayes:
    def __init__(self, laplace_smooth=1, X=None, y=None):
        self.laplace_smooth = laplace_smooth
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        n, d = X.shape

        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()





class VQNB:
    def __init__(self, k, laplace_smooth=1, X=None, y=None):
        self.k = k
        self.laplace_smooth = laplace_smooth
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()

