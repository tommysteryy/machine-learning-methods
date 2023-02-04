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

        # print(f"X looks like {X}")

        self.n0 = np.sum(y == 0)
        self.n1 = np.sum(y == 1)

        # print(f"Calculated n1 and n0 to be {self.n1} and {self.n0}")

        ## p(y) or theta_y 
        self.theta_y1 = self.n1 / n
        self.theta_y0 = self.n0 / n 
        # print(f"theta_y is {self.theta_y}")

        ## theta_x_1 is a vector of d x 1, where theta_x_j is the P(x_j = 1 | y = 1)
        X_ones = X[y == 1]
        X_zeros = X[y == 0]

        self.theta_x_1 = (np.sum(X_ones, axis = 0)+ self.laplace_smooth )/ (self.n1 + self.laplace_smooth*2)
        self.theta_x_0 = (np.sum(X_zeros, axis = 0) + self.laplace_smooth) / (self.n0 + self.laplace_smooth*2)
        # print(f"theta_x looks like {self.theta_x}")


    def predict(self, X):
        n, d = X.shape

        preds = -1*np.ones(n)

        for i in range(n):
            x_i = X[i]
            p_x_y1 = self.theta_y1 * np.inner(x_i, self.theta_x_1)
            p_x_y0 = self.theta_y0 * np.inner(x_i, self.theta_x_0)

            preds[i] = p_x_y1 > p_x_y0

        return preds



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

