import numpy as np
from utils import euclidean_dist_squared, plot2Dclusters


class KMeans:
    def __init__(self, X, k, plot=False, log=True):
        self.rng = np.random.default_rng()
        if X is not None:
            self.fit(X, k, plot=plot, log=log)

    def get_assignments(self, X):
        D2 = euclidean_dist_squared(X, self.w)
        # D2[np.isnan(D2)] = np.inf  # shouldn't be necessary tbh
        return np.argmin(D2, axis=1)

    def update_means(self, X, y):
        for k_i in range(self.k):
            matching = y == k_i
            if matching.any():
                self.w[k_i] = X[matching].mean(axis=0)

    def fit(self, X, k, plot=False, log=True, plot_fn=None):
        self.k = k
        n, self.d = X.shape
        assert n >= k

        self.w = w = self.rng.choice(X, k, replace=False)  # k by d
        y = np.zeros(n)
        changes = n

        while changes != 0:
            y_old = y
            y = self.get_assignments(X)
            changes = np.sum(y != y_old)

            self.update_means(X, y)

            if plot and self.d == 2:
                from matplotlib import pyplot as plt
                plot2Dclusters(X, y, w)
                plt.pause(1)
                plt.clf()

            if log:
                print(f"Changes: {changes:>7,}")
                print(f"Logging the loss: {self.loss(X, y)}")

        if plot and self.d == 2:
            plot2Dclusters(
                X, y, w, filename=f"{plot_fn or type(self).__name__.lower()}.png"
            )

    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)

        n, self.d = X.shape
        k, d = w.shape
        
        sum_of_squared_distances = 0

        for cluster in range(k):
            cluster_mean = w[cluster,]
    
            for example in range(n):
                if y[example] == cluster:
                    sum_of_squared_distances += np.linalg.norm(X[example,] - cluster_mean)
        
        return sum_of_squared_distances

        



