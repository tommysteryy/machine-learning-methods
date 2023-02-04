import numpy as np
from utils import euclidean_dist_squared, plot2Dclusters


class KMeans:
    def __init__(self, k, X=None, **kwargs):
        self.k = k
        self.rng = np.random.default_rng()
        if X is not None:
            self.fit(X, **kwargs)

    def get_assignments(self, X):
        D2 = euclidean_dist_squared(X, self.w)
        return np.argmin(D2, axis=1)

    def update_means(self, X, y):
        for k_i in range(self.k):
            matching = y == k_i
            if matching.any():
                self.w[k_i] = X[matching].mean(axis=0)

    def fit(
        self, X, plot=False, log=True, plot_fn=None, change_thresh=0, loss_thresh=None
    ):
        k = self.k
        n, self.d = X.shape
        assert n >= k

        self.w = w = self.rng.choice(X, k, replace=False)

        y = np.zeros(n, dtype=np.int32)
        changes = n
        if loss_thresh is not None:
            old_loss = self.loss(X, y)

        while changes > change_thresh:
            y_old = y
            y = self.get_assignments(X)
            changes = np.sum(y != y_old)
            self.update_means(X, y)

            if plot:
                if self.d == 2:
                    from matplotlib import pyplot as plt

                    plot2Dclusters(X, y, w)
                    plt.pause(1)
                    plt.clf()

            if log:
                print(f"Changes: {changes:>7,}")
                print(f"Loss: {self.loss(X, y):10.1f}")

            if loss_thresh is not None:
                loss = self.loss(X, y)
                if old_loss - loss < loss_thresh:
                    break

        if plot:
            if self.d == 2:
                plot2Dclusters(
                    X, y, w, filename=f"{plot_fn or type(self).__name__.lower()}.png"
                )

    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)
        return np.sum((X - w[y]) ** 2)
