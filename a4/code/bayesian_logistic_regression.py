from pathlib import Path

import numpy as np
from scipy import stats


class BayesianLogisticRegression:
    def __init__(self, *args, **kwargs):
        if args or kwargs:
            self.sample_weights(*args, **kwargs)

    def sample_weights(self, X, y, n_samples=10000, lam=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.lam = 1
        n, d = X.shape

        samples = np.zeros((n_samples, d))
        w = np.zeros(d)
        log_p = self.log_likelihood(X, y, w, self.lam)
        n_accept = 0

        for i in range(1, n_samples):
            w_hat = w + rng.normal(loc=0, scale=1, size=d)
            log_phat = self.log_likelihood(X, y, w_hat, self.lam)

            log_r = log_phat - log_p
            if np.log(np.random.rand()) < log_r:
                w = w_hat
                n_accept += 1
                log_p = log_phat
                print(f"Accepted sample {i}, acceptance rate = {n_accept/i:%}")
            samples[i, :] = w

        print(f"Done; acceptance rate {n_accept / n_samples:%}")

        return samples


    def log_likelihood(self, X, y, w, lam):
        yXw = y * (X @ w)
        return (
            -1 * np.sum(np.log(1 + np.exp(-yXw)))
            - (lam / 2) * np.linalg.norm(w, ord=2) ** 2
        )

    def plot_weights(self, sample_weights, figname=None):
        import matplotlib.pyplot as plt

        pos = [119, 144, 152, 162, 184, 196]
        neg = [30, 75, 106, 109, 123, 124]
        neu = [79, 167, 182, 213, 222, 255]

        fig, ax = plt.subplots(6, 3, figsize=(8, 8), constrained_layout=True)
        for i, pos in enumerate(pos):
            ax[i, 0].hist(sample_weights[:, pos], bins=50)
            ax[i, 0].set_title("Positive Variable")
        for i, neg in enumerate(neg):
            ax[i, 1].hist(sample_weights[:, neg], bins=50)
            ax[i, 1].set_title("Negative Variable")
        for i, neu in enumerate(neu):
            ax[i, 2].hist(sample_weights[:, neu], bins=50)
            ax[i, 2].set_title("Neutral Variable")

        if figname is None:
            plt.show()
        else:
            fn = f"../figs/{figname}.png"
            fig.savefig(fn, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved to {fn}")
