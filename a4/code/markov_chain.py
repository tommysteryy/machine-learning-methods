import numpy as np
from scipy import stats


class MarkovChain:
    def __init__(self, p1=None, pt=None):
        if p1 is not None and pt is not None:
            self.fit(p1, pt)

    def fit(self, p1, pt):
        self.p1 = p1
        self.pt = pt

    def sample(self, n_samples, d, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        samples = np.zeros((n_samples, d), dtype=int)
        raise NotImplementedError()

        return samples

    def marginals(self, d):
        M = np.zeros((self.p1.size, d))
        raise NotImplementedError()

        return M

    def mode(self, d):
        raise NotImplementedError()


    # TODO: method here for mc-conditional-exact
