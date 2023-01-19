import numpy as np
from utils import euclidean_dist_squared


class LeastSquares:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        self.w = np.linalg.solve(X.T @ X, X.T @ y)

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        return X @ self.w




class LeastSquaresBias:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)


    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()



def gaussianRBF_feats(X, bases, sigma):
    # Not mandatory, but might be nicer to implement this separately.
    raise NotImplementedError()



class LeastSquaresRBFL2:
    def __init__(self, X=None, y=None, lam=1, sigma=1):
        self.lam = lam
        self.sigma = sigma
        if X is not None and y is not None:
            self.fit(X, y)


    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()

