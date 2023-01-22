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
        ## used some tips from 
        ## https://stackoverflow.com/questions/8486294/how-do-i-add-an-extra-column-to-a-numpy-array
        n, d = X.shape
        col_of_ones = np.ones((n, 1))
        X_bias = np.append(col_of_ones, X, axis = 1)
        self.w = np.linalg.solve(X_bias.T @ X_bias, X_bias.T @ y)


    def predict(self, X):
        if self.w is None:
            raise RuntimeError("You must fit the model first!")
        n, d = X.shape
        X_bias = np.append(np.ones((n, 1)), X, axis = 1)
        return X_bias @ self.w



def gaussianRBF_feats(X, bases, sigma):
    # Not mandatory, but might be nicer to implement this separately.
    D2 = euclidean_dist_squared(X, bases)
    return np.exp(-1 * (D2 / 2*sigma*2))



class LeastSquaresRBFL2:
    def __init__(self, X=None, y=None, lam=1, sigma=1):
        self.lam = lam
        self.sigma = sigma
        if X is not None and y is not None:
            self.fit(X, y)


    def fit(self, X, y):
        self.bases = X
        Z = gaussianRBF_feats(X, self.bases, self.sigma)
        n, d = Z.shape
        self.w = np.linalg.solve(Z.T @ Z + self.lam*np.identity(n), Z.T @ y)

    def predict(self, X):
        Z = gaussianRBF_feats(X, self.bases, self.sigma)
        return Z @ self.w

