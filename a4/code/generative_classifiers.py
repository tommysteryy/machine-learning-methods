import sys

import numpy as np
from scipy import stats

from student_t import MultivariateT


class GDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)
            

    def fit(self, X, y):
        self.c = 10
        n, d = X.shape

        self.p = np.zeros(self.c)
        self.mus = np.zeros((self.c, d))
        
        ## self.sigmas[c] = d x d matrix with covariances.
        self.sigmas = np.zeros((self.c, d, d))

        for cls in range(self.c):
            y_c = y[y == cls]
            x_c = X[y == cls]

            n_c = len(y_c)

            p_c = n_c / n
            mu_c = x_c.mean(axis=0)

            sigma_c = np.zeros((d, d))
            for i in range(x_c.shape[0]):
                x_i_c = x_c[i]
                x_i_c_centered = (x_i_c - mu_c).reshape((d, 1))
                sigma_c += (x_i_c_centered) @ (x_i_c_centered).T
            sigma_c = (1/n_c) * sigma_c

            self.p[cls] = p_c
            self.mus[cls] = mu_c
            self.sigmas[cls] = sigma_c

    def nll(self, X):
        """
        if X is n x d, nll(X) returns n x c matrix NLL where:
            NLL[i, c] = - log p(x^i | self.mu_c, self.sigma_c)
        """
        n, d = X.shape
        NLL = np.zeros((n, self.c))

        for i in range(n):

            x_i = X[i].reshape((d, 1))
            nlls_x_i = np.zeros(self.c)

            for cls in range(self.c):
                mu_c = self.mus[cls].reshape((d, 1))
                p_c = self.p[cls]
                sigma_c = self.sigmas[cls]

                nll = (x_i - mu_c).T @ np.linalg.inv(sigma_c) @ (x_i - mu_c) + np.log(np.linalg.det(sigma_c))

                nlls_x_i[cls] = (0.5)*nll - np.log(p_c)

            NLL[i] = nlls_x_i
            
        return NLL
    
    def predict(self, Xtest):
        NLLs = self.nll(Xtest)
        return NLLs.argmin(axis=1)


    
            

        



class TDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        n, _ = X.shape

        self.c = len(np.unique(y))
        self.ps = np.zeros(self.c)
        self.Ts = []
        for cls in range(self.c):
            X_c = X[y == cls]
            n_c = X_c.shape[0]

            self.ps[cls] = n_c / n
            self.Ts.append(MultivariateT(X_c))
            

    def predict(self, Xtest):
        t, _ = Xtest.shape
        probs_all = np.zeros((self.c, t))

        for cls in range(self.c):
            T_dist = self.Ts[cls]
            probs_for_cls = -np.log(self.ps[cls]) + T_dist.log_prob(Xtest)
            probs_all[cls] = probs_for_cls

        # print(probs_all)

        return probs_all.argmin(axis=0)

