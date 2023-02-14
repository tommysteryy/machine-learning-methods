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

        # self.n0 = np.sum(y == 0)
        # self.n1 = np.sum(y == 1)
        self.n1 = 0
    
        for i in range(n):
            if y[i] == 1:
                self.n1 += 1

        ## self.p_xy_k = vector of length d, where p_xy_k[d] = Pr(x_j = 1 | y = k)
        self.p_xy_1 = np.zeros(d)
        self.p_xy_0 = np.zeros(d)

        for j in range(d):
            for i in range(n):
                x_i = X[i]
                if (y[i] == 1) and (x_i[j] == 1):
                    self.p_xy_1[j] += 1
                if (y[i] == 0) and (x_i[j] == 1):
                    self.p_xy_0[j] += 1
        
        self.p_xy_1 /= self.n1
        self.p_xy_0 /= (n - self.n1)

        self.p_y = self.n1 / n


    def predict(self, X):
        n, d = X.shape

        preds = -1*np.ones(n) 

        for i in range(n):
            x_i = X[i]

            prob_y_1 = self.p_y
            prob_y_0 = 1 - self.p_y

            for j in range(d):
                if x_i[j] == 1:
                    prob_y_0 *= self.p_xy_0[j]
                    prob_y_1 *= self.p_xy_1[j]
            
            preds[i] = prob_y_1 > prob_y_0
        
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

