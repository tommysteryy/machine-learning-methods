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

        self.n0 = (n - self.n1)

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
        
        ## Add laplace smoothing in TWO steps because the vectorized operations messed up my vectors
        self.p_xy_1 /= (self.n1 + self.laplace_smooth)
        self.p_xy_1 += self.laplace_smooth / (self.n1 + self.laplace_smooth)

        ## Add laplace smoothing in TWO steps because the vectorized operations messed up my vectors
        self.p_xy_0 /= (self.n0 + self.laplace_smooth)
        self.p_xy_0 += self.laplace_smooth / (self.n0 + self.laplace_smooth)

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

        n, d = X.shape

        ## Fit KMeans and get assignments z^i for each x^i
        kmeans = KMeans(self.k, X)
        self.z = kmeans.get_assignments(X)
        print(self.z)

        ## calculate p_ys
        self.n1 = 0

        for i in range(n):
            if y[i] == 1:
                self.n1 += 1
        self.n0 = n - self.n1

        self.p_y1 = (self.n1 + self.laplace_smooth) / (n + 2*self.laplace_smooth)
        self.p_y0 = (self.n0 + self.laplace_smooth) / (n + 2*self.laplace_smooth)
    
        ## calculate p_z_y0 and p_z_y1
        self.p_z_y0 = np.zeros(self.k)
        self.p_z_y1 = np.zeros(self.k)

        for i in range(n):
            if (y[i] == 1):
                self.p_z_y1[self.z[i]] += 1
            elif (y[i] == 0):
                self.p_z_y0[self.z[i]] += 1


        self.p_z_y0 /= (self.n0 + self.laplace_smooth)
        self.p_z_y0 += self.laplace_smooth / (self.n0 + self.laplace_smooth)
        self.p_z_y1 /= (self.n1 + self.laplace_smooth)
        self.p_z_y1 += self.laplace_smooth / (self.n1 + self.laplace_smooth)

        ## calculate p_xj_yz
        self.p_xj_yz = np.zeros((d, 2, self.k))

        ## calculate conditional counts across y and z
        self.y_z = np.zeros((2, self.k))
        
        for i in range(n):
            x_i = X[i]
            for j in range(d):
                if (x_i[j] == 1):
                    self.p_xj_yz[j, y[i], self.z[i]] += 1 
            
            self.y_z[y[i], self.z[i]] += 1

        ## change p_xj_yz from counts to probabilities
        for j in range(d):
            self.p_xj_yz[j] = np.divide(self.p_xj_yz[j], self.y_z)


    def predict(self, X):

        n, d = X.shape

        ## need to calculate both p(x, y = 0) and p (x, y = 1) and directly compare them.

        ## Hence, we need:
        # p_y1: p(y = 1)
        # p_y0: p(y = 0)

        # p_z_y0: a k x 1 array A where A[k] p(k | y = 0)
        # p_z_y1: a k x 1 array B where B[k] p(k | y = 1)

        # p_xj_yz: d x 2 x k array A where A[j, y, z] = p(x_j | y, z)

        preds = -1*np.ones(n)

        ## Calculate:
        # p_x_0 = p(x, y=0)
        # p_x_1 = p(x, y=1)
        for i in range(n):

            x_i = X[i]

            p_x_0 = 0
            p_x_1 = 0

            for K in range(self.k):

                p_k_y0 = self.p_z_y0[K]
                p_k_y1 = self.p_z_y1[K]

                for j in range(d):
                    if x_i[j] == 1:
                        p_k_y0 *= self.p_xj_yz[j, 0, K]
                        p_k_y1 *= self.p_xj_yz[j, 1, K]
                    
                p_x_0 += p_k_y0
                p_x_1 += p_k_y1
            
            p_x_0 *= self.p_y0
            p_x_1 *= self.p_y1

            preds[i] = p_x_1 > p_x_0

        return preds

