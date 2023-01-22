import numpy as np
import matplotlib.pyplot as plt
from utils import plot2Dclusters

from kmeans import KMeans

# If you want, you could write this function to compute pairwise L1 distances
def l1_distances(X1, X2):
    ## Got some help from https://sparrow.dev/pairwise-distance-in-numpy/
    ## With X1 dimensions of m x n, and X2 dimensions of p x n,
    ## returns dist matrix with dimension n x p, where dist[i, j] = l1-distance between X1[i:] (row i) with X2[j:] (row j)
    return np.sum(np.abs(X1[:, None, :] - X2[None, :, :]), axis = -1)



class KMedians(KMeans):
    # We can reuse most of the code structure from KMeans, rather than copy-pasting,
    # by just overriding these few methods. Object-orientation!
 
    def get_assignments(self, X):
        D1 = l1_distances(X, self.w)
        return np.argmin(D1, axis=1)


    def update_means(self, X, y):
        for k_i in range(self.k):
            same_k = y == k_i
            self.w[k_i] = np.median(X[same_k], axis = 0)


    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)

        n, self.d = X.shape
        k, d = w.shape
        
        sum_of_l1_distances = 0

        for cluster in range(k):
            cluster_median = w[cluster,]
    
            for example in range(n):
                if y[example] == cluster:
                    sum_of_l1_distances += np.linalg.norm(X[example,] - cluster_median, 1)
        
        return sum_of_l1_distances

