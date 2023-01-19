import numpy as np
import matplotlib.pyplot as plt
from utils import plot2Dclusters

from kmeans import KMeans

# If you want, you could write this function to compute pairwise L1 distances
def l1_distances(X1, X2):
    raise NotImplementedError()



class KMedians(KMeans):
    # We can reuse most of the code structure from KMeans, rather than copy-pasting,
    # by just overriding these few methods. Object-orientation!

    def get_assignments(self, X):
        raise NotImplementedError()


    def update_means(self, X, y):
        raise NotImplementedError()


    def loss(self, X, y=None):
        w = self.w
        if y is None:
            y = self.get_assignments(X)

        raise NotImplementedError()

