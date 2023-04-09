import sys

import numpy as np
from scipy import stats

from student_t import MultivariateT


class GDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, Xtest):
        raise NotImplementedError()



class TDA:
    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self.fit(X, y)

    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, Xtest):
        raise NotImplementedError()

