import numpy as np
from optimize import find_min


class SoftmaxClassifier:
    def __init__(self, X, y, add_bias=True, **opt_args):
        self.add_bias = add_bias
        self.opt_args = opt_args

        if X is not None and y is not None:
            self.fit(X, y)


    def fit(self, X, y, add_bias=None, **opt_args):
        # You can assume that:
        #   X.shape == (n, d)
        #   y.shape == (n,)
        #   y's values are integers from 0 to k-1
        if add_bias is not None:
            self.add_bias = add_bias
        opt_args = {**self.opt_args, **opt_args}  # merge dicts, second one overrides

        raise NotImplementedError()


    def loss_and_grad(self, w, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()

