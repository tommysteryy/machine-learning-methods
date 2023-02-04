import numpy as np
from optimize import find_min


class LogisticRegression:
    def __init__(self, X=None, y=None, add_bias=True, **opt_args):
        self.add_bias = add_bias
        self.opt_args = opt_args
        if X is not None:
            if y is not None:
                self.fit(X, y)

    def featurize(self, X):
        if self.add_bias:
            bias_column = np.ones((X.shape[0], 1))
            return np.concatenate((bias_column, X), axis=1)
        return X

    def center(self, y):
        y = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        return y

    def uncenter(self, yhat):
        yhat = (yhat + 1) * (self.y_max - self.y_min) / 2 + self.y_min
        return yhat

    def fit(self, X, y, add_bias=None, **opt_args):
        if add_bias is not None:
            self.add_bias = add_bias
        opt_args = {**(self.opt_args), **opt_args}
        X = self.featurize(X)
        n, d = X.shape
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        y = self.center(y)
        w_start = np.zeros(d)
        self.w, _ = find_min(
            (lambda w: self.loss_and_grad(w, X, y)), w_start, **opt_args
        )

    def loss_and_grad(self, w, X, y):
        yXw = y * (X @ w)
        f = np.logaddexp(0, -yXw).sum()
        with np.errstate(over="ignore"):
            g_bits = -y / (1 + np.exp(yXw))
        g = X.T @ g_bits
        return (f, g)

    def decision_function(self, X):
        return self.featurize(X) @ self.w

    def predict(self, X):
        return self.uncenter(np.sign(self.decision_function(X)))


class OneVsRestClassifier:
    def __init__(self, classifier_cls, X=None, y=None):
        self.classifier_cls = classifier_cls
        if X is not None:
            if y is not None:
                self.fit(X, y)

    def fit(self, X, y):
        self.n_classes = np.max(y) + 1
        self.classifiers = [
            self.classifier_cls(X, (y == c).astype(np.int32))
            for c in range(self.n_classes)
        ]

    def predict(self, X):
        decs = np.array([clf.decision_function(X) for clf in self.classifiers])
        return np.argmax(decs, axis=0)
