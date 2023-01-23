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
        n, d = X.shape
        y_unique_classes = np.unique(y)
        k = len(y_unique_classes)

        W_initial_guess = np.zeros([d, k])
        W_initial_guess = W_initial_guess.flatten()

        W_optimal, f_optimal = find_min(self.loss_and_grad, W_initial_guess, X, y, check_grad=True)
        W_optimal = W_optimal.reshape((d, k))

        self.W = W_optimal


    def loss_and_grad(self, w, X, y):
        n, d = X.shape
        y_unique = np.unique(y)
        k = len(y_unique)

        W = w.reshape((d, k))

        ## Make the M matrix: M is n x k, where M[i, c] = exp(x^i * w_c)
        M = np.exp(X @ W)

        ## Make P and Y
        # P is a n x k matrix, where P[i, c] = P(y^i = c | x^i, W)
        #Y is a n x k matrix, where Y[i, c] = 1 if y^u == c, and 0 if not

        P = np.zeros([n, k])
        Y = np.zeros([n, k])

        for i in range(n):
            sum_of_probabilities_for_x_i = np.sum(M[i,:])
            for c in range(k):
                prob_for_class_c = M[i, c]
                probability_calculated = prob_for_class_c / sum_of_probabilities_for_x_i

                P[i, c] = probability_calculated

                Y[i, c] = 1 if y[i] == c else 0
        
        f_total = 0
        for i in range(n):
            y_i = y[i]
            w_yi_T_x_i = np.log(M[i, y_i])
            inner_sum = np.sum(M[i,:])
            f_total += -1 * w_yi_T_x_i + np.log(inner_sum)

        W_grad = X.T @ (P-Y)
 
        w_grad_flattened = W_grad.flatten()

        return f_total, w_grad_flattened 


    def predict(self, X):
        return np.argmax(X @ self.W, axis = 1)

