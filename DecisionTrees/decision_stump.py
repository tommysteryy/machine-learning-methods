from operator import index, is_
import numpy as np
import utils


class DecisionStumpEquality:
    """
    This is a decision stump that branches on whether the value of X is
    "almost equal to" some threshold.

    This probably isn't a thing you want to actually do, it's just an example.
    """

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None

    def fit(self, X, y):
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to equate to
                t = np.round(X[i, j])

                # Find most likely class for each split
                is_almost_equal = np.round(X[:, j]) == t
                y_yes_mode = utils.mode(y[is_almost_equal])
                y_no_mode = utils.mode(y[~is_almost_equal])  # ~ is "logical not"

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[np.round(X[:, j]) != t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode

    def predict(self, X):
        n, d = X.shape
        X = np.round(X)

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        y_hat = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] == self.t_best:
                y_hat[i] = self.y_hat_yes
            else:
                y_hat[i] = self.y_hat_no

        return y_hat

class DecisionStumpErrorRate:
    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None
    
    def fit(self, X, y):
        
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
            
                # Choose threshold
                #    Here, I rounded to help with generalisability of my model and
                #    running time for Q6.5. But I recognize that you don't need to.
                t = np.round(X[i,j], 3)

                is_more_than = X[:, j] > t
                y_yes_mode = utils.mode(y[is_more_than])
                y_no_mode = utils.mode(y[~is_more_than])  

                # Make predictions
                y_pred = y_yes_mode * np.ones(n)
                y_pred[X[:, j] <=  t] = y_no_mode

                # Compute error
                errors = np.sum(y_pred != y)

                if errors < minError: # This is the lowest error, store this value
                    minError = errors
                    self.j_best = j
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
                    
    def predict(self, X):

        n,d = X.shape

        if self.j_best is None:
            return self.y_hat_yes * np.ones(n)

        res = np.zeros(n)

        for i in range(n):
            if X[i, self.j_best] > self.t_best:
                res[i] = self.y_hat_yes
            else:
                res[i] = self.y_hat_no
        
        return res




def entropy(p):
    """
    A helper function that computes the entropy of the
    discrete distribution p (stored in a 1D numpy array).
    The elements of p should add up to 1.
    This function ensures lim p-->0 of p log(p) = 0
    which is mathematically true, but numerically results in NaN
    because log(0) returns -Inf.
    """
    plogp = 0 * p  # initialize full of zeros
    plogp[p > 0] = p[p > 0] * np.log(p[p > 0])  # only do the computation when p>0
    return -np.sum(plogp)

    

# class DecisionStumpInfoGain(DecisionStumpErrorRate):
#     # This is not required, but one way to simplify the code is
#     # to have this class inherit from DecisionStumpErrorRate.
#     # Which methods (init, fit, predict) do you need to overwrite?
#     y_hat_yes = None
#     y_hat_no = None
#     j_best = None
#     t_best = None

#     def calcProbForEntropy(self, y):
#         """
#         Takes in 1D array with n unique integers (classes), and returns an array of 
#         size n, that adds to 1, and contains the probability of each different class 
#         within that array.
#         Used to supply into entropy() right after.
#         """
#         n_classes = np.unique(y)

#         count = np.bincount(y, minlength = n_classes.size)
#         total = y.size

#         p_for_entropy = np.zeros(n_classes.size)

#         index_pos = 0
#         for class_i in n_classes:
#             count_of_class_i = count[class_i]
#             p_for_entropy[index_pos] = count_of_class_i/total
#             index_pos += 1
        
#         return p_for_entropy    
    
#     def fit(self, X, y):
        
#         n, d = X.shape

#         # Get an array with the number of 0's, number of 1's, etc.
#         count = np.bincount(y)
#         total = y.size

#         # Get the index of the largest value in count.
#         # Thus, y_mode is the mode (most popular value) of y
#         y_mode = np.argmax(count)

#         self.y_hat_yes = y_mode
#         self.y_hat_no = None
#         self.j_best = None
#         self.t_best = None

#         # If all the labels are the same, no need to split further
#         if np.unique(y).size <= 1:
#             return
        
#         ## Calculating Initial Error
#         minEntropy = entropy(self.calcProbForEntropy(y))

#         # Loop over features looking for the best split
#         for j in range(d):
#             for i in range(n):
#                 # Choose value to split on 
#                 t = np.round(X[i,j], 3)

#                 # Split, and find mode for each split
#                 is_more_than = X[:, j] > t
#                 y_yes = y[is_more_than]
#                 y_no = y[~is_more_than]
#                 y_yes_mode = utils.mode(y_yes)      
#                 y_no_mode = utils.mode(y_no) 

#                 # Compute new entropy
#                 y_yes_entropy = entropy(self.calcProbForEntropy(y_yes))
#                 y_no_entropy = entropy(self.calcProbForEntropy(y_no))

#                 currEntropy = y_yes.size/total * y_yes_entropy + y_no.size/total * y_no_entropy

#                 # Compare to minimum entropy so far
#                 if currEntropy < minEntropy:
#                     minEntropy = currEntropy
#                     self.j_best = j
#                     self.t_best = t
#                     self.y_hat_yes = y_yes_mode
#                     self.y_hat_no = y_no_mode

#     def predict(self, X):
#         return super().predict(X)

class DecisionStumpInfoGain(DecisionStumpErrorRate):

    y_hat_yes = None
    y_hat_no = None
    j_best = None
    t_best = None
    
    # Helper function to make entropy calculations easier
    def calcProbForEntropy(self, y):
        """
        Takes in 1D array with n unique integers (classes), and returns an array of 
        size n, that adds to 1, and contains the probability of each different class 
        within that array.
        Used to supply into entropy() right after.
        """
        n_classes = np.unique(y)

        count = np.bincount(y, minlength = n_classes.size)
        total = y.size

        p_for_entropy = np.zeros(n_classes.size)

        index_pos = 0
        for class_i in n_classes:
            count_of_class_i = count[class_i]
            p_for_entropy[index_pos] = count_of_class_i/total
            index_pos += 1
        
        return p_for_entropy    
    
    def fit(self, X, y):
        
        n, d = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)
        total = y.size

        # Get the index of the largest value in count.
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count)

        self.y_hat_yes = y_mode
        self.y_hat_no = None
        self.j_best = None
        self.t_best = None

        if np.unique(y).size <= 1:
            return
        
        # Calculating Initial Error
        minEntropy = entropy(self.calcProbForEntropy(y))

        # Loop over features looking for the best split
        for j in range(d):
            for i in range(n):
                # Choose value to split on 
                # Same reason to round as Q6.2
                t = np.round(X[i,j], 3)

                # Split, and find mode for each split
                is_more_than = X[:, j] > t
                y_yes = y[is_more_than]
                y_no = y[~is_more_than]
                y_yes_mode = utils.mode(y_yes)      
                y_no_mode = utils.mode(y_no) 

                # Compute new entropy
                y_yes_entropy = entropy(self.calcProbForEntropy(y_yes))
                y_no_entropy = entropy(self.calcProbForEntropy(y_no))

                currEntropy = y_yes.size/total * y_yes_entropy + y_no.size/total * y_no_entropy

                # Compare to minimum entropy so far
                if currEntropy < minEntropy:
                    minEntropy = currEntropy
                    self.j_best = j
                    
                    self.t_best = t
                    self.y_hat_yes = y_yes_mode
                    self.y_hat_no = y_no_mode
        

    def predict(self, X):
        return super().predict(X)
