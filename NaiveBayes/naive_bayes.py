import numpy as np


class NaiveBayes:
    """
    Naive Bayes implementation.
    Assumes the feature are binary.
    Also assumes the labels go from 0,1,...k-1
    """

    p_y = None
    p_xy = None

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def fit(self, X, y):
        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        
        p_xy = np.ones((d, k))

        for cls in range(k):
            rows_of_class_k = y == cls
            class_k_domain = X[rows_of_class_k]
            num_examples_in_class_k = class_k_domain.shape[0]
            
            for ft in range(d):
                feature_values_count = np.bincount(class_k_domain[:,ft], minlength=2)
                p_xy[ft, cls] = feature_values_count[1]/num_examples_in_class_k

                # print(f"Examining feature = {ft} in y = {cls}:\n")
                # print(f"There are {feature_values_count[1]} total occurances of this feature in this column")
                # print(f"There are  {feature_values_count[0]} examples without this feature")
                # print(f"resulting in probability = {feature_values_count[1]/num_examples_in_class_k}")


    


        self.p_y = p_y
        self.p_xy = p_xy

    def predict(self, X):
        n, d = X.shape
        k = self.num_classes
        p_xy = self.p_xy
        p_y = self.p_y

        y_pred = np.zeros(n)
        for i in range(n):

            probs = p_y.copy()  # initialize with the p(y) terms
            for j in range(d):
                if X[i, j] != 0:
                    probs *= p_xy[j, :]
                else:
                    probs *= 1 - p_xy[j, :]

            y_pred[i] = np.argmax(probs)

        return y_pred


class NaiveBayesLaplace(NaiveBayes):
    def __init__(self, num_classes, beta=0):
        super().__init__(num_classes)
        self.beta = beta

    def fit(self, X, y):
        """YOUR CODE FOR Q3.4"""

        n, d = X.shape

        # Compute the number of class labels
        k = self.num_classes

        # Compute the probability of each class i.e p(y==c), aka "baseline -ness"
        counts = np.bincount(y)
        p_y = counts / n

        """YOUR CODE HERE FOR Q3.3"""

        # Compute the conditional probabilities i.e.
        # p(x_ij=1 | y_i==c) as p_xy[j, c]
        # p(x_ij=0 | y_i==c) as 1 - p_xy[j, c]
        
        p_xy = np.ones((d, k))

        for cls in range(k):
            rows_of_class_k = y == cls
            class_k_domain = X[rows_of_class_k]
            num_examples_in_class_k = class_k_domain.shape[0]
            
            for ft in range(d):
                feature_values_count = np.bincount(class_k_domain[:,ft], minlength=2)
                # if ((feature_values_count[1] == 0) or (feature_values_count[0] == 0) ):
                #     print(f"Feature {ft} for y = {cls} is all the same value")
                p_xy[ft, cls] = (feature_values_count[1] + self.beta) / (num_examples_in_class_k + k*self.beta)

        self.p_y = p_y
        self.p_xy = p_xy
